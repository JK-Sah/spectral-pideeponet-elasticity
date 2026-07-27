#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
route_b.py

Route-B training/evaluation: the body-force-AND-material-field to displacement
operator  (f, E(x)) -> u  for 2D linear elasticity with a spatially varying
Young's modulus (fixed Poisson ratio), homogeneous Dirichlet BCs.

Models
------
* SpectralPIDeepONetE (hero): the paper's exact-BC spectral DeepONet, extended
  so the branch ingests BOTH the sine-projection of f AND a cosine-projection
  of log E(x) (cosine so the constant/mean-stiffness mode is retained).  The
  tensor-product sine trunk still makes u=0 on the boundary exact for every
  weight.  The operator is now nonlinear in the input (u ~ K(E)^{-1} f), which
  the branch MLP supplies.
* FNOEfield (strong baseline): a Fourier neural operator taking (fx,fy,E,x,y)
  as a 5-channel field input; no structural BC enforcement.

Physics
-------
The physics-informed loss uses the *variable-coefficient* Navier-Cauchy
residual: stress is formed pointwise from nodal lambda(x),mu(x) and then its
divergence is taken by finite differences, so coefficient variation is captured
without needing explicit gradients of lambda,mu.

Usage
-----
    python route_b.py --data data/hetero_field.npz --model spectral_e --w_pde 1e-4
    python route_b.py --data data/hetero_field.npz --model fno_e     --w_pde 0
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cmame_extended_study import (
    PhysicsConfig, make_grid, build_modes, sine_basis, project_onto_sine,
    fd_first_derivatives, MLP, SpectralConv2d, relative_l2, set_seed, get_device,
)

NU = 0.30


# ----------------------------------------------------------------------
# Cosine featurization of the (log) material field
# ----------------------------------------------------------------------
def build_cos_modes(P):
    cands = [(i, j) for i in range(0, 8) for j in range(0, 8)]
    cands = sorted(cands, key=lambda t: (t[0] ** 2 + t[1] ** 2, t[0], t[1]))
    return cands[:P]                      # (0,0) first -> captures the mean


def cosine_basis(grid, modes):
    x = grid[..., 0:1]; y = grid[..., 1:2]
    feats = [torch.cos(i * math.pi * x) * torch.cos(j * math.pi * y)
             for (i, j) in modes]
    return torch.cat(feats, dim=-1)       # [res,res,P]


def project_onto_cosine(field, modes, grid):
    """field [B,res,res] -> [B,P] trapezoid inner products with cosine modes."""
    psi = cosine_basis(grid, modes).to(field.device)          # [res,res,P]
    res = field.shape[1]; h = 1.0 / (res - 1)
    w = torch.ones(res, device=field.device); w[0] = w[-1] = 0.5
    w2 = (w[:, None] * w[None, :])[None, :, :, None] * h * h    # [1,res,res,1]
    return torch.sum(w2 * field[..., None] * psi[None], dim=(1, 2))


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
class SpectralPIDeepONetE(nn.Module):
    """Exact-BC spectral DeepONet whose branch ingests f and log E(x)."""
    def __init__(self, modes, cos_modes, phys, hidden=192, depth=4):
        super().__init__()
        self.modes = modes
        self.cos_modes = cos_modes
        self.M = len(modes)
        self.P = len(cos_modes)
        self.branch = MLP(2 * self.M + self.P, 2 * self.M, hidden, depth)
        grid = make_grid(phys.res)
        self.register_buffer("grid", grid)
        self.register_buffer("phi", sine_basis(grid, modes))

    def forward(self, f_grid, E):
        feat_f = project_onto_sine(f_grid, self.modes, self.grid)
        logE = torch.log(E.clamp_min(1e-6))
        feat_E = project_onto_cosine(logE, self.cos_modes, self.grid)
        feat = torch.cat([feat_f, feat_E], dim=-1)
        coeff = self.branch(feat)
        cx, cy = coeff[:, :self.M], coeff[:, self.M:]
        ux = torch.einsum("bm,ijm->bij", cx, self.phi)
        uy = torch.einsum("bm,ijm->bij", cy, self.phi)
        return torch.stack([ux, uy], dim=-1)


class FNOEfield(nn.Module):
    """FNO over (fx, fy, E, x, y) -> (ux, uy). No structural BC."""
    def __init__(self, phys, modes=12, width=32, n_layers=4):
        super().__init__()
        self.phys = phys
        mm = min(modes, phys.res // 2)
        self.fc0 = nn.Linear(5, width)
        self.convs = nn.ModuleList(
            [SpectralConv2d(width, width, mm, mm) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 2)
        self.register_buffer("grid", make_grid(phys.res))

    def forward(self, f_grid, E):
        B = f_grid.shape[0]
        g = self.grid.unsqueeze(0).expand(B, -1, -1, -1)
        x = torch.cat([f_grid, E.unsqueeze(-1), g], dim=-1)     # (B,H,W,5)
        x = self.fc0(x).permute(0, 3, 1, 2)
        for conv, w in zip(self.convs, self.ws):
            x = F.gelu(conv(x) + w(x))
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)


# ----------------------------------------------------------------------
# Variable-coefficient physics
# ----------------------------------------------------------------------
def lame_field(E, nu=NU):
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam, mu


def hetero_strain_stress(u, E, h, nu=NU):
    """Interior (res-2) strain and variable-coefficient stress; both [B,.,.,3]."""
    ux_x, ux_y, uy_x, uy_y = fd_first_derivatives(u, h)
    eps_xx, eps_yy = ux_x, uy_y
    eps_xy = 0.5 * (ux_y + uy_x)
    lam, mu = lame_field(E[:, 1:-1, 1:-1], nu)
    sig_xx = (lam + 2 * mu) * eps_xx + lam * eps_yy
    sig_yy = lam * eps_xx + (lam + 2 * mu) * eps_yy
    sig_xy = 2 * mu * eps_xy
    eps = torch.stack([eps_xx, eps_yy, eps_xy], dim=-1)
    sig = torch.stack([sig_xx, sig_yy, sig_xy], dim=-1)
    return eps, sig


def hetero_residual_mse(u, f, E, h, nu=NU):
    """Mean square of div(sigma)+f with variable coefficients, on the res-4 grid."""
    _, sig = hetero_strain_stress(u, E, h, nu)
    sxx, syy, sxy = sig[..., 0], sig[..., 1], sig[..., 2]     # [B,res-2,res-2]
    dsxx_dx = ((sxx[:, :, 2:] - sxx[:, :, :-2]) / (2 * h))[:, 1:-1, :]
    dsxy_dy = ((sxy[:, 2:, :] - sxy[:, :-2, :]) / (2 * h))[:, :, 1:-1]
    dsxy_dx = ((sxy[:, :, 2:] - sxy[:, :, :-2]) / (2 * h))[:, 1:-1, :]
    dsyy_dy = ((syy[:, 2:, :] - syy[:, :-2, :]) / (2 * h))[:, :, 1:-1]
    f_in = f[:, 2:-2, 2:-2, :]
    res_x = dsxx_dx + dsxy_dy + f_in[..., 0]
    res_y = dsxy_dx + dsyy_dy + f_in[..., 1]
    return torch.mean(res_x ** 2 + res_y ** 2)


def strain_energy_field(u, E, h, nu=NU):
    eps, sig = hetero_strain_stress(u, E, h, nu)
    dens = 0.5 * (sig[..., 0] * eps[..., 0] + sig[..., 1] * eps[..., 1]
                  + 2 * sig[..., 2] * eps[..., 2])
    return torch.sum(dens, dim=(1, 2)) * h * h


# ----------------------------------------------------------------------
# Data / eval / train
# ----------------------------------------------------------------------
def load_data(path):
    z = np.load(path, allow_pickle=True)
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    return (t(z["f_tr"]), t(z["E_tr"]), t(z["u_tr"]),
            t(z["f_te"]), t(z["E_te"]), t(z["u_te"]))


@torch.no_grad()
def evaluate(model, f, E, u_true, phys, device, batch=128):
    model.eval()
    h = phys.h
    preds = []
    t0 = time.perf_counter()
    for i in range(0, f.shape[0], batch):
        preds.append(model(f[i:i+batch].to(device), E[i:i+batch].to(device)).cpu())
    infer_ms = 1000.0 * (time.perf_counter() - t0) / f.shape[0]
    up = torch.cat(preds, 0)
    eps_p, sig_p = hetero_strain_stress(up, E, h)
    eps_t, sig_t = hetero_strain_stress(u_true, E, h)
    en_p = strain_energy_field(up, E, h)
    en_t = strain_energy_field(u_true, E, h)
    return {
        "rel_l2_u": relative_l2(up, u_true),
        "rel_l2_strain": relative_l2(eps_p, eps_t),
        "rel_l2_stress": relative_l2(sig_p, sig_t),
        "rel_energy": (torch.abs(en_p - en_t) / (torch.abs(en_t) + 1e-12)).mean().item(),
        "residual_mse": hetero_residual_mse(up, f, E, h).item(),
        "infer_ms_per_sample": infer_ms,
    }


def build_model(name, phys, args):
    if name == "spectral_e":
        return SpectralPIDeepONetE(build_modes(args.modes), build_cos_modes(args.cos_modes),
                                   phys, hidden=args.hidden, depth=args.depth)
    if name == "fno_e":
        return FNOEfield(phys, modes=args.fno_modes, width=args.fno_width)
    raise ValueError(name)


def train(args):
    set_seed(args.seed)
    device = get_device(args.device)
    phys = PhysicsConfig(res=args.res, nu=args.nu)
    f_tr, E_tr, u_tr, f_te, E_te, u_te = load_data(args.data)
    print(f"data: train {tuple(f_tr.shape)}  test {tuple(f_te.shape)}  device {device}")

    model = build_model(args.model, phys, args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 3), gamma=0.5)

    f_trd, E_trd, u_trd = f_tr.to(device), E_tr.to(device), u_tr.to(device)
    n = f_tr.shape[0]
    best = float("inf"); best_state = None
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for b0 in range(0, n, args.batch):
            idx = perm[b0:b0 + args.batch]
            pred = model(f_trd[idx], E_trd[idx])
            loss_data = torch.mean((pred - u_trd[idx]) ** 2)
            loss = loss_data
            if args.w_pde > 0:
                loss = loss + args.w_pde * hetero_residual_mse(pred, f_trd[idx], E_trd[idx], phys.h)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sched.step()
        if ep % args.eval_every == 0 or ep == 1 or ep == args.epochs:
            m = evaluate(model, f_te, E_te, u_te, phys, device)
            if m["rel_l2_u"] < best:
                best = m["rel_l2_u"]; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"[{args.model}] ep {ep:4d}/{args.epochs}  u={m['rel_l2_u']:.4f} "
                  f"sig={m['rel_l2_stress']:.4f}  resMSE={m['residual_mse']:.3e}  best={best:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    final = evaluate(model, f_te, E_te, u_te, phys, device)
    final["best_rel_l2_u"] = best
    final["n_params"] = n_params
    final["model"] = args.model
    final["w_pde"] = args.w_pde
    tag = f"{args.model}_wpde{args.w_pde:g}_seed{args.seed}"
    torch.save({"state": model.state_dict(), "args": vars(args), "final": final},
               out / f"{tag}.pt")
    with open(out / f"{tag}.json", "w") as fh:
        json.dump(final, fh, indent=2)
    print("FINAL", json.dumps(final, indent=2))
    print("saved", out / f"{tag}.json")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/hetero_field.npz")
    p.add_argument("--model", default="spectral_e", choices=["spectral_e", "fno_e"])
    p.add_argument("--w_pde", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--res", type=int, default=29)
    p.add_argument("--nu", type=float, default=0.30)
    p.add_argument("--modes", type=int, default=16)       # sine modes for f / trunk
    p.add_argument("--cos_modes", type=int, default=16)   # cosine modes for log E
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--fno_modes", type=int, default=12)
    p.add_argument("--fno_width", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=111)
    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--out", default="runs/route_b")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
