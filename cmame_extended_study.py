#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cmame_extended_study.py

CMAME Extended Comparison Study — V2
Adds four new experiments to the original PI-Spectral-DeepONet study:

  1. Fourier Neural Operator (FNO) baseline comparison
  2. Finite-element-method (FEM) solve-time comparison
  3. Noise robustness test (noisy body-force inputs)
  4. Out-of-distribution (OOD) generalization to higher sine modes

This script is self-contained. It reproduces the original three models and adds
the new experiments.

Requirements:
    pip install torch numpy scipy pandas matplotlib

Usage:
    # Quick test (reduced epochs, fewer samples):
    python cmame_extended_study.py --mode quick

Outputs (in ./results_extended/):
    figures/Figure_7_fno_comparison.{pdf,png}
    figures/Figure_8_fem_timing.{pdf,png}
    figures/Figure_9_noise_robustness.{pdf,png}
    figures/Figure_10_ood_generalization.{pdf,png}
    tables/extended_main_comparison.csv
    tables/fem_timing_summary.csv
    tables/noise_robustness.csv
    tables/ood_generalization.csv
"""

import os
import time
import math
import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found. FEM timing will be skipped.")
    print("Install with: pip install scipy")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

# ==========================================================================
# 0. REPRODUCIBILITY
# ==========================================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


# ==========================================================================
# 1. CONFIGURATION
# ==========================================================================

@dataclass
class PhysicsConfig:
    res: int = 29
    E: float = 1.0
    nu: float = 0.30
    plane: str = "strain"

    @property
    def h(self) -> float:
        return 1.0 / (self.res - 1)

    @property
    def lame(self) -> Tuple[float, float]:
        E, nu = self.E, self.nu
        mu = E / (2.0 * (1.0 + nu))
        if self.plane.lower() == "stress":
            lam = E * nu / (1.0 - nu ** 2)
        else:
            lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return lam, mu


# ==========================================================================
# 2. DATA GENERATION  (matches original study exactly)
# ==========================================================================

def make_grid(res: int, device=None) -> torch.Tensor:
    xs = torch.linspace(0.0, 1.0, res, device=device)
    ys = torch.linspace(0.0, 1.0, res, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1)


def build_modes(max_modes: int) -> List[Tuple[int, int]]:
    candidates = []
    max_pq = max(8, int(np.ceil(np.sqrt(max_modes))) + 4)
    for p in range(1, max_pq + 1):
        for q in range(1, max_pq + 1):
            candidates.append((p, q))
    candidates = sorted(candidates,
                        key=lambda t: (t[0] ** 2 + t[1] ** 2, t[0], t[1]))
    return candidates[:max_modes]


def sine_basis(grid: torch.Tensor,
               modes: List[Tuple[int, int]]) -> torch.Tensor:
    x = grid[..., 0:1]
    y = grid[..., 1:2]
    feats = []
    for p, q in modes:
        feats.append(torch.sin(p * math.pi * x) *
                     torch.sin(q * math.pi * y))
    return torch.cat(feats, dim=-1)


def generate_coefficients(n: int, modes: List[Tuple[int, int]],
                           scale: float, seed: int):
    rng = np.random.default_rng(seed)
    M = len(modes)
    ax = np.zeros((n, M), dtype=np.float32)
    ay = np.zeros((n, M), dtype=np.float32)
    for k, (p, q) in enumerate(modes):
        d = 1.0 / (p ** 2 + q ** 2)
        ax[:, k] = rng.normal(0.0, scale * d, size=n)
        ay[:, k] = rng.normal(0.0, scale * d, size=n)
    return ax, ay


def analytical_solution_and_force(coeff_x, coeff_y, modes, phys):
    lam, mu = phys.lame
    res = phys.res
    x = np.linspace(0.0, 1.0, res, dtype=np.float64)
    y = np.linspace(0.0, 1.0, res, dtype=np.float64)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    n, M = coeff_x.shape

    ux = np.zeros((n, res, res), dtype=np.float64)
    uy = np.zeros_like(ux)
    ux_x = np.zeros_like(ux); ux_y = np.zeros_like(ux)
    uy_x = np.zeros_like(ux); uy_y = np.zeros_like(ux)
    ux_xx = np.zeros_like(ux); ux_yy = np.zeros_like(ux)
    uy_xx = np.zeros_like(ux); uy_yy = np.zeros_like(ux)
    ux_xy = np.zeros_like(ux); uy_xy = np.zeros_like(ux)

    for k, (p, q) in enumerate(modes):
        P, Q = p * math.pi, q * math.pi
        sx, cx = np.sin(P * xx), np.cos(P * xx)
        sy, cy = np.sin(Q * yy), np.cos(Q * yy)
        phi    = sx * sy
        phi_x  = P * cx * sy
        phi_y  = Q * sx * cy
        phi_xx = -(P**2) * phi
        phi_yy = -(Q**2) * phi
        phi_xy = P * Q * cx * cy
        ax = coeff_x[:, k][:, None, None]
        ay = coeff_y[:, k][:, None, None]
        ux += ax * phi;  uy += ay * phi
        ux_x += ax * phi_x;  ux_y += ax * phi_y
        uy_x += ay * phi_x;  uy_y += ay * phi_y
        ux_xx += ax * phi_xx; ux_yy += ax * phi_yy
        uy_xx += ay * phi_xx; uy_yy += ay * phi_yy
        ux_xy += ax * phi_xy; uy_xy += ay * phi_xy

    lap_ux = ux_xx + ux_yy
    lap_uy = uy_xx + uy_yy
    div_x  = ux_xx + uy_xy
    div_y  = ux_xy + uy_yy
    fx = -(mu * lap_ux + (lam + mu) * div_x)
    fy = -(mu * lap_uy + (lam + mu) * div_y)

    eps_xx = ux_x; eps_yy = uy_y
    eps_xy = 0.5 * (ux_y + uy_x)
    sig_xx = (lam + 2*mu) * eps_xx + lam * eps_yy
    sig_yy = lam * eps_xx + (lam + 2*mu) * eps_yy
    sig_xy = 2*mu * eps_xy

    u   = np.stack([ux, uy], axis=-1).astype(np.float32)
    f   = np.stack([fx, fy], axis=-1).astype(np.float32)
    eps = np.stack([eps_xx, eps_yy, eps_xy], axis=-1).astype(np.float32)
    sig = np.stack([sig_xx, sig_yy, sig_xy], axis=-1).astype(np.float32)
    return {"u": u, "f": f, "eps": eps, "sig": sig}


def make_dataset(n_train, n_test, true_modes, coeff_scale, phys,
                 seed_train=42, seed_test=999):
    modes = build_modes(true_modes)
    ax_tr, ay_tr = generate_coefficients(
        n_train, modes, coeff_scale, seed_train)
    ax_te, ay_te = generate_coefficients(
        n_test,  modes, coeff_scale, seed_test)
    tr = analytical_solution_and_force(ax_tr, ay_tr, modes, phys)
    te = analytical_solution_and_force(ax_te, ay_te, modes, phys)
    return {
        "train_f":   torch.tensor(tr["f"]),
        "train_u":   torch.tensor(tr["u"]),
        "train_eps": torch.tensor(tr["eps"]),
        "train_sig": torch.tensor(tr["sig"]),
        "test_f":    torch.tensor(te["f"]),
        "test_u":    torch.tensor(te["u"]),
        "test_eps":  torch.tensor(te["eps"]),
        "test_sig":  torch.tensor(te["sig"]),
        "modes":     modes,
    }


# ==========================================================================
# 3. FEATURE EXTRACTION AND FD OPERATORS  (same as original)
# ==========================================================================

def project_onto_sine(f_grid, modes, grid):
    phi  = sine_basis(grid, modes).to(f_grid.device)
    res  = f_grid.shape[1]
    h    = 1.0 / (res - 1)
    wx   = torch.ones(res, device=f_grid.device)
    wy   = torch.ones(res, device=f_grid.device)
    wx[0] = wx[-1] = 0.5;  wy[0] = wy[-1] = 0.5
    w2   = (wy[:, None] * wx[None, :])[None, :, :, None] * h * h
    fx_  = f_grid[..., 0:1];  fy_ = f_grid[..., 1:2]
    phb  = phi[None, :, :, :]
    cx   = 4.0 * torch.sum(w2 * fx_ * phb, dim=(1, 2))
    cy   = 4.0 * torch.sum(w2 * fy_ * phb, dim=(1, 2))
    return torch.cat([cx, cy], dim=-1)


def fd_first_derivatives(u, h):
    ux = u[..., 0];  uy = u[..., 1]
    ux_x = (ux[:, 1:-1, 2:] - ux[:, 1:-1, :-2]) / (2*h)
    ux_y = (ux[:, 2:, 1:-1] - ux[:, :-2, 1:-1]) / (2*h)
    uy_x = (uy[:, 1:-1, 2:] - uy[:, 1:-1, :-2]) / (2*h)
    uy_y = (uy[:, 2:, 1:-1] - uy[:, :-2, 1:-1]) / (2*h)
    return ux_x, ux_y, uy_x, uy_y


def fd_strain_stress(u, phys):
    lam, mu = phys.lame
    ux_x, ux_y, uy_x, uy_y = fd_first_derivatives(u, phys.h)
    eps_xx = ux_x;  eps_yy = uy_y
    eps_xy = 0.5 * (ux_y + uy_x)
    sig_xx = (lam + 2*mu) * eps_xx + lam * eps_yy
    sig_yy = lam * eps_xx + (lam + 2*mu) * eps_yy
    sig_xy = 2*mu * eps_xy
    eps = torch.stack([eps_xx, eps_yy, eps_xy], dim=-1)
    sig = torch.stack([sig_xx, sig_yy, sig_xy], dim=-1)
    return eps, sig


def fd_elasticity_operator(u, phys):
    lam, mu = phys.lame;  h = phys.h
    ux = u[..., 0];  uy = u[..., 1]
    ux_xx = (ux[:, 1:-1, 2:] - 2*ux[:, 1:-1, 1:-1] + ux[:, 1:-1, :-2]) / h**2
    ux_yy = (ux[:, 2:, 1:-1] - 2*ux[:, 1:-1, 1:-1] + ux[:, :-2, 1:-1]) / h**2
    uy_xx = (uy[:, 1:-1, 2:] - 2*uy[:, 1:-1, 1:-1] + uy[:, 1:-1, :-2]) / h**2
    uy_yy = (uy[:, 2:, 1:-1] - 2*uy[:, 1:-1, 1:-1] + uy[:, :-2, 1:-1]) / h**2
    ux_xy = (ux[:, 2:, 2:] - ux[:, 2:, :-2] - ux[:, :-2, 2:] + ux[:, :-2, :-2]) / (4*h**2)
    uy_xy = (uy[:, 2:, 2:] - uy[:, 2:, :-2] - uy[:, :-2, 2:] + uy[:, :-2, :-2]) / (4*h**2)
    Lx = (lam + 2*mu)*ux_xx + mu*ux_yy + (lam+mu)*uy_xy
    Ly = mu*uy_xx + (lam + 2*mu)*uy_yy + (lam+mu)*ux_xy
    return torch.stack([Lx, Ly], dim=-1)


def strain_energy(u, phys):
    eps, sig = fd_strain_stress(u, phys)
    density = 0.5 * (sig[..., 0]*eps[..., 0] +
                     sig[..., 1]*eps[..., 1] +
                     2*sig[..., 2]*eps[..., 2])
    return torch.sum(density, dim=(1, 2)) * phys.h**2


# ==========================================================================
# 4. MODELS
# ==========================================================================

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, depth, act="silu"):
        super().__init__()
        acts = {"silu": nn.SiLU, "gelu": nn.GELU, "tanh": nn.Tanh}
        A = acts[act]
        layers = []
        last = in_dim
        for _ in range(depth):
            layers += [nn.Linear(last, hidden), A()]
            last = hidden
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---- Spectral PI-DeepONet (exact BC via sine basis) ----------------------

class SpectralPIDeepONet(nn.Module):
    """
    Branch network maps spectral features of f to sine coefficients.
    Displacement is reconstructed as sum of sine basis functions.
    Homogeneous Dirichlet BC is exact by construction.
    """
    def __init__(self, modes, phys, hidden=128, depth=3):
        super().__init__()
        self.modes = modes
        self.phys  = phys
        self.M     = len(modes)
        self.branch = MLP(2*self.M, 2*self.M, hidden, depth)
        grid = make_grid(phys.res)
        self.register_buffer("grid", grid)
        self.register_buffer("phi",  sine_basis(grid, modes))

    def forward(self, f_grid):
        feats  = project_onto_sine(f_grid, self.modes, self.grid)
        coeff  = self.branch(feats)
        cx, cy = coeff[:, :self.M], coeff[:, self.M:]
        phi    = self.phi
        ux     = torch.einsum("bm,ijm->bij", cx, phi)
        uy     = torch.einsum("bm,ijm->bij", cy, phi)
        return torch.stack([ux, uy], dim=-1)


# ---- Fourier Neural Operator (NEW) ---------------------------------------

class SpectralConv2d(nn.Module):
    """2D Fourier integral operator layer."""
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        sc = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(
            sc * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(
            sc * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))

    def cmul(self, a, w):
        return torch.einsum("bixy,ioxy->boxy", a, w)

    def forward(self, x):
        B, C, H, W = x.shape
        xft  = torch.fft.rfft2(x)
        out  = torch.zeros(B, self.w1.shape[1], H, W//2+1,
                            dtype=torch.cfloat, device=x.device)
        m1, m2 = self.modes1, self.modes2
        out[:, :, :m1, :m2]  = self.cmul(xft[:, :, :m1, :m2],  self.w1)
        out[:, :, -m1:, :m2] = self.cmul(xft[:, :, -m1:, :m2], self.w2)
        return torch.fft.irfft2(out, s=(H, W))


class FNO2dElasticity(nn.Module):
    """
    Fourier Neural Operator for 2D elasticity.
    Input:  body-force field  (B, res, res, 2)
    Output: displacement field (B, res, res, 2)

    Note: this model does NOT enforce boundary conditions exactly.
    The BC error is part of its displacement error. This is the standard
    FNO as published by Li et al. (2021).
    """
    def __init__(self, phys, modes=12, width=20, n_layers=4):
        super().__init__()
        self.phys = phys
        max_modes = min(modes, phys.res // 2)  # rfft2 limit
        self.fc0   = nn.Linear(4, width)        # (fx, fy, x, y) → width
        self.convs = nn.ModuleList([
            SpectralConv2d(width, width, max_modes, max_modes)
            for _ in range(n_layers)])
        self.ws    = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.fc1   = nn.Linear(width, 128)
        self.fc2   = nn.Linear(128, 2)
        grid = make_grid(phys.res)
        self.register_buffer("grid", grid)

    def forward(self, f_grid):
        B = f_grid.shape[0]
        g = self.grid.unsqueeze(0).expand(B, -1, -1, -1)
        x = torch.cat([f_grid, g], dim=-1)     # (B, H, W, 4)
        x = self.fc0(x)                         # (B, H, W, width)
        x = x.permute(0, 3, 1, 2)              # (B, width, H, W)
        for conv, w in zip(self.convs, self.ws):
            x = F.gelu(conv(x) + w(x))
        x = x.permute(0, 2, 3, 1)              # (B, H, W, width)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)                      # (B, H, W, 2)


# ==========================================================================
# 5. FEM SOLVER  (NEW — for timing comparison only)
# ==========================================================================

def q4_element_stiffness(hx: float, hy: float,
                          lam: float, mu: float) -> np.ndarray:
    """
    8×8 stiffness matrix for a rectangular Q4 element, plane strain.
    Assembled with 2×2 Gauss quadrature.
    """
    gp = np.array([-1.0/np.sqrt(3), 1.0/np.sqrt(3)])
    gw = np.array([1.0, 1.0])
    C  = np.array([[lam+2*mu, lam, 0],
                   [lam, lam+2*mu, 0],
                   [0,   0,        mu]])
    ke = np.zeros((8, 8))
    for xi, wi in zip(gp, gw):
        for eta, wj in zip(gp, gw):
            dN_dxi  = np.array([-(1-eta), (1-eta), (1+eta), -(1+eta)]) / 4.0
            dN_deta = np.array([-(1-xi),  -(1+xi), (1+xi),  (1-xi)])   / 4.0
            dN_dx = (2.0/hx) * dN_dxi
            dN_dy = (2.0/hy) * dN_deta
            detJ  = (hx/2.0) * (hy/2.0)
            B = np.zeros((3, 8))
            for k in range(4):
                B[0, 2*k]   = dN_dx[k]
                B[1, 2*k+1] = dN_dy[k]
                B[2, 2*k]   = dN_dy[k]
                B[2, 2*k+1] = dN_dx[k]
            ke += detJ * wi * wj * (B.T @ C @ B)
    return ke


def fem_assemble_and_time(f_sample_np: np.ndarray,
                           phys: PhysicsConfig,
                           n_rep: int = 20) -> Dict:
    """
    Assemble Q4 FEM stiffness matrix and time the sparse direct solve
    for a single body-force sample.

    Parameters
    ----------
    f_sample_np : (res, res, 2) body-force array
    phys        : PhysicsConfig
    n_rep       : number of timing repetitions

    Returns
    -------
    dict with 'assembly_ms', 'solve_ms', 'total_ms', 'n_dof', 'n_elements'
    """
    if not HAS_SCIPY:
        return {"assembly_ms": float("nan"), "solve_ms": float("nan"),
                "total_ms": float("nan"), "n_dof": 0, "n_elements": 0}

    lam, mu = phys.lame
    res     = phys.res
    nx = ny = res - 1
    hx = hy = 1.0 / nx
    n_nodes = res * res
    n_dof   = 2 * n_nodes

    # --- Assembly ---
    t_asm0 = time.perf_counter()
    ke = q4_element_stiffness(hx, hy, lam, mu)
    rows, cols, vals = [], [], []
    f_global = np.zeros(n_dof)

    for ey in range(ny):
        for ex in range(nx):
            n1 = ey * res + ex
            n2 = n1 + 1
            n3 = (ey + 1) * res + ex + 1
            n4 = (ey + 1) * res + ex
            nodes = [n1, n2, n3, n4]
            dofs  = [d for n in nodes for d in (2*n, 2*n+1)]
            for i, di in enumerate(dofs):
                for j, dj in enumerate(dofs):
                    rows.append(di); cols.append(dj); vals.append(ke[i, j])
            # Lumped nodal force contribution
            for k, n in enumerate(nodes):
                iy, ix = divmod(n, res)
                f_global[2*n]   += f_sample_np[iy, ix, 0] * hx * hy / 4.0
                f_global[2*n+1] += f_sample_np[iy, ix, 1] * hx * hy / 4.0

    K = sp.csr_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))
    t_asm = (time.perf_counter() - t_asm0) * 1000.0  # ms

    # --- Apply Dirichlet BCs ---
    bc_dofs = set()
    for i in range(res):                       # bottom
        bc_dofs.update([2*i, 2*i+1])
    for i in range(res):                       # top
        n = (res-1)*res + i
        bc_dofs.update([2*n, 2*n+1])
    for j in range(res):                       # left
        n = j * res
        bc_dofs.update([2*n, 2*n+1])
    for j in range(res):                       # right
        n = j * res + (res-1)
        bc_dofs.update([2*n, 2*n+1])

    free     = [i for i in range(n_dof) if i not in bc_dofs]
    K_free   = K[np.ix_(free, free)]
    f_free   = f_global[free]

    # Warm-up
    spla.spsolve(K_free.tocsc(), f_free)

    # Timed solves
    solve_times = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        spla.spsolve(K_free.tocsc(), f_free)
        solve_times.append(time.perf_counter() - t0)

    solve_ms = float(np.mean(solve_times) * 1000.0)
    return {
        "assembly_ms": round(t_asm, 3),
        "solve_ms":    round(solve_ms, 4),
        "total_ms":    round(t_asm + solve_ms, 3),
        "n_dof":       n_dof,
        "n_elements":  nx * ny,
    }


# ==========================================================================
# 6. TRAINING AND EVALUATION  (same as original, extended for FNO)
# ==========================================================================

def relative_l2(pred, true, eps=1e-12):
    num = torch.linalg.norm((pred - true).reshape(pred.shape[0], -1), dim=1)
    den = torch.linalg.norm(true.reshape(true.shape[0], -1), dim=1) + eps
    return (num / den).mean().item()


def pde_mse(u_pred, f_grid, phys):
    Lu  = fd_elasticity_operator(u_pred, phys)
    f_i = f_grid[:, 1:-1, 1:-1, :]
    return torch.mean((Lu + f_i) ** 2).item()


@torch.no_grad()
def evaluate_model(model, f, u_true, phys, device, batch=64):
    model.eval()
    preds, times = [], []
    for i in range(0, f.shape[0], batch):
        fb = f[i:i+batch].to(device)
        t0 = time.perf_counter()
        p  = model(fb)
        times.append(time.perf_counter() - t0)
        preds.append(p.cpu())
    u_pred = torch.cat(preds, dim=0)

    eps_p, sig_p = fd_strain_stress(u_pred, phys)
    eps_t, sig_t = fd_strain_stress(u_true, phys)

    e_u   = relative_l2(u_pred, u_true)
    e_eps = relative_l2(eps_p,  eps_t)
    e_sig = relative_l2(sig_p,  sig_t)

    en_p = strain_energy(u_pred, phys)
    en_t = strain_energy(u_true, phys)
    e_en = ((en_p - en_t).abs() / (en_t.abs() + 1e-12)).mean().item()

    pde  = pde_mse(u_pred, f, phys)

    n_s      = f.shape[0]
    infer_ms = 1000.0 * sum(times) / n_s

    return {"disp_l2":    e_u,
            "strain_l2":  e_eps,
            "stress_l2":  e_sig,
            "energy_err": e_en,
            "pde_mse":    pde,
            "infer_ms":   infer_ms}


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, dataset, phys, device,
                epochs=800, batch=32, lr=1e-3,
                w_data=1.0, w_pde=0.0,
                sched_step=400, sched_gamma=0.5,
                eval_every=20, run_name="run"):
    """
    Generic training loop. Works for all model types.
    Set w_pde=0 for data-only training.
    """
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=sched_step, gamma=sched_gamma)

    train_f = dataset["train_f"].to(device)
    train_u = dataset["train_u"].to(device)
    test_f  = dataset["test_f"]
    test_u  = dataset["test_u"]

    N  = train_f.shape[0]
    best_err, best_state = float("inf"), None
    history = []

    print(f"\n{'='*60}")
    print(f" Training: {run_name}  |  params={count_params(model):,}")
    print(f" epochs={epochs}  w_pde={w_pde}  lr={lr}")
    print(f"{'='*60}")

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N, device=device)
        loss_sum = 0.0; nb = 0

        for b0 in range(0, N, batch):
            idx  = perm[b0:b0+batch]
            fb   = train_f[idx];  ub = train_u[idx]
            pred = model(fb)
            l_d  = torch.mean((pred - ub) ** 2)
            l_p  = 0.0
            if w_pde > 0:
                Lu  = fd_elasticity_operator(pred, phys)
                fi  = fb[:, 1:-1, 1:-1, :]
                l_p = torch.mean((Lu + fi) ** 2)
            loss = w_data * l_d + w_pde * l_p
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item(); nb += 1

        sched.step()

        if ep % eval_every == 0 or ep == epochs:
            m = evaluate_model(model, test_f, test_u, phys, device)
            history.append({"epoch": ep, **m})
            if m["disp_l2"] < best_err:
                best_err   = m["disp_l2"]
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
            print(f"  ep {ep:5d}/{epochs}  loss={loss_sum/nb:.3e}"
                  f"  disp={m['disp_l2']:.4f}  best={best_err:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    final = evaluate_model(model, test_f, test_u, phys, device)
    final["best_disp_l2"] = best_err
    final["n_params"]     = count_params(model)
    return model, final, history


# ==========================================================================
# 7. NOISE ROBUSTNESS EXPERIMENT  (NEW)
# ==========================================================================

@torch.no_grad()
def noise_robustness_study(models_dict, dataset, phys, device,
                            noise_levels=None):
    """
    Evaluate models on body forces with increasing Gaussian noise.

    noise_level = σ × std(f), applied independently per sample.
    Returns a dict: {model_name: [err at each noise level]}
    """
    if noise_levels is None:
        noise_levels = [0.00, 0.01, 0.02, 0.05, 0.10, 0.20]

    test_f = dataset["test_f"]
    test_u = dataset["test_u"]
    f_std  = test_f.std().item()

    results = {"noise_level": noise_levels}
    for name, model in models_dict.items():
        model.eval()
        errs = []
        for sigma in noise_levels:
            if sigma == 0.0:
                f_in = test_f
            else:
                noise = torch.randn_like(test_f) * sigma * f_std
                f_in  = test_f + noise
            m = evaluate_model(model, f_in, test_u, phys, device)
            errs.append(m["disp_l2"])
            print(f"  {name:25s}  σ={sigma:.2f}  disp_L2={m['disp_l2']:.4f}")
        results[name] = errs
    return results


# ==========================================================================
# 8. OOD GENERALIZATION EXPERIMENT  (NEW)
# ==========================================================================

@torch.no_grad()
def ood_generalization_study(models_dict, phys, device,
                              mode_counts=None, coeff_scale=0.04,
                              n_test=200):
    """
    Evaluate models on test sets generated with increasing numbers of
    sine modes. Models were trained on `true_modes` modes; we test on
    higher mode counts to assess out-of-distribution generalization.

    Returns dict: {model_name: [err at each mode count]}
    """
    if mode_counts is None:
        mode_counts = [12, 16, 20, 24, 28]

    results = {"mode_count": mode_counts}
    for name, model in models_dict.items():
        model.eval()
        errs = []
        for M in mode_counts:
            ds = make_dataset(n_train=1, n_test=n_test,
                              true_modes=M, coeff_scale=coeff_scale,
                              phys=phys, seed_train=1, seed_test=77+M)
            m = evaluate_model(model, ds["test_f"], ds["test_u"],
                               phys, device)
            errs.append(m["disp_l2"])
            print(f"  {name:25s}  modes={M:3d}  disp_L2={m['disp_l2']:.4f}")
        results[name] = errs
    return results


# ==========================================================================
# 9. FIGURE GENERATION  (NEW — publication-quality)
# ==========================================================================

COLORS = {
    "PI-Spectral DeepONet": "#2166ac",
    "Data-only Spectral":   "#d73027",
    "FNO":                  "#4dac26",
}
MARKERS = {
    "PI-Spectral DeepONet": "o",
    "Data-only Spectral":   "s",
    "FNO":                  "^",
}


def fig7_fno_comparison(comparison_data: dict, fig_dir: Path):
    """
    Bar chart comparing all three models across all five mechanics metrics.
    Mirrors Figure 2 of the original manuscript but adds the FNO column.
    """
    metrics     = ["disp_l2", "strain_l2", "stress_l2", "energy_err", "pde_mse"]
    metric_labs = ["Disp. L2", "Strain L2", "Stress L2", "Energy", "PDE MSE"]
    models      = ["PI-Spectral DeepONet", "Data-only Spectral", "FNO"]
    colors_bar  = [COLORS[m] for m in models]

    vals = {model: [comparison_data[model][m] for m in metrics]
            for model in models}

    x   = np.arange(len(metrics))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (model, color) in enumerate(zip(models, colors_bar)):
        ax.bar(x + (i - 1) * w, vals[model], w,
               label=model, color=color, alpha=0.85)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labs)
    ax.set_ylabel("Error / residual (log scale)")
    ax.set_title("Extended comparison: PI-Spectral DeepONet vs "
                 "Data-only vs FNO")
    ax.legend(loc="upper right")
    ax.grid(axis="y", which="both", ls="--", alpha=0.4)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"Figure_7_fno_comparison.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved Figure 7")


def fig8_fem_timing(fem_data: dict, neural_infer_ms: dict, fig_dir: Path):
    """
    Horizontal bar chart: FEM total vs neural operator inference time.
    """
    labels = (["FEM (assembly\n+ solve)"]
              + [f"{k}\n(inference)" for k in neural_infer_ms])
    values = ([fem_data["total_ms"]]
              + list(neural_infer_ms.values()))
    colors_b = (["#b2182b"]
                + [COLORS.get(k, "#888888") for k in neural_infer_ms])

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, values, color=colors_b, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("Time per sample (ms, log scale)")
    ax.set_title("Computational cost: FEM vs neural operator inference")

    for bar, val in zip(bars, values):
        ax.text(val * 1.15, bar.get_y() + bar.get_height()/2,
                f"{val:.3g} ms", va="center", fontsize=9)

    # Speedup annotation
    fem_ms = fem_data["total_ms"]
    for k, t in neural_infer_ms.items():
        speedup = fem_ms / t
        print(f"  Speedup vs FEM  [{k}]: {speedup:.0f}×")

    ax.grid(axis="x", which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"Figure_8_fem_timing.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved Figure 8")


def fig9_noise_robustness(noise_data: dict, fig_dir: Path):
    """
    Line plot: displacement L2 error vs noise level for all models.
    """
    noise_levels = noise_data["noise_level"]
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for name in [k for k in noise_data if k != "noise_level"]:
        ax.plot([100*s for s in noise_levels], noise_data[name],
                marker=MARKERS.get(name, "o"),
                color=COLORS.get(name, "#888888"),
                label=name, lw=2, ms=7)

    ax.set_xlabel("Body-force noise level (% of signal std)")
    ax.set_ylabel(r"Displacement relative $L^2$ error")
    ax.set_title("Robustness to noisy body-force inputs")
    ax.legend()
    ax.grid(ls="--", alpha=0.4)
    ax.set_yscale("log")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"Figure_9_noise_robustness.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved Figure 9")


def fig10_ood_generalization(ood_data: dict, train_modes: int, fig_dir: Path):
    """
    Line plot: displacement L2 error vs test mode count (OOD generalization).
    """
    mode_counts = ood_data["mode_count"]
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for name in [k for k in ood_data if k != "mode_count"]:
        ax.plot(mode_counts, ood_data[name],
                marker=MARKERS.get(name, "o"),
                color=COLORS.get(name, "#888888"),
                label=name, lw=2, ms=7)

    ax.axvline(train_modes, color="gray", ls="--", lw=1.5,
               label=f"Training modes (K={train_modes})")
    ax.set_xlabel("Test sine-mode count K")
    ax.set_ylabel(r"Displacement relative $L^2$ error")
    ax.set_title("Out-of-distribution generalization to higher spatial frequencies")
    ax.legend()
    ax.grid(ls="--", alpha=0.4)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"Figure_10_ood_generalization.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved Figure 10")


# ==========================================================================
# 10. MAIN STUDY
# ==========================================================================

def make_config(mode: str):
    """Return (train_modes, n_train, n_test, epochs_pi, epochs_fno)."""
    if mode == "quick":
        return dict(train_modes=12, n_train=300, n_test=60,
                    epochs_pi=200, epochs_fno=150,
                    fno_width=16, fno_modes=10)
    else:  # publication
        return dict(train_modes=16, n_train=3000, n_test=200,
                    epochs_pi=800, epochs_fno=600,
                    fno_width=20, fno_modes=12)


def run(args):
    set_seed(42)
    device = get_device(args.device)
    cfg    = make_config(args.mode)
    phys   = PhysicsConfig()

    root = Path(args.out_dir)
    fig_dir = root / "figures"
    tab_dir = root / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDevice: {device}")
    print(f"Mode:   {args.mode}")
    print(f"Config: {cfg}")
    lam, mu = phys.lame
    print(f"Lamé:   λ={lam:.4f}, μ={mu:.4f}\n")

    # ------------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------------
    print("Generating dataset ...")
    dataset = make_dataset(
        n_train=cfg["n_train"], n_test=cfg["n_test"],
        true_modes=cfg["train_modes"], coeff_scale=0.04,
        phys=phys, seed_train=42, seed_test=999)
    modes = dataset["modes"]
    M     = len(modes)
    print(f"  train={cfg['n_train']}  test={cfg['n_test']}"
          f"  modes={cfg['train_modes']}")

    # ------------------------------------------------------------------
    # MODEL 1: PI-Spectral DeepONet
    # ------------------------------------------------------------------
    pi_model = SpectralPIDeepONet(modes, phys, hidden=192, depth=4).to(device)
    pi_model, pi_metrics, _ = train_model(
        pi_model, dataset, phys, device,
        epochs=cfg["epochs_pi"], batch=32, lr=1e-3,
        w_data=1.0, w_pde=1e-4,
        sched_step=cfg["epochs_pi"]//2, sched_gamma=0.5,
        eval_every=20, run_name="PI-Spectral-DeepONet")

    # ------------------------------------------------------------------
    # MODEL 2: Data-only Spectral DeepONet
    # ------------------------------------------------------------------
    do_model = SpectralPIDeepONet(modes, phys, hidden=192, depth=4).to(device)
    do_model, do_metrics, _ = train_model(
        do_model, dataset, phys, device,
        epochs=cfg["epochs_pi"], batch=32, lr=1e-3,
        w_data=1.0, w_pde=0.0,
        sched_step=cfg["epochs_pi"]//2, sched_gamma=0.5,
        eval_every=20, run_name="Data-only-Spectral")

    # ------------------------------------------------------------------
    # MODEL 3: FNO
    # ------------------------------------------------------------------
    fno_model = FNO2dElasticity(
        phys, modes=cfg["fno_modes"], width=cfg["fno_width"],
        n_layers=4).to(device)
    fno_model, fno_metrics, _ = train_model(
        fno_model, dataset, phys, device,
        epochs=cfg["epochs_fno"], batch=32, lr=1e-3,
        w_data=1.0, w_pde=0.0,
        sched_step=cfg["epochs_fno"]//2, sched_gamma=0.5,
        eval_every=20, run_name="FNO")

    # ------------------------------------------------------------------
    # MAIN COMPARISON TABLE
    # ------------------------------------------------------------------
    metric_keys = ["disp_l2", "strain_l2", "stress_l2",
                   "energy_err", "pde_mse", "infer_ms", "n_params"]
    rows = []
    for name, m in [("PI-Spectral DeepONet", pi_metrics),
                    ("Data-only Spectral",   do_metrics),
                    ("FNO",                  fno_metrics)]:
        row = {"model": name}
        row.update({k: m.get(k, float("nan")) for k in metric_keys})
        rows.append(row)
        print(f"\n{name}:")
        for k in metric_keys:
            print(f"  {k:20s} = {m.get(k, float('nan')):.6g}")

    df_main = pd.DataFrame(rows)
    df_main.to_csv(tab_dir / "extended_main_comparison.csv", index=False)

    comparison_data = {
        "PI-Spectral DeepONet": pi_metrics,
        "Data-only Spectral":   do_metrics,
        "FNO":                  fno_metrics,
    }

    # ------------------------------------------------------------------
    # FEM TIMING
    # ------------------------------------------------------------------
    print("\nRunning FEM timing experiment ...")
    f_sample = dataset["test_f"][0].numpy()
    fem_data = fem_assemble_and_time(f_sample, phys, n_rep=20)
    print(f"  FEM assembly: {fem_data['assembly_ms']:.1f} ms")
    print(f"  FEM solve:    {fem_data['solve_ms']:.4f} ms")
    print(f"  FEM total:    {fem_data['total_ms']:.1f} ms")
    print(f"  Neural infer: PI={pi_metrics['infer_ms']:.5f} ms"
          f"  FNO={fno_metrics['infer_ms']:.5f} ms")

    neural_infer = {
        "PI-Spectral DeepONet": pi_metrics["infer_ms"],
        "Data-only Spectral":   do_metrics["infer_ms"],
        "FNO":                  fno_metrics["infer_ms"],
    }
    pd.DataFrame([{**{"method": "FEM"}, **fem_data},
                  *[{"method": k, "solve_ms": v}
                    for k, v in neural_infer.items()]]
                 ).to_csv(tab_dir / "fem_timing_summary.csv", index=False)

    # ------------------------------------------------------------------
    # NOISE ROBUSTNESS
    # ------------------------------------------------------------------
    print("\nRunning noise robustness study ...")
    noise_levels = [0.00, 0.01, 0.02, 0.05, 0.10, 0.20]
    models_dict  = {
        "PI-Spectral DeepONet": pi_model,
        "Data-only Spectral":   do_model,
        "FNO":                  fno_model,
    }
    noise_data = noise_robustness_study(
        models_dict, dataset, phys, device, noise_levels)
    pd.DataFrame(noise_data).to_csv(
        tab_dir / "noise_robustness.csv", index=False)

    # ------------------------------------------------------------------
    # OOD GENERALIZATION
    # ------------------------------------------------------------------
    print("\nRunning OOD generalization study ...")
    ood_mode_counts = [cfg["train_modes"],
                       cfg["train_modes"] + 4,
                       cfg["train_modes"] + 8,
                       cfg["train_modes"] + 12,
                       cfg["train_modes"] + 16]
    ood_data = ood_generalization_study(
        models_dict, phys, device,
        mode_counts=ood_mode_counts, n_test=100)
    pd.DataFrame(ood_data).to_csv(
        tab_dir / "ood_generalization.csv", index=False)

    # ------------------------------------------------------------------
    # FIGURES
    # ------------------------------------------------------------------
    print("\nGenerating figures ...")
    fig7_fno_comparison(comparison_data, fig_dir)
    fig8_fem_timing(fem_data, neural_infer, fig_dir)
    fig9_noise_robustness(noise_data, fig_dir)
    fig10_ood_generalization(ood_data, cfg["train_modes"], fig_dir)

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STUDY COMPLETE")
    print("="*60)
    print(f"Results saved to: {root.resolve()}")
    print(f"New figures:  {fig_dir}")
    print(f"New tables:   {tab_dir}")
    print("\nCopy the new figures to your manuscript figures/ directory:")
    for fn in ["Figure_7_fno_comparison.pdf",
               "Figure_8_fem_timing.pdf",
               "Figure_9_noise_robustness.pdf",
               "Figure_10_ood_generalization.pdf"]:
        print(f"  {fig_dir}/{fn}")


def parse_args():
    p = argparse.ArgumentParser(
        description="CMAME Extended Study: FNO + FEM + Noise + OOD")
    p.add_argument("--mode",    default="quick",
                   choices=["quick", "publication"])
    p.add_argument("--device",  default="auto")
    p.add_argument("--out_dir", default="results_extended")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
