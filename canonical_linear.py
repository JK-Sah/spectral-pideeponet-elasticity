#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
canonical_linear.py

One canonical configuration for the linear manufactured benchmark, run for
every spectral variant under an identical protocol, with seed statistics.

Two referee points are addressed here.

(1) The least-squares-anchored spectral model reaches 0.4% displacement error
    in the calibration table but is absent from the headline comparison, which
    quotes 8.4% as the best learned result.  The anchored model is a trainable
    physics-informed architecture and belongs in the main comparison.  It is
    trained here in the main configuration, alongside the plain branch and the
    closed-form read-out, so all three are directly comparable.

(2) The reported PI-spectral error varies across tables (8.39%, 3.54%, 9.6%)
    by more than seed noise explains, because the runs used different training
    sizes and settings.  Everything below uses one configuration -- stated in
    full in the output -- and reports mean and standard deviation over seeds.

Model selection uses a separate validation set drawn from the same generator
with its own seed, so the 3000-sample training configuration of the published
benchmark is preserved exactly while checkpoint selection stops touching test.

    python canonical_linear.py --seeds 42 43 44 --out results/canonical.json
"""

import argparse, json, time
from pathlib import Path
import numpy as np
import torch

import cmame_extended_study as st
from cmame_extended_study import (PhysicsConfig, build_modes, SpectralPIDeepONet,
                                  FNO2dElasticity, make_dataset, relative_l2,
                                  set_seed, get_device)

# Canonical configuration -- the single setting used for every number reported.
CFG = dict(n_train=3000, n_val=400, n_test=200, true_modes=16, coeff_scale=0.04,
           hidden=192, depth=4, epochs=800, epochs_fno=600,
           fno_width=20, fno_modes=12, fno_layers=4,
           batch=32, lr=1e-3, weight_decay=1e-6,
           w_data=1.0, w_pde=1e-4, sched_step=400, sched_gamma=0.5,
           eval_every=20, seed_train=42, seed_val=7, seed_test=999,
           selection="min validation displacement error")


class AnchoredSpectral(torch.nn.Module):
    """Least-squares-anchored branch: linear read-out initialized at the
    closed-form optimum, plus an MLP correction whose output layer starts at
    zero, so training begins exactly at the closed-form solution."""

    def __init__(self, modes, phys, train_f, train_u, hidden=192, depth=4):
        super().__init__()
        self.modes = modes
        grid = st.make_grid(phys.res)
        self.grid = grid
        self.register_buffer("phi", st.sine_basis(grid, modes))
        M = len(modes)
        self.lin = torch.nn.Linear(2 * M, 2 * M)
        self.mlp = st.MLP(2 * M, 2 * M, hidden, depth)
        last = [m for m in self.mlp.modules() if isinstance(m, torch.nn.Linear)][-1]
        torch.nn.init.zeros_(last.weight); torch.nn.init.zeros_(last.bias)
        W, b = closed_form_readout(modes, grid, train_f, train_u)
        with torch.no_grad():
            self.lin.weight.copy_(torch.tensor(W, dtype=torch.float32))
            self.lin.bias.copy_(torch.tensor(b, dtype=torch.float32))

    def forward(self, f):
        feats = st.project_onto_sine(f, self.modes, self.grid.to(f.device))
        c = self.lin(feats) + self.mlp(feats)
        M = len(self.modes)
        ux = torch.einsum("nm,ijm->nij", c[:, :M], self.phi)
        uy = torch.einsum("nm,ijm->nij", c[:, M:], self.phi)
        return torch.stack([ux, uy], dim=-1)


def closed_form_readout(modes, grid, train_f, train_u, ridge=1e-8):
    """Optimal linear map from sine features of f to sine coefficients of u."""
    M = len(modes)
    X = st.project_onto_sine(train_f, modes, grid).numpy().astype(np.float64)
    Yx = st.project_onto_sine(train_u[..., 0:1].repeat(1, 1, 1, 2), modes,
                              grid).numpy()[:, :M]
    Yy = st.project_onto_sine(train_u[..., 1:2].repeat(1, 1, 1, 2), modes,
                              grid).numpy()[:, :M]
    Y = np.concatenate([Yx, Yy], axis=1).astype(np.float64)
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    Wb = np.linalg.solve(Xb.T @ Xb + ridge * np.eye(Xb.shape[1]), Xb.T @ Y)
    return Wb[:-1].T, Wb[-1]


class ClosedFormModel(torch.nn.Module):
    def __init__(self, W, b, modes, phys):
        super().__init__()
        self.modes = modes
        grid = st.make_grid(phys.res)
        self.grid = grid
        self.register_buffer("phi", st.sine_basis(grid, modes))
        self.register_buffer("W", torch.tensor(W, dtype=torch.float32))
        self.register_buffer("b", torch.tensor(b, dtype=torch.float32))

    def forward(self, f):
        feats = st.project_onto_sine(f, self.modes, self.grid.to(f.device))
        c = feats @ self.W.T + self.b
        M = len(self.modes)
        ux = torch.einsum("nm,ijm->nij", c[:, :M], self.phi)
        uy = torch.einsum("nm,ijm->nij", c[:, M:], self.phi)
        return torch.stack([ux, uy], dim=-1)


def evaluate(model, f, u, dev, bs=100):
    model.eval()
    with torch.no_grad():
        pr = torch.cat([model(f[i:i+bs].to(dev)).cpu() for i in range(0, f.shape[0], bs)], 0)
    return float(relative_l2(pr, u))


def batched_ms_per_sample(model, f, batch=128, warm=3, reps=10):
    """Amortized cost per sample when many queries are issued together."""
    model.eval(); n = min(batch, f.shape[0])
    with torch.no_grad():
        for _ in range(warm):
            model(f[:n])
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); model(f[:n]); ts.append((time.perf_counter()-t0)*1e3)
    return float(np.median(ts)) / n


def single_query_ms(model, f1, warm=5, reps=50):
    model.eval()
    with torch.no_grad():
        for _ in range(warm):
            model(f1)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); model(f1); ts.append((time.perf_counter()-t0)*1e3)
    return float(np.median(ts))


def train(model, ds_tr, ds_va, phys, dev, seed, w_pde, tag, epochs=None):
    """Train with the canonical settings; select the checkpoint on validation."""
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"],
                           weight_decay=CFG["weight_decay"])
    epochs = epochs or CFG["epochs"]
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 2),
                                          gamma=CFG["sched_gamma"])
    f_tr, u_tr = ds_tr[0].to(dev), ds_tr[1].to(dev)
    n = f_tr.shape[0]
    best, best_ep, best_state = 9e9, -1, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        for b in range(0, n, CFG["batch"]):
            idx = perm[b:b + CFG["batch"]]
            fb = f_tr[idx]
            pred = model(fb)
            loss = CFG["w_data"] * torch.mean((pred - u_tr[idx]) ** 2)
            if w_pde > 0:
                # Identical Navier-Cauchy residual to the main study's trainer.
                Lu = st.fd_elasticity_operator(pred, phys)
                loss = loss + w_pde * torch.mean((Lu + fb[:, 1:-1, 1:-1, :]) ** 2)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sch.step()
        if ep % CFG["eval_every"] == 0 or ep == epochs:
            e = evaluate(model, ds_va[0], ds_va[1], dev)
            if e < best:
                best, best_ep = e, ep
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best, best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--trunk", type=int, default=None,
                    help="spectral trunk capacity M (default: equal to the "
                         "data-generation mode count K)")
    ap.add_argument("--ckpt_dir", default="")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results_revision/canonical_linear.json")
    a = ap.parse_args()
    dev = get_device(a.device)
    phys = PhysicsConfig()
    M_trunk = a.trunk or CFG["true_modes"]
    modes = build_modes(M_trunk)
    print(f"device={dev}\ncanonical config: {CFG}\ntrunk M={M_trunk}")

    d_main = make_dataset(CFG["n_train"], CFG["n_test"], CFG["true_modes"],
                          CFG["coeff_scale"], phys,
                          seed_train=CFG["seed_train"], seed_test=CFG["seed_test"])
    d_val = make_dataset(CFG["n_val"], 1, CFG["true_modes"], CFG["coeff_scale"],
                         phys, seed_train=CFG["seed_val"], seed_test=12345)
    tr = (d_main["train_f"], d_main["train_u"])
    va = (d_val["train_f"], d_val["train_u"])
    te = (d_main["test_f"], d_main["test_u"])
    print(f"train={tr[0].shape[0]} val={va[0].shape[0]} test={te[0].shape[0]}")

    f1 = te[0][:1].clone()
    results = {"config": CFG, "runs": []}

    # Closed-form read-out: no training, no seed dependence beyond the data.
    W, b = closed_form_readout(modes, st.make_grid(phys.res), tr[0], tr[1])
    cf = ClosedFormModel(W, b, modes, phys).to(dev)
    e_cf = evaluate(cf, te[0], te[1], dev)
    cf = cf.cpu()
    ms_cf = single_query_ms(cf, f1)
    msb_cf = batched_ms_per_sample(cf, te[0])
    npar_cf = int(W.size + b.size)
    print(f"[closed-form ] test={e_cf:.5f}  single={ms_cf:.3f} ms  "
          f"batched={msb_cf:.4f} ms  params={npar_cf}")
    results["closed_form"] = dict(test_error=e_cf, single_query_ms=ms_cf,
                                  batched_ms_per_sample=msb_cf, n_params=npar_cf)

    for seed in a.seeds:
        for tag, w_pde in (("pi_spectral_plain", CFG["w_pde"]),
                           ("data_only_spectral", 0.0),
                           ("pi_spectral_anchored", CFG["w_pde"]),
                           ("fno", CFG["w_pde"])):
            set_seed(seed)
            if tag == "pi_spectral_anchored":
                m = AnchoredSpectral(modes, phys, tr[0], tr[1],
                                     CFG["hidden"], CFG["depth"]).to(dev)
            elif tag == "fno":
                m = FNO2dElasticity(phys, modes=CFG["fno_modes"],
                                    width=CFG["fno_width"],
                                    n_layers=CFG["fno_layers"]).to(dev)
            else:
                m = SpectralPIDeepONet(modes, phys, hidden=CFG["hidden"],
                                       depth=CFG["depth"]).to(dev)
            npar = sum(p.numel() for p in m.parameters())
            t0 = time.time()
            n_ep = CFG["epochs_fno"] if tag == "fno" else CFG["epochs"]
            m, e_val, ep = train(m, tr, va, phys, dev, seed, w_pde, tag, epochs=n_ep)
            e_te = evaluate(m, te[0], te[1], dev)
            m = m.cpu()
            ms = single_query_ms(m, f1)
            msb = batched_ms_per_sample(m, te[0])
            if a.ckpt_dir:
                Path(a.ckpt_dir).mkdir(parents=True, exist_ok=True)
                torch.save(dict(state_dict=m.state_dict(), model=tag, seed=seed,
                                trunk=M_trunk),
                           Path(a.ckpt_dir)/f"{tag}_M{M_trunk}_seed{seed}.pt")
            m.to(dev)
            rec = dict(model=tag, seed=seed, trunk=M_trunk, w_pde=w_pde,
                       val_error=e_val, test_error=e_te, selected_epoch=ep,
                       n_params=int(npar), single_query_ms=ms,
                       batched_ms_per_sample=msb, minutes=(time.time()-t0)/60)
            results["runs"].append(rec)
            print(f"[{tag:21s} M{M_trunk} s{seed}] val={e_val:.5f} "
                  f"test={e_te:.5f} ep*={ep} params={npar} "
                  f"single={ms:.3f}ms batched={msb:.4f}ms ({rec['minutes']:.1f}min)")

    # Aggregate
    agg = {}
    for tag in ("pi_spectral_plain", "data_only_spectral",
                "pi_spectral_anchored", "fno"):
        es = [r["test_error"] for r in results["runs"] if r["model"] == tag]
        if es:
            agg[tag] = dict(mean=float(np.mean(es)), std=float(np.std(es)),
                            n_seeds=len(es), values=es)
            print(f"[AGG {tag:21s}] {np.mean(es):.5f} +/- {np.std(es):.5f} "
                  f"over {len(es)} seeds")
    results["aggregate"] = agg
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))
    print("wrote", a.out)
    print("CANONICAL_DONE")


if __name__ == "__main__":
    main()
