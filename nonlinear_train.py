#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUPERSEDED -- retained for provenance, do not use for new results.

This script selects the reported error by taking the minimum over periodic
evaluations of the TEST set, with no validation split, and saves no
checkpoint.  That is model selection on the test set: the number it prints is
a best-epoch-on-test value, not a held-out estimate.  The referee on
CM-26-0573 identified this, correctly.

Use nonlinear_train_v2.py instead, which splits a validation set out of the
training pool, selects the checkpoint on validation only, evaluates the test
set exactly once, and saves the selected weights.  Re-running the corrected
protocol changed the reported finite-strain errors by less than 0.15
percentage points -- the measured optimism was zero -- but the protocol here
is not sound and this file should not be used.


nonlinear_train.py

Train data-only neural operators (spectral DeepONet, FNO) on the finite-strain
f -> u map.  The Navier-Cauchy physics residual is linear-elasticity-specific
and does not apply here, so training is supervised only.  This gives the
neural-operator accuracy to compare against the classical baselines
(nonlinear_rom.py POD+regression, the POD floor, and Newton-FEM).

    python nonlinear_train.py --data data/nonlinear.npz --model spectral --modes 64
    python nonlinear_train.py --data data/nonlinear.npz --model fno
"""

import argparse, json, time
from pathlib import Path
import numpy as np
import torch

from cmame_extended_study import (PhysicsConfig, build_modes, SpectralPIDeepONet,
                                  FNO2dElasticity, relative_l2, set_seed, get_device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/nonlinear.npz")
    ap.add_argument("--model", default="spectral", choices=["spectral", "fno"])
    ap.add_argument("--modes", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--fno_width", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=111)
    ap.add_argument("--out", default="runs/nonlinear")
    a = ap.parse_args()

    set_seed(a.seed); dev = get_device(a.device)
    z = np.load(a.data)
    t = lambda x: torch.tensor(x, dtype=torch.float32)
    f_tr, u_tr, f_te, u_te = t(z["f_tr"]), t(z["u_tr"]), t(z["f_te"]), t(z["u_te"])
    phys = PhysicsConfig(res=f_tr.shape[1], nu=0.30)

    if a.model == "spectral":
        model = SpectralPIDeepONet(build_modes(a.modes), phys, a.hidden, a.depth).to(dev)
    else:
        model = FNO2dElasticity(phys, modes=12, width=a.fno_width, n_layers=4).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-6)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, a.epochs//3), gamma=0.5)
    f_trd, u_trd = f_tr.to(dev), u_tr.to(dev); n = f_tr.shape[0]
    best = 9e9

    for ep in range(1, a.epochs+1):
        model.train(); perm = torch.randperm(n, device=dev)
        for b in range(0, n, a.batch):
            idx = perm[b:b+a.batch]
            loss = torch.mean((model(f_trd[idx]) - u_trd[idx])**2)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sch.step()
        if ep % 25 == 0 or ep == a.epochs:
            model.eval()
            with torch.no_grad():
                pr = torch.cat([model(f_te[i:i+128].to(dev)).cpu()
                                for i in range(0, f_te.shape[0], 128)], 0)
            err = relative_l2(pr, u_te)
            best = min(best, err)
            print(f"[{a.model}] ep {ep} rel_l2_u={err:.4f} best={best:.4f}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(model=a.model, modes=a.modes, rel_l2_u=best, n_params=npar),
              open(out/f"{a.model}_seed{a.seed}.json", "w"), indent=2)
    print(f"FINAL {a.model}: rel_l2_u={best:.4f} params={npar}")


if __name__ == "__main__":
    main()
