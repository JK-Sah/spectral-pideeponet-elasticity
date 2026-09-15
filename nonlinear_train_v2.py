#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nonlinear_train_v2.py

Finite-strain operator training with a held-out protocol.

The previous script (nonlinear_train.py) evaluated the test set every 25
epochs and reported the minimum observed test error.  That is model selection
on the test set: the reported figure is a best-epoch-on-test value, not a
held-out estimate, and no checkpoint was saved.

This version:
  - carves a validation split out of the training set; the test set is
    untouched during training,
  - selects the checkpoint by validation error only,
  - evaluates the test set exactly once, with the selected checkpoint,
  - saves that checkpoint so downstream timing uses trained weights,
  - also records the old best-epoch-on-test statistic, purely so the revision
    can quantify how much optimism the earlier protocol introduced,
  - writes the full configuration (architecture, split sizes, optimizer,
    schedule, seed, epoch budget, selection rule) into the result file.

    python nonlinear_train_v2.py --data data/nonlinear.npz --model spectral --seed 111
"""

import argparse, json, time
from pathlib import Path
import numpy as np
import torch

from cmame_extended_study import (PhysicsConfig, build_modes, SpectralPIDeepONet,
                                  FNO2dElasticity, relative_l2, set_seed, get_device)


def evaluate(model, f, u, dev, bs=128):
    model.eval()
    with torch.no_grad():
        pr = torch.cat([model(f[i:i+bs].to(dev)).cpu() for i in range(0, f.shape[0], bs)], 0)
    return float(relative_l2(pr, u))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/nonlinear.npz")
    ap.add_argument("--model", default="spectral", choices=["spectral", "fno"])
    ap.add_argument("--modes", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--fno_width", type=int, default=32)
    ap.add_argument("--fno_modes", type=int, default=12)
    ap.add_argument("--fno_layers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--val_frac", type=float, default=0.20)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=111)
    ap.add_argument("--out", default="runs/nonlinear_v2")
    a = ap.parse_args()

    set_seed(a.seed); dev = get_device(a.device)
    z = np.load(a.data)
    t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)
    f_all, u_all = t(z["f_tr"]), t(z["u_tr"])
    f_te, u_te = t(z["f_te"]), t(z["u_te"])

    # Deterministic train/validation split of the training pool.
    n_all = f_all.shape[0]
    g = np.random.default_rng(a.seed)
    perm = g.permutation(n_all)
    n_val = int(round(a.val_frac * n_all))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    f_tr, u_tr = f_all[tr_idx], u_all[tr_idx]
    f_va, u_va = f_all[val_idx], u_all[val_idx]
    print(f"[split] train={len(tr_idx)}  val={len(val_idx)}  test={f_te.shape[0]}")

    phys = PhysicsConfig(res=f_tr.shape[1], nu=0.30)
    if a.model == "spectral":
        model = SpectralPIDeepONet(build_modes(a.modes), phys, a.hidden, a.depth).to(dev)
        arch = dict(kind="SpectralPIDeepONet", modes=a.modes, hidden=a.hidden, depth=a.depth)
    else:
        model = FNO2dElasticity(phys, modes=a.fno_modes, width=a.fno_width,
                                n_layers=a.fno_layers).to(dev)
        arch = dict(kind="FNO2dElasticity", modes=a.fno_modes, width=a.fno_width,
                    n_layers=a.fno_layers)
    npar = sum(p.numel() for p in model.parameters())

    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, a.epochs // 3), gamma=0.5)
    f_trd, u_trd = f_tr.to(dev), u_tr.to(dev)
    n = f_tr.shape[0]

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ck = out / f"{a.model}_seed{a.seed}.pt"
    best_val, best_ep = 9e9, -1
    best_on_test_optimistic = 9e9        # diagnostic only, never used for selection
    hist = []
    t0 = time.time()

    for ep in range(1, a.epochs + 1):
        model.train()
        p = torch.randperm(n, device=dev)
        for b in range(0, n, a.batch):
            idx = p[b:b + a.batch]
            loss = torch.mean((model(f_trd[idx]) - u_trd[idx]) ** 2)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sch.step()
        if ep % a.eval_every == 0 or ep == a.epochs:
            e_val = evaluate(model, f_va, u_va, dev)
            e_te = evaluate(model, f_te, u_te, dev)     # recorded, not selected on
            best_on_test_optimistic = min(best_on_test_optimistic, e_te)
            hist.append(dict(epoch=ep, val=e_val, test=e_te))
            if e_val < best_val:
                best_val, best_ep = e_val, ep
                torch.save(dict(state_dict=model.state_dict(), arch=arch,
                                epoch=ep, val=e_val), ck)
            print(f"[{a.model}] ep {ep:4d} val={e_val:.4f} (best {best_val:.4f} "
                  f"@{best_ep}) test={e_te:.4f}")

    # Single, final evaluation of the held-out test set with the selected model.
    model.load_state_dict(torch.load(ck, map_location=dev)["state_dict"])
    test_heldout = evaluate(model, f_te, u_te, dev)
    val_final = evaluate(model, f_va, u_va, dev)

    rec = dict(
        model=a.model, arch=arch, n_params=int(npar), seed=a.seed,
        protocol=dict(selection="min validation error", test_evaluations=1,
                      val_frac=a.val_frac, eval_every=a.eval_every,
                      epochs=a.epochs, batch=a.batch, lr=a.lr, weight_decay=a.wd,
                      optimizer="Adam", scheduler=f"StepLR(step={max(1,a.epochs//3)},gamma=0.5)",
                      n_train=int(len(tr_idx)), n_val=int(len(val_idx)),
                      n_test=int(f_te.shape[0])),
        selected_epoch=int(best_ep), val_error=val_final,
        test_error_heldout=test_heldout,
        test_error_best_epoch_optimistic=best_on_test_optimistic,
        optimism_gap=test_heldout - best_on_test_optimistic,
        checkpoint=str(ck), train_minutes=(time.time() - t0) / 60.0, history=hist,
    )
    json.dump(rec, open(out / f"{a.model}_seed{a.seed}.json", "w"), indent=2)
    print(f"FINAL {a.model} seed={a.seed}: held-out test={test_heldout:.4f} "
          f"(val={val_final:.4f}, selected epoch {best_ep}); "
          f"old best-on-test would have reported {best_on_test_optimistic:.4f} "
          f"[optimism {test_heldout - best_on_test_optimistic:+.4f}]")


if __name__ == "__main__":
    main()
