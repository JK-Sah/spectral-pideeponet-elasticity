#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ablation_canonical.py

The three ablations of Fig. 5 (residual weight, trunk capacity, training-set
size), re-run under the canonical protocol of canonical_linear.py: identical
data generator, optimizer, schedule and epoch budget, checkpoint selected on
the seed-7 validation set, the seed-999 test set evaluated once, three
initialization seeds.  Only the swept quantity differs from the canonical
configuration.

The earlier ablation selected checkpoints on the test set and used a
different schedule, which is where the 3.54% figure came from (the w_PDE=1e-3
point of the residual-weight sweep).  Here the residual weight is chosen on
validation error alone, so the sweep can also say which weight the canonical
configuration should have used.

    python ablation_canonical.py --kind wpde  --value 1e-3 --out runs/abl/wpde_1e-3.json
    python ablation_canonical.py --kind trunk --value 8
    python ablation_canonical.py --kind ntrain --value 300
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch

import cmame_extended_study as st
from cmame_extended_study import (PhysicsConfig, build_modes, SpectralPIDeepONet,
                                  make_dataset, set_seed, get_device)
from canonical_linear import CFG, train, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["wpde", "trunk", "ntrain"])
    ap.add_argument("--value", required=True, type=float)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=None, help="smoke tests only")
    ap.add_argument("--w_pde", type=float, default=None,
                    help="residual weight for the trunk and n_train sweeps (default: canonical)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = get_device(a.device)
    phys = PhysicsConfig()
    w_pde = a.value if a.kind == "wpde" else (CFG["w_pde"] if a.w_pde is None else a.w_pde)
    trunk = int(a.value) if a.kind == "trunk" else CFG["true_modes"]
    n_train = int(a.value) if a.kind == "ntrain" else CFG["n_train"]
    modes = build_modes(trunk)

    d_main = make_dataset(n_train, CFG["n_test"], CFG["true_modes"], CFG["coeff_scale"],
                          phys, seed_train=CFG["seed_train"], seed_test=CFG["seed_test"])
    d_val = make_dataset(CFG["n_val"], 1, CFG["true_modes"], CFG["coeff_scale"], phys,
                         seed_train=CFG["seed_val"], seed_test=12345)
    tr = (d_main["train_f"], d_main["train_u"])
    va = (d_val["train_f"], d_val["train_u"])
    te = (d_main["test_f"], d_main["test_u"])
    print(f"[{a.kind}={a.value:g}] w_pde={w_pde:g} trunk={trunk} n_train={n_train} "
          f"val={va[0].shape[0]} test={te[0].shape[0]} device={dev}")

    runs = []
    for seed in a.seeds:
        set_seed(seed)
        m = SpectralPIDeepONet(modes, phys, hidden=CFG["hidden"], depth=CFG["depth"]).to(dev)
        t0 = time.time()
        m, e_val, ep = train(m, tr, va, phys, dev, seed, w_pde, "pi_spectral_plain",
                             epochs=a.epochs)
        e_te = evaluate(m, te[0], te[1], dev)
        runs.append(dict(seed=seed, val_error=e_val, test_error=e_te, selected_epoch=ep,
                         n_params=int(sum(p.numel() for p in m.parameters())),
                         minutes=(time.time() - t0) / 60))
        print(f"  seed {seed}: val={e_val:.5f} test={e_te:.5f} ep*={ep} "
              f"({runs[-1]['minutes']:.1f} min)")

    val = [r["val_error"] for r in runs]; tst = [r["test_error"] for r in runs]
    out = dict(kind=a.kind, value=a.value, w_pde=w_pde, trunk=trunk, n_train=n_train,
               protocol=dict(CFG, selection="min validation displacement error",
                             test_evaluations=1),
               runs=runs,
               val_mean=float(np.mean(val)), val_std=float(np.std(val)),
               test_mean=float(np.mean(tst)), test_std=float(np.std(tst)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"[AGG {a.kind}={a.value:g}] val {out['val_mean']:.5f} +/- {out['val_std']:.5f}  "
          f"test {out['test_mean']:.5f} +/- {out['test_std']:.5f}")
    print("ABLATION_DONE")


if __name__ == "__main__":
    main()
