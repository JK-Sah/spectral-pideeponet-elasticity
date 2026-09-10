#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
canonical_metrics.py

Two measurements the revision needs.

(1) Full metric suite for the plain physics-informed spectral branch in the
    canonical configuration, over three seeds.  The manuscript previously
    reported seed-repeatability statistics of 3.5% displacement error while
    the headline comparison used 8.4%; the two came from different settings.
    These numbers replace that table so the whole paper quotes one setting.

(2) Boundary error of each model, measured rather than asserted.  The
    spectral trunk satisfies the homogeneous Dirichlet condition by
    construction; the FNO does not enforce it, and the manuscript should
    state how large the resulting violation actually is instead of writing
    "not enforced".

    python canonical_metrics.py --seeds 42 43 44 --out results/metrics.json
"""

import argparse, json, time
from pathlib import Path
import numpy as np
import torch

import cmame_extended_study as st
from cmame_extended_study import (PhysicsConfig, build_modes, SpectralPIDeepONet,
                                  FNO2dElasticity, make_dataset, evaluate_model,
                                  set_seed, get_device)
from canonical_linear import CFG, train, evaluate


def boundary_linf(model, f, dev, bs=100):
    """Max |u| predicted on the domain boundary (exact BC => 0)."""
    model.eval()
    worst = 0.0
    with torch.no_grad():
        for i in range(0, f.shape[0], bs):
            p = model(f[i:i+bs].to(dev)).cpu().numpy()
            edges = np.concatenate([np.abs(p[:, 0, :, :]).ravel(),
                                    np.abs(p[:, -1, :, :]).ravel(),
                                    np.abs(p[:, :, 0, :]).ravel(),
                                    np.abs(p[:, :, -1, :]).ravel()])
            worst = max(worst, float(edges.max()))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results_revision/canonical_metrics.json")
    a = ap.parse_args()
    dev = get_device(a.device)
    phys = PhysicsConfig()
    modes = build_modes(CFG["true_modes"])

    d = make_dataset(CFG["n_train"], CFG["n_test"], CFG["true_modes"],
                     CFG["coeff_scale"], phys,
                     seed_train=CFG["seed_train"], seed_test=CFG["seed_test"])
    dv = make_dataset(CFG["n_val"], 1, CFG["true_modes"], CFG["coeff_scale"],
                      phys, seed_train=CFG["seed_val"], seed_test=12345)
    tr, va = (d["train_f"], d["train_u"]), (dv["train_f"], dv["train_u"])
    te_f, te_u = d["test_f"], d["test_u"]
    print(f"train={tr[0].shape[0]} val={va[0].shape[0]} test={te_f.shape[0]}")

    out = {"config": CFG, "spectral_runs": [], "boundary": {}}

    # ---- (1) plain PI-spectral, full metric suite, three seeds ----
    for seed in a.seeds:
        set_seed(seed)
        m = SpectralPIDeepONet(modes, phys, hidden=CFG["hidden"],
                               depth=CFG["depth"]).to(dev)
        m, e_val, ep = train(m, tr, va, phys, dev, seed, CFG["w_pde"],
                             "pi_spectral_plain", epochs=CFG["epochs"])
        met = evaluate_model(m, te_f, te_u, phys, dev)
        met = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
               for k, v in (met.items() if isinstance(met, dict)
                            else zip(["e_u","e_eps","e_sig","e_en","pde"], met))}
        met.update(seed=seed, selected_epoch=int(ep), val_error=e_val)
        out["spectral_runs"].append(met)
        print(f"[spectral s{seed}] ep*={ep} " +
              " ".join(f"{k}={v:.6f}" for k, v in met.items()
                       if isinstance(v, float)))

    # ---- (2) boundary error: spectral (exact) vs FNO (not enforced) ----
    set_seed(a.seeds[0])
    sp = SpectralPIDeepONet(modes, phys, hidden=CFG["hidden"],
                            depth=CFG["depth"]).to(dev)
    sp, _, _ = train(sp, tr, va, phys, dev, a.seeds[0], CFG["w_pde"],
                     "spectral_bc", epochs=CFG["epochs"])
    out["boundary"]["spectral_linf"] = boundary_linf(sp, te_f, dev)

    set_seed(a.seeds[0])
    fn = FNO2dElasticity(phys, modes=CFG["fno_modes"], width=CFG["fno_width"],
                         n_layers=CFG["fno_layers"]).to(dev)
    fn, _, _ = train(fn, tr, va, phys, dev, a.seeds[0], CFG["w_pde"],
                     "fno_bc", epochs=CFG["epochs_fno"])
    out["boundary"]["fno_linf"] = boundary_linf(fn, te_f, dev)
    out["boundary"]["u_scale_rms"] = float(
        np.sqrt((te_u.numpy() ** 2).mean()))
    out["boundary"]["fno_linf_relative"] = (
        out["boundary"]["fno_linf"] / out["boundary"]["u_scale_rms"])
    print(f"[boundary] spectral Linf={out['boundary']['spectral_linf']:.3e}  "
          f"FNO Linf={out['boundary']['fno_linf']:.3e}  "
          f"({100*out['boundary']['fno_linf_relative']:.2f}% of RMS |u|)")

    es = [r["e_u"] for r in out["spectral_runs"] if "e_u" in r]
    if es:
        out["aggregate_displacement"] = dict(mean=float(np.mean(es)),
                                             std=float(np.std(es)))
        print(f"[AGG spectral displacement] {np.mean(es):.5f} +/- {np.std(es):.5f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=str))
    print("wrote", a.out)
    print("METRICS_DONE")


if __name__ == "__main__":
    main()
