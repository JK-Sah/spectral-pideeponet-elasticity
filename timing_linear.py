#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
timing_linear.py

Per-query cost of the linear-benchmark models, measured on a dedicated CPU
node from trained checkpoints.

The accuracy sweep runs on a GPU node whose CPU is shared with other jobs, so
timings taken there scatter by up to an order of magnitude and are not usable
in the paper.  This script re-measures the same trained checkpoints on an
uncontended CPU allocation under one protocol:

  - fixed thread count, set for BLAS and torch and recorded;
  - trained weights loaded from disk;
  - real held-out test inputs;
  - warm-up repetitions discarded, then many timed repetitions;
  - median and interquartile range reported, across seeds as well as
    repetitions, so contention shows up as spread rather than as a number.

Reports single-query latency (batch 1, the quantity comparable with a linear
solve) and batched throughput per sample, separately.

    python timing_linear.py --ckpt_dir runs/linear/ckpt --trunk 16 --threads 4
"""

import argparse, json, os, time
from pathlib import Path
import numpy as np


def timed(fn, warm, reps):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    a = np.array(ts)
    return dict(median_ms=float(np.median(a)), min_ms=float(a.min()),
                iqr_ms=float(np.percentile(a, 75) - np.percentile(a, 25)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="runs/linear/ckpt")
    ap.add_argument("--trunk", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default="results_revision/timing_linear.json")
    a = ap.parse_args()

    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(a.threads)
    import torch
    torch.set_num_threads(a.threads)
    import cmame_extended_study as st
    from cmame_extended_study import (PhysicsConfig, build_modes,
                                      SpectralPIDeepONet, FNO2dElasticity,
                                      make_dataset)
    from canonical_linear import CFG, AnchoredSpectral, closed_form_readout, ClosedFormModel

    phys = PhysicsConfig()
    modes = build_modes(a.trunk)
    d = make_dataset(CFG["n_train"], CFG["n_test"], CFG["true_modes"],
                     CFG["coeff_scale"], phys,
                     seed_train=CFG["seed_train"], seed_test=CFG["seed_test"])
    te_f = d["test_f"]
    f1 = te_f[:1].clone()
    nb = min(a.batch, te_f.shape[0])
    out = {"protocol": dict(threads=a.threads, batch=nb, warm=a.warm, reps=a.reps,
                            trunk=a.trunk, weights="trained checkpoints",
                            inputs="held-out test fields",
                            statistic="median over repetitions; across-seed spread reported")}
    print(f"[protocol] threads={a.threads} batch={nb} warm={a.warm} reps={a.reps} M={a.trunk}")

    # closed-form read-out (no checkpoint; refit from the same training data)
    W, b = closed_form_readout(modes, st.make_grid(phys.res), d["train_f"], d["train_u"])
    cf = ClosedFormModel(W, b, modes, phys).eval()
    with torch.no_grad():
        s_cf = timed(lambda: cf(f1), a.warm, a.reps)
        b_cf = timed(lambda: cf(te_f[:nb]), a.warm, max(10, a.reps // 5))
    out["closed_form"] = dict(single_query=s_cf,
                              batched_per_sample={k: v / nb for k, v in b_cf.items()},
                              n_params=int(W.size + b.size))
    print(f"[closed-form        ] single {s_cf['median_ms']:8.3f} ms  "
          f"batched/sample {b_cf['median_ms']/nb:8.4f} ms")

    for tag in ("pi_spectral_plain", "pi_spectral_anchored",
                "data_only_spectral", "fno"):
        singles, batches, npar = [], [], None
        for seed in a.seeds:
            ck = Path(a.ckpt_dir) / f"{tag}_M{a.trunk}_seed{seed}.pt"
            if not ck.exists():
                continue
            blob = torch.load(ck, map_location="cpu")
            if tag == "fno":
                m = FNO2dElasticity(phys, modes=CFG["fno_modes"],
                                    width=CFG["fno_width"], n_layers=CFG["fno_layers"])
            elif tag == "pi_spectral_anchored":
                m = AnchoredSpectral(modes, phys, d["train_f"], d["train_u"],
                                     CFG["hidden"], CFG["depth"])
            else:
                m = SpectralPIDeepONet(modes, phys, hidden=CFG["hidden"],
                                       depth=CFG["depth"])
            m.load_state_dict(blob["state_dict"]); m.eval()
            npar = sum(p.numel() for p in m.parameters())
            with torch.no_grad():
                singles.append(timed(lambda: m(f1), a.warm, a.reps)["median_ms"])
                batches.append(timed(lambda: m(te_f[:nb]), a.warm,
                                     max(10, a.reps // 5))["median_ms"] / nb)
        if not singles:
            print(f"[{tag:20s}] no checkpoints found"); continue
        out[tag] = dict(n_params=int(npar), n_seeds=len(singles),
                        single_query_ms=dict(median=float(np.median(singles)),
                                             lo=float(min(singles)), hi=float(max(singles))),
                        batched_ms_per_sample=dict(median=float(np.median(batches)),
                                                   lo=float(min(batches)), hi=float(max(batches))))
        print(f"[{tag:20s}] single {np.median(singles):8.3f} ms "
              f"[{min(singles):.3f}-{max(singles):.3f}]   "
              f"batched/sample {np.median(batches):8.4f} ms "
              f"[{min(batches):.4f}-{max(batches):.4f}]")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)
    print("TIMINGLINEAR_DONE")


if __name__ == "__main__":
    main()
