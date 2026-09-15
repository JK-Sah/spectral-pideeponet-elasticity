#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
timing_v2.py

Per-query cost of the learned and classical finite-strain methods under a
single, stated protocol.

The earlier measurement timed the neural models in batches of 128 with eight
threads on freshly instantiated weights and divided total runtime by the
sample count, while the classical figures were individual solves.  That
compares batched throughput against single-solve latency, so the two are not
the same quantity.

Reported here, separately and explicitly:

  single-query latency   one input, one output, batch size 1.  This is the
                         quantity that is comparable to a Newton solve or a
                         reduced-order solve, both of which are inherently
                         one query at a time.
  batched throughput     amortized cost per sample when many queries are
                         issued together (batch 128).  Favourable to the
                         neural models and reported as such.

Protocol, applied identically to every method:
  - fixed thread count, set for numpy/BLAS and torch and recorded;
  - trained checkpoints loaded from disk (not random weights);
  - real test inputs (not torch.randn);
  - warm-up repetitions discarded, then a fixed number of timed repetitions;
  - median and interquartile range reported, not just the minimum.

    python timing_v2.py --data data/nonlinear.npz --ckpt_dir runs/nonlinear_v2 \
        --threads 8 --out results_revision/nonlinear/timing_v2.json
"""

import argparse, json, os, time
from pathlib import Path
import numpy as np


def _timed(fn, warm, reps):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.array(ts)
    return dict(median_ms=float(np.median(ts)), min_ms=float(ts.min()),
                iqr_ms=float(np.percentile(ts, 75) - np.percentile(ts, 25)),
                reps=int(reps), warmup=int(warm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/nonlinear.npz")
    ap.add_argument("--ckpt_dir", default="runs/nonlinear_v2")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--n_single", type=int, default=50)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=111)
    ap.add_argument("--out", default="results_revision/nonlinear/timing_v2.json")
    a = ap.parse_args()

    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(a.threads)
    import torch
    torch.set_num_threads(a.threads)
    from cmame_extended_study import (PhysicsConfig, build_modes, SpectralPIDeepONet,
                                      FNO2dElasticity, get_device)
    from nonlinear_fem_fast import FastNonlinearFEM

    dev = torch.device("cpu")
    z = np.load(a.data)
    f_te = np.asarray(z["f_te"], dtype=np.float32)
    res = f_te.shape[1]
    ft = torch.tensor(f_te)
    phys = PhysicsConfig(res=res, nu=0.30)
    out = {"protocol": dict(threads=a.threads, device="cpu", batch=a.batch,
                            warmup=a.warm, reps=a.reps,
                            inputs="held-out test fields",
                            weights="trained checkpoints",
                            statistic="median over repetitions (IQR reported)"),
           "methods": {}}
    print(f"[protocol] threads={a.threads} batch={a.batch} warm={a.warm} reps={a.reps}")

    # ---------------- neural models, from trained checkpoints ----------------
    for name in ("spectral", "fno"):
        ck = Path(a.ckpt_dir) / f"{name}_seed{a.seed}.pt"
        if not ck.exists():
            print(f"[skip] {name}: no checkpoint at {ck}")
            continue
        blob = torch.load(ck, map_location=dev)
        arch = blob["arch"]
        if arch["kind"] == "SpectralPIDeepONet":
            model = SpectralPIDeepONet(build_modes(arch["modes"]), phys,
                                       arch["hidden"], arch["depth"])
        else:
            model = FNO2dElasticity(phys, modes=arch["modes"], width=arch["width"],
                                    n_layers=arch["n_layers"])
        model.load_state_dict(blob["state_dict"]); model.eval()
        npar = sum(p.numel() for p in model.parameters())

        one = ft[:1].clone()
        with torch.no_grad():
            single = _timed(lambda: model(one), a.warm, max(a.reps, a.n_single))
            nb = min(a.batch, ft.shape[0])
            bt = _timed(lambda: model(ft[:nb]), a.warm, a.reps)
        bt_per = {k: (v / nb if k.endswith("_ms") else v) for k, v in bt.items()}
        out["methods"][name] = dict(kind="neural", n_params=int(npar),
                                    single_query=single,
                                    batched_total=bt, batch_size=int(nb),
                                    batched_per_sample=bt_per)
        print(f"[{name:8s}] single-query {single['median_ms']:8.3f} ms   "
              f"batched/sample {bt_per['median_ms']:8.3f} ms   "
              f"ratio {single['median_ms']/bt_per['median_ms']:6.1f}x")

    # ---------------- classical: Newton-FEM, single query ----------------
    fem = FastNonlinearFEM(res)
    fg = f_te[0].astype(np.float64)
    fem_t = _timed(lambda: fem.solve(fg, n_steps=6), max(1, a.warm // 2), max(3, a.reps // 4))
    _, info = fem.solve(fg, n_steps=6, collect_stats=True)
    out["methods"]["newton_fem_vectorized"] = dict(
        kind="classical", single_query=fem_t, newton_iters=int(info["newton_iters"]),
        breakdown_ms=info["breakdown_ms"], note="vectorized assembly; 6 load steps")
    print(f"[newton-fem] single-query {fem_t['median_ms']:8.1f} ms "
          f"({info['newton_iters']} Newton its)")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)
    print("TIMINGV2_DONE")


if __name__ == "__main__":
    main()
