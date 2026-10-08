#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
timing_nonlinear_v3.py

Every finite-strain timing in one job, on one exclusive allocation, at the same
four threads and under the same protocol as the linear benchmarks:

  * Newton-FEM, POD-Galerkin (r=32) and POD-DEIM (r=32, m=128/192/256): 5
    warm-up solves, then the median of 50 timed solves cycling through the
    first 50 test cases; plus the Newton-FEM wall-clock breakdown averaged over
    those cases;
  * FNO and spectral DeepONet, from the trained checkpoints: single-query
    latency (10 warm-ups, median of 100) and batched throughput (batch 128,
    median of 20, per sample), denormals flushed;
  * the equivalence checks and the scikit-fem cross-check of fem_validation.py,
    re-run here so they come from the same allocation;
  * the offline cost of the reduced models beyond the shared training corpus:
    the POD decomposition, the reduced trajectories that supply the DEIM force
    snapshots, and the DEIM basis and sample-mesh construction for each m.

The earlier finite-strain timings were taken on shared nodes; the linear ones
showed that contention can inflate a timing tenfold.

    python timing_nonlinear_v3.py --data data/nonlinear.npz --ckpt_dir runs/nonlinear_v2_long --out ...
"""
import argparse, json, os, time
from pathlib import Path

import numpy as np
import torch

from nonlinear_fem_fast import FastNonlinearFEM
from pod_deim import build_pod, collect_trajectory_forces, reduced_galerkin_solve
from pod_deim_v2 import build_rom_v2, reduced_deim_solve_v2
from cmame_extended_study import PhysicsConfig, build_modes, SpectralPIDeepONet, FNO2dElasticity
import fem_validation as fv

N_STEPS, TOL, MAXIT = 6, 1e-8, 60          # reduced models
FEM_TOL, FEM_MAXIT = 1e-9, 50              # reference Newton-FEM, as the data were generated


def med_solve(fn, inputs, warm=5, reps=50):
    for i in range(warm):
        fn(inputs[i % len(inputs)])
    ts = []
    for i in range(reps):
        t0 = time.perf_counter(); fn(inputs[i % len(inputs)]); ts.append(1e3 * (time.perf_counter() - t0))
    return dict(median_ms=float(np.median(ts)), q25=float(np.percentile(ts, 25)),
                q75=float(np.percentile(ts, 75)), reps=reps, warm=warm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/nonlinear.npz")
    ap.add_argument("--ckpt_dir", default="runs/nonlinear_v2_long")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--m", type=int, nargs="+", default=[128, 192, 256])
    ap.add_argument("--n_cases", type=int, default=50)
    ap.add_argument("--out", default="results_revision/nonlinear/timing_v3.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.set_flush_denormal(True)
    out = dict(protocol=dict(threads=a.threads, omp=os.environ.get("OMP_NUM_THREADS"),
                             flush_denormal=True, classical="5 warm-up, median of 50",
                             neural_single="10 warm-up, median of 100",
                             neural_batched="batch 128, median of 20, per sample"))

    z = np.load(a.data)
    f_all, u_all, f_te = z["f_tr"], z["u_tr"], z["f_te"]
    perm = np.random.default_rng(111).permutation(f_all.shape[0])
    tr_idx = perm[int(round(0.2 * f_all.shape[0])):]
    fem = FastNonlinearFEM(f_all.shape[1]); free = fem.free
    cases = [f_te[i].astype(np.float64) for i in range(a.n_cases)]

    # Newton-FEM: cost and wall-clock breakdown
    out["newton_fem"] = med_solve(lambda f: fem.solve(f, n_steps=N_STEPS, tol=FEM_TOL,
                                                      maxit=FEM_MAXIT), cases)
    acc = {}
    for f in cases[:20]:
        _, info = fem.solve(f, n_steps=N_STEPS, tol=FEM_TOL, maxit=FEM_MAXIT, collect_stats=True)
        for k, v in info["breakdown_ms"].items():
            acc[k] = acc.get(k, 0.0) + v
    tot = sum(acc.values())
    out["newton_fem"]["breakdown_frac"] = {k: v / tot for k, v in acc.items()}
    print(f"[newton-fem] {out['newton_fem']['median_ms']:.1f} ms  breakdown "
          f"{ {k: round(v, 3) for k, v in out['newton_fem']['breakdown_frac'].items()} }")

    # Reduced models, built from the training part of the seed-111 split
    S = np.asarray(u_all[tr_idx].reshape(len(tr_idx), -1)[:, free], dtype=np.float64)
    t0 = time.perf_counter()
    V = np.asarray(build_pod(S, 32)[0], dtype=np.float64)
    out["offline_s"] = dict(pod_svd=time.perf_counter() - t0, n_snapshots=int(S.shape[0]))
    out["pod_galerkin_r32"] = med_solve(
        lambda f: reduced_galerkin_solve(fem, f, V, n_steps=N_STEPS, tol=TOL, maxit=MAXIT), cases)
    print(f"[pod-galerkin] {out['pod_galerkin_r32']['median_ms']:.1f} ms")
    t0 = time.perf_counter()
    Fs = collect_trajectory_forces(fem, V, [f_all[i].astype(np.float64) for i in tr_idx[:40]],
                                   n_steps=N_STEPS)
    out["offline_s"]["deim_force_snapshots"] = time.perf_counter() - t0
    out["pod_deim_r32"] = {}
    for m in a.m:
        t0 = time.perf_counter()
        rom = build_rom_v2(fem, S, 32, m, Fs)
        out["offline_s"][f"deim_build_m{m}"] = time.perf_counter() - t0
        conv = [f for f in cases if reduced_deim_solve_v2(fem, f, rom, n_steps=N_STEPS)[0] is not None]
        rec = med_solve(lambda f: reduced_deim_solve_v2(fem, f, rom, n_steps=N_STEPS), conv)
        rec["n_cases_converged"] = len(conv)
        out["pod_deim_r32"][str(m)] = rec
        print(f"[pod-deim m={m}] {rec['median_ms']:.1f} ms over {len(conv)} converging cases")

    # Neural operators from their checkpoints
    phys = PhysicsConfig(res=f_all.shape[1], nu=0.30)
    ft = torch.tensor(f_te, dtype=torch.float32)
    out["neural"] = {}
    for ck in sorted(Path(a.ckpt_dir).glob("*.pt")):
        d = torch.load(ck, map_location="cpu"); arch = d["arch"]
        if arch["kind"] == "SpectralPIDeepONet":
            mdl = SpectralPIDeepONet(build_modes(arch["modes"]), phys, arch["hidden"], arch["depth"])
        else:
            mdl = FNO2dElasticity(phys, modes=arch["modes"], width=arch["width"],
                                  n_layers=arch["n_layers"])
        mdl.load_state_dict(d["state_dict"]); mdl.eval()
        with torch.no_grad():
            single = med_solve(lambda i: mdl(ft[int(i):int(i) + 1]), list(range(100)), warm=10, reps=100)
            ts = []
            for _ in range(3):
                mdl(ft[:128])
            for _ in range(20):
                t0 = time.perf_counter(); mdl(ft[:128]); ts.append(1e3 * (time.perf_counter() - t0))
        out["neural"][ck.stem] = dict(single_query_ms=single["median_ms"],
                                      batched_ms_per_sample=float(np.median(ts)) / 128,
                                      n_params=int(sum(p.numel() for p in mdl.parameters())))
        print(f"[{ck.stem}] single {single['median_ms']:.3f} ms  batched "
              f"{out['neural'][ck.stem]['batched_ms_per_sample']:.4f} ms/sample")

    # Equivalence of the vectorized solver and the scikit-fem cross-check, same allocation
    out["equivalence"] = fv.part_A()
    out["skfem_crosscheck"] = fv.part_D()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("TIMINGNL3_DONE")


if __name__ == "__main__":
    main()
