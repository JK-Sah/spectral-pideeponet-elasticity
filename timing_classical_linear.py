#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
timing_classical_linear.py

Re-times the classical baselines of the linear benchmark on the same CPU
allocation and thread count used for the learned models, so the whole linear
ledger comes from one machine under one protocol.

The published linear timings were taken on a 4-core workstation while the
corrected neural timings were taken on the cluster.  Those differ by a factor
of three to four on identical work, so quoting them in one table would repeat
the cross-protocol comparison the review objected to.  Everything here is
measured where timing_linear.py measured the learned models.

Three costs are reported for the finite-element solver, because which one is
appropriate depends on the workload:

  amortized   assembly and factorization once, spread over the test set, plus
              the per-sample triangular solve.  The many-query figure.
  per query   assemble, factorize and solve for every sample.  The figure
              that applies when the operator changes per sample.
  solve only  the triangular solve alone, given an existing factorization.
              The true single-query latency in the many-query setting.

    python timing_classical_linear.py --threads 4
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
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--meshes", type=int, nargs="+", default=[15, 29, 57, 113])
    ap.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--out", default="results_revision/linear/timing_classical.json")
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(a.threads)

    import scipy.sparse.linalg as spla
    from revision_new_benchmarks import FemSolver, lame
    from rom_baseline import assemble_K_free
    from cmame_extended_study import PhysicsConfig, make_dataset
    from canonical_linear import CFG

    phys = PhysicsConfig()
    lam, mu = phys.lame

    def forcing_at(res, n=1):
        """Manufactured forcing sampled on a res x res grid.

        The sine coefficients depend only on the seed, so every resolution
        sees the same physical field and the mesh sweep varies discretization
        alone."""
        ph = PhysicsConfig(res=res)
        dd = make_dataset(CFG["n_train"], max(n, 1), CFG["true_modes"],
                          CFG["coeff_scale"], ph,
                          seed_train=CFG["seed_train"], seed_test=CFG["seed_test"])
        return dd["test_f"].numpy().astype(np.float64), ph

    f_te, _ = forcing_at(phys.res, a.n_test)
    out = {"protocol": dict(threads=a.threads, warm=a.warm, reps=a.reps,
                            n_test=a.n_test,
                            note="same allocation and thread count as timing_linear.py")}
    print(f"[protocol] threads={a.threads} warm={a.warm} reps={a.reps}")

    # ---------------- finite element, three protocols ----------------
    fem_rows = []
    for res in a.meshes:
        fg = forcing_at(res)[0][0]           # forcing on this mesh's own grid
        t0 = time.perf_counter(); soln = FemSolver(res, lam, mu)
        setup_ms = (time.perf_counter() - t0) * 1e3
        solve_only = timed(lambda: soln.solve(fg), a.warm, a.reps)
        per_query = timed(lambda: FemSolver(res, lam, mu).solve(fg),
                          max(1, a.warm // 3), max(3, a.reps // 10))
        amortized = setup_ms / a.n_test + solve_only["median_ms"]
        rec = dict(res=res, n_dof=2 * res * res, setup_ms=setup_ms,
                   solve_only_ms=solve_only["median_ms"],
                   amortized_ms=amortized, per_query_ms=per_query["median_ms"])
        fem_rows.append(rec)
        print(f"[fem {res-1:3d}x{res-1:<3d}] setup {setup_ms:8.1f} ms | "
              f"solve-only {solve_only['median_ms']:7.3f} | "
              f"amortized {amortized:7.3f} | per-query {per_query['median_ms']:8.2f}")
    out["fem"] = fem_rows

    # ---------------- POD-Galerkin reduced-order model ----------------
    from scipy.linalg import lu_factor, lu_solve
    res = 29
    soln = FemSolver(res, lam, mu)
    K_free, free = assemble_K_free(res, lam, mu)
    snaps = np.array([soln.solve(f_te[i])[0].reshape(-1)[free]
                      for i in range(min(200, a.n_test))])
    U, sv, _ = np.linalg.svd(snaps.T, full_matrices=False)
    bf = soln.rhs(f_te[0])[free]
    rom_rows = []
    for r in a.ranks:
        V = U[:, :r]
        luf = lu_factor(V.T @ (K_free @ V))
        online = timed(lambda: V @ lu_solve(luf, V.T @ bf), a.warm, a.reps)
        rom_rows.append(dict(rank=r, online_ms=online["median_ms"]))
        print(f"[rom r={r:4d}] online {online['median_ms']:8.4f} ms")
    out["rom"] = rom_rows

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)
    print("TIMINGCLASSICAL_DONE")


if __name__ == "__main__":
    main()
