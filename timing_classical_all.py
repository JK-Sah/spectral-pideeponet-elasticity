#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
timing_classical_all.py

Every classical per-query cost the paper quotes, on one allocation under one
protocol: the fixed-operator (manufactured) benchmark across its mesh sweep,
and the heterogeneous modulus-field benchmark at the matched mesh.

This supersedes timing_classical_linear.py, which measured only the
fixed-operator side, and the timing incidentally taken inside rom_field.py,
which measured only the heterogeneous side.  Quoting the two together left the
paper reporting one quantity -- the cost of a finite-element solve when the
operator changes per sample -- as both 11 ms and 36.5 ms at the same mesh.

Neither figure was measured wrongly; they measure different things.  The larger
one re-ran the solver's constructor for every query, so it charged each query
for building the mesh, deriving the boundary-condition index set and assembling
the element topology -- work that is fixed for the whole sweep and that no
per-query workflow would repeat.  The smaller one precomputed that topology and
timed only what genuinely changes: re-assembling K(E), re-factorizing it and
solving.  The second is the honest per-query figure, and it is the one this
script reports; the inflated variant is recorded beside it as
per_query_with_setup_ms so the difference stays visible rather than becoming a
discrepancy between tables.

The two benchmarks also used different assembly implementations, which is the
other half of the discrepancy.  Here both go through the vectorized COO
assembly of HeteroFieldQ4, which reproduces the fixed-operator solver exactly
when the modulus field is uniform -- the equivalence is asserted below rather
than assumed.

Reported per mesh, for the fixed-operator benchmark:

  setup            assemble and factorize once.  Offline; not a query cost.
  solve_only       the triangular solve given that factorization.  The
                   single-query latency when the operator is reused.
  amortized        setup spread over the test set, plus solve_only.
  per_query        re-assemble, re-factorize and solve, mesh topology
                   precomputed.  The figure that applies when the operator
                   changes per sample.

    python timing_classical_all.py --threads 4
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
    ap.add_argument("--het_ranks", type=int, nargs="+",
                    default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--het_data", default="data/hetero_field.npz")
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--out", default="results_revision/linear/timing_classical_all.json")
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(a.threads)

    from scipy.linalg import lu_factor, lu_solve
    from revision_new_benchmarks import FemSolver
    from rom_baseline import assemble_K_free
    from hetero_field import HeteroFieldQ4, nodal_to_elem
    from cmame_extended_study import PhysicsConfig, make_dataset
    from canonical_linear import CFG

    phys = PhysicsConfig()
    lam, mu = phys.lame
    nu = phys.nu

    def forcing_at(res, n=1):
        """Manufactured forcing on a res x res grid.

        The sine coefficients depend only on the seed, so every resolution sees
        the same physical field and the mesh sweep varies discretization alone.
        """
        ph = PhysicsConfig(res=res)
        dd = make_dataset(CFG["n_train"], max(n, 1), CFG["true_modes"],
                          CFG["coeff_scale"], ph,
                          seed_train=CFG["seed_train"], seed_test=CFG["seed_test"])
        return dd["test_f"].numpy().astype(np.float64), ph

    f_te, _ = forcing_at(phys.res, a.n_test)
    out = {"protocol": dict(threads=a.threads, warm=a.warm, reps=a.reps,
                            n_test=a.n_test,
                            assembly="vectorized COO (HeteroFieldQ4), uniform E "
                                     "for the fixed-operator benchmark",
                            note="one exclusive allocation; same node and thread "
                                 "count as the learned-model timings")}
    print(f"[protocol] threads={a.threads} warm={a.warm} reps={a.reps}")

    # ---- the two assembly paths must agree before either is timed --------
    res0 = 29
    fs0 = FemSolver(res0, lam, mu)
    hf0 = HeteroFieldQ4(res0, nu=nu)
    u_ref = fs0.solve(f_te[0])[0]
    u_vec = hf0.solve(f_te[0], np.ones((res0 - 1, res0 - 1)))[0]
    rel = float(np.linalg.norm(u_ref - u_vec) / np.linalg.norm(u_ref))
    print(f"[check] uniform-E vectorized assembly vs fixed-operator solver: "
          f"rel diff {rel:.3e}")
    assert rel < 1e-10, f"assembly paths disagree ({rel:.3e}); timings would not be comparable"
    out["equivalence_rel_diff"] = rel

    # ---- fixed-operator benchmark, mesh sweep ---------------------------
    fem_rows = []
    for res in a.meshes:
        fg = forcing_at(res)[0][0]
        t0 = time.perf_counter(); soln = FemSolver(res, lam, mu)
        setup_ms = (time.perf_counter() - t0) * 1e3
        solve_only = timed(lambda: soln.solve(fg), a.warm, a.reps)

        hf = HeteroFieldQ4(res, nu=nu)
        E1 = np.ones((res - 1, res - 1))
        per_query = timed(lambda: hf.solve(fg, E1), a.warm, a.reps)
        with_setup = timed(lambda: FemSolver(res, lam, mu).solve(fg),
                           max(1, a.warm // 3), max(3, a.reps // 10))

        amortized = setup_ms / a.n_test + solve_only["median_ms"]
        fem_rows.append(dict(res=res, n_dof=2 * res * res, setup_ms=setup_ms,
                             solve_only_ms=solve_only["median_ms"],
                             amortized_ms=amortized,
                             per_query_ms=per_query["median_ms"],
                             per_query_with_setup_ms=with_setup["median_ms"]))
        print(f"[fem {res-1:3d}x{res-1:<3d}] setup {setup_ms:8.1f} | "
              f"solve-only {solve_only['median_ms']:7.3f} | "
              f"amortized {amortized:7.3f} | per-query {per_query['median_ms']:7.3f} "
              f"| per-query+setup {with_setup['median_ms']:8.2f}")
    out["fem"] = fem_rows

    # ---- fixed-operator reduced model (operator reused offline) ---------
    res = 29
    soln = FemSolver(res, lam, mu)
    K_free, free = assemble_K_free(res, lam, mu)
    snaps = np.array([soln.solve(f_te[i])[0].reshape(-1)[free]
                      for i in range(min(200, a.n_test))])
    U, _, _ = np.linalg.svd(snaps.T, full_matrices=False)
    bf = soln.rhs(f_te[0])[free]
    rom_rows = []
    for r in a.ranks:
        if r > U.shape[1] or r > snaps.shape[0]:
            print(f"[rom  r={r:4d}] skipped: only {min(U.shape[1], snaps.shape[0])} "
                  f"independent snapshots available")
            continue
        V = U[:, :r]
        luf = lu_factor(V.T @ (K_free @ V))
        online = timed(lambda: V @ lu_solve(luf, V.T @ bf), a.warm, a.reps)
        rom_rows.append(dict(rank=r, online_ms=online["median_ms"]))
        print(f"[rom  r={r:4d}] online {online['median_ms']:8.4f} ms")
    out["rom"] = rom_rows

    # ---- offline cost of each classical construction -------------------
    # The break-even column divides these by the per-query saving, so they have
    # to come from the same protocol as the per-query numbers rather than from
    # whichever script happened to print a setup time.
    res = 29
    n_snap = min(200, a.n_test)
    t0 = time.perf_counter()
    soln_o = FemSolver(res, lam, mu)
    fem_offline_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    snaps_o = np.array([soln_o.solve(f_te[i])[0].reshape(-1)[free]
                        for i in range(n_snap)])
    snap_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    Uo, _, _ = np.linalg.svd(snaps_o.T, full_matrices=False)
    svd_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    Vo = Uo[:, :32]
    _ = lu_factor(Vo.T @ (K_free @ Vo))
    proj_ms = (time.perf_counter() - t0) * 1e3

    out["offline"] = dict(
        n_snapshots=n_snap,
        fem_assemble_factorize_ms=fem_offline_ms,
        rom_snapshot_ms=snap_ms, rom_svd_ms=svd_ms,
        rom_projection_r32_ms=proj_ms,
        rom_total_ms=snap_ms + svd_ms + proj_ms,
        note="the reduced model is charged for generating its own snapshots, "
             "which on this manufactured family is the dominant offline term")
    print(f"[offline] fem assemble+factorize {fem_offline_ms:8.1f} ms | "
          f"rom snapshots {snap_ms:8.1f} + svd {svd_ms:7.1f} + project(r=32) "
          f"{proj_ms:6.1f} = {snap_ms + svd_ms + proj_ms:8.1f} ms")

    # ---- heterogeneous benchmark: operator changes per query -----------
    hp = Path(a.het_data)
    if not hp.exists():
        print(f"[warn] {hp} not found; skipping the heterogeneous block")
        out["hetero"] = None
    else:
        z = np.load(hp)

        def pick(*names):
            for n in names:
                if n in z:
                    return n
            raise KeyError(f"none of {names} in {list(z.keys())}")

        # float64 throughout: a float32 snapshot matrix gives a rank-deficient
        # POD basis whose trailing columns are noise, which would time garbage.
        E_te = z[pick("E_te", "E_test")].astype(np.float64)
        f_te_h = z[pick("f_te", "f_test")].astype(np.float64)
        u_basis = z[pick("u_tr", "u_train")].astype(np.float64)
        res_h = f_te_h.shape[1]
        hf = HeteroFieldQ4(res_h, nu=nu)
        free_h = hf.free
        print(f"[hetero] res={res_h} n_test={len(E_te)} keys={list(z.keys())}")

        Ee0 = nodal_to_elem(E_te[0])
        fem_q = timed(lambda: hf.solve(f_te_h[0], Ee0), a.warm, a.reps)
        print(f"[hetero fem ] per-query {fem_q['median_ms']:8.3f} ms "
              f"(re-assemble + re-factorize + solve)")

        # POD basis from the training snapshots, never the test ones.
        snaps_h = u_basis.reshape(u_basis.shape[0], -1)[:, free_h]
        Uh, _, _ = np.linalg.svd(snaps_h.T, full_matrices=False)
        bh = hf.rhs(f_te_h[0])[free_h]
        het_rows = []
        for r in a.het_ranks:
            if r > Uh.shape[1] or r > snaps_h.shape[0]:
                print(f"[hetero rom r={r:4d}] skipped: only "
                      f"{min(Uh.shape[1], snaps_h.shape[0])} snapshots available")
                continue
            Vr = Uh[:, :r]

            def one_query(Vr=Vr):
                K = hf.assemble(Ee0)
                Kf = K[np.ix_(free_h, free_h)]
                Kr = Vr.T @ (Kf @ Vr)
                return Vr @ lu_solve(lu_factor(Kr), Vr.T @ bh)

            online = timed(one_query, a.warm, a.reps)
            het_rows.append(dict(rank=r, online_ms=online["median_ms"]))
            print(f"[hetero rom r={r:4d}] online {online['median_ms']:8.3f} ms")
        out["hetero"] = dict(res=res_h, fem_per_query_ms=fem_q["median_ms"],
                             rom=het_rows)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)
    print("TIMINGCLASSICALALL_DONE")


if __name__ == "__main__":
    main()
