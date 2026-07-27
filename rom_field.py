#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rom_field.py

Strong classical baseline for the Route-B random-E(x)-field benchmark:
a global POD-Galerkin reduced-order model.

Unlike the fixed-operator case (where one offline factorization serves every
query), here the operator K(E) changes each sample, so the reduced model must
(i) assemble K(E) for the query, (ii) project it onto the POD basis, and
(iii) solve the r x r system.  This script reports, per POD rank:

  * the POD projection FLOOR (best possible ROM accuracy at that rank),
  * the actual Galerkin-ROM error vs the res-113 FEM ground truth,
  * the online per-query cost (assemble + project + solve),

plus, as reference Pareto points, the res-29 full FEM error/cost vs the same
ground truth.  The point of interest is the Kolmogorov n-width: for a smooth
high-dimensional random field the POD spectrum decays slowly, so matching the
surrogate's accuracy needs a large rank, and the per-query projection cost then
erodes any speed advantage over simply re-solving the FEM.

    python rom_field.py --data data/hetero_field.npz --ranks 8 16 32 64 128
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from hetero_field import HeteroFieldQ4, nodal_to_elem


def rel_l2_block(pred, true):
    num = np.linalg.norm((pred - true).reshape(pred.shape[0], -1), axis=1)
    den = np.linalg.norm(true.reshape(true.shape[0], -1), axis=1) + 1e-12
    return float(np.mean(num / den))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/hetero_field.npz")
    ap.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--res", type=int, default=29)
    ap.add_argument("--nu", type=float, default=0.30)
    ap.add_argument("--out", default="results_revision/rom_field.json")
    args = ap.parse_args()
    np.seterr(divide="ignore", over="ignore", invalid="ignore")

    z = np.load(args.data)
    f_tr, E_tr, u_tr = z["f_tr"], z["E_tr"], z["u_tr"]
    f_te, E_te, u_te = z["f_te"], z["E_te"], z["u_te"]
    res = args.res
    solver = HeteroFieldQ4(res, nu=args.nu)
    free = solver.free
    Ntr, Nte = u_tr.shape[0], u_te.shape[0]

    # POD basis from training displacement snapshots (free dofs).
    Str = u_tr.reshape(Ntr, -1)[:, free].T                 # [nfree, Ntr]
    t0 = time.perf_counter()
    U, sv, _ = np.linalg.svd(Str, full_matrices=False)
    svd_ms = 1000 * (time.perf_counter() - t0)
    rmax = max(args.ranks)
    V = U[:, :rmax]                                         # [nfree, rmax]

    u_te_free = u_te.reshape(Nte, -1)[:, free]              # [Nte, nfree]

    print(f"POD: {Ntr} snapshots, SVD {svd_ms:.0f} ms")
    print(f"singular values (first 8): {np.array2string(sv[:8], precision=2)}")
    energy = np.cumsum(sv**2) / np.sum(sv**2)
    for r in args.ranks:
        if r <= len(energy):
            print(f"  rank {r:4d}: captured POD energy {energy[r-1]:.4f}")

    # Precompute test RHS (free) once.
    b_free = np.stack([solver.rhs(f_te[i])[free] for i in range(Nte)])

    print(f"\n{'rank':>5} {'POD_floor':>10} {'ROM_err(vsGT)':>13} "
          f"{'online_ms':>10}")
    rows = []
    for r in args.ranks:
        Vr = V[:, :r]
        # POD projection floor (Euclidean-optimal) vs ground truth.
        proj = u_te_free @ Vr @ Vr.T
        floor = rel_l2_block(proj, u_te_free)

        # Galerkin ROM: per-query assemble + project + solve.
        u_pred = np.zeros_like(u_te)
        t0 = time.perf_counter()
        for i in range(Nte):
            K = solver.assemble(nodal_to_elem(E_te[i]))
            Kf = K[np.ix_(free, free)]
            KV = Kf @ Vr                                    # [nfree, r]
            Kr = Vr.T @ KV                                  # [r, r]
            br = Vr.T @ b_free[i]
            c = lu_solve(lu_factor(Kr), br)
            full = np.zeros(2 * res * res); full[free] = Vr @ c
            u_pred[i] = full.reshape(res, res, 2)
        online_ms = 1000 * (time.perf_counter() - t0) / Nte
        err = rel_l2_block(u_pred, u_te)
        print(f"{r:>5} {floor:>10.4f} {err:>13.4f} {online_ms:>10.3f}")
        rows.append(dict(rank=r, pod_floor=floor, rom_err=err, online_ms=online_ms))

    # Reference: full res-29 FEM error vs res-113 ground truth + per-query cost.
    t0 = time.perf_counter()
    fem_err_num = 0.0
    u_fem = np.zeros_like(u_te)
    for i in range(Nte):
        uf, _ = solver.solve(f_te[i], nodal_to_elem(E_te[i]))
        u_fem[i] = uf
    fem_ms = 1000 * (time.perf_counter() - t0) / Nte
    fem_err = rel_l2_block(u_fem, u_te)
    print(f"\nres-{res} full FEM vs res-113 GT: err {fem_err:.4f}, "
          f"{fem_ms:.3f} ms/query (assemble+factorize+solve, no reuse)")

    print("\nSurrogate reference (Route-B, from training runs):")
    print("  FNO+PDE ......... 0.189 @ ~0.03 ms/query")
    print("  spectral (M-sweep in progress)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(ranks=rows, fem_res29_err=fem_err, fem_res29_ms=fem_ms,
                   singular_values=sv[:64].tolist()), open(args.out, "w"), indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
