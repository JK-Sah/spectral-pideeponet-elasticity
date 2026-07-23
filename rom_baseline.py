#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rom_baseline.py

Strong classical baseline for the linear, fixed-operator benchmark:
a projection-based reduced-order model (POD--Galerkin) built on the same
Q4 FEM discretization used elsewhere in this repository.

Rationale (referee response, R-strong-baselines):
The neural operators in this study are, on the linear fixed-operator
problem, compared mainly against a full FEM solve and a closed-form linear
read-out.  The recognized *strong* classical baseline for a many-query
linear PDE is a reduced-order model: assemble and factorize once offline,
build a POD basis from a handful of snapshots, and Galerkin-project the
governing operator onto that basis so each online query is an r x r solve
with r ~ O(10).  This script implements that baseline and reports the full
resource ledger (offline cost, online cost, memory, disk) so it can be
placed on the same accuracy-vs-cost axes as the learned surrogates.

Pure numpy/scipy for the ROM; imports the manufactured-data helpers from
the main study so the evaluation set is byte-for-byte the one the neural
operators see.

Usage:
    python rom_baseline.py                 # K=16 benchmark, sweep basis size
    python rom_baseline.py --n_train 3000 --ranks 2 4 8 16 24 32
"""

import argparse
import time
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Reuse the verified Q4 machinery and Lame helper.
from revision_new_benchmarks import q4_element_stiffness, FemSolver, lame

# Reuse the exact manufactured-data pipeline the neural operators train on.
from continuum_elasticity_pideeponet_publication_study import (
    PhysicsConfig, DataConfig, build_modes, generate_coefficients,
    analytical_solution_and_force,
)


def assemble_K_free(res, lam, mu):
    """Assemble the global Q4 stiffness and return (K_free, free_dofs),
    using the SAME node/DOF/BC ordering as FemSolver so that loads produced
    by FemSolver.rhs() are consistent with this operator."""
    nx = ny = res - 1
    hx = hy = 1.0 / nx
    n_dof = 2 * res * res
    ke = q4_element_stiffness(hx, hy, lam, mu)
    rows, cols, vals = [], [], []
    for ey in range(ny):
        for ex in range(nx):
            n1 = ey * res + ex
            nodes = (n1, n1 + 1, (ey + 1) * res + ex + 1, (ey + 1) * res + ex)
            dofs = [d for n in nodes for d in (2 * n, 2 * n + 1)]
            for i, di in enumerate(dofs):
                for j, dj in enumerate(dofs):
                    rows.append(di); cols.append(dj); vals.append(ke[i, j])
    K = sp.csr_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))
    bc = set()
    for i in range(res):
        for n in (i, (res - 1) * res + i, i * res, i * res + res - 1):
            bc.update((2 * n, 2 * n + 1))
    free = np.array(sorted(set(range(n_dof)) - bc))
    K_free = K[np.ix_(free, free)].tocsc()
    return K_free, free


def make_manufactured(n_train, n_test, true_modes=16, res=29, nu=0.30,
                      coeff_scale=0.04, seed_train=42, seed_test=999):
    """Identical distribution to the main study's K=16 benchmark."""
    phys = PhysicsConfig(res=res, E=1.0, nu=nu, plane="strain")
    modes = build_modes(true_modes)
    axtr, aytr = generate_coefficients(n_train, modes, coeff_scale, seed_train)
    axte, ayte = generate_coefficients(n_test, modes, coeff_scale, seed_test)
    tr = analytical_solution_and_force(axtr, aytr, modes, phys)
    te = analytical_solution_and_force(axte, ayte, modes, phys)
    return phys, tr, te


def rel_l2(pred, true):
    """Per-sample relative L2 over the full [n,res,res,2] block, then mean."""
    num = np.linalg.norm((pred - true).reshape(pred.shape[0], -1), axis=1)
    den = np.linalg.norm(true.reshape(true.shape[0], -1), axis=1) + 1e-12
    return float(np.mean(num / den))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--true_modes", type=int, default=16)
    ap.add_argument("--res", type=int, default=29)
    ap.add_argument("--nu", type=float, default=0.30)
    ap.add_argument("--ranks", type=int, nargs="+",
                    default=[2, 4, 8, 16, 24, 32])
    args = ap.parse_args()

    # numpy 2.0 on Apple Accelerate emits spurious FP warnings inside BLAS
    # matmul; all quantities here are finite (verified: rank-32 ROM reproduces
    # the FEM solution to its discretization floor). Silence the cosmetic noise.
    np.seterr(divide="ignore", over="ignore", invalid="ignore")

    lam, mu = lame(nu=args.nu)
    phys, tr, te = make_manufactured(args.n_train, args.n_test,
                                     args.true_modes, args.res, args.nu)
    res = args.res

    # ---- offline: assemble + factorize once, build snapshots by FEM ----
    t0 = time.perf_counter()
    K_free, free = assemble_K_free(res, lam, mu)
    solver = FemSolver(res, lam, mu)            # shares ordering; used for RHS + snapshots
    assemble_ms = 1000.0 * (time.perf_counter() - t0)

    # Snapshots = FEM solutions of the TRAINING forcings (no analytical oracle).
    t0 = time.perf_counter()
    S = np.zeros((free.size, args.n_train))
    for i in range(args.n_train):
        u, _ = solver.solve(tr["f"][i])                 # [res,res,2]
        S[:, i] = u.reshape(-1)[free]
    snapshot_ms = 1000.0 * (time.perf_counter() - t0)

    # POD basis via thin SVD of the snapshot matrix.
    t0 = time.perf_counter()
    U, svals, _ = np.linalg.svd(S, full_matrices=False)
    svd_ms = 1000.0 * (time.perf_counter() - t0)

    # Reference test displacement (analytical) and FEM test solution.
    u_true_te = te["u"]                                 # [n_test,res,res,2], analytical
    f_te = te["f"]

    # Precompute the free-DOF loads for the test set once (shared across ranks).
    b_full = np.zeros((args.n_test, free.size))
    for i in range(args.n_test):
        b_full[i] = solver.rhs(f_te[i])[free]

    print(f"Offline: assemble+factor {assemble_ms:.1f} ms | "
          f"{args.n_train} snapshots {snapshot_ms:.1f} ms | SVD {svd_ms:.1f} ms")
    print(f"Singular-value decay (first 8): "
          f"{np.array2string(svals[:8], precision=2)}")
    print()
    print(f"{'rank':>5} {'relL2_u(vs analytic)':>22} "
          f"{'online_us/sample':>17} {'basis_MB':>9} {'basis_disk_MB':>14}")

    rows = []
    for r in args.ranks:
        from scipy.linalg import lu_factor, lu_solve
        V = U[:, :r]                                    # [n_free, r]
        Kr = V.T @ (K_free @ V)                         # [r,r] reduced operator
        lufac = lu_factor(Kr)

        # ---- online: r x r solve per test sample ----
        u_rom = np.zeros_like(u_true_te)
        t0 = time.perf_counter()
        for i in range(args.n_test):
            br = V.T @ b_full[i]                         # [r]
            c = lu_solve(lufac, br)                      # [r]
            uf = V @ c                                   # [n_free]
            full = np.zeros(2 * res * res)
            full[free] = uf
            u_rom[i] = full.reshape(res, res, 2)
        online_us = 1e6 * (time.perf_counter() - t0) / args.n_test

        err = rel_l2(u_rom, u_true_te)
        basis_mb = V.nbytes / 1e6                        # in-RAM (float64)
        disk_mb = (V.astype(np.float32).nbytes) / 1e6    # float32 on disk
        print(f"{r:>5} {err:>22.6f} {online_us:>17.1f} "
              f"{basis_mb:>9.3f} {disk_mb:>14.3f}")
        rows.append(dict(rank=r, rel_l2_u=err, online_us=online_us,
                         basis_mb=basis_mb, disk_mb=disk_mb))

    # Context: the learned operators on this same benchmark (from the manuscript)
    print()
    print("For comparison (same K=16 benchmark, displacement rel L2):")
    print("  PI-spectral (plain MLP) .......... 0.0958")
    print("  PI-spectral (LS-anchored) ........ 0.0040")
    print("  Data-only spectral ............... 0.2146")
    print("  Capacity-matched FNO ............. 0.133")
    print("  FEM Q4 28x28 (vs analytic) ....... 0.0099")

    Path("results_revision").mkdir(exist_ok=True)
    import json
    with open("results_revision/rom_baseline.json", "w") as fh:
        json.dump(dict(config=vars(args), assemble_ms=assemble_ms,
                       snapshot_ms=snapshot_ms, svd_ms=svd_ms,
                       singular_values=svals[:32].tolist(), ranks=rows), fh,
                  indent=2)
    print("\nSaved results_revision/rom_baseline.json")


if __name__ == "__main__":
    main()
