#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOTE -- the DEIM online stage in this file is NOT fully reduced.

reduced_deim_solve() forms the full displacement vector each iteration,
assembles the sampled tangent into a full n_dof x n_dof matrix, and contracts
it with the full-dimensional POD basis, so its online cost scales with the
number of free degrees of freedom rather than with the number of
interpolation points.  The POD-Galerkin routines here are unaffected and are
still used.

Use pod_deim_v2.py for the hyper-reduced solve: it indexes everything on the
sample mesh, so the online cost is O(m * n_sample * r) per Newton iteration.


pod_deim.py

Reduced-order models for the finite-strain hyperelastic benchmark, built as
the strong classical baseline the neural operator must beat.

Step A (this file, verified first): POD-Galerkin reduced Newton -- reduce the
unknowns to a POD subspace but still assemble the full nonlinear residual and
tangent each Newton iteration.  This is accurate (down to the POD projection
floor) but NOT fast: full-order assembly gives no speedup, which is exactly
why nonlinear ROMs need hyper-reduction (DEIM, added next).

Step B (added after A verifies): DEIM hyper-reduction, so the reduced residual
and tangent are evaluated from a small sample mesh and the online solve is
genuinely cheap.

    python pod_deim.py --selftest --data data/nonlinear.npz
"""

import argparse
import time

import numpy as np
import scipy.sparse as sp

from nonlinear_fem import NonlinearFEM, neohookean_P, neohookean_tangent


# ----------------------------------------------------------------------
# Step B: DEIM hyper-reduction
# ----------------------------------------------------------------------
def internal_force_free(fem, u_free):
    u = np.zeros(fem.n_dof); u[fem.free] = u_free
    return fem.assemble(u, tangent=False)[fem.free]


def deim_snapshots(fem, u_tr_free, n_sub=300, alphas=(0.5, 1.0)):
    """Internal-force snapshots along the loading path (u scaled by alpha),
    subsampled over the training set -- captures how R_int varies with u."""
    N = u_tr_free.shape[0]
    idx = np.linspace(0, N - 1, min(n_sub, N)).astype(int)
    cols = []
    for i in idx:
        for a in alphas:
            cols.append(internal_force_free(fem, a * u_tr_free[i]))
    return np.array(cols).T                                   # [n_free, n_snap]


def deim_select(U):
    """Greedy DEIM interpolation points for basis U [n, m] -> m row indices."""
    m = U.shape[1]
    idx = [int(np.argmax(np.abs(U[:, 0])))]
    for j in range(1, m):
        Uj = U[np.ix_(idx, list(range(j)))]
        c, *_ = np.linalg.lstsq(Uj, U[idx, j], rcond=None)
        res = U[:, j] - U[:, :j] @ c
        idx.append(int(np.argmax(np.abs(res))))
    return np.array(idx)


def sample_mesh(fem, deim_full_dofs):
    """Elements touching the DEIM dofs' nodes, and the dofs those elements own.
    All elements contributing to R_int at a DEIM dof are included, so the
    internal force at the DEIM dofs is assembled EXACTLY from this sample mesh."""
    nodes = set(int(d) // 2 for d in deim_full_dofs)
    elems = [e for e, dofs in enumerate(fem.elems)
             if any((int(n) in nodes) for n in dofs[0::2] // 2)]
    sdofs = sorted(set(int(d) for e in elems for d in fem.elems[e]))
    return elems, np.array(sdofs)


def assemble_subset(fem, u_full, elem_list, tangent=True):
    """Assemble R_int (and K_t) from only elem_list (the sample mesh)."""
    lam, mu = fem.lam, fem.mu
    R = np.zeros(fem.n_dof)
    rows, cols, vals = [], [], []
    for e in elem_list:
        dofs = fem.elems[e]
        ue = u_full[dofs]
        fe = np.zeros(8); Ke = np.zeros((8, 8)) if tangent else None
        for (N, G, w) in fem.gp_ops:
            grad = G @ ue
            F = np.array([[1 + grad[0], grad[1]], [grad[2], 1 + grad[3]]])
            fe += G.T @ neohookean_P(F, lam, mu).reshape(-1) * w
            if tangent:
                Ke += G.T @ neohookean_tangent(F, lam, mu) @ G * w
        R[dofs] += fe
        if tangent:
            for a in range(8):
                for b in range(8):
                    rows.append(dofs[a]); cols.append(dofs[b]); vals.append(Ke[a, b])
    if tangent:
        K = sp.csr_matrix((vals, (rows, cols)), shape=(fem.n_dof, fem.n_dof))
        return R, K
    return R


def collect_trajectory_forces(fem, V, f_list, n_steps=6, tol=1e-8, maxit=60):
    """Run POD-Galerkin on each forcing and save the internal force R_int(V a) at
    every Newton iterate -- the EXACT states DEIM must interpolate (far better
    snapshots than scaling the converged solution)."""
    free = fem.free; r = V.shape[1]; cols = []
    for f in f_list:
        f_ext = fem.external_force(f)[free]; a = np.zeros(r)
        for s in range(1, n_steps + 1):
            fext = f_ext * (s / n_steps)
            for it in range(maxit):
                u = np.zeros(fem.n_dof); u[free] = V @ a
                R_int, K = fem.assemble(u, tangent=True)
                cols.append(R_int[free].copy())
                Rr = V.T @ (R_int[free] - fext)
                if np.linalg.norm(Rr) < tol:
                    break
                a = a + np.linalg.solve(V.T @ (K[np.ix_(free, free)] @ V), -Rr)
    return np.array(cols).T


def build_deim_rom(fem, u_tr_free, r, m, Fs=None):
    """Offline: POD basis V, DEIM basis/points, precomputed operators + sample mesh.
    Fs = internal-force snapshot matrix [n_free, n_snap]; if None, fall back to the
    (weaker) scaled-solution snapshots."""
    V, _ = build_pod(u_tr_free, r)
    if Fs is None:
        Fs = deim_snapshots(fem, u_tr_free)
    UU, _, _ = np.linalg.svd(Fs, full_matrices=False)
    U = UU[:, :m]
    p = deim_select(U)                                        # indices in free space
    PU = U[p, :]
    Dop = V.T @ U @ np.linalg.inv(PU)                         # [r, m]
    deim_full = fem.free[p]                                   # DEIM dofs in full space
    elems, sdofs = sample_mesh(fem, deim_full)
    return dict(V=V, U=U, p=p, Dop=Dop, deim_full=deim_full,
                elems=elems, sdofs=sdofs, r=r, m=m)


def reduced_deim_solve(fem, f_grid, rom, n_steps=6, tol=1e-8, maxit=60):
    """Online: hyper-reduced Newton -- assemble ONLY the sample mesh each step."""
    free = fem.free; V = rom["V"]; Dop = rom["Dop"]; p = rom["p"]; elems = rom["elems"]
    Vtf = V.T @ fem.external_force(f_grid)[free]
    a = np.zeros(rom["r"])
    t0 = time.perf_counter(); nit = 0
    for s in range(1, n_steps + 1):
        fext_r = Vtf * (s / n_steps)
        for it in range(maxit):
            u_full = np.zeros(fem.n_dof); u_full[free] = V @ a
            Rs, Ks = assemble_subset(fem, u_full, elems, tangent=True)
            PtR = Rs[free][p]                                  # R_int at DEIM dofs
            Rr = Dop @ PtR - fext_r
            if np.linalg.norm(Rr) < tol:
                break
            PtKV = (Ks[np.ix_(free[p], free)] @ V)            # [m, r]
            Kr = Dop @ PtKV
            a = a + np.linalg.solve(Kr, -Rr)
            nit += 1
        else:
            return None, dict(converged=False, step=s)
    dt = 1000 * (time.perf_counter() - t0)
    u = np.zeros(fem.n_dof); u[free] = V @ a
    return u.reshape(fem.res, fem.res, 2), dict(converged=True, newton=nit, ms=dt,
                                                n_sample_elems=len(elems))


def build_pod(snapshots_free, r):
    """POD basis from free-dof solution snapshots [N, n_free] -> V [n_free, r]."""
    U, sv, _ = np.linalg.svd(snapshots_free.T, full_matrices=False)
    return U[:, :r], sv


def reduced_galerkin_solve(fem, f_grid, V, n_steps=6, tol=1e-8, maxit=60):
    """POD-Galerkin reduced Newton.  Returns u[res,res,2], info.
    Full-order residual/tangent are assembled and then projected (no speedup);
    this is the accuracy reference for the hyper-reduced version."""
    free = fem.free
    f_ext_full = fem.external_force(f_grid)[free]
    r = V.shape[1]
    a = np.zeros(r)
    t0 = time.perf_counter()
    nit = 0
    for s in range(1, n_steps + 1):
        fext = f_ext_full * (s / n_steps)
        for it in range(maxit):
            u_full = np.zeros(fem.n_dof)
            u_full[free] = V @ a
            R_int, K_t = fem.assemble(u_full, tangent=True)
            R = R_int[free] - fext
            Rr = V.T @ R
            if np.linalg.norm(Rr) < tol:
                break
            Kf = K_t[np.ix_(free, free)]
            Kr = V.T @ (Kf @ V)                       # reduced tangent [r,r]
            a = a + np.linalg.solve(Kr, -Rr)
            nit += 1
        else:
            return None, dict(converged=False, step=s)
    dt = 1000 * (time.perf_counter() - t0)
    u = np.zeros(fem.n_dof); u[free] = V @ a
    return u.reshape(fem.res, fem.res, 2), dict(converged=True, newton=nit, ms=dt)


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def selftest(data):
    z = np.load(data)
    u_tr = z["u_tr"]; f_tr = z["f_tr"]; f_te = z["f_te"]; u_te = z["u_te"]
    res = u_tr.shape[1]
    fem = NonlinearFEM(res, nu=0.30)
    free = fem.free
    S = u_tr.reshape(u_tr.shape[0], -1)[:, free]          # [N, n_free]
    print(f"snapshots {S.shape}, free dofs {free.size}")

    # POD projection floor of the nonlinear solution set (upper bound on ROM accuracy).
    for r in (16, 32, 64):
        V, sv = build_pod(S, r)
        Ute = u_te.reshape(u_te.shape[0], -1)[:, free]
        proj = Ute @ V @ V.T
        floor = np.mean(np.linalg.norm(Ute - proj, axis=1) / (np.linalg.norm(Ute, axis=1) + 1e-30))
        print(f"  r={r:3d}: POD projection floor = {floor:.4f}")

    # POD-Galerkin reduced Newton on a few test forcings vs the full FEM solution.
    r = 32
    V, _ = build_pod(S, r)
    n_test = 5
    errs_gt = []; times = []
    for i in range(n_test):
        u_rom, info = reduced_galerkin_solve(fem, f_te[i].astype(np.float64), V)
        if not info["converged"]:
            print(f"  sample {i}: reduced Newton DID NOT CONVERGE"); continue
        e = rel(u_rom, u_te[i])
        errs_gt.append(e); times.append(info["ms"])
        print(f"  sample {i}: reduced-Galerkin (r={r}) vs FEM GT = {e:.4f}  "
              f"({info['newton']} Newton, {info['ms']:.0f} ms)")
    print(f"POD-Galerkin (r={r}): mean err vs FEM {np.mean(errs_gt):.4f}, "
          f"mean {np.mean(times):.0f} ms/query")
    print("Step A OK if reduced-Galerkin error ~ POD floor (reduction is correct); "
          "note it is NOT faster than FEM -- that is what DEIM will fix.")

    # ---- Step B: DEIM hyper-reduction (proper trajectory snapshots) ----
    print("\nCollecting DEIM trajectory snapshots (POD-Galerkin iterates on 40 forcings)...")
    t0 = time.perf_counter()
    Fs = collect_trajectory_forces(fem, V, [f_tr[i].astype(np.float64)
                                            for i in range(40)])
    print(f"  {Fs.shape[1]} force snapshots in {time.perf_counter()-t0:.0f} s")
    for m in (80, 128):
        rom = build_deim_rom(fem, S, r, m, Fs=Fs)
        errs_gt = []; errs_gal = []; times = []
        for i in range(n_test):
            u_deim, info = reduced_deim_solve(fem, f_te[i].astype(np.float64), rom)
            u_gal, _ = reduced_galerkin_solve(fem, f_te[i].astype(np.float64), rom["V"])
            if not info["converged"]:
                print(f"  m={m} sample {i}: DID NOT CONVERGE"); continue
            errs_gt.append(rel(u_deim, u_te[i])); errs_gal.append(rel(u_deim, u_gal))
            times.append(info["ms"])
        if errs_gt:
            print(f"POD-DEIM r={r} m={m}: err vs FEM {np.mean(errs_gt):.4f}, "
                  f"vs POD-Galerkin {np.mean(errs_gal):.4f}, {np.mean(times):.0f} ms/query, "
                  f"sample mesh {len(rom['elems'])}/{len(fem.elems)}")
    print("Step B OK if DEIM ~ POD-Galerkin (faithful) AND faster than FEM.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data", default="data/nonlinear.npz")
    a = ap.parse_args()
    selftest(a.data)
