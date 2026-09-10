#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pod_deim_v2.py

Hyper-reduced POD-DEIM whose online stage is actually reduced.

In the previous implementation the online Newton loop
  - formed the full displacement vector  u_full[free] = V @ a  every iteration,
    costing O(n_free * r);
  - assembled the sampled tangent into a full n_dof x n_dof sparse matrix;
  - contracted that with the full-dimensional basis,  K[free[p], free] @ V,
    costing O(m * n_free * r).
So the online cost still scaled with the number of free degrees of freedom
rather than with the number of interpolation points, which is the whole point
of hyper-reduction.

This version restricts every online operation to the sample mesh:
  - sample-mesh dofs are indexed locally; the displacement needed by the
    residual is  u_s = V_s @ a  with V_s of shape (n_sample_dofs, r);
  - residual and tangent are assembled, vectorized, only over sample elements
    into local arrays of size n_sample_dofs;
  - the tangent contraction is  (K_local[p_local, :] @ V_s), cost
    O(m * n_sample_dofs * r), independent of the full mesh;
  - the full field is reconstructed once, after convergence.

It also reports what the referee asked for: the fraction of elements retained
in the sample mesh, the sample-dof count, and the resulting online complexity,
swept over the number of interpolation points and mesh resolution so the
small-mesh limitation is visible rather than asserted.

    python pod_deim_v2.py --selftest --data data/nonlinear.npz
"""

import argparse, json, time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from nonlinear_fem_fast import FastNonlinearFEM, P_batch, A_batch
from pod_deim import build_pod, deim_select, collect_trajectory_forces


# ---------------------------------------------------------------- offline
def build_sample_mesh(fem, deim_full_dofs):
    """Elements touching any DEIM dof's node.  Every element contributing to
    R_int at a DEIM dof is retained, so the residual at those dofs is exact."""
    nodes = set(int(d) // 2 for d in deim_full_dofs)
    mask = np.isin(fem.elem_nodes, list(nodes)).any(axis=1)
    elems = np.flatnonzero(mask)
    sdofs = np.unique(fem.elem_dofs[elems].ravel())
    return elems, sdofs


def build_rom_v2(fem, u_tr_free, r, m, Fs):
    """Offline stage.  Returns operators sized by the sample mesh, not the mesh."""
    V, _ = build_pod(np.asarray(u_tr_free, dtype=np.float64), r)
    V = np.asarray(V, dtype=np.float64)
    UU, _, _ = np.linalg.svd(np.asarray(Fs, dtype=np.float64), full_matrices=False)
    U = UU[:, :m]
    p = deim_select(U)                       # rows in free-dof space
    Dop = V.T @ U @ np.linalg.inv(U[p, :])   # (r, m)
    deim_full = fem.free[p]
    elems, sdofs = build_sample_mesh(fem, deim_full)

    # Local indexing over sample dofs.
    loc = -np.ones(fem.n_dof, dtype=np.int64)
    loc[sdofs] = np.arange(sdofs.size)
    elem_dofs_local = loc[fem.elem_dofs[elems]]              # (n_se, 8)

    # V_s: rows of V for sample dofs; constrained dofs contribute zero.
    free_pos = -np.ones(fem.n_dof, dtype=np.int64)
    free_pos[fem.free] = np.arange(fem.free.size)
    V_s = np.zeros((sdofs.size, r))
    fp = free_pos[sdofs]
    V_s[fp >= 0] = V[fp[fp >= 0]]

    p_local = loc[fem.free[p]]
    assert (p_local >= 0).all(), "DEIM dofs must lie in the sample mesh"

    n_se = elems.size
    rows = np.repeat(elem_dofs_local, 8, axis=1).ravel()
    cols = np.tile(elem_dofs_local, (1, 8)).ravel()
    return dict(V=V, V_s=V_s, Dop=Dop, p=p, p_local=p_local, r=r, m=m,
                elems=elems, sdofs=sdofs, elem_dofs_local=elem_dofs_local,
                rows=rows, cols=cols, n_sample_dofs=int(sdofs.size),
                n_sample_elems=int(n_se), n_elem_total=int(fem.n_elem),
                sample_frac=float(n_se) / fem.n_elem)


# ---------------------------------------------------------------- online
def _assemble_sample(fem, rom, u_s):
    """Residual and tangent over the sample mesh only, in local indices."""
    ns = rom["n_sample_dofs"]
    ue = u_s[rom["elem_dofs_local"]]                          # (n_se, 8)
    grad = np.einsum("gcj,ej->egc", fem.G, ue, optimize=True)
    Fl = grad + np.array([1.0, 0.0, 0.0, 1.0])
    P = P_batch(Fl, fem.lam, fem.mu)
    A = A_batch(Fl, fem.lam, fem.mu)
    w = fem.detJ
    fe = np.einsum("gca,egc->ea", fem.G, P, optimize=True) * w
    R = np.bincount(rom["elem_dofs_local"].ravel(), weights=fe.ravel(), minlength=ns)
    Ke = np.einsum("gca,egcd,gdb->eab", fem.G, A, fem.G, optimize=True) * w
    K = sp.coo_matrix((Ke.ravel(), (rom["rows"], rom["cols"])), shape=(ns, ns)).tocsr()
    return R, K


def reduced_deim_solve_v2(fem, f_grid, rom, n_steps=6, tol=1e-8, maxit=60,
                          collect_stats=False):
    V, V_s, Dop = rom["V"], rom["V_s"], rom["Dop"]
    p_local = rom["p_local"]
    Vtf = V.T @ fem.external_force(f_grid)[fem.free]
    a = np.zeros(rom["r"])
    t0 = time.perf_counter()
    nit = 0
    for s in range(1, n_steps + 1):
        fext_r = Vtf * (s / n_steps)
        for _ in range(maxit):
            u_s = V_s @ a                              # O(n_sample_dofs * r)
            R, K = _assemble_sample(fem, rom, u_s)     # O(n_sample_elems)
            Rr = Dop @ R[p_local] - fext_r
            if np.linalg.norm(Rr) < tol:
                break
            PtKV = K[p_local, :] @ V_s                 # O(m * n_sample_dofs * r)
            a += np.linalg.solve(Dop @ PtKV, -Rr)
            nit += 1
        else:
            return None, dict(converged=False, step=s)
    ms = 1000 * (time.perf_counter() - t0)
    u = np.zeros(fem.n_dof)
    u[fem.free] = V @ a                                # reconstruction, once
    info = dict(converged=True, newton=nit, ms=ms,
                n_sample_elems=rom["n_sample_elems"],
                n_sample_dofs=rom["n_sample_dofs"],
                sample_frac=rom["sample_frac"])
    return u.reshape(fem.res, fem.res, 2), info


# ---------------------------------------------------------------- study
def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


def study(data, r=32, m_list=(64, 128, 192, 256), n_traj=40, n_test=10, n_steps=6):
    """Full hyper-reduction study: accuracy, cost, and how much of the mesh the
    sample mesh actually retains, swept over the number of interpolation points."""
    from pod_deim import reduced_galerkin_solve
    z = np.load(data)
    f_tr, u_tr, f_te = z["f_tr"], z["u_tr"], z["f_te"]
    fem = FastNonlinearFEM(f_tr.shape[1]); free = fem.free
    print(f"mesh {fem.res-1}x{fem.res-1}: {fem.n_elem} elements, {free.size} free dofs")

    S = np.asarray(u_tr.reshape(u_tr.shape[0], -1)[:, free], dtype=np.float64)
    V, sv = build_pod(S, r); V = np.asarray(V, dtype=np.float64)
    print(f"[offline] POD r={r} from {S.shape[0]} snapshots, "
          f"energy {sv[:r].sum()/sv.sum():.5f}")
    Fs = collect_trajectory_forces(fem, V, [f_tr[i].astype(np.float64)
                                            for i in range(n_traj)], n_steps=n_steps)
    print(f"[offline] internal-force snapshots {Fs.shape}")

    refs, t_fem = [], []
    for i in range(n_test):
        fg = f_te[i].astype(np.float64)
        t0 = time.perf_counter(); u, _ = fem.solve(fg, n_steps=n_steps)
        t_fem.append(1000 * (time.perf_counter() - t0)); refs.append(u)
    print(f"[fem] median {np.median(t_fem):.1f} ms")

    eg, tg = [], []
    for i in range(n_test):
        t0 = time.perf_counter()
        ug, _ = reduced_galerkin_solve(fem, f_te[i].astype(np.float64), V, n_steps=n_steps)
        tg.append(1000 * (time.perf_counter() - t0))
        eg.append(rel(ug, refs[i]) if ug is not None else np.nan)
    print(f"[galerkin r={r}] err {np.nanmean(eg):.4f}  {np.median(tg):.1f} ms")

    out = []
    for m in m_list:
        rom = build_rom_v2(fem, S, r, m, Fs)
        errs, ts, its, ndiv = [], [], [], 0
        for i in range(n_test):
            u, info = reduced_deim_solve_v2(fem, f_te[i].astype(np.float64), rom,
                                            n_steps=n_steps)
            if u is None:
                ndiv += 1; continue
            errs.append(rel(u, refs[i])); ts.append(info["ms"]); its.append(info["newton"])
        rec = dict(r=r, m=m, n_elem=rom["n_elem_total"],
                   sample_elems=rom["n_sample_elems"], sample_frac=rom["sample_frac"],
                   sample_dofs=rom["n_sample_dofs"], n_free=int(free.size),
                   sample_dof_frac=rom["n_sample_dofs"] / float(free.size),
                   n_diverged=ndiv, n_test=n_test,
                   err=float(np.mean(errs)) if errs else None,
                   ms=float(np.median(ts)) if ts else None,
                   newton=float(np.mean(its)) if its else None,
                   speedup_vs_fem=(float(np.median(t_fem)) / float(np.median(ts))
                                   if ts else None))
        e = f"{rec['err']:.4f}" if rec["err"] is not None else "  --  "
        ms = f"{rec['ms']:7.1f}" if rec["ms"] else "     --"
        print(f"[deim m={m:4d}] elems {rec['sample_elems']:4d}/{rec['n_elem']} "
              f"({100*rec['sample_frac']:4.1f}%)  dofs {rec['sample_dofs']:5d}/{free.size} "
              f"({100*rec['sample_dof_frac']:4.1f}%)  err {e}  {ms} ms  div {ndiv}/{n_test}")
        out.append(rec)
    return dict(mesh=dict(res=int(fem.res), n_elem=int(fem.n_elem),
                          n_free=int(free.size)),
                fem_ms=float(np.median(t_fem)),
                galerkin=dict(r=r, err=float(np.nanmean(eg)), ms=float(np.median(tg))),
                deim=out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/nonlinear.npz")
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--m", type=int, nargs="+", default=[64, 128, 192, 256])
    ap.add_argument("--n_traj", type=int, default=40)
    ap.add_argument("--n_test", type=int, default=10)
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    res = study(a.data, r=a.r, m_list=tuple(a.m), n_test=a.n_test, n_traj=a.n_traj)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2))
        print("wrote", a.out)
    print("DEIMV2_DONE")
