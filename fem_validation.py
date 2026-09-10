#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fem_validation.py

Validation of the finite-strain classical baseline, answering the referee's
question of whether the reported Newton-FEM cost is intrinsic or an artifact
of an interpreted implementation.

Four parts:
  A  the vectorized solver reproduces the reference solver to round-off;
  B  cost of both implementations across mesh resolutions;
  C  Newton iteration and load-step statistics, plus a wall-clock breakdown
     over constitutive evaluation, assembly, factorization and solve;
  D  a cross-check against scikit-fem, an established finite-element library,
     assembling and solving the same problem on the same hardware.

On (D): the cross-check reuses the same Neo-Hookean constitutive routines, so
it does not independently re-derive the material model -- that is covered by
the finite-difference tangent test.  What it does check is the part the
referee questioned: whether our assembly and solve infrastructure is
competitive with an established library on the same mesh and hardware.

    python fem_validation.py --out results_revision/nonlinear/fem_validation.json
"""

import argparse, json, time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from revision_new_benchmarks import grid_xy
from nonlinear_fem import NonlinearFEM
from nonlinear_fem_fast import FastNonlinearFEM, P_batch, A_batch


def test_field(res, amp=6.0):
    xx, yy = grid_xy(res)
    return np.stack([amp * np.sin(np.pi * xx) * np.sin(np.pi * yy),
                     amp * 0.4 * np.sin(np.pi * xx) * np.sin(2 * np.pi * yy)],
                    axis=-1)


def timeit(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t0)
    return min(ts) * 1000.0, out


# ---------------------------------------------------------------- A
def part_A(res_list=(15, 29)):
    out = []
    for res in res_list:
        ref, fast = NonlinearFEM(res), FastNonlinearFEM(res)
        np.random.seed(0)
        # Scale the probe by the element size so every element keeps J > 0;
        # an unscaled probe inverts elements at finer meshes and makes log(J)
        # undefined in both implementations, which reports as nan rather than
        # as a discrepancy.
        u = 0.02 * fast.hx * np.random.randn(fast.n_dof)
        u[np.setdiff1d(np.arange(fast.n_dof), fast.free)] = 0.0
        R0, K0 = ref.assemble(u, tangent=True)
        R1, K1 = fast.assemble(u, tangent=True)
        f = test_field(res)
        u0, i0 = ref.solve(f, n_steps=6)
        u1, i1 = fast.solve(f, n_steps=6)
        rec = dict(res=res, n_dof=int(fast.n_dof), n_elem=int(fast.n_elem),
                   d_residual=float(np.linalg.norm(R1-R0)/(np.linalg.norm(R0)+1e-300)),
                   d_tangent=float(spla.norm(K1-K0)/spla.norm(K0)),
                   d_solution=float(np.linalg.norm(u1-u0)/np.linalg.norm(u0)),
                   newton_ref=int(i0["newton_iters"]), newton_fast=int(i1["newton_iters"]))
        print(f"[A] res={res:3d}  dR={rec['d_residual']:.2e}  dK={rec['d_tangent']:.2e}"
              f"  du={rec['d_solution']:.2e}  newton {rec['newton_ref']}/{rec['newton_fast']}")
        out.append(rec)
    return out


# ---------------------------------------------------------------- B
def part_B(res_list=(15, 29, 43, 57), ref_max_res=43, n_steps=6, reps=3):
    out = []
    for res in res_list:
        f = test_field(res)
        fast = FastNonlinearFEM(res)
        t_fast, _ = timeit(lambda: fast.solve(f, n_steps=n_steps), reps)
        t_ref = None
        if res <= ref_max_res:
            ref = NonlinearFEM(res)
            t_ref, _ = timeit(lambda: ref.solve(f, n_steps=n_steps), max(1, reps - 2))
        rec = dict(res=res, n_dof=int(fast.n_dof), n_elem=int(fast.n_elem),
                   ms_vectorized=t_fast, ms_reference=t_ref,
                   speedup=(t_ref / t_fast if t_ref else None))
        s = f"{rec['speedup']:.0f}x" if rec["speedup"] else "--"
        r = f"{t_ref:9.1f}" if t_ref else "        --"
        print(f"[B] res={res:3d} dofs={fast.n_dof:6d}  ref={r} ms  "
              f"fast={t_fast:8.2f} ms  speedup={s}")
        out.append(rec)
    return out


# ---------------------------------------------------------------- C
def part_C(res=29, n_steps=6):
    fast = FastNonlinearFEM(res)
    f = test_field(res)
    _, info = fast.solve(f, n_steps=n_steps, collect_stats=True)
    print(f"[C] res={res} total {info['ms']:.2f} ms, {info['newton_iters']} Newton its "
          f"over {n_steps} load steps: {[d['newton_iters'] for d in info['per_step']]}")
    for k, v in info["breakdown_ms"].items():
        print(f"      {k:<13} {v:8.2f} ms  {info['breakdown_frac'][k]*100:5.1f}%")
    return dict(res=res, n_steps=n_steps, total_ms=info["ms"],
                newton_iters=int(info["newton_iters"]),
                per_step=info["per_step"], breakdown_ms=info["breakdown_ms"],
                breakdown_frac=info["breakdown_frac"])


# ---------------------------------------------------------------- D
def part_D(res=29, n_steps=6, reps=3):
    """Same problem assembled and solved with scikit-fem's assembler."""
    try:
        import skfem
        from skfem import MeshQuad, ElementVector, ElementQuad1, Basis
        from skfem.helpers import grad
    except Exception as e:
        print(f"[D] scikit-fem unavailable ({e}); skipping cross-check")
        return dict(available=False, error=str(e))

    fast = FastNonlinearFEM(res)
    lam, mu = fast.lam, fast.mu
    n = res
    # Same unit square, same Q4 grid.
    xs = np.linspace(0, 1, n)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    p = np.vstack([X.ravel(), Y.ravel()])
    t = []
    for ey in range(n - 1):
        for ex in range(n - 1):
            n1 = ey * n + ex
            t.append([n1, n1 + 1, (ey + 1) * n + ex + 1, (ey + 1) * n + ex])
    m = MeshQuad(p, np.array(t).T)
    e = ElementVector(ElementQuad1())
    basis = Basis(m, e, intorder=2)
    D = basis.get_dofs().all()
    free = np.setdiff1d(np.arange(basis.N), D)

    @skfem.LinearForm
    def residual(v, w):
        G = grad(w["uh"])                                  # (2,2,nelem,nqp)
        Fl = np.stack([G[0, 0] + 1.0, G[0, 1], G[1, 0], G[1, 1] + 1.0], axis=-1)
        P = P_batch(Fl, lam, mu)
        gv = grad(v)
        return (P[..., 0] * gv[0, 0] + P[..., 1] * gv[0, 1]
                + P[..., 2] * gv[1, 0] + P[..., 3] * gv[1, 1])

    @skfem.BilinearForm
    def tangent(u, v, w):
        G = grad(w["uh"])
        Fl = np.stack([G[0, 0] + 1.0, G[0, 1], G[1, 0], G[1, 1] + 1.0], axis=-1)
        A = A_batch(Fl, lam, mu)                            # (nelem,nqp,4,4)
        gu, gv = grad(u), grad(v)
        gu4 = [gu[0, 0], gu[0, 1], gu[1, 0], gu[1, 1]]
        gv4 = [gv[0, 0], gv[0, 1], gv[1, 0], gv[1, 1]]
        tot = 0.0
        for i in range(4):
            for j in range(4):
                tot = tot + A[..., i, j] * gu4[j] * gv4[i]
        return tot

    fgrid = test_field(res)
    fext_ours = fast.external_force(fgrid)
    # Map our node-ordered load vector onto skfem's dof numbering.
    dofs_x = basis.get_dofs(lambda x: np.ones(x.shape[1], dtype=bool)).all()  # unused
    order = np.lexsort((np.round(basis.doflocs[0], 12), np.round(basis.doflocs[1], 12)))
    fext = np.zeros(basis.N)
    ours_nodes = np.arange(n * n)
    # our node k = iy*n+ix at (xs[ix], xs[iy]); skfem dof order via lexsort(y,x)
    fx_idx = order[0::2] if basis.doflocs.shape[1] == basis.N else None
    # Robust mapping: match coordinates and component.
    xy = np.round(basis.doflocs.T, 12)
    key = {}
    for i in range(basis.N):
        key.setdefault((xy[i, 0], xy[i, 1]), []).append(i)
    for k in ours_nodes:
        ix, iy = k % n, k // n
        ids = key[(round(xs[ix], 12), round(xs[iy], 12))]
        fext[ids[0]] += fext_ours[2 * k]
        fext[ids[1]] += fext_ours[2 * k + 1]

    def run():
        uh = basis.zeros()
        nit = 0
        for s in range(1, n_steps + 1):
            fs = fext * (s / n_steps)
            for _ in range(50):
                R = skfem.asm(residual, basis, uh=basis.interpolate(uh))
                r = (R - fs)[free]
                if np.linalg.norm(r) < 1e-9:
                    break
                K = skfem.asm(tangent, basis, uh=basis.interpolate(uh))
                du = spla.spsolve(K.tocsr()[free][:, free].tocsc(), -r)
                uh[free] += du
                nit += 1
        return uh, nit

    t_sk, (uh, nit) = timeit(run, reps)
    u_ours, info = fast.solve(fgrid, n_steps=n_steps)
    # Compare solutions on matching coordinates.
    ours_flat = u_ours.reshape(-1, 2)
    err_num, err_den = 0.0, 0.0
    for k in ours_nodes:
        ix, iy = k % n, k // n
        ids = key[(round(xs[ix], 12), round(xs[iy], 12))]
        d = np.array([uh[ids[0]], uh[ids[1]]]) - ours_flat[k]
        err_num += d @ d; err_den += ours_flat[k] @ ours_flat[k]
    rel = float(np.sqrt(err_num / max(err_den, 1e-300)))
    t_fast, _ = timeit(lambda: fast.solve(fgrid, n_steps=n_steps), reps)
    print(f"[D] scikit-fem {t_sk:9.1f} ms ({nit} its)   ours {t_fast:8.2f} ms "
          f"({info['newton_iters']} its)   solution rel diff = {rel:.2e}")
    return dict(available=True, res=res, ms_skfem=t_sk, ms_ours=t_fast,
                ratio_skfem_over_ours=t_sk / t_fast, newton_skfem=int(nit),
                newton_ours=int(info["newton_iters"]), solution_rel_diff=rel,
                skfem_version=skfem.__version__)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_revision/nonlinear/fem_validation.json")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    res_b = (15, 29) if a.quick else (15, 29, 43, 57)
    result = dict(
        A_equivalence=part_A((15,) if a.quick else (15, 29)),
        B_scaling=part_B(res_b, ref_max_res=29 if a.quick else 43),
        C_statistics=part_C(),
        D_crosscheck=part_D(res=29, reps=1 if a.quick else 3),
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print("wrote", a.out)
    print("FEMVALIDATION_DONE")
