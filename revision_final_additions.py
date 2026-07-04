#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
revision_final_additions.py

Final additions:

  1. LSFEM: an equal-order bilinear least-squares FEM for the
     first-order (mixed) stress--displacement system, giving a direct
     stress approximation.  Run:  python revision_final_additions.py --lsfem_selftest
                      python revision_final_additions.py --lsfem_stress_table

  2. Heterogeneous-inclusion benchmark: circular inclusion of
     per-sample stiffness contrast k, so the solution operator
     (f, k) -> u is NONLINEAR in its input.  FEM ground truth requires
     per-sample factorization (K(k) = K_out + k*K_in), which also makes
     the many-query surrogate timing argument concrete.
     Run:  python revision_final_additions.py --hetero_selftest
           python revision_final_additions.py --hetero_gen
           python revision_final_additions.py --hetero_fem_timing

Training jobs for the heterogeneous case are driven by
revision_chunk_runner.py (jobs 'het_*').
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# reuse the verified Q4 machinery
from revision_new_benchmarks import q4_element_stiffness, FemSolver, lame, grid_xy

OUT = Path("results_revision")

GAUSS = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])


# ======================================================================
# 1. LSFEM: least-squares FEM for the first-order system
#       A sigma - eps(u) = 0,   div sigma + f = 0,   u = 0 on boundary
#    Equal-order bilinear (Q4) nodal spaces for u (2) and sigma (3).
# ======================================================================

class LsfemSolver:
    NF = 5  # fields per node: ux, uy, sxx, syy, sxy

    def __init__(self, res, lam, mu, wdiv=None):
        """wdiv: weight on the div-sigma residual. Default 5h (mesh-
        dependent weighting is standard practice for first-order
        LSFEM and balances the two residual norms)."""
        if wdiv is None:
            wdiv = 5.0 / (res - 1)
        self.wdiv = wdiv
        self.res = res
        nx = ny = res - 1
        hx = hy = 1.0 / nx
        n_dof = self.NF * res * res
        Ct = np.array([[lam + 2 * mu, lam, 0.0],
                       [lam, lam + 2 * mu, 0.0],
                       [0.0, 0.0, 2 * mu]])
        A = np.linalg.inv(Ct)  # compliance, tensor-shear convention
        # element matrices at 2x2 Gauss points
        rows, cols, vals = [], [], []
        self.rhs_blocks = []   # (dofs, B2 stacked, weights) for RHS assembly
        detJ = (hx / 2) * (hy / 2)
        gp_data = []
        for xi in GAUSS:
            for eta in GAUSS:
                N = np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                              (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)]) / 4.0
                dN_dxi = np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)]) / 4.0
                dN_deta = np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]) / 4.0
                dNx = (2 / hx) * dN_dxi
                dNy = (2 / hy) * dN_deta
                gp_data.append((N, dNx, dNy))
        # per-gauss-point B matrices (5 residual rows x 20 element dofs)
        # dof order within element: node-major [ux uy sxx syy sxy] x 4
        B1s, B2s = [], []
        for (N, dNx, dNy) in gp_data:
            B1 = np.zeros((3, 20))  # A sigma - eps(u)
            B2 = np.zeros((2, 20))  # div sigma  (+ f goes to RHS)
            for a in range(4):
                c = self.NF * a
                # -eps(u) part
                B1[0, c + 0] -= dNx[a]                      # -eps_xx
                B1[1, c + 1] -= dNy[a]                      # -eps_yy
                B1[2, c + 0] -= 0.5 * dNy[a]                # -eps_xy
                B1[2, c + 1] -= 0.5 * dNx[a]
                # A sigma part
                for i in range(3):
                    B1[i, c + 2] += A[i, 0] * N[a]
                    B1[i, c + 3] += A[i, 1] * N[a]
                    B1[i, c + 4] += A[i, 2] * N[a]
                # div sigma
                B2[0, c + 2] += dNx[a]
                B2[0, c + 4] += dNy[a]
                B2[1, c + 4] += dNx[a]
                B2[1, c + 3] += dNy[a]
            B1s.append(B1)
            B2s.append(wdiv * B2)
        ke = detJ * sum(B1.T @ B1 + B2.T @ B2 for B1, B2 in zip(B1s, B2s))
        self.B2s, self.gp_N = B2s, [g[0] for g in gp_data]
        self.detJ = detJ
        # assemble
        elem_dofs = []
        for ey in range(ny):
            for ex in range(nx):
                n1 = ey * res + ex
                nodes = (n1, n1 + 1, (ey + 1) * res + ex + 1, (ey + 1) * res + ex)
                dofs = [self.NF * n + c for n in nodes for c in range(self.NF)]
                elem_dofs.append((nodes, dofs))
                for i, di in enumerate(dofs):
                    for j, dj in enumerate(dofs):
                        rows.append(di); cols.append(dj); vals.append(ke[i, j])
        K = sp.csr_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))
        self.elem_dofs = elem_dofs
        # essential BC: u = 0 on boundary nodes (sigma left free)
        bc = set()
        for i in range(res):
            for n in (i, (res - 1) * res + i, i * res, i * res + res - 1):
                bc.update((self.NF * n + 0, self.NF * n + 1))
        self.free = np.array(sorted(set(range(n_dof)) - bc))
        t0 = time.perf_counter()
        self.lu = spla.splu(K[np.ix_(self.free, self.free)].tocsc())
        self.factor_ms = 1000 * (time.perf_counter() - t0)
        self.n_dof = n_dof

    def solve(self, f_grid):
        """f_grid: [res,res,2] nodal body force. Returns u [res,res,2],
        sigma [res,res,3] (direct nodal stress dofs), solve time (ms)."""
        F = np.zeros(self.n_dof)
        for (nodes, dofs) in self.elem_dofs:
            fx = f_grid[[n // self.res for n in nodes],
                        [n % self.res for n in nodes], 0]
            fy = f_grid[[n // self.res for n in nodes],
                        [n % self.res for n in nodes], 1]
            fe = np.zeros(20)
            for B2, N in zip(self.B2s, self.gp_N):
                fgp = np.array([N @ fx, N @ fy])
                # minimize ||w(div sigma + f)||^2 -> RHS -= w*B2^T (w*f);
                # B2s already carry one factor of wdiv
                fe -= self.detJ * (B2.T @ (self.wdiv * fgp))
            for i, d in enumerate(dofs):
                F[d] += fe[i]
        t0 = time.perf_counter()
        x_free = self.lu.solve(F[self.free])
        dt = 1000 * (time.perf_counter() - t0)
        x = np.zeros(self.n_dof)
        x[self.free] = x_free
        x = x.reshape(self.res, self.res, self.NF)
        return x[..., 0:2], x[..., 2:5], dt


def analytical_fields(res, lam, mu, p=2, q=1, ax=0.01, ay=-0.02):
    """Manufactured sine mode: returns u, sigma, f on the grid."""
    P, Q = p * math.pi, q * math.pi
    xx, yy = grid_xy(res)
    sxsy = np.sin(P * xx) * np.sin(Q * yy)
    phi_x = P * np.cos(P * xx) * np.sin(Q * yy)
    phi_y = Q * np.sin(P * xx) * np.cos(Q * yy)
    phi_xy = P * Q * np.cos(P * xx) * np.cos(Q * yy)
    lap = -(P ** 2 + Q ** 2) * sxsy
    ux, uy = ax * sxsy, ay * sxsy
    exx, eyy = ax * phi_x, ay * phi_y
    exy = 0.5 * (ax * phi_y + ay * phi_x)
    sxx = (lam + 2 * mu) * exx + lam * eyy
    syy = lam * exx + (lam + 2 * mu) * eyy
    sxy = 2 * mu * exy
    fx = -(mu * lap * ax + (lam + mu) * (-(P ** 2) * sxsy * ax + phi_xy * ay))
    fy = -(mu * lap * ay + (lam + mu) * (phi_xy * ax + -(Q ** 2) * sxsy * ay))
    return (np.stack([ux, uy], -1), np.stack([sxx, syy, sxy], -1),
            np.stack([fx, fy], -1))


def q4_stress_from_u(u, lam, mu):
    """Central-difference stress from a nodal displacement field
    (the post-processing used for the Q4 displacement solver)."""
    res = u.shape[0]
    h = 1.0 / (res - 1)
    ux_x = np.gradient(u[..., 0], h, axis=1)
    ux_y = np.gradient(u[..., 0], h, axis=0)
    uy_x = np.gradient(u[..., 1], h, axis=1)
    uy_y = np.gradient(u[..., 1], h, axis=0)
    exx, eyy, exy = ux_x, uy_y, 0.5 * (ux_y + uy_x)
    return np.stack([(lam + 2 * mu) * exx + lam * eyy,
                     lam * exx + (lam + 2 * mu) * eyy,
                     2 * mu * exy], -1)


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def lsfem_selftest():
    lam, mu = lame()
    print("LSFEM convergence vs manufactured solution (u error / sigma error):")
    prev = None
    for res in (15, 29, 57):
        u_t, s_t, f = analytical_fields(res, lam, mu)
        solver = LsfemSolver(res, lam, mu)
        u, s, _ = solver.solve(f)
        eu, es = rel(u, u_t), rel(s, s_t)
        r = "" if prev is None else "  rates=%.2f/%.2f" % (
            math.log2(prev[0] / eu), math.log2(prev[1] / es))
        print(f"  res={res:4d}  u={eu:.3e}  sigma={es:.3e}{r}")
        prev = (eu, es)


def lsfem_stress_table(n_test=200, seed=999):
    """Stress accuracy: Q4 displacement FEM (post-processed) vs LSFEM
    (direct stress dofs), manufactured K=16 test set, res 29 and 57."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import cmame_extended_study as st
    lam, mu = lame()
    modes = st.build_modes(16)
    ax, ay = st.generate_coefficients(n_test, modes, 0.04, seed)
    rows = []
    for res in (29, 57):
        phys = st.PhysicsConfig(res=res)
        data = st.analytical_solution_and_force(ax, ay, modes, phys)
        q4 = FemSolver(res, lam, mu)
        ls = LsfemSolver(res, lam, mu)
        e_q4u = e_q4s = e_lsu = e_lss = 0.0
        t_q4 = t_ls = 0.0
        for i in range(n_test):
            f = data["f"][i].astype(np.float64)
            u_t = data["u"][i].astype(np.float64)
            s_t = data["sig"][i].astype(np.float64)
            uq, dt1 = q4.solve(f)
            sq = q4_stress_from_u(uq, lam, mu)
            ul, sl, dt2 = ls.solve(f)
            e_q4u += rel(uq, u_t); e_q4s += rel(sq, s_t)
            e_lsu += rel(ul, u_t); e_lss += rel(sl, s_t)
            t_q4 += dt1; t_ls += dt2
        n = n_test
        rows.append(dict(res=res,
                         q4_u=e_q4u / n, q4_sigma=e_q4s / n,
                         lsfem_u=e_lsu / n, lsfem_sigma=e_lss / n,
                         q4_solve_ms=t_q4 / n, lsfem_solve_ms=t_ls / n,
                         q4_factor_ms=q4.factor_ms, lsfem_factor_ms=ls.factor_ms))
        print(rows[-1])
    OUT.mkdir(exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT / "lsfem_stress_table.csv", index=False)


# ======================================================================
# 2. HETEROGENEOUS-INCLUSION BENCHMARK (nonlinear in the input k)
#    E(x) = 1 outside, k inside a circular inclusion; nu fixed.
#    K(k) = K_out + k * K_in  ->  per-sample factorization.
# ======================================================================

INCL_C, INCL_R = (0.5, 0.5), 0.25


class HeteroQ4:
    def __init__(self, res, nu=0.30):
        lam, mu = lame(nu=nu)
        self.res = res
        nx = ny = res - 1
        hx = hy = 1.0 / nx
        n_dof = 2 * res * res
        ke = q4_element_stiffness(hx, hy, lam, mu)
        ro, co, vo = [], [], []   # outside
        ri, ci, vi = [], [], []   # inside
        for ey in range(ny):
            for ex in range(nx):
                cx, cy = (ex + 0.5) * hx, (ey + 0.5) * hy
                inside = (cx - INCL_C[0]) ** 2 + (cy - INCL_C[1]) ** 2 < INCL_R ** 2
                n1 = ey * res + ex
                nodes = (n1, n1 + 1, (ey + 1) * res + ex + 1, (ey + 1) * res + ex)
                dofs = [d for n in nodes for d in (2 * n, 2 * n + 1)]
                R, C, V = (ri, ci, vi) if inside else (ro, co, vo)
                for i, di in enumerate(dofs):
                    for j, dj in enumerate(dofs):
                        R.append(di); C.append(dj); V.append(ke[i, j])
        self.K_out = sp.csr_matrix((vo, (ro, co)), shape=(n_dof, n_dof))
        self.K_in = sp.csr_matrix((vi, (ri, ci)), shape=(n_dof, n_dof))
        bc = set()
        for i in range(res):
            for n in (i, (res - 1) * res + i, i * res, i * res + res - 1):
                bc.update((2 * n, 2 * n + 1))
        self.free = np.array(sorted(set(range(n_dof)) - bc))
        self.n_dof = n_dof
        self.hx = hx

    def rhs(self, f_grid):
        res = self.res
        w = np.full((res, res), 4.0)
        w[0, :] = w[-1, :] = w[:, 0] = w[:, -1] = 2.0
        w[0, 0] = w[0, -1] = w[-1, 0] = w[-1, -1] = 1.0
        area = self.hx * self.hx / 4.0
        F = np.zeros(self.n_dof)
        F[0::2] = (w * f_grid[..., 0] * area).reshape(-1)
        F[1::2] = (w * f_grid[..., 1] * area).reshape(-1)
        return F

    def solve(self, f_grid, k):
        """Per-sample: K(k) = K_out + k*K_in must be re-factorized."""
        K = (self.K_out + k * self.K_in).tocsc()
        F = self.rhs(f_grid)
        t0 = time.perf_counter()
        lu = spla.splu(K[np.ix_(self.free, self.free)])
        u_free = lu.solve(F[self.free])
        dt = 1000 * (time.perf_counter() - t0)
        u = np.zeros(self.n_dof)
        u[self.free] = u_free
        return u.reshape(self.res, self.res, 2), dt


def material_fields(res, k, nu=0.30):
    """Nodal lambda(x), mu(x) maps for the residual loss."""
    xx, yy = grid_xy(res)
    E = np.where((xx - INCL_C[0]) ** 2 + (yy - INCL_C[1]) ** 2 < INCL_R ** 2,
                 k, 1.0)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lam, mu


def hetero_selftest():
    """k=1 must reproduce the homogeneous solver; plus mesh convergence."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    lam, mu = lame()
    res = 29
    u_t, _, f = analytical_fields(res, lam, mu)
    hom = FemSolver(res, lam, mu)
    het = HeteroQ4(res)
    u1, _ = hom.solve(f)
    u2, _ = het.solve(f, 1.0)
    print("k=1 vs homogeneous solver: rel diff = %.2e" % rel(u2, u1))
    # mesh convergence for k=4 (no analytical solution; compare to res 113)
    rng = np.random.default_rng(0)
    fr = rng.standard_normal((113, 113, 2))
    # smooth the random forcing so restriction is meaningful
    from scipy.ndimage import gaussian_filter
    fr = gaussian_filter(fr, sigma=(8, 8, 0)) * 50
    uref, _ = HeteroQ4(113).solve(fr, 4.0)
    for r, step in ((29, 4), (57, 2)):
        u, _ = HeteroQ4(r).solve(fr[::113 // (r - 1) * 0 + step, ::step], 4.0) \
            if False else HeteroQ4(r).solve(fr[::step, ::step], 4.0)
        print(f"  res={r}: rel diff to res=113 ref = "
              f"{rel(u, uref[::step, ::step]):.3e}")


def hetero_gen_part(tag, n, seed, res=29, refine=2, nu=0.30,
                    start=0, stop=None):
    """Generate a slice of the heterogeneous ground truth and save it
    to /tmp/femgt_hetero_{tag}_{start}_{stop}.npz (chunk-friendly)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import cmame_extended_study as st
    res_f = refine * (res - 1) + 1
    modes = st.build_modes(16)
    physf = st.PhysicsConfig(res=res_f, nu=nu)
    solver = HeteroQ4(res_f, nu=nu)
    ax, ay = st.generate_coefficients(n, modes, 0.04, seed)
    rng = np.random.default_rng(seed + 7)
    ks = np.exp(rng.uniform(np.log(0.2), np.log(5.0), n))
    stop = n if stop is None else min(stop, n)
    ax, ay = ax[start:stop], ay[start:stop]
    data = st.analytical_solution_and_force(ax, ay, modes, physf)
    m = stop - start
    u = np.zeros((m, res, res, 2), dtype=np.float32)
    t_sum = 0.0
    for i in range(m):
        uf, dt = solver.solve(data["f"][i].astype(np.float64), ks[start + i])
        u[i] = uf[::refine, ::refine]
        t_sum += dt
    fn = f"/tmp/femgt_hetero_{tag}_{start}_{stop}.npz"
    np.savez_compressed(fn, f=data["f"][:, ::refine, ::refine].astype(np.float32),
                        u=u, k=ks[start:stop].astype(np.float32))
    print(f"saved {fn}  (mean per-sample factorize+solve "
          f"{t_sum/m:.2f} ms at res {res_f})")


def hetero_merge():
    import glob
    out = {}
    for tag in ("tr", "te"):
        parts = sorted(glob.glob(f"/tmp/femgt_hetero_{tag}_*.npz"),
                       key=lambda p: int(p.split("_")[-2]))
        fs, us, ks = [], [], []
        for p in parts:
            z = np.load(p)
            fs.append(z["f"]); us.append(z["u"]); ks.append(z["k"])
        out[f"f_{tag}"] = np.concatenate(fs)
        out[f"u_{tag}"] = np.concatenate(us)
        out[f"k_{tag}"] = np.concatenate(ks)
    np.savez_compressed("/tmp/femgt_hetero.npz", **out)
    print("merged:", {k: v.shape for k, v in out.items()})


def hetero_fem_timing(n=50, nu=0.30):
    """Per-sample FEM cost when the operator changes with k (no
    factorization reuse possible) at several meshes."""
    rows = []
    for res in (29, 57):
        solver = HeteroQ4(res, nu=nu)
        rng = np.random.default_rng(1)
        xx, yy = grid_xy(res)
        t = 0.0
        for i in range(n):
            f = np.stack([np.sin((i % 3 + 1) * math.pi * xx) * np.sin(math.pi * yy),
                          np.cos(i) * np.sin(math.pi * xx) * np.sin(2 * math.pi * yy)], -1)
            _, dt = solver.solve(f, float(np.exp(rng.uniform(np.log(0.2), np.log(5)))))
            t += dt
        rows.append(dict(res=res, per_sample_ms=t / n))
        print(rows[-1])
    OUT.mkdir(exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT / "hetero_fem_timing.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lsfem_selftest", action="store_true")
    p.add_argument("--lsfem_stress_table", action="store_true")
    p.add_argument("--hetero_selftest", action="store_true")
    p.add_argument("--hetero_gen_part", nargs=4, metavar=("TAG", "N", "START", "STOP"))
    p.add_argument("--hetero_merge", action="store_true")
    p.add_argument("--hetero_fem_timing", action="store_true")
    a = p.parse_args()
    if a.lsfem_selftest: lsfem_selftest()
    if a.lsfem_stress_table: lsfem_stress_table()
    if a.hetero_selftest: hetero_selftest()
    if a.hetero_gen_part:
        tag, n, s0, s1 = a.hetero_gen_part
        seed = 42 if tag == "tr" else 999
        hetero_gen_part(tag, int(n), seed, start=int(s0), stop=int(s1))
    if a.hetero_merge: hetero_merge()
    if a.hetero_fem_timing: hetero_fem_timing()
