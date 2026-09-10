#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nonlinear_fem_fast.py

Vectorized total-Lagrangian Neo-Hookean Q4 solver: same discretization and
same converged solution as NonlinearFEM in nonlinear_fem.py, but with the
element and Gauss-point loops replaced by array operations.

The original assembly ran an interpreted loop over elements, an inner
2x2x2x2 loop inside the tangent, and an 8x8 triplet loop per element --
roughly 1e5 interpreted operations per assembly.  Reported per-query cost
therefore measured interpreter overhead rather than the intrinsic cost of the
method, which is not a fair baseline for a paper whose premise is that
learned methods should be compared against classical methods as a competent
practitioner would implement them.

Everything here is per-element-batched:
  - deformation gradient, stress and tangent evaluated for all elements and
    all Gauss points at once,
  - element residual and stiffness contracted with einsum,
  - the global sparse matrix built from precomputed index arrays in one call.

The solver also records the statistics the reviewer asked for: Newton
iterations per load step, and a wall-clock breakdown over constitutive
evaluation, assembly, factorization and triangular solve.

Verify against the reference implementation (must agree to round-off):
    python nonlinear_fem_fast.py --verify
Benchmark both implementations:
    python nonlinear_fem_fast.py --bench
"""

import argparse
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from revision_new_benchmarks import lame, grid_xy
from nonlinear_fem import NonlinearFEM


# ---------------------------------------------------------------- constitutive
# Flat 2x2 ordering throughout is [00, 01, 10, 11], matching F.reshape(-1) in
# the reference implementation so the two agree index for index.

def _kinematics(Fflat):
    """J and G = F^{-T} (flat, same ordering) for a batch of deformation gradients."""
    F00, F01, F10, F11 = Fflat[..., 0], Fflat[..., 1], Fflat[..., 2], Fflat[..., 3]
    J = F00 * F11 - F01 * F10
    G = np.stack([F11 / J, -F10 / J, -F01 / J, F00 / J], axis=-1)
    return J, G


def P_batch(Fflat, lam, mu):
    """First Piola-Kirchhoff stress, batched.  P = mu(F - F^{-T}) + lam ln J F^{-T}."""
    J, G = _kinematics(Fflat)
    return mu * (Fflat - G) + lam * np.log(J)[..., None] * G


def A_batch(Fflat, lam, mu):
    """Elasticity tensor A_iJkL = dP_iJ/dF_kL, batched, as (..., 4, 4).

    A_iJkL = mu d_ik d_JL + (mu - lam ln J) G_kJ G_iL + lam G_iJ G_kL,  G = F^{-T}.
    The 16 component assignments are a Python loop, but each one is a single
    array operation over every element and Gauss point.
    """
    J, G = _kinematics(Fflat)
    c = mu - lam * np.log(J)
    A = np.empty(Fflat.shape[:-1] + (4, 4))
    for i in range(2):
        for Jj in range(2):
            for k in range(2):
                for L in range(2):
                    A[..., 2 * i + Jj, 2 * k + L] = (
                        mu * float(i == k and Jj == L)
                        + c * G[..., 2 * k + Jj] * G[..., 2 * i + L]
                        + lam * G[..., 2 * i + Jj] * G[..., 2 * k + L]
                    )
    return A


# ---------------------------------------------------------------- solver
class FastNonlinearFEM:
    """Drop-in replacement for NonlinearFEM with vectorized assembly."""

    def __init__(self, res, E=1.0, nu=0.30):
        self.res = res
        self.lam, self.mu = lame(E=E, nu=nu)
        nx = ny = res - 1
        hx = hy = 1.0 / nx
        self.hx = hx
        self.n_dof = 2 * res * res

        # Reference-element operators, stacked over the four Gauss points.
        g = 1.0 / np.sqrt(3.0)
        gps = [(-g, -g), (g, -g), (g, g), (-g, g)]
        xi_n = np.array([-1, 1, 1, -1])
        et_n = np.array([-1, -1, 1, 1])
        self.detJ = hx * hy / 4.0
        Ns, Gs = [], []
        for xi, et in gps:
            N = 0.25 * (1 + xi * xi_n) * (1 + et * et_n)
            dNdX = 0.25 * xi_n * (1 + et * et_n) * (2.0 / hx)
            dNdY = 0.25 * et_n * (1 + xi * xi_n) * (2.0 / hy)
            G = np.zeros((4, 8))
            for i in range(4):
                G[0, 2 * i] = dNdX[i]
                G[1, 2 * i] = dNdY[i]
                G[2, 2 * i + 1] = dNdX[i]
                G[3, 2 * i + 1] = dNdY[i]
            Ns.append(N)
            Gs.append(G)
        self.N = np.array(Ns)                  # (4gp, 4nodes)
        self.G = np.array(Gs)                  # (4gp, 4comp, 8dof)

        # Connectivity, identical ordering to the reference implementation.
        elem_nodes = np.empty((nx * ny, 4), dtype=np.int64)
        e = 0
        for ey in range(ny):
            for ex in range(nx):
                n1 = ey * res + ex
                elem_nodes[e] = (n1, n1 + 1, (ey + 1) * res + ex + 1, (ey + 1) * res + ex)
                e += 1
        self.elem_nodes = elem_nodes
        ed = np.empty((elem_nodes.shape[0], 8), dtype=np.int64)
        ed[:, 0::2] = 2 * elem_nodes
        ed[:, 1::2] = 2 * elem_nodes + 1
        self.elem_dofs = ed
        self.n_elem = ed.shape[0]

        # Precomputed COO index arrays: Ke[e,a,b] -> (ed[e,a], ed[e,b]).
        self.rows = np.repeat(ed, 8, axis=1).ravel()
        self.cols = np.tile(ed, (1, 8)).ravel()

        bc = set()
        for i in range(res):
            for n in (i, (res - 1) * res + i, i * res, i * res + res - 1):
                bc.update((2 * n, 2 * n + 1))
        self.free = np.array(sorted(set(range(self.n_dof)) - bc))

    # -- loads ---------------------------------------------------------
    def external_force(self, f_grid):
        fx = f_grid[..., 0].reshape(-1)[self.elem_nodes]      # (ne, 4)
        fy = f_grid[..., 1].reshape(-1)[self.elem_nodes]
        fxg = fx @ self.N.T                                    # (ne, gp)
        fyg = fy @ self.N.T
        fe = np.zeros((self.n_elem, 8))
        fe[:, 0::2] = np.einsum("eg,gn->en", fxg, self.N, optimize=True) * self.detJ
        fe[:, 1::2] = np.einsum("eg,gn->en", fyg, self.N, optimize=True) * self.detJ
        return np.bincount(self.elem_dofs.ravel(), weights=fe.ravel(),
                           minlength=self.n_dof)

    # -- residual and tangent -----------------------------------------
    def assemble(self, u, tangent=True, timers=None):
        w = self.detJ
        ue = u[self.elem_dofs]                                 # (ne, 8)
        t0 = time.perf_counter()
        grad = np.einsum("gcj,ej->egc", self.G, ue, optimize=True)            # (ne, gp, 4)
        Fflat = grad + np.array([1.0, 0.0, 0.0, 1.0])
        P = P_batch(Fflat, self.lam, self.mu)
        A = A_batch(Fflat, self.lam, self.mu) if tangent else None
        t1 = time.perf_counter()

        fe = np.einsum("gca,egc->ea", self.G, P, optimize=True) * w
        R = np.bincount(self.elem_dofs.ravel(), weights=fe.ravel(),
                        minlength=self.n_dof)
        if not tangent:
            if timers is not None:
                timers["constitutive"] += t1 - t0
                timers["assembly"] += time.perf_counter() - t1
            return R

        Ke = np.einsum("gca,egcd,gdb->eab", self.G, A, self.G, optimize=True) * w
        K = sp.coo_matrix((Ke.ravel(), (self.rows, self.cols)),
                          shape=(self.n_dof, self.n_dof)).tocsr()
        if timers is not None:
            timers["constitutive"] += t1 - t0
            timers["assembly"] += time.perf_counter() - t1
        return R, K

    # -- Newton --------------------------------------------------------
    def solve(self, f_grid, n_steps=1, tol=1e-9, maxit=50, verbose=False,
              collect_stats=False):
        f_ext_full = self.external_force(f_grid)
        u = np.zeros(self.n_dof)
        free = self.free
        timers = dict(constitutive=0.0, assembly=0.0, factorize=0.0, solve=0.0)
        per_step = []
        t0 = time.perf_counter()
        total_newton = 0
        for s in range(1, n_steps + 1):
            f_ext = f_ext_full * (s / n_steps)
            its, rlast = 0, np.nan
            for it in range(maxit):
                R, K = self.assemble(u, tangent=True, timers=timers)
                res_vec = (R - f_ext)[free]
                rnorm = np.linalg.norm(res_vec)
                rlast = rnorm
                if verbose:
                    print(f"  step {s}/{n_steps} it {it} |R|={rnorm:.3e}")
                if rnorm < tol:
                    break
                ta = time.perf_counter()
                Kff = K[np.ix_(free, free)].tocsc()
                lu = spla.splu(Kff)
                tb = time.perf_counter()
                du = lu.solve(-res_vec)
                tc = time.perf_counter()
                timers["factorize"] += tb - ta
                timers["solve"] += tc - tb
                u[free] += du
                total_newton += 1
                its += 1
            else:
                return u.reshape(self.res, self.res, 2), dict(converged=False, step=s)
            per_step.append(dict(step=s, newton_iters=its, final_residual=float(rlast)))
        dt = 1000 * (time.perf_counter() - t0)
        info = dict(converged=True, newton_iters=total_newton, ms=dt)
        if collect_stats:
            info["per_step"] = per_step
            info["breakdown_ms"] = {k: 1000 * v for k, v in timers.items()}
            acc = sum(timers.values())
            info["breakdown_frac"] = {k: (v / acc if acc else 0.0)
                                      for k, v in timers.items()}
        return u.reshape(self.res, self.res, 2), info


# ---------------------------------------------------------------- checks
def _test_field(res, amp):
    xx, yy = grid_xy(res)
    return np.stack([amp * np.sin(np.pi * xx) * np.sin(np.pi * yy),
                     amp * 0.4 * np.sin(np.pi * xx) * np.sin(2 * np.pi * yy)],
                    axis=-1)


def verify(res=29):
    """Vectorized implementation must reproduce the reference to round-off."""
    np.random.seed(0)
    ref = NonlinearFEM(res)
    fast = FastNonlinearFEM(res)
    print(f"mesh {res-1}x{res-1}, {fast.n_elem} elements, {fast.n_dof} dofs, "
          f"{fast.free.size} free")

    # (a) identical connectivity and boundary sets
    same_dofs = all(np.array_equal(np.asarray(d), fast.elem_dofs[i])
                    for i, d in enumerate(ref.elems))
    print(f"(a) connectivity identical: {same_dofs}; free dofs identical: "
          f"{np.array_equal(ref.free, fast.free)}")

    # (b) residual and tangent at a random admissible state.  The perturbation
    #     is scaled by the element size so the displacement gradient stays small
    #     and every element keeps J > 0; an unscaled probe inverts elements at
    #     fine meshes and makes log(J) undefined in either implementation.
    u = 0.02 * fast.hx * np.random.randn(fast.n_dof)
    u[np.setdiff1d(np.arange(fast.n_dof), fast.free)] = 0.0
    R0, K0 = ref.assemble(u, tangent=True)
    R1, K1 = fast.assemble(u, tangent=True)
    dR = np.linalg.norm(R1 - R0) / (np.linalg.norm(R0) + 1e-300)
    dK = spla.norm(K1 - K0) / spla.norm(K0)
    print(f"(b) residual rel diff = {dR:.3e}   tangent rel diff = {dK:.3e}")

    # (c) external force
    f = _test_field(res, 6.0)
    dF = (np.linalg.norm(fast.external_force(f) - ref.external_force(f))
          / np.linalg.norm(ref.external_force(f)))
    print(f"(c) external force rel diff = {dF:.3e}")

    # (d) converged nonlinear solutions agree
    u_ref, i_ref = ref.solve(f, n_steps=6)
    u_fast, i_fast = fast.solve(f, n_steps=6, collect_stats=True)
    du = np.linalg.norm(u_fast - u_ref) / np.linalg.norm(u_ref)
    print(f"(d) converged solution rel diff = {du:.3e}  "
          f"(Newton its ref={i_ref['newton_iters']} fast={i_fast['newton_iters']}, "
          f"peak |u|/h = {np.abs(u_fast).max()/fast.hx:.2f})")
    # (e) same comparison at the converged large-deformation state
    uc = u_ref.reshape(-1).copy()
    Rr, Kr = ref.assemble(uc, tangent=True)
    Rf, Kf = fast.assemble(uc, tangent=True)
    dR2 = np.linalg.norm(Rf - Rr) / (np.linalg.norm(Rr) + 1e-300)
    dK2 = spla.norm(Kf - Kr) / spla.norm(Kr)
    print(f"(e) at converged deformed state: residual {dR2:.3e}  tangent {dK2:.3e}")

    ok = (dR < 1e-10 and dK < 1e-10 and du < 1e-10 and same_dofs
          and dR2 < 1e-10 and dK2 < 1e-10)
    print("VERIFY", "PASS" if ok else "FAIL")
    return ok


def bench(res=29, n_steps=6, reps=3):
    f = _test_field(res, 6.0)
    ref = NonlinearFEM(res)
    fast = FastNonlinearFEM(res)

    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); ref.solve(f, n_steps=n_steps)
        t.append(time.perf_counter() - t0)
    t_ref = min(t)

    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); _, info = fast.solve(f, n_steps=n_steps,
                                                       collect_stats=True)
        t.append(time.perf_counter() - t0)
    t_fast = min(t)

    print(f"\nmesh {res-1}x{res-1} ({fast.n_dof} dofs), {n_steps} load steps, "
          f"best of {reps}")
    print(f"  reference (interpreted loops): {1000*t_ref:9.1f} ms")
    print(f"  vectorized                   : {1000*t_fast:9.1f} ms")
    print(f"  speedup                      : {t_ref/t_fast:9.1f}x")
    print(f"  Newton iterations            : {info['newton_iters']}"
          f"  (per step: {[d['newton_iters'] for d in info['per_step']]})")
    print("  breakdown (ms / fraction):")
    for k, v in info["breakdown_ms"].items():
        print(f"    {k:<13} {v:8.2f}   {info['breakdown_frac'][k]*100:5.1f}%")
    return t_ref, t_fast, info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--res", type=int, default=29)
    a = ap.parse_args()
    if a.verify or not a.bench:
        verify(a.res)
    if a.bench:
        bench(a.res)
