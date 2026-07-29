#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nonlinear_fem.py

Finite-strain hyperelastic FEM: compressible Neo-Hookean, plane strain,
total-Lagrangian Q4 elements, Newton-Raphson with optional load stepping.
Homogeneous Dirichlet boundary, body-force loading -- the nonlinear
counterpart of the linear benchmark used elsewhere in this repository.

The map body-force -> displacement is now genuinely nonlinear, which is the
regime where linear-subspace reduced-order models fundamentally struggle and
a learned operator has its clearest opening.

Strain energy (plane strain):  W = mu/2 (tr(F^T F) - 2 - 2 ln J) + lam/2 (ln J)^2
First Piola-Kirchhoff stress:   P = mu (F - F^{-T}) + lam ln J F^{-T}
Node/dof ordering matches revision_new_benchmarks.FemSolver so results are
directly comparable to the linear solver.

Run the verification suite:
    python nonlinear_fem.py --selftest
"""

import argparse
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from revision_new_benchmarks import lame, FemSolver, grid_xy


def neohookean_P(F, lam, mu):
    """First Piola-Kirchhoff stress for compressible Neo-Hookean (2x2 F)."""
    J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    FinvT = np.array([[F[1, 1], -F[1, 0]], [-F[0, 1], F[0, 0]]]) / J   # F^{-T}
    return mu * (F - FinvT) + lam * np.log(J) * FinvT


def neohookean_tangent(F, lam, mu):
    """Analytical first elasticity tensor A_iJkL = dP_iJ/dF_kL as a 4x4 matrix
    (index 2i+J, 2k+L; ordering 11,12,21,22).  Derived from
    P = mu(F - F^{-T}) + lam ln J F^{-T}:
        A_iJkL = mu d_ik d_JL + (mu - lam ln J) G_kJ G_iL + lam G_iJ G_kL,  G = F^{-T}.
    """
    J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    G = np.array([[F[1, 1], -F[1, 0]], [-F[0, 1], F[0, 0]]]) / J        # F^{-T}
    c = mu - lam * np.log(J)
    A = np.zeros((4, 4))
    for i in range(2):
        for Jj in range(2):
            for k in range(2):
                for L in range(2):
                    A[2*i+Jj, 2*k+L] = (mu * (i == k and Jj == L)
                                        + c * G[k, Jj] * G[i, L]
                                        + lam * G[i, Jj] * G[k, L])
    return A


def tangent_fd(F, lam, mu, eps=1e-7):
    """4x4 dP/dF (rows/cols ordered 11,12,21,22) by central finite difference."""
    A = np.zeros((4, 4))
    Ff = F.reshape(-1)
    for j in range(4):
        Fp = Ff.copy(); Fp[j] += eps
        Fm = Ff.copy(); Fm[j] -= eps
        Pp = neohookean_P(Fp.reshape(2, 2), lam, mu).reshape(-1)
        Pm = neohookean_P(Fm.reshape(2, 2), lam, mu).reshape(-1)
        A[:, j] = (Pp - Pm) / (2 * eps)
    return A


class NonlinearFEM:
    def __init__(self, res, E=1.0, nu=0.30):
        self.res = res
        self.lam, self.mu = lame(E=E, nu=nu)
        nx = ny = res - 1
        hx = hy = 1.0 / nx
        self.hx = hx
        n_dof = 2 * res * res
        self.n_dof = n_dof

        # Reference-element operators (identical for every element on a uniform grid).
        g = 1.0 / np.sqrt(3.0)
        gps = [(-g, -g), (g, -g), (g, g), (-g, g)]
        xi_n = np.array([-1, 1, 1, -1]); et_n = np.array([-1, -1, 1, 1])
        detJ = hx * hy / 4.0
        self.gp_ops = []
        for xi, et in gps:
            N = 0.25 * (1 + xi * xi_n) * (1 + et * et_n)
            dNdxi = 0.25 * xi_n * (1 + et * et_n)
            dNdet = 0.25 * et_n * (1 + xi * xi_n)
            dNdX = dNdxi * (2.0 / hx)
            dNdY = dNdet * (2.0 / hy)
            G = np.zeros((4, 8))                 # [ux_X, ux_Y, uy_X, uy_Y] = G u_e
            for i in range(4):
                G[0, 2 * i] = dNdX[i]
                G[1, 2 * i] = dNdY[i]
                G[2, 2 * i + 1] = dNdX[i]
                G[3, 2 * i + 1] = dNdY[i]
            self.gp_ops.append((N, G, detJ))     # weight = detJ * 1 (Gauss wt 1)

        # Connectivity + boundary dofs (same convention as FemSolver).
        elems = []
        for ey in range(ny):
            for ex in range(nx):
                n1 = ey * res + ex
                nodes = (n1, n1 + 1, (ey + 1) * res + ex + 1, (ey + 1) * res + ex)
                dofs = np.array([d for n in nodes for d in (2 * n, 2 * n + 1)])
                elems.append(dofs)
        self.elems = elems
        bc = set()
        for i in range(res):
            for n in (i, (res - 1) * res + i, i * res, i * res + res - 1):
                bc.update((2 * n, 2 * n + 1))
        self.free = np.array(sorted(set(range(n_dof)) - bc))

    def external_force(self, f_grid):
        """Consistent nodal load vector from a body-force field f_grid[res,res,2]."""
        F = np.zeros(self.n_dof)
        fx = f_grid[..., 0].reshape(-1)      # node n = iy*res+ix
        fy = f_grid[..., 1].reshape(-1)
        for dofs in self.elems:
            nodes = dofs[0::2] // 2
            fxe = fx[nodes]; fye = fy[nodes]
            fe = np.zeros(8)
            for (N, _, w) in self.gp_ops:
                fxg = N @ fxe; fyg = N @ fye
                fe[0::2] += N * fxg * w
                fe[1::2] += N * fyg * w
            F[dofs] += fe
        return F

    def assemble(self, u, tangent=True):
        """Internal force R_int(u) and (optionally) tangent stiffness K_t(u)."""
        R = np.zeros(self.n_dof)
        rows, cols, vals = [], [], []
        lam, mu = self.lam, self.mu
        for dofs in self.elems:
            ue = u[dofs]
            fe = np.zeros(8)
            Ke = np.zeros((8, 8)) if tangent else None
            for (N, G, w) in self.gp_ops:
                grad = G @ ue
                F = np.array([[1 + grad[0], grad[1]], [grad[2], 1 + grad[3]]])
                P = neohookean_P(F, lam, mu).reshape(-1)
                fe += G.T @ P * w
                if tangent:
                    AA = neohookean_tangent(F, lam, mu)
                    Ke += G.T @ AA @ G * w
            R[dofs] += fe
            if tangent:
                for a in range(8):
                    for b in range(8):
                        rows.append(dofs[a]); cols.append(dofs[b]); vals.append(Ke[a, b])
        if tangent:
            K = sp.csr_matrix((vals, (rows, cols)), shape=(self.n_dof, self.n_dof))
            return R, K
        return R

    def solve(self, f_grid, n_steps=1, tol=1e-9, maxit=50, verbose=False):
        """Newton-Raphson with n_steps load increments. Returns u[res,res,2], info."""
        f_ext_full = self.external_force(f_grid)
        u = np.zeros(self.n_dof)
        free = self.free
        t0 = time.perf_counter()
        total_newton = 0
        for s in range(1, n_steps + 1):
            f_ext = f_ext_full * (s / n_steps)
            for it in range(maxit):
                R, K = self.assemble(u, tangent=True)
                res_vec = (R - f_ext)[free]
                rnorm = np.linalg.norm(res_vec)
                if verbose:
                    print(f"  step {s}/{n_steps} it {it} |R|={rnorm:.3e}")
                if rnorm < tol:
                    break
                du = spla.spsolve(K[np.ix_(free, free)].tocsc(), -res_vec)
                u[free] += du
                total_newton += 1
            else:
                return u.reshape(self.res, self.res, 2), dict(converged=False, step=s)
        dt = 1000 * (time.perf_counter() - t0)
        return u.reshape(self.res, self.res, 2), dict(converged=True,
                                                      newton_iters=total_newton, ms=dt)


# ----------------------------------------------------------------------
def selftest():
    np.random.seed(0)
    res = 15
    fem = NonlinearFEM(res, nu=0.30)
    lam, mu = fem.lam, fem.mu
    xx, yy = grid_xy(res)

    # (1) Global tangent consistency: K_t == dR/du by finite difference.
    u0 = 0.01 * np.random.randn(fem.n_dof)
    u0[np.setdiff1d(np.arange(fem.n_dof), fem.free)] = 0.0
    R0, K0 = fem.assemble(u0, tangent=True)
    free = fem.free
    Kfd = np.zeros((free.size, free.size))
    eps = 1e-6
    for j in range(free.size):
        up = u0.copy(); up[free[j]] += eps
        Rp = fem.assemble(up, tangent=False)
        Kfd[:, j] = (Rp - R0)[free] / eps
    Kd = np.asarray(K0[np.ix_(free, free)].todense())
    rel = np.linalg.norm(Kd - Kfd) / np.linalg.norm(Kd)
    print(f"(1) tangent vs FD(dR/du): rel diff = {rel:.2e}  (want << 1e-2)")

    # (2) Small-force limit IS linear elasticity: the initial tangent K(u=0) is
    #     the linear elastic stiffness, so the small-strain solution equals the
    #     linear solve with the SAME consistent load (rules out the lumped-vs-
    #     consistent load difference from confusing the check).
    amp = 1e-4
    f = np.stack([amp * np.sin(np.pi * xx) * np.sin(np.pi * yy),
                  amp * 0.5 * np.sin(2 * np.pi * xx) * np.sin(np.pi * yy)], axis=-1)
    u_nl, info = fem.solve(f, n_steps=1)
    f_ext = fem.external_force(f)
    _, K0 = fem.assemble(np.zeros(fem.n_dof), tangent=True)
    ul = np.zeros(fem.n_dof)
    ul[free] = spla.spsolve(K0[np.ix_(free, free)].tocsc(), f_ext[free])
    d = np.linalg.norm(u_nl.reshape(-1) - ul) / (np.linalg.norm(ul) + 1e-30)
    print(f"(2) small-force NL vs linear (same load): rel diff = {d:.2e}  "
          f"(want << 1e-3; Newton its={info.get('newton_iters')})")

    # (3) Genuine nonlinearity at large force: u(2f) != 2 u(f).
    big = 6.0
    fb = np.stack([big * np.sin(np.pi * xx) * np.sin(np.pi * yy),
                   big * 0.4 * np.sin(np.pi * xx) * np.sin(2 * np.pi * yy)], axis=-1)
    u1, i1 = fem.solve(fb, n_steps=8)
    u2, i2 = fem.solve(2 * fb, n_steps=12)
    nl = np.linalg.norm(u2 - 2 * u1) / np.linalg.norm(2 * u1)
    maxstrain = np.abs(u1).max() / fem.hx
    print(f"(3) large force converged={i1['converged'] and i2['converged']}, "
          f"|u(2f)-2u(f)|/|2u(f)| = {nl:.3f} (want >> 0: nonlinear); "
          f"peak |u|/h ~ {maxstrain:.1f}")
    print("Foundation OK if: (1) << 1e-2, (2) small, (3) converged and nonlinearity clear.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    selftest()
