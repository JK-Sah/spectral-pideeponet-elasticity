#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hetero_field.py

Route-B benchmark physics: 2D linear elasticity with a *spatially varying*
Young's modulus field E(x) drawn independently per sample (fixed Poisson
ratio).  This is the many-query regime in which the surrogate has a genuine
structural advantage over classical methods:

  * The operator K(E) changes with every sample, so a full FEM solve must
    re-assemble AND re-factorize for each query -- factorization reuse (the
    fair baseline on the fixed-operator problem) is unavailable.
  * The parametric input is the whole field E(x) (O(10^3) dof), so a global
    POD-Galerkin ROM has no low-dimensional affine structure to exploit
    without EIM/DEIM hyper-reduction; a neural operator conditioned on E(x)
    can amortize across the family.

Key fact used for fast assembly: at fixed nu, both Lame parameters are linear
in E, so each element stiffness is  Ke(E_e) = E_e * Ke0  with a single unit
matrix Ke0 (uniform grid).  Hence K(E) = sum_e E_e * scatter(Ke0), assembled
in one vectorized shot per sample and re-factorized with splu.

Pure numpy/scipy.  Run the self-test first:
    python hetero_field.py --selftest
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import gaussian_filter

from revision_new_benchmarks import q4_element_stiffness, FemSolver, lame, grid_xy


class HeteroFieldQ4:
    """Q4 plane-strain FEM with a per-element modulus field E_elem.
    Assembly is O(nnz) per sample; the matrix is re-factorized each solve."""

    def __init__(self, res, nu=0.30):
        self.res = res
        self.nu = nu
        nx = ny = res - 1
        hx = hy = 1.0 / nx
        self.hx = hx
        n_dof = 2 * res * res
        self.n_dof = n_dof
        lam1, mu1 = lame(E=1.0, nu=nu)
        ke0 = q4_element_stiffness(hx, hy, lam1, mu1)     # unit-E element stiffness
        self.ke0_flat = ke0.reshape(-1)                    # [64]

        # Precompute global row/col indices and the element ownership of each
        # of the 64 entries per element, so K(E) is a single scaled COO build.
        n_el = nx * ny
        rows = np.empty(n_el * 64, dtype=np.int64)
        cols = np.empty(n_el * 64, dtype=np.int64)
        base = np.empty(n_el * 64, dtype=np.float64)
        elem = np.empty(n_el * 64, dtype=np.int64)
        e = 0
        for ey in range(ny):
            for ex in range(nx):
                n1 = ey * res + ex
                nodes = (n1, n1 + 1, (ey + 1) * res + ex + 1, (ey + 1) * res + ex)
                dofs = np.array([d for n in nodes for d in (2 * n, 2 * n + 1)])
                sl = slice(e * 64, (e + 1) * 64)
                rows[sl] = np.repeat(dofs, 8)
                cols[sl] = np.tile(dofs, 8)
                base[sl] = self.ke0_flat
                elem[sl] = e
                e += 1
        self.rows, self.cols, self.base, self.elem = rows, cols, base, elem
        self.n_el, self.nx, self.ny = n_el, nx, ny

        bc = set()
        for i in range(res):
            for n in (i, (res - 1) * res + i, i * res, i * res + res - 1):
                bc.update((2 * n, 2 * n + 1))
        self.free = np.array(sorted(set(range(n_dof)) - bc))

    def assemble(self, E_elem):
        """E_elem: [ny, nx] per-element modulus. Returns csc K over all dof."""
        vals = self.base * E_elem.reshape(-1)[self.elem]
        K = sp.csr_matrix((vals, (self.rows, self.cols)),
                          shape=(self.n_dof, self.n_dof))
        return K.tocsc()

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

    def solve(self, f_grid, E_elem):
        """Re-assemble + re-factorize + solve. Returns u[res,res,2], time(ms)."""
        F = self.rhs(f_grid)
        t0 = time.perf_counter()
        K = self.assemble(E_elem)
        lu = spla.splu(K[np.ix_(self.free, self.free)])
        u_free = lu.solve(F[self.free])
        dt = 1000.0 * (time.perf_counter() - t0)
        u = np.zeros(self.n_dof)
        u[self.free] = u_free
        return u.reshape(self.res, self.res, 2), dt


def nodal_to_elem(E_nodal):
    """Average the 4 corner nodal moduli to a per-element modulus."""
    return 0.25 * (E_nodal[:-1, :-1] + E_nodal[1:, :-1]
                   + E_nodal[:-1, 1:] + E_nodal[1:, 1:])


def random_E_field(n, res, seed, corr=6.0, log_contrast=math.log(5.0)):
    """Smooth log-normal modulus fields on the [res,res] node grid.
    corr = Gaussian-filter sigma (correlation length in grid cells);
    log_contrast sets the spread so E mostly lies in [1/5, 5]."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n, res, res))
    g = gaussian_filter(g, sigma=(0, corr, corr))
    g = (g - g.mean(axis=(1, 2), keepdims=True)) / (g.std(axis=(1, 2), keepdims=True) + 1e-12)
    E = np.exp((log_contrast / 2.0) * g)               # log-normal, median 1
    return E.astype(np.float64)                        # [n,res,res]


def selftest():
    lam, mu = lame()
    res = 29
    # (1) E == 1 must reproduce the homogeneous factorization-reuse solver.
    from revision_new_benchmarks import selftest as _  # noqa (ensure import ok)
    xx, yy = grid_xy(res)
    f = np.stack([np.sin(2 * math.pi * xx) * np.sin(math.pi * yy),
                  np.sin(math.pi * xx) * np.sin(3 * math.pi * yy)], axis=-1)
    hom = FemSolver(res, lam, mu)
    fld = HeteroFieldQ4(res)
    u_hom, _ = hom.solve(f)
    u_one, dt1 = fld.solve(f, np.ones((res - 1, res - 1)))
    d = np.linalg.norm(u_one - u_hom) / np.linalg.norm(u_hom)
    print(f"E==1 vs homogeneous solver: rel diff = {d:.2e}  (assemble+solve {dt1:.1f} ms)")

    # (2) Mesh convergence of a random-field solve (coarse vs fine reference).
    E_fine = random_E_field(1, 113, seed=0)[0]
    xf, yf = grid_xy(113)
    ff = np.stack([np.sin(math.pi * xf) * np.sin(math.pi * yf),
                   0.5 * np.sin(2 * math.pi * xf) * np.sin(math.pi * yf)], axis=-1)
    uref, dtf = HeteroFieldQ4(113).solve(ff, nodal_to_elem(E_fine))
    print(f"  res=113 reference solve: assemble+factorize+solve {dtf:.1f} ms")
    for r, step in ((29, 4), (57, 2)):
        Er = E_fine[::step, ::step]
        u, dt = HeteroFieldQ4(r).solve(ff[::step, ::step], nodal_to_elem(Er))
        rel = np.linalg.norm(u - uref[::step, ::step]) / np.linalg.norm(uref[::step, ::step])
        print(f"  res={r:3d}: rel diff to res=113 = {rel:.3e}  "
              f"(per-sample assemble+factorize+solve {dt:.1f} ms)")
    print("Expect E==1 diff ~1e-12 and decreasing coarse-vs-fine diff.")
    print("Note the per-sample ms: this cost recurs for EVERY query (no reuse).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        selftest()


if __name__ == "__main__":
    main()
