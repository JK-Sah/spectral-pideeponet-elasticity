#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
revision_new_benchmarks.py

New experiments addressing:

  R1-2 / R2-5 : forcings OUTSIDE the sine family (Gaussian bumps, patch loads)
                with FEM-generated ground truth  -> --exp forcing
  R2-3        : capacity-matched FNO comparison   -> --exp fno_matched
  R3-2        : Poisson ratio sweep toward 0.5    -> --exp nu
  R3-3 / R3-4 : accuracy-vs-cost Pareto data      -> --exp pareto
  R2-4        : OOD ablation (trunk capacity vs physics) -> --exp ood_ablation

Reuses models/training from cmame_extended_study.py (same folder).
FEM ground truth uses one-time splu factorization (fair protocol).

Usage:
  python revision_new_benchmarks.py --selftest          # numpy-only FEM check
  python revision_new_benchmarks.py --exp forcing
  python revision_new_benchmarks.py --exp nu
  python revision_new_benchmarks.py --exp fno_matched
  python revision_new_benchmarks.py --exp pareto
  python revision_new_benchmarks.py --exp ood_ablation
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

OUT = Path("results_revision")


# ======================================================================
# 1. FEM (pure numpy/scipy; factorization reused across samples)
# ======================================================================

def q4_element_stiffness(hx, hy, lam, mu):
    gp = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
    C = np.array([[lam + 2 * mu, lam, 0.0],
                  [lam, lam + 2 * mu, 0.0],
                  [0.0, 0.0, mu]])
    ke = np.zeros((8, 8))
    for xi in gp:
        for eta in gp:
            dN_dxi = np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)]) / 4.0
            dN_deta = np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]) / 4.0
            dN_dx = (2.0 / hx) * dN_dxi
            dN_dy = (2.0 / hy) * dN_deta
            detJ = (hx / 2.0) * (hy / 2.0)
            B = np.zeros((3, 8))
            for k in range(4):
                B[0, 2 * k] = dN_dx[k]
                B[1, 2 * k + 1] = dN_dy[k]
                B[2, 2 * k] = dN_dy[k]
                B[2, 2 * k + 1] = dN_dx[k]
            ke += detJ * B.T @ C @ B
    return ke


class FemSolver:
    """Q4 plane-strain FEM on the unit square, homogeneous Dirichlet BC.
    Assembles and LU-factorizes ONCE; solve() handles per-sample RHS."""

    def __init__(self, res, lam, mu):
        self.res = res
        nx = ny = res - 1
        hx = hy = 1.0 / nx
        self.hx, self.hy = hx, hy
        n_dof = 2 * res * res
        ke = q4_element_stiffness(hx, hy, lam, mu)
        rows, cols, vals = [], [], []
        elems = []
        for ey in range(ny):
            for ex in range(nx):
                n1 = ey * res + ex
                nodes = (n1, n1 + 1, (ey + 1) * res + ex + 1, (ey + 1) * res + ex)
                dofs = [d for n in nodes for d in (2 * n, 2 * n + 1)]
                elems.append((nodes, dofs))
                for i, di in enumerate(dofs):
                    for j, dj in enumerate(dofs):
                        rows.append(di); cols.append(dj); vals.append(ke[i, j])
        K = sp.csr_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))
        bc = set()
        for i in range(res):
            for n in (i, (res - 1) * res + i, i * res, i * res + res - 1):
                bc.update((2 * n, 2 * n + 1))
        self.free = np.array(sorted(set(range(n_dof)) - bc))
        t0 = time.perf_counter()
        self.lu = spla.splu(K[np.ix_(self.free, self.free)].tocsc())
        self.factor_ms = 1000.0 * (time.perf_counter() - t0)
        self.elems = elems
        self.n_dof = n_dof

    def rhs(self, f_grid):
        """Quarter-lumped nodal load (same protocol as the original study).
        f_grid: [res,res,2] indexed [iy,ix,comp]."""
        res, hx, hy = self.res, self.hx, self.hy
        f_glob = np.zeros(self.n_dof)
        w = np.full((res, res), 4.0)          # interior nodes belong to 4 elems
        w[0, :] = w[-1, :] = w[:, 0] = w[:, -1] = 2.0
        w[0, 0] = w[0, -1] = w[-1, 0] = w[-1, -1] = 1.0
        area = hx * hy / 4.0
        f_glob[0::2] = (w * f_grid[..., 0] * area).reshape(-1, order="C")[
            np.arange(res * res)]
        f_glob[1::2] = (w * f_grid[..., 1] * area).reshape(-1)[np.arange(res * res)]
        # note: node n = iy*res + ix matches reshape of [iy,ix]
        return f_glob

    def solve(self, f_grid):
        """Returns u [res,res,2] and per-RHS solve time in ms."""
        f_glob = self.rhs(f_grid)
        t0 = time.perf_counter()
        u_free = self.lu.solve(f_glob[self.free])
        dt = 1000.0 * (time.perf_counter() - t0)
        u = np.zeros(self.n_dof)
        u[self.free] = u_free
        return u.reshape(self.res, self.res, 2), dt


def lame(E=1.0, nu=0.30):
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


# ======================================================================
# 2. NON-BANDLIMITED FORCINGS (outside the sine family)
# ======================================================================

def grid_xy(res):
    x = np.linspace(0, 1, res)
    yy, xx = np.meshgrid(x, x, indexing="ij")
    return xx, yy


def gaussian_bump_forcing(n, res, seed, max_bumps=3, amp=1.0):
    """Random superpositions of Gaussian bumps per component."""
    rng = np.random.default_rng(seed)
    xx, yy = grid_xy(res)
    f = np.zeros((n, res, res, 2), dtype=np.float64)
    meta = []
    for i in range(n):
        rec = []
        for c in range(2):
            for _ in range(rng.integers(1, max_bumps + 1)):
                x0, y0 = rng.uniform(0.2, 0.8, 2)
                s = rng.uniform(0.05, 0.15)
                a = amp * rng.uniform(-1.0, 1.0)
                f[i, ..., c] += a * np.exp(-(((xx - x0) ** 2 + (yy - y0) ** 2)
                                             / (2 * s ** 2)))
                rec.append(dict(comp=c, x0=x0, y0=y0, s=s, a=a))
        meta.append(rec)
    return f, meta


def patch_forcing(n, res, seed, amp=1.0):
    """Piecewise-constant loads on random rectangles (discontinuous)."""
    rng = np.random.default_rng(seed)
    xx, yy = grid_xy(res)
    f = np.zeros((n, res, res, 2), dtype=np.float64)
    for i in range(n):
        for c in range(2):
            x0, y0 = rng.uniform(0.1, 0.6, 2)
            w, h = rng.uniform(0.15, 0.35, 2)
            a = amp * rng.uniform(-1.0, 1.0)
            mask = (xx >= x0) & (xx <= x0 + w) & (yy >= y0) & (yy <= y0 + h)
            f[i, ..., c] += a * mask
    return f, None


def eval_forcing_on_grid(kind, n, res, seed, amp):
    if kind == "bumps":
        return gaussian_bump_forcing(n, res, seed, amp=amp)[0]
    if kind == "patch":
        return patch_forcing(n, res, seed, amp=amp)[0]
    raise ValueError(kind)


def fem_ground_truth(kind, n, res_coarse, seed, nu=0.30, refine=4, amp=2.0,
                     verbose=True):
    """FEM reference on a refined mesh, restricted to the coarse grid.
    refine=4: res_fine = 4*(res_coarse-1)+1 (nested grids)."""
    res_f = refine * (res_coarse - 1) + 1
    lam, mu = lame(nu=nu)
    f_fine = eval_forcing_on_grid(kind, n, res_f, seed, amp)
    f_coarse = f_fine[:, ::refine, ::refine, :]
    solver = FemSolver(res_f, lam, mu)
    u = np.zeros((n, res_coarse, res_coarse, 2), dtype=np.float64)
    t_sum = 0.0
    for i in range(n):
        uf, dt = solver.solve(f_fine[i])
        u[i] = uf[::refine, ::refine, :]
        t_sum += dt
        if verbose and (i + 1) % 100 == 0:
            print(f"  FEM ground truth {i+1}/{n}")
    return (f_coarse.astype(np.float32), u.astype(np.float32),
            dict(res_fine=res_f, factor_ms=solver.factor_ms,
                 mean_solve_ms=t_sum / n))


# ======================================================================
# 3. SELF-TEST (numpy-only): FEM vs manufactured analytical solution
# ======================================================================

def selftest():
    """Verify the FEM pipeline against the manufactured sine solution."""
    lam, mu = lame()
    print("FEM convergence vs analytical sine solution, mode (p,q)=(2,1):")
    p, q = 2, 1
    P, Q = p * math.pi, q * math.pi
    prev = None
    for res in (29, 57, 113):
        xx, yy = grid_xy(res)
        phi = np.sin(P * xx) * np.sin(Q * yy)
        phi_xy = P * Q * np.cos(P * xx) * np.cos(Q * yy)
        ax, ay = 0.01, -0.02
        ux, uy = ax * phi, ay * phi
        lap = -(P ** 2 + Q ** 2) * phi
        fx = -(mu * lap * ax + (lam + mu) * (-(P ** 2) * phi * ax + phi_xy * ay))
        fy = -(mu * lap * ay + (lam + mu) * (phi_xy * ax + -(Q ** 2) * phi * ay))
        f = np.stack([fx, fy], axis=-1)
        u_true = np.stack([ux, uy], axis=-1)
        solver = FemSolver(res, lam, mu)
        u_fem, _ = solver.solve(f)
        err = (np.linalg.norm(u_fem - u_true)
               / np.linalg.norm(u_true))
        rate = "" if prev is None else f"  rate={math.log2(prev/err):.2f}"
        print(f"  res={res:4d}  rel L2 err={err:.3e}{rate}")
        prev = err
    print("Expect err << 1 and rate ~ 2 (Q4 is O(h^2)). If so, FEM GT is sound.")


# ======================================================================
# 4. TORCH EXPERIMENTS (import the existing study lazily)
# ======================================================================

def load_study():
    import torch  # noqa
    import cmame_extended_study as st
    return st


def to_dataset(st, f_tr, u_tr, f_te, u_te, n_modes, phys):
    import torch
    return {
        "train_f": torch.tensor(f_tr), "train_u": torch.tensor(u_tr),
        "test_f": torch.tensor(f_te), "test_u": torch.tensor(u_te),
        "modes": st.build_modes(n_modes),
    }


def pick_fno_width(st, phys, modes, target_params):
    """Smallest width whose FNO param count is closest to target."""
    best = None
    for w in range(4, 65):
        m = st.FNO2dElasticity(phys, modes=modes, width=w)
        p = st.count_params(m)
        d = abs(p - target_params)
        if best is None or d < best[2]:
            best = (w, p, d)
    return best[0], best[1]


def run_forcing(args):
    """R1-2/R2-5: train + evaluate on non-sine forcings with FEM ground truth."""
    st = load_study()
    import torch
    device = st.get_device("auto")
    phys = st.PhysicsConfig(res=args.res, nu=args.nu)
    OUT.mkdir(exist_ok=True)

    for kind in ("bumps", "patch"):
        print(f"\n### Forcing family: {kind} ###")
        f_tr, u_tr, info = fem_ground_truth(kind, args.n_train, args.res,
                                            seed=42, nu=args.nu)
        f_te, u_te, _ = fem_ground_truth(kind, args.n_test, args.res,
                                         seed=999, nu=args.nu)
        print(f"  fine mesh res={info['res_fine']}, "
              f"mean FEM solve {info['mean_solve_ms']:.3f} ms/RHS")
        ds = to_dataset(st, f_tr, u_tr, f_te, u_te, args.model_modes, phys)
        modes = ds["modes"]
        results = {}
        runs = [
            ("PI-Spectral", st.SpectralPIDeepONet(modes, phys, hidden=192, depth=4),
             dict(w_pde=args.w_pde)),
            ("Data-only Spectral", st.SpectralPIDeepONet(modes, phys, hidden=192, depth=4),
             dict(w_pde=0.0)),
        ]
        fno_w, fno_p = pick_fno_width(st, phys, args.fno_modes,
                                      target_params=123_680)
        runs.append((f"FNO (matched, w={fno_w}, {fno_p:,}p)",
                     st.FNO2dElasticity(phys, modes=args.fno_modes, width=fno_w),
                     dict(w_pde=0.0)))
        for name, model, kw in runs:
            model = model.to(device)
            _, final, _ = st.train_model(
                model, ds, phys, device, epochs=args.epochs,
                run_name=f"{kind}_{name}", **kw)
            results[name] = final
        df_rows = [{"model": k, **v} for k, v in results.items()]
        import pandas as pd
        pd.DataFrame(df_rows).to_csv(OUT / f"forcing_{kind}_results.csv",
                                     index=False)
        print(json.dumps({k: {m: round(float(x), 5) for m, x in v.items()}
                          for k, v in results.items()}, indent=2))


def run_nu(args):
    """R3-2: robustness in Poisson ratio toward the incompressible limit.
    Uses the original manufactured sine benchmark at each nu."""
    st = load_study()
    import pandas as pd
    device = st.get_device("auto")
    rows = []
    for nu in (0.30, 0.40, 0.45, 0.49, 0.499):
        phys = st.PhysicsConfig(res=args.res, nu=nu)
        ds = st.make_dataset(args.n_train, args.n_test, args.model_modes,
                             0.04, phys)
        for name, w_pde in (("PI-Spectral", args.w_pde),
                            ("Data-only Spectral", 0.0)):
            model = st.SpectralPIDeepONet(ds["modes"], phys,
                                          hidden=192, depth=4).to(device)
            _, final, _ = st.train_model(model, ds, phys, device,
                                         epochs=args.epochs, w_pde=w_pde,
                                         run_name=f"nu{nu}_{name}")
            rows.append({"nu": nu, "model": name, **final})
            print(f"  nu={nu}  {name}: disp={final['disp_l2']:.4f}")
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "nu_sweep_results.csv", index=False)


def run_fno_matched(args):
    """R2-3: FNO with mode cutoff >= true modes and matched parameters,
    on the ORIGINAL 16-mode manufactured benchmark."""
    st = load_study()
    import pandas as pd
    device = st.get_device("auto")
    phys = st.PhysicsConfig(res=args.res, nu=args.nu)
    ds = st.make_dataset(args.n_train, args.n_test, 16, 0.04, phys)
    rows = []
    # grid is 29 -> rfft y-modes max 15; use 14 to stay safe
    for fmodes in (12, 14):
        w, p = pick_fno_width(st, phys, fmodes, target_params=123_680)
        model = st.FNO2dElasticity(phys, modes=fmodes, width=w).to(device)
        _, final, _ = st.train_model(model, ds, phys, device,
                                     epochs=args.epochs, w_pde=0.0,
                                     run_name=f"FNO_m{fmodes}_w{w}")
        rows.append({"model": f"FNO modes={fmodes} width={w}",
                     "n_params_target": p, **final})
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "fno_matched_results.csv", index=False)


def run_pareto(args):
    """R3-3/R3-4: accuracy vs per-sample cost. FEM at several meshes vs the
    analytical solution on the manufactured benchmark; amortized timing."""
    st = load_study()
    import pandas as pd
    phys = st.PhysicsConfig(res=args.res, nu=args.nu)
    modes = st.build_modes(16)
    ax, ay = st.generate_coefficients(args.n_test, modes, 0.04, 999)
    rows = []
    for res in (15, 29, 57, 113):
        physr = st.PhysicsConfig(res=res, nu=args.nu)
        data = st.analytical_solution_and_force(ax, ay, modes, physr)
        lam, mu = physr.lame
        t0 = time.perf_counter()
        solver = FemSolver(res, lam, mu)
        setup_ms = 1000 * (time.perf_counter() - t0)
        errs, t_solve = [], 0.0
        for i in range(args.n_test):
            u_fem, dt = solver.solve(data["f"][i].astype(np.float64))
            t_solve += dt
            errs.append(np.linalg.norm(u_fem - data["u"][i])
                        / np.linalg.norm(data["u"][i]))
        amort = setup_ms / args.n_test + t_solve / args.n_test
        rows.append({"method": f"FEM res={res}", "disp_l2": float(np.mean(errs)),
                     "per_sample_ms": amort, "setup_ms": setup_ms})
        print(f"  FEM res={res}: err={np.mean(errs):.4f}  {amort:.4f} ms/sample")
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "pareto_fem.csv", index=False)
    print("Merge with model errors/timings from the main study for the figure.")


def run_ood_ablation(args):
    """R2-4: separate sine-capacity ceiling from physics benefit.
    Train PI + data-only with trunk modes 16 vs 24 on 16-mode data,
    evaluate on higher-mode (K up to 28) test sets."""
    st = load_study()
    import pandas as pd
    device = st.get_device("auto")
    phys = st.PhysicsConfig(res=args.res, nu=args.nu)
    ds16 = st.make_dataset(args.n_train, args.n_test, 16, 0.04, phys)
    rows = []
    for trunk_modes in (16, 24):
        modes = st.build_modes(trunk_modes)
        for name, w_pde in (("PI", args.w_pde), ("data-only", 0.0)):
            model = st.SpectralPIDeepONet(modes, phys,
                                          hidden=192, depth=4).to(device)
            ds = dict(ds16); ds["modes"] = modes
            _, final, _ = st.train_model(model, ds, phys, device,
                                         epochs=args.epochs, w_pde=w_pde,
                                         run_name=f"trunk{trunk_modes}_{name}")
            for K in (16, 20, 24, 28):
                dK = st.make_dataset(1, args.n_test, K, 0.04, phys)
                m = st.evaluate_model(model, dK["test_f"], dK["test_u"],
                                      phys, device)
                rows.append({"trunk_modes": trunk_modes, "model": name,
                             "test_K": K, **m})
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "ood_ablation_results.csv", index=False)


# ======================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", choices=["forcing", "nu", "fno_matched",
                                     "pareto", "ood_ablation"])
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--res", type=int, default=29)
    p.add_argument("--nu", type=float, default=0.30)
    p.add_argument("--n_train", type=int, default=1000)
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--model_modes", type=int, default=16)
    p.add_argument("--fno_modes", type=int, default=14)
    p.add_argument("--w_pde", type=float, default=1e-4)
    args = p.parse_args()
    if args.selftest:
        selftest()
        return
    {"forcing": run_forcing, "nu": run_nu, "fno_matched": run_fno_matched,
     "pareto": run_pareto, "ood_ablation": run_ood_ablation}[args.exp](args)


if __name__ == "__main__":
    main()
