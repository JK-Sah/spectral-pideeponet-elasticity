#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_forcing_v2.py

Non-sine forcing families (Gaussian bumps, discontinuous patch loads) at the
canonical sizes -- 3000 training, 400 validation, 200 test samples -- with the
FEM reference on the 4x-refined mesh restricted to the 29x29 working grid,
plus the classical baselines on the same test set:

  * Q4 FEM on the 15x15, 29x29 and 57x57 node meshes, against the reference;
  * POD-Galerkin reduced-order models built from FEM snapshots of the training
    forcings at the working resolution, ranks 16..256;
  * the closed-form least-squares read-out at trunk capacity M=16;
  * the representation floor of the M=16 sine trial space.

Seeds: train 42 (the first 1000 samples are the previous training set), val 7,
test 999 (identical to the previous test set).

    python gen_forcing_v2.py --kind bumps --out_dir data
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch

import cmame_extended_study as st
from revision_new_benchmarks import FemSolver, lame, eval_forcing_on_grid
from rom_baseline import assemble_K_free
from canonical_linear import closed_form_readout

RES, REFINE, AMP = 29, 4, 2.0


def reference(kind, n, seed, nu=0.30):
    """Forcing on the refined grid and its FEM solution restricted to RES."""
    res_f = REFINE * (RES - 1) + 1
    lam, mu = lame(nu=nu)
    f_fine = eval_forcing_on_grid(kind, n, res_f, seed, AMP)
    solver = FemSolver(res_f, lam, mu)
    u = np.stack([solver.solve(f_fine[i])[0][::REFINE, ::REFINE] for i in range(n)])
    return f_fine, u


def rel(pred, true):
    num = np.linalg.norm((pred - true).reshape(len(pred), -1), axis=1)
    den = np.linalg.norm(true.reshape(len(true), -1), axis=1) + 1e-300
    return float(np.mean(num / den))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["bumps", "patch"])
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_val", type=int, default=400)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--res_out", default="results_revision/aux")
    a = ap.parse_args()
    lam, mu = lame(nu=0.30)
    t0 = time.time()

    sets = {}
    for name, n, seed in (("tr", a.n_train, 42), ("va", a.n_val, 7), ("te", a.n_test, 999)):
        f_fine, u = reference(a.kind, n, seed)
        sets[name] = (f_fine, u)
        print(f"[{a.kind}] {name}: {n} reference solves done ({(time.time()-t0)/60:.1f} min)")
    f_c = {k: v[0][:, ::REFINE, ::REFINE, :].astype(np.float32) for k, v in sets.items()}
    u_c = {k: v[1].astype(np.float32) for k, v in sets.items()}
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    fn = out / f"forcing_{a.kind}_v2.npz"
    np.savez_compressed(fn, f_tr=f_c["tr"], u_tr=u_c["tr"], f_va=f_c["va"], u_va=u_c["va"],
                        f_te=f_c["te"], u_te=u_c["te"])
    print("saved", fn)

    u_te = u_c["te"].astype(np.float64)
    res = dict(kind=a.kind, n_train=a.n_train, n_val=a.n_val, n_test=a.n_test,
               reference=f"Q4 FEM on the {REFINE*(RES-1)+1}x{REFINE*(RES-1)+1} node mesh")

    # FEM at several meshes: forcing sampled on that mesh's nodes from the fine grid.
    fem = {}
    f_fine_te = sets["te"][0]
    for r_ in (15, 29, 57):
        step = (REFINE * (RES - 1)) // (r_ - 1)
        solver = FemSolver(r_, lam, mu)
        sub = (r_ - 1) // (RES - 1) if r_ > RES else None
        preds = []
        for i in range(a.n_test):
            u_r, _ = solver.solve(f_fine_te[i, ::step, ::step, :])
            if r_ >= RES:
                preds.append(u_r[::(r_ - 1) // (RES - 1), ::(r_ - 1) // (RES - 1)])
            else:   # coarser mesh: compare at its own nodes, which are working-grid nodes
                preds.append(u_r)
        preds = np.stack(preds)
        ref = u_te if r_ >= RES else u_te[:, ::(RES - 1) // (r_ - 1), ::(RES - 1) // (r_ - 1)]
        fem[f"{r_-1}x{r_-1}"] = rel(preds, ref)
        print(f"[fem {r_-1}x{r_-1}] error {fem[f'{r_-1}x{r_-1}']:.4f}")
    res["fem_error"] = fem

    # POD-Galerkin from FEM snapshots of the training forcings at the working resolution.
    K_free, free = assemble_K_free(RES, lam, mu)
    solver = FemSolver(RES, lam, mu)
    S = np.stack([solver.solve(f_c["tr"][i].astype(np.float64))[0].reshape(-1)[free]
                  for i in range(a.n_train)], axis=1)
    U, sv, _ = np.linalg.svd(S, full_matrices=False)
    B = np.stack([solver.rhs(f_c["te"][i].astype(np.float64))[free] for i in range(a.n_test)])
    rom = {}
    for r in (16, 32, 64, 128, 256):
        if r > U.shape[1]:
            continue
        V = U[:, :r]; Kr = V.T @ (K_free @ V)
        A = np.linalg.solve(Kr, V.T @ B.T)                    # (r, n_test)
        uf = np.zeros((a.n_test, 2 * RES * RES)); uf[:, free] = (V @ A).T
        rom[str(r)] = rel(uf.reshape(a.n_test, RES, RES, 2), u_te)
        print(f"[rom r={r}] error {rom[str(r)]:.4f}")
    res["rom_error_by_rank"] = rom
    res["rom_snapshots"] = a.n_train

    # Closed-form read-out and representation floor, trunk M=16.
    modes = st.build_modes(16); grid = st.make_grid(RES)
    tr_f, tr_u = torch.tensor(f_c["tr"]), torch.tensor(u_c["tr"])
    W, b = closed_form_readout(modes, grid, tr_f, tr_u)
    X = st.project_onto_sine(torch.tensor(f_c["te"]), modes, grid).numpy()
    c = X @ W.T + b
    phi = st.sine_basis(grid, modes).numpy()                  # (res,res,M)
    M = len(modes)
    pred = np.stack([np.einsum("nm,ijm->nij", c[:, :M], phi),
                     np.einsum("nm,ijm->nij", c[:, M:], phi)], -1)
    res["closed_form_error"] = rel(pred, u_te)
    # L2-optimal projection of the true solutions onto the sine trial space
    P = phi.reshape(-1, M); G = np.linalg.pinv(P)
    proj = np.stack([(P @ (G @ u_te[i, ..., k].reshape(-1))).reshape(RES, RES)
                     for i in range(a.n_test) for k in (0, 1)]).reshape(a.n_test, 2, RES, RES)
    res["representation_floor"] = rel(np.moveaxis(proj, 1, -1), u_te)
    print(f"[closed-form] {res['closed_form_error']:.4f}  [floor] {res['representation_floor']:.4f}")
    res["minutes"] = (time.time() - t0) / 60
    Path(a.res_out).mkdir(parents=True, exist_ok=True)
    Path(a.res_out, f"forcing_{a.kind}_classical.json").write_text(json.dumps(res, indent=2))
    print("GENFORCING_DONE")


if __name__ == "__main__":
    main()
