#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hetero_fem_refine.py

Finite-element accuracy under mesh refinement on the smooth and rough modulus
fields, against the 113x113-node reference the benchmarks are scored on.

The material and forcing fields of the 400 test samples are regenerated
exactly as hetero_field_gen.py drew them (forcing seed 999, field seed 1122,
correlation 6 or 1 working-grid cells).  Each sample is then solved with Q4 on
the 15x15, 29x29 and 57x57 node meshes, sampling E and f at that mesh's nodes
from the fine grid, and compared with the stored reference at the working-grid
nodes the coarser mesh shares.  The 29x29 result must reproduce the
finite-element row of the ledger; that is checked and recorded.

Per-query costs are not timed here: the assemble-factorize-solve cost of a Q4
mesh does not depend on the modulus values, so the exclusive-allocation figures
of timing_classical_all.py apply unchanged.

    python hetero_fem_refine.py --data data_rough/hetero_field.npz --corr 1.0 --out ...
"""
import argparse, json, time
from pathlib import Path

import numpy as np

from hetero_field import HeteroFieldQ4, random_E_field, nodal_to_elem
from continuum_elasticity_pideeponet_publication_study import (
    PhysicsConfig, build_modes, generate_coefficients, analytical_solution_and_force)

RES, REFINE = 29, 4


def rel(a, b):
    num = np.linalg.norm((a - b).reshape(len(a), -1), axis=1)
    den = np.linalg.norm(b.reshape(len(b), -1), axis=1) + 1e-300
    return float(np.mean(num / den))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--corr", type=float, required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--n_eval", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    res_f = REFINE * (RES - 1) + 1
    z = np.load(a.data)
    u_ref = z["u_te"].astype(np.float64); E_c = z["E_te"]
    n_eval = a.n_eval or a.n

    modes = build_modes(16)
    ax, ay = generate_coefficients(a.n, modes, 0.04, a.seed)
    E_fine = random_E_field(a.n, res_f, seed=a.seed + 123, corr=REFINE * a.corr)
    f_fine = analytical_solution_and_force(ax[:n_eval], ay[:n_eval], modes,
                                           PhysicsConfig(res=res_f, nu=0.30))["f"]
    regen = float(np.max(np.abs(E_fine[:n_eval, ::REFINE, ::REFINE] - E_c[:n_eval])))
    print(f"regenerated fields: max |E - stored E| on the working grid = {regen:.2e}")

    out = dict(data=a.data, corr=a.corr, n_eval=n_eval, field_regeneration_max_abs_diff=regen,
               reference=f"Q4 on the {res_f}x{res_f} node mesh", errors={})
    for r in (15, 29, 57):
        step = (res_f - 1) // (r - 1)
        solver = HeteroFieldQ4(r, nu=0.30)
        preds = []
        for i in range(n_eval):
            u, _ = solver.solve(f_fine[i, ::step, ::step].astype(np.float64),
                                nodal_to_elem(E_fine[i, ::step, ::step]))
            preds.append(u)
        preds = np.stack(preds)
        if r >= RES:
            k = (r - 1) // (RES - 1); err = rel(preds[:, ::k, ::k], u_ref[:n_eval])
        else:
            k = (RES - 1) // (r - 1); err = rel(preds, u_ref[:n_eval, ::k, ::k])
        out["errors"][f"{r-1}x{r-1}"] = err
        print(f"[{r-1}x{r-1}] error vs reference {err:.4f}  ({(time.time()-t0)/60:.1f} min)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("FEMREFINE_DONE")


if __name__ == "__main__":
    main()
