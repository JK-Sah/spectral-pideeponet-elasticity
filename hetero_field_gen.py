#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hetero_field_gen.py

Ground-truth generator for the Route-B random-E(x)-field benchmark.

Each sample is a pair (f, E) -> u, where the forcing f is drawn from the same
K=16 sine family used elsewhere in the paper (so heterogeneity is the ONLY new
axis relative to the fixed-operator benchmark), and E(x) is a smooth log-normal
modulus field.  The reference displacement is a Q4 FEM solve on a refined mesh
(refine * (res-1) + 1 nodes), restricted to the res working grid.

Chunk-friendly for Slurm array jobs:
    python hetero_field_gen.py --tag tr --n 2000 --seed 42 --start 0   --stop 250
    ...
    python hetero_field_gen.py --merge --tags tr te --out data/hetero_field.npz

Outputs npz chunks with f[m,res,res,2], E[m,res,res] (nodal), u[m,res,res,2].
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np

from hetero_field import HeteroFieldQ4, random_E_field, nodal_to_elem
from continuum_elasticity_pideeponet_publication_study import (
    PhysicsConfig, build_modes, generate_coefficients, analytical_solution_and_force,
)

DATA = Path(os.environ.get("CMAME_DATA", "data"))


def gen_part(tag, n, seed, start, stop, res=29, refine=4, nu=0.30,
             coeff_scale=0.04, corr=6.0):
    res_f = refine * (res - 1) + 1
    stop = min(stop, n)
    modes = build_modes(16)
    physf = PhysicsConfig(res=res_f, nu=nu)

    # Deterministic full draws, then slice -> reproducible regardless of chunking.
    ax, ay = generate_coefficients(n, modes, coeff_scale, seed)
    E_all = random_E_field(n, res_f, seed=seed + 123, corr=refine * corr)

    sl = slice(start, stop)
    forc = analytical_solution_and_force(ax[sl], ay[sl], modes, physf)  # uses f only
    f_fine = forc["f"].astype(np.float64)                               # [m,res_f,res_f,2]
    E_fine = E_all[sl]                                                  # [m,res_f,res_f]

    solver = HeteroFieldQ4(res_f, nu=nu)
    m = stop - start
    u = np.zeros((m, res, res, 2), dtype=np.float32)
    t_sum = 0.0
    for i in range(m):
        uf, dt = solver.solve(f_fine[i], nodal_to_elem(E_fine[i]))
        u[i] = uf[::refine, ::refine].astype(np.float32)
        t_sum += dt
        if (i + 1) % 50 == 0:
            print(f"  [{tag} {start}:{stop}] {i+1}/{m}  (mean {t_sum/(i+1):.1f} ms/solve)")

    f_c = f_fine[:, ::refine, ::refine, :].astype(np.float32)
    E_c = E_fine[:, ::refine, ::refine].astype(np.float32)
    DATA.mkdir(parents=True, exist_ok=True)
    fn = DATA / f"hetero_field_{tag}_{start}_{stop}.npz"
    np.savez_compressed(fn, f=f_c, E=E_c, u=u,
                        meta=dict(res=res, refine=refine, nu=nu, seed=seed))
    print(f"saved {fn}  (mean per-sample assemble+factorize+solve "
          f"{t_sum/m:.1f} ms at res_fine={res_f}; NO factorization reuse)")


def merge(tags, out):
    data = {}
    for tag in tags:
        parts = sorted(glob.glob(str(DATA / f"hetero_field_{tag}_*.npz")),
                       key=lambda p: int(Path(p).stem.split("_")[-2]))
        if not parts:
            print(f"WARNING: no chunks for tag {tag}")
            continue
        fs, es, us = [], [], []
        for p in parts:
            z = np.load(p, allow_pickle=True)
            fs.append(z["f"]); es.append(z["E"]); us.append(z["u"])
        data[f"f_{tag}"] = np.concatenate(fs)
        data[f"E_{tag}"] = np.concatenate(es)
        data[f"u_{tag}"] = np.concatenate(us)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    print("merged:", {k: v.shape for k, v in data.items()}, "->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="tr")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=10**9)
    ap.add_argument("--res", type=int, default=29)
    ap.add_argument("--refine", type=int, default=4)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--tags", nargs="+", default=["tr", "te"])
    ap.add_argument("--out", default="data/hetero_field.npz")
    args = ap.parse_args()
    if args.merge:
        merge(args.tags, args.out)
    else:
        gen_part(args.tag, args.n, args.seed, args.start, args.stop,
                 res=args.res, refine=args.refine)


if __name__ == "__main__":
    main()
