#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nonlinear_gen.py

Ground-truth generator for the finite-strain hyperelastic benchmark:
smooth body force f (K=16 damped sine modes, scaled into the finite-strain
regime) -> displacement u, solved with the Newton-Raphson Neo-Hookean FEM of
nonlinear_fem.py at the res working grid.  The map is genuinely nonlinear in
f, which is the regime a linear reduced-order model cannot span.

Chunk-friendly for Slurm arrays:
    python nonlinear_gen.py --tag tr --n 2000 --seed 42 --start 0 --stop 250 --scale 40
    python nonlinear_gen.py --merge --tags tr te --out data/nonlinear.npz
"""

import argparse
import glob
import math
import os
from pathlib import Path

import numpy as np

from nonlinear_fem import NonlinearFEM
from revision_new_benchmarks import grid_xy
from cmame_extended_study import build_modes

DATA = Path(os.environ.get("CMAME_DATA", "data"))


def make_forcing(n, seed, res, scale, K=16):
    """n smooth body-force fields: damped random K-mode sine expansion x scale."""
    modes = build_modes(K)
    xx, yy = grid_xy(res)
    Phi = np.stack([np.sin(p * math.pi * xx) * np.sin(q * math.pi * yy)
                    for (p, q) in modes], axis=0)          # [K,res,res]
    damp = np.array([1.0 / (p * p + q * q) for (p, q) in modes])
    rng = np.random.default_rng(seed)
    cx = rng.normal(0, 1, (n, K)) * damp
    cy = rng.normal(0, 1, (n, K)) * damp
    fx = np.einsum("nk,kij->nij", cx, Phi)
    fy = np.einsum("nk,kij->nij", cy, Phi)
    return scale * np.stack([fx, fy], axis=-1)             # [n,res,res,2]


def gen_part(tag, n, seed, start, stop, res=29, nu=0.30, scale=40.0, base_steps=6):
    stop = min(stop, n)
    f_all = make_forcing(n, seed, res, scale)              # deterministic full draw
    fem = NonlinearFEM(res, nu=nu)
    m = stop - start
    u = np.zeros((m, res, res, 2), dtype=np.float32)
    conv = np.zeros(m, dtype=bool)
    iters = np.zeros(m, dtype=np.int32)
    t_sum = 0.0
    import time
    for i in range(m):
        f = f_all[start + i].astype(np.float64)
        info = {"converged": False}
        for ns in (base_steps, 2 * base_steps, 4 * base_steps):   # adaptive load stepping
            t0 = time.perf_counter()
            ui, info = fem.solve(f, n_steps=ns)
            dt = time.perf_counter() - t0
            if info["converged"]:
                break
        u[i] = ui.astype(np.float32)
        conv[i] = info["converged"]
        iters[i] = info.get("newton_iters", -1)
        t_sum += dt
        if (i + 1) % 25 == 0:
            print(f"  [{tag} {start}:{stop}] {i+1}/{m}  conv={conv[:i+1].mean():.2f} "
                  f"mean {t_sum/(i+1):.2f} s/solve")
    DATA.mkdir(parents=True, exist_ok=True)
    fn = DATA / f"nonlinear_{tag}_{start}_{stop}.npz"
    np.savez_compressed(fn, f=f_all[start:stop].astype(np.float32), u=u,
                        conv=conv, iters=iters,
                        meta=dict(res=res, nu=nu, scale=scale))
    print(f"saved {fn}  ({conv.mean()*100:.1f}% converged; "
          f"mean per-sample Newton solve {t_sum/m:.2f} s at res {res})")


def merge(tags, out):
    data = {}
    for tag in tags:
        parts = sorted(glob.glob(str(DATA / f"nonlinear_{tag}_*.npz")),
                       key=lambda p: int(Path(p).stem.split("_")[-2]))
        if not parts:
            print(f"WARNING no chunks for {tag}"); continue
        fs, us, cs = [], [], []
        for p in parts:
            z = np.load(p, allow_pickle=True)
            fs.append(z["f"]); us.append(z["u"]); cs.append(z["conv"])
        f = np.concatenate(fs); u = np.concatenate(us); c = np.concatenate(cs)
        # keep only converged samples
        data[f"f_{tag}"] = f[c]; data[f"u_{tag}"] = u[c]
        print(f"  {tag}: {c.sum()}/{len(c)} converged kept")
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
    ap.add_argument("--scale", type=float, default=10.0)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--tags", nargs="+", default=["tr", "te"])
    ap.add_argument("--out", default="data/nonlinear.npz")
    a = ap.parse_args()
    if a.merge:
        merge(a.tags, a.out)
    else:
        gen_part(a.tag, a.n, a.seed, a.start, a.stop, res=a.res, scale=a.scale)


if __name__ == "__main__":
    main()
