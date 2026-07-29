#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nonlinear_ledger.py

Final accuracy-cost ledger for the finite-strain hyperelastic benchmark, with
all classical methods timed on the same CPU using the analytical tangent:
    Newton-FEM (reference) | POD-Galerkin | POD-DEIM | + neural rows.

    python nonlinear_ledger.py --data data/nonlinear.npz --r 32 --m 128
"""

import argparse, json, os, time
import numpy as np

from nonlinear_fem import NonlinearFEM
from pod_deim import (build_pod, collect_trajectory_forces, build_deim_rom,
                      reduced_galerkin_solve, reduced_deim_solve)


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/nonlinear.npz")
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--n_test", type=int, default=8)
    ap.add_argument("--n_traj", type=int, default=40)
    ap.add_argument("--out", default="runs/nonlinear/ledger.json")
    ap.add_argument("--neural_dir", default="runs/nonlinear")
    a = ap.parse_args()

    z = np.load(a.data)
    u_tr, f_tr, f_te, u_te = z["u_tr"], z["f_tr"], z["f_te"], z["u_te"]
    res = u_tr.shape[1]
    fem = NonlinearFEM(res, nu=0.30)
    S = u_tr.reshape(u_tr.shape[0], -1)[:, fem.free]
    nt = a.n_test
    rows = {}

    # Newton-FEM (analytical tangent): reference solution; measure per-query cost.
    t = []
    for i in range(nt):
        t0 = time.perf_counter()
        _u, info = fem.solve(f_te[i].astype(np.float64), n_steps=6)
        t.append(1000 * (time.perf_counter() - t0))
    rows["Newton-FEM"] = dict(err=0.0, ms=float(np.mean(t)), note="reference")
    print(f"Newton-FEM: {np.mean(t):.0f} ms/query (reference, err=0)")

    # POD-Galerkin (reduced unknowns, full assembly -> accurate, no speedup).
    V, _ = build_pod(S, a.r)
    e, t = [], []
    for i in range(nt):
        u, info = reduced_galerkin_solve(fem, f_te[i].astype(np.float64), V)
        if u is None: continue
        e.append(rel(u, u_te[i])); t.append(info["ms"])
    rows[f"POD-Galerkin(r={a.r})"] = dict(err=float(np.mean(e)), ms=float(np.mean(t)))
    print(f"POD-Galerkin r={a.r}: err {np.mean(e):.4f}, {np.mean(t):.0f} ms/query")

    # POD-DEIM (hyper-reduced).
    Fs = collect_trajectory_forces(fem, V, [f_tr[i].astype(np.float64)
                                            for i in range(a.n_traj)])
    rom = build_deim_rom(fem, S, a.r, a.m, Fs=Fs)
    e, t = [], []
    for i in range(nt):
        u, info = reduced_deim_solve(fem, f_te[i].astype(np.float64), rom)
        if u is None: continue
        e.append(rel(u, u_te[i])); t.append(info["ms"])
    rows[f"POD-DEIM(r={a.r},m={a.m})"] = dict(
        err=float(np.mean(e)), ms=float(np.mean(t)),
        sample_mesh=f"{len(rom['elems'])}/{len(fem.elems)}")
    print(f"POD-DEIM r={a.r} m={a.m}: err {np.mean(e):.4f}, {np.mean(t):.0f} ms/query, "
          f"sample mesh {len(rom['elems'])}/{len(fem.elems)}")

    # Neural rows (accuracy from training JSONs; inference cost is ~1 ms, fixed).
    for name, fn in (("FNO", "fno_seed111.json"), ("Spectral", "spectral_seed111.json")):
        p = os.path.join(a.neural_dir, fn)
        if os.path.exists(p):
            d = json.load(open(p))
            rows[name] = dict(err=d.get("rel_l2_u", -1),
                              ms=1000.0 / d.get("throughput_samples_per_s", 1000)
                              if d.get("throughput_samples_per_s") else 1.0,
                              params=d.get("n_params"))
            print(f"{name}: err {rows[name]['err']:.4f}, ~{rows[name]['ms']:.2f} ms/query "
                  f"(inference cost fixed, resolution-independent)")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=2)
    print("\nsaved", a.out)


if __name__ == "__main__":
    main()
