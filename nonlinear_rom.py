#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nonlinear_rom.py

Non-intrusive classical reduced-order model for the finite-strain benchmark:
project the displacement onto a POD basis, and learn the (nonlinear) map from
the body-force sine features to the POD coefficients with a low-degree
polynomial ridge regression.  This is the nonlinear analogue of the linear
paper's closed-form least-squares read-out -- a cheap, interpretable classical
reduction with no neural network -- and it is the baseline the neural operator
must beat on this problem.

    python nonlinear_rom.py --data data/nonlinear.npz --ranks 16 32 64 --degrees 1 2 3
"""

import argparse
import json
import math
import time
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

from hetero_field import HeteroFieldQ4          # reuse its res-29 free-dof indexing
from cmame_extended_study import build_modes


def sine_features(f, K=16):
    """Project body-force field f[N,res,res,2] onto K sine modes -> [N,2K]."""
    N, res = f.shape[0], f.shape[1]
    x = np.linspace(0, 1, res)
    yy, xx = np.meshgrid(x, x, indexing="ij")
    modes = build_modes(K)
    Phi = np.stack([np.sin(p*math.pi*xx)*np.sin(q*math.pi*yy) for (p, q) in modes], 0)
    w = np.ones(res); w[0] = w[-1] = 0.5
    W = (w[:, None]*w[None, :]) * (1.0/(res-1))**2
    cx = 4.0*np.einsum("ij,nij,kij->nk", W, f[..., 0], Phi)
    cy = 4.0*np.einsum("ij,nij,kij->nk", W, f[..., 1], Phi)
    return np.concatenate([cx, cy], axis=1)


def poly_features(X, degree):
    """[1, x_i, x_i x_j, ...] up to given degree (numpy, no sklearn)."""
    N, d = X.shape
    cols = [np.ones((N, 1))]
    for deg in range(1, degree + 1):
        for idx in combinations_with_replacement(range(d), deg):
            cols.append(np.prod(X[:, idx], axis=1, keepdims=True))
    return np.concatenate(cols, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/nonlinear.npz")
    ap.add_argument("--ranks", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--degrees", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--out", default="results_revision/nonlinear_rom.json")
    a = ap.parse_args()

    z = np.load(a.data)
    ftr, utr, fte, ute = z["f_tr"], z["u_tr"], z["f_te"], z["u_te"]
    res = utr.shape[1]
    free = HeteroFieldQ4(res).free

    Xtr = sine_features(ftr); Xte = sine_features(fte)          # [N,32]
    Utr = utr.reshape(utr.shape[0], -1)[:, free]                # [N,nfree]
    Ute = ute.reshape(ute.shape[0], -1)[:, free]
    # standardize features (helps polynomial conditioning)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    Xtr = (Xtr - mu)/sd; Xte = (Xte - mu)/sd

    U, sv, _ = np.linalg.svd(Utr.T, full_matrices=False)

    def rel(pred, true):
        n = np.linalg.norm(pred-true, axis=1); d = np.linalg.norm(true, axis=1)+1e-12
        return float(np.mean(n/d))

    print(f"{'rank':>5} {'degree':>7} {'n_feat':>7} {'displ_err':>10} {'online_us':>10} {'disk_MB':>8}")
    rows = []
    for r in a.ranks:
        V = U[:, :r]
        Ctr = Utr @ V                                          # POD coeffs [N,r]
        for deg in a.degrees:
            Ptr = poly_features(Xtr, deg); Pte = poly_features(Xte, deg)
            nf = Ptr.shape[1]
            W = np.linalg.solve(Ptr.T@Ptr + a.alpha*np.eye(nf), Ptr.T@Ctr)   # ridge
            t0 = time.perf_counter()
            Cte = Pte @ W
            upred = Cte @ V.T
            online_us = 1e6*(time.perf_counter()-t0)/Xte.shape[0]
            err = rel(upred, Ute)
            disk = (V.size + W.size)*8/1e6
            print(f"{r:>5} {deg:>7} {nf:>7} {err:>10.4f} {online_us:>10.1f} {disk:>8.3f}")
            rows.append(dict(rank=r, degree=deg, n_feat=nf, err=err,
                             online_us=online_us, disk_MB=disk))
    print("\nPOD floor (best rank-r linear-subspace error):")
    for r in a.ranks:
        V = U[:, :r]; proj = Ute@V@V.T
        print(f"  rank {r}: {rel(proj, Ute):.4f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(rows=rows, singular_values=sv[:64].tolist()), open(a.out, "w"), indent=2)
    print("saved", a.out)


if __name__ == "__main__":
    main()
