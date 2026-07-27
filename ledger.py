#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ledger.py

Master resource ledger (INN/Park-et-al. template): accuracy, wall-time,
peak RAM, and on-disk operator size for each CLASSICAL method, across a
mesh-resolution sweep, on both benchmarks.

Peak RAM is measured per method in ISOLATION: the driver re-invokes this
script as a subprocess for each (method,res,rank) and reads the child's
ru_maxrss, so the numbers are not contaminated by other methods.  On-disk
size is the deterministic operator footprint (LU factors for FEM; basis +
reduced operator for ROM).  Neural-operator rows are pulled from their own
eval JSONs (measured on GPU) and merged by the driver.

    python ledger.py --driver --data data/hetero_field.npz --out results_revision/ledger.json
    python ledger.py --single --bench A --method fem   --res 29
    python ledger.py --single --bench B --method rom   --res 29 --rank 64 --data ...
"""

import argparse, json, resource, subprocess, sys, time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from revision_new_benchmarks import FemSolver, lame
from rom_baseline import make_manufactured, assemble_K_free
from hetero_field import HeteroFieldQ4, nodal_to_elem


def peak_rss_mb():
    # ru_maxrss units differ: KB on Linux, bytes on macOS.
    m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return m / 1024.0 if sys.platform != "darwin" else m / 1024.0 / 1024.0


def rel_l2(pred, true):
    n = np.linalg.norm((pred-true).reshape(pred.shape[0], -1), axis=1)
    d = np.linalg.norm(true.reshape(true.shape[0], -1), axis=1) + 1e-12
    return float(np.mean(n/d))


# ---------------- single-method runners (isolated processes) --------------
def run_A(method, res, rank, ntest=200):
    """Homogeneous manufactured K=16 benchmark."""
    lam, mu = lame(nu=0.30)
    phys, tr, te = make_manufactured(1000, ntest, 16, res, 0.30)
    if method == "fem":
        t0 = time.perf_counter(); solver = FemSolver(res, lam, mu)
        factor_s = time.perf_counter()-t0
        u = np.zeros_like(te["u"]); tsolve = 0.0
        for i in range(ntest):
            t1 = time.perf_counter(); uu,_ = solver.solve(te["f"][i]); tsolve += time.perf_counter()-t1
            u[i] = uu
        err = rel_l2(u, te["u"])
        amort_ms = 1000*(factor_s/ntest + tsolve/ntest)               # amortized
        disk_mb = (solver.lu.L.nnz + solver.lu.U.nnz)*8/1e6
        return dict(err=err, time_ms=amort_ms, disk_MB=disk_mb)
    if method == "rom":
        K_free, free = assemble_K_free(res, lam, mu)
        solver = FemSolver(res, lam, mu)
        S = np.stack([solver.solve(tr["f"][i])[0].reshape(-1)[free] for i in range(1000)]).T
        U,_,_ = np.linalg.svd(S, full_matrices=False); V = U[:, :rank]
        from scipy.linalg import lu_factor, lu_solve
        Kr = V.T @ (K_free @ V); lf = lu_factor(Kr)
        u = np.zeros_like(te["u"]); t0 = time.perf_counter()
        for i in range(ntest):
            br = V.T @ solver.rhs(te["f"][i])[free]
            full = np.zeros(2*res*res); full[free] = V @ lu_solve(lf, br)
            u[i] = full.reshape(res,res,2)
        time_ms = 1000*(time.perf_counter()-t0)/ntest
        err = rel_l2(u, te["u"])
        disk_mb = (V.size + Kr.size)*8/1e6
        return dict(err=err, time_ms=time_ms, disk_MB=disk_mb)
    raise ValueError(method)


def run_B(method, res, rank, data, ntest=None):
    """Heterogeneous E(x) benchmark (res-113 GT)."""
    z = np.load(data)
    f_te, E_te, u_te = z["f_te"], z["E_te"], z["u_te"]
    N = ntest or u_te.shape[0]
    f_te, E_te, u_te = f_te[:N], E_te[:N], u_te[:N]
    solver = HeteroFieldQ4(res, nu=0.30)
    free = solver.free
    if method == "fem":
        u = np.zeros((N, res, res, 2)); t0 = time.perf_counter(); disk=0
        for i in range(N):
            uu, _ = solver.solve(f_te[i] if res==29 else _restr(f_te[i],res), nodal_to_elem(_field(E_te[i],res)))
            u[i] = uu
        time_ms = 1000*(time.perf_counter()-t0)/N
        # storage: one LU factorization (must be recomputed each query, but its size is representative)
        K = solver.assemble(nodal_to_elem(_field(E_te[0],res)))
        lu = spla.splu(K[np.ix_(free,free)])
        disk = (lu.L.nnz+lu.U.nnz)*8/1e6
        err = rel_l2(u if res==29 else u, u_te) if res==29 else None
        return dict(err=err, time_ms=time_ms, disk_MB=disk, note="per-query (no reuse)")
    if method == "rom":
        u_tr = z["u_tr"]; Ntr = u_tr.shape[0]
        Str = u_tr.reshape(Ntr,-1)[:, free].T
        U,_,_ = np.linalg.svd(Str, full_matrices=False); V = U[:, :rank]
        from scipy.linalg import lu_factor, lu_solve
        b = np.stack([solver.rhs(f_te[i])[free] for i in range(N)])
        u = np.zeros((N,res,res,2)); t0 = time.perf_counter()
        for i in range(N):
            K = solver.assemble(nodal_to_elem(E_te[i])); Kf = K[np.ix_(free,free)]
            Kr = V.T @ (Kf @ V); c = lu_solve(lu_factor(Kr), V.T @ b[i])
            full = np.zeros(2*res*res); full[free] = V @ c; u[i] = full.reshape(res,res,2)
        time_ms = 1000*(time.perf_counter()-t0)/N
        err = rel_l2(u, u_te); disk = (V.size + rank*rank)*8/1e6
        return dict(err=err, time_ms=time_ms, disk_MB=disk)
    raise ValueError(method)


def _field(E29, res):   # upsample/rescale a 29-nodal E to res (nearest); GT was res113
    if res == E29.shape[0]: return E29
    import scipy.ndimage as ndi
    return ndi.zoom(E29, res/E29.shape[0], order=1)


def _restr(f29, res):
    import scipy.ndimage as ndi
    return np.stack([ndi.zoom(f29[...,c], res/f29.shape[0], order=1) for c in range(2)], -1)


# ---------------- driver -------------------------------------------------
def single():
    ap = argparse.ArgumentParser(); ap.add_argument("--single", action="store_true")
    ap.add_argument("--bench"); ap.add_argument("--method"); ap.add_argument("--res", type=int)
    ap.add_argument("--rank", type=int, default=0); ap.add_argument("--data", default="")
    a = ap.parse_args()
    r = run_A(a.method, a.res, a.rank) if a.bench == "A" else run_B(a.method, a.res, a.rank, a.data)
    r["peak_rss_MB"] = peak_rss_mb()
    print("LEDGER_JSON " + json.dumps(r))


def driver():
    ap = argparse.ArgumentParser(); ap.add_argument("--driver", action="store_true")
    ap.add_argument("--data", required=True); ap.add_argument("--out", default="results_revision/ledger.json")
    a = ap.parse_args()
    jobs = ([("A","fem",r,0) for r in (15,29,57,113)] +
            [("A","rom",29,k) for k in (16,32,64)] +
            [("B","fem",r,0) for r in (29,57,113)] +
            [("B","rom",29,k) for k in (32,64,128)])
    rows = []
    for bench, method, res, rank in jobs:
        cmd = [sys.executable, __file__, "--single", "--bench", bench,
               "--method", method, "--res", str(res), "--rank", str(rank), "--data", a.data]
        out = subprocess.run(cmd, capture_output=True, text=True)
        line = [l for l in out.stdout.splitlines() if l.startswith("LEDGER_JSON")]
        rec = json.loads(line[0][len("LEDGER_JSON "):]) if line else {"error": out.stderr[-400:]}
        rec.update(bench=bench, method=method, res=res, rank=rank)
        rows.append(rec)
        print(f"{bench} {method} res={res} rank={rank}: {rec}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=2)
    print("saved", a.out)


if __name__ == "__main__":
    if "--driver" in sys.argv: driver()
    elif "--single" in sys.argv: single()
    else: print("use --driver or --single")
