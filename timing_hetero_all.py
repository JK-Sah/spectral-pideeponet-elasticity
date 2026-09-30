#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
timing_hetero_all.py

Per-query cost of every method on a varying-operator benchmark -- classical and
learned, smooth field or rough -- on one allocation under the protocol used for
the fixed-operator benchmark.

The crossover argument rests on a cost comparison: the reduced model's
projection cost is said to cliff past the finite-element solve once the modulus
field is rough enough to need a high rank, and the neural operator is said to
take the frontier there.  Both sides of that comparison were previously timed
inside the scripts that produced them, at whatever thread count and contention
those jobs happened to have, and the figure caption nonetheless claims all
timings are on the same CPU.

Re-measuring the smooth field under control moved the rank-256 reduced model
from 69 ms to 7.2 ms, a factor of 9.5, which is larger than the effect the
argument turns on.  The rough-field numbers come from the same uncontrolled
setting and need the same treatment before anything is concluded from them.

Everything here is measured in one job: fixed thread count set for BLAS and
torch, denormals flushed, trained weights loaded from disk, real held-out
inputs, warm-up repetitions discarded, median over timed repetitions.

    python timing_hetero_all.py --data data_rough/hetero_field.npz \
        --ckpt_dir runs/rough --ranks 16 32 64 128 256 384 512
"""

import argparse, json, os, time
from argparse import Namespace
from pathlib import Path
import numpy as np


def subnormal_fraction(model):
    """Fraction of parameter entries that are subnormal but not zero.

    Training under weight decay drives many weights below the smallest normal
    float32, and on this hardware arithmetic touching them takes a slow path.
    Denormals are flushed for every measurement here, but the fraction is
    recorded so that a cost difference between architecturally identical models
    can be checked against it rather than guessed at.
    """
    tiny = float(np.finfo(np.float32).tiny)
    n_sub = n_tot = 0
    for prm in model.parameters():
        a = prm.detach().abs()
        n_sub += int(((a > 0) & (a < tiny)).sum())
        n_tot += a.numel()
    return n_sub / max(n_tot, 1)


def timed(fn, warm, reps):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    a = np.array(ts)
    return dict(median_ms=float(np.median(a)), min_ms=float(a.min()),
                iqr_ms=float(np.percentile(a, 75) - np.percentile(a, 25)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/hetero_field.npz")
    ap.add_argument("--ckpt_dir", default="runs/rough")
    ap.add_argument("--ranks", type=int, nargs="+",
                    default=[16, 32, 64, 128, 256, 384, 512])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--nu", type=float, default=0.30)
    ap.add_argument("--out", default="results_revision/hetero/timing_rough.json")
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(a.threads)

    import torch
    torch.set_num_threads(a.threads)
    torch.set_flush_denormal(True)
    from scipy.linalg import lu_factor, lu_solve
    from hetero_field import HeteroFieldQ4, nodal_to_elem
    from cmame_extended_study import PhysicsConfig
    from route_b import build_model

    z = np.load(a.data)
    E_te = z["E_te"].astype(np.float64)
    f_te = z["f_te"].astype(np.float64)
    u_tr = z["u_tr"].astype(np.float64)
    res = f_te.shape[1]
    hf = HeteroFieldQ4(res, nu=a.nu)
    free = hf.free
    Ee0 = nodal_to_elem(E_te[0])
    b0 = hf.rhs(f_te[0])[free]

    out = {"protocol": dict(threads=a.threads, warm=a.warm, reps=a.reps,
                            batch=a.batch, flush_denormal=True, res=res,
                            data=str(a.data),
                            note="one allocation; same protocol as the "
                                 "fixed-operator benchmark"),
           "n_test": int(len(E_te))}
    print(f"[protocol] threads={a.threads} warm={a.warm} reps={a.reps} "
          f"res={res} data={a.data}")

    # ---- finite element: the operator changes, so no factorization reuse ----
    fem = timed(lambda: hf.solve(f_te[0], Ee0), a.warm, a.reps)
    out["fem_per_query_ms"] = fem["median_ms"]
    print(f"[fem   ] per-query {fem['median_ms']:8.3f} ms  "
          f"(re-assemble + re-factorize + solve)")

    # ---- reduced model: assemble, project, solve, per query ----------------
    snaps = u_tr.reshape(u_tr.shape[0], -1)[:, free]
    t0 = time.perf_counter()
    U, sv, _ = np.linalg.svd(snaps.T, full_matrices=False)
    svd_ms = 1000 * (time.perf_counter() - t0)
    print(f"[rom   ] basis from {snaps.shape[0]} training snapshots, "
          f"SVD {svd_ms:.0f} ms")
    rom_rows = []
    for r in a.ranks:
        if r > U.shape[1] or r > snaps.shape[0]:
            print(f"[rom r={r:4d}] skipped: only {min(U.shape[1], snaps.shape[0])} available")
            continue
        Vr = U[:, :r]

        def one_query(Vr=Vr):
            K = hf.assemble(Ee0)
            Kf = K[np.ix_(free, free)]
            Kr = Vr.T @ (Kf @ Vr)
            return Vr @ lu_solve(lu_factor(Kr), Vr.T @ b0)

        t = timed(one_query, a.warm, a.reps)
        rom_rows.append(dict(rank=r, online_ms=t["median_ms"]))
        flag = "  <-- past the FEM solve" if t["median_ms"] > fem["median_ms"] else ""
        print(f"[rom r={r:4d}] online {t['median_ms']:8.3f} ms{flag}")
    out["rom"] = rom_rows
    out["rom_svd_ms"] = svd_ms

    # ---- learned operators, from their saved checkpoints -------------------
    phys = PhysicsConfig(res=res, nu=a.nu)
    f_t = torch.from_numpy(f_te).float()
    E_t = torch.from_numpy(E_te).float()
    nb = min(a.batch, f_t.shape[0])
    neural = {}
    for ck in sorted(Path(a.ckpt_dir).glob("*.pt")):
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        margs = Namespace(**blob["args"])
        m = build_model(margs.model, phys, margs)
        m.load_state_dict(blob["state"])
        m.eval()
        npar = sum(p.numel() for p in m.parameters())
        with torch.no_grad():
            s = timed(lambda: m(f_t[:1], E_t[:1]), a.warm, a.reps)
            b = timed(lambda: m(f_t[:nb], E_t[:nb]), a.warm, a.reps)
        err = blob.get("final", {}).get("rel_l2_u")
        # Key by checkpoint, not by model name: a run directory may hold both
        # the physics-informed and the data-only variant of one architecture,
        # and keying by model silently kept whichever was globbed last.  Their
        # cost is the same -- identical architecture and parameter count, and
        # denormals are flushed -- but their accuracy is not, and pairing one
        # variant's error with the other's timing is the kind of mismatch this
        # script exists to remove.
        neural[ck.stem] = dict(model=margs.model,
                               subnormal_fraction=subnormal_fraction(m),
                               w_pde=getattr(margs, "w_pde", None),
                               single_query_ms=s["median_ms"],
                               batched_ms_per_sample=b["median_ms"] / nb,
                               n_params=int(npar), rel_l2_u=err,
                               checkpoint=ck.name)
        print(f"[{ck.stem:34s}] single {s['median_ms']:8.3f} ms  "
              f"batched/sample {b['median_ms']/nb:8.4f} ms  "
              f"params {npar}  subnormal {neural[ck.stem]['subnormal_fraction']:.3e}  "
              f"err {err}")
    out["neural"] = neural

    # ---- what the crossover argument actually needs ------------------------
    print("\n[frontier] per-query cost against the finite-element solve "
          f"({fem['median_ms']:.2f} ms):")
    for r in rom_rows:
        print(f"  ROM r={r['rank']:4d}  {r['online_ms']:8.3f} ms  "
              f"= {r['online_ms']/fem['median_ms']:5.2f} x FEM")
    for k, v in neural.items():
        print(f"  {k:34s} {v['single_query_ms']:8.3f} ms  "
              f"= {v['single_query_ms']/fem['median_ms']:5.2f} x FEM"
              + (f"   err {v['rel_l2_u']:.4f}" if v.get("rel_l2_u") else ""))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("\nwrote", a.out)
    print("TIMINGHETEROALL_DONE")


if __name__ == "__main__":
    main()
