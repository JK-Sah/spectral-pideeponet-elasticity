#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
timing_linear_v3.py

Per-query cost of the linear-benchmark models, measured under a stated
protocol and checked for sensitivity to the two choices that protocol makes.

The previous pass (timing_linear.py) left two artifacts in the batched column
which cannot be architectural:

  - the plain and data-only spectral branches are the same network with the
    same parameter count, yet they timed 3x apart at M=64;
  - the plain branch's batched cost varied 5.7x across initialization seeds
    at M=16, which weights alone cannot explain.

Both are subnormal arithmetic.  Training drives most of the branch MLP's
hidden weights below the smallest normal float32 (about 1.2e-38): in several
checkpoints 70-99 per cent of the entries in a hidden layer are subnormal but
nonzero.  On the Xeon used here every multiply touching one of those values
takes a microcoded slow path, so two identical architectures differ by a
factor of three according to how far their weights happened to decay.  The
effect is a property of the hardware's denormal handling, not of the
operator, and it disappears when denormals are flushed to zero.

This script therefore measures every combination of the two protocol choices
and records them all:

  - ordering.  "blocked" times each model to completion before the next, so
    its weights stay resident in cache; this is how a surrogate is used in
    service and is the condition reported in the paper.  "interleaved" times
    the models round-robin, one repetition each per pass, so that drift over
    the run is charged to all of them equally; it is the diagnostic that
    isolated the artifact above, and it is pessimistic in absolute terms
    because cycling between models evicts each one's weights from cache.
  - denormal handling.  "as_is" leaves the hardware default; "flush_denormal"
    sets flush-to-zero, which any CPU inference deployment would do.

Common to every condition: fixed thread count set for BLAS and torch, trained
weights loaded from disk, real held-out test inputs, warm-up repetitions
discarded, the same repetition count for single-query and batched, median
reported, and each seed's value kept rather than only the spread.  The
subnormal fraction of every checkpoint is recorded alongside its timings.

    python timing_linear_v3.py --ckpt_dir runs/linear/ckpt --trunk 16 --threads 4
"""

import argparse, json, os, time
from pathlib import Path
import numpy as np


def subnormal_fraction(model):
    """Fraction of parameter entries that are subnormal but not zero."""
    import torch
    n_sub = n_tot = 0
    tiny = np.finfo(np.float32).tiny
    for p in model.parameters():
        a = p.detach().abs()
        n_sub += int(((a > 0) & (a < tiny)).sum())
        n_tot += a.numel()
    return n_sub / max(n_tot, 1)


def blocked(fns, warm, reps):
    """Time each callable to completion before moving to the next.

    This is the ordering a surrogate actually sees in service: one model
    evaluated many times, its weights resident in cache.  It is the protocol
    reported in the paper.
    """
    out = {}
    for name, fn in fns.items():
        for _ in range(warm):
            fn()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1e3)
        a = np.array(ts)
        out[name] = dict(median_ms=float(np.median(a)), min_ms=float(a.min()),
                         iqr_ms=float(np.percentile(a, 75) - np.percentile(a, 25)))
    return out


def interleaved(fns, warm, reps):
    """Time callables in round-robin passes, one repetition each per pass.

    Any drift over the run is charged to every model equally, which is what
    makes this the diagnostic ordering.  It is pessimistic in absolute terms:
    cycling through every model between repetitions evicts each one's weights
    from cache, so each call pays a cold-cache cost that service would not.
    """
    names = list(fns)
    for name in names:
        for _ in range(warm):
            fns[name]()
    ts = {name: [] for name in names}
    for _ in range(reps):
        for name in names:
            t0 = time.perf_counter()
            fns[name]()
            ts[name].append((time.perf_counter() - t0) * 1e3)
    out = {}
    for name in names:
        a = np.array(ts[name])
        out[name] = dict(median_ms=float(np.median(a)), min_ms=float(a.min()),
                         iqr_ms=float(np.percentile(a, 75) - np.percentile(a, 25)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="runs/linear/ckpt")
    ap.add_argument("--trunk", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--order", choices=["blocked", "interleaved", "both"],
                    default="both")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default="results_revision/timing_linear_v3.json")
    a = ap.parse_args()

    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(a.threads)
    import torch
    torch.set_num_threads(a.threads)
    import cmame_extended_study as st
    from cmame_extended_study import (PhysicsConfig, build_modes,
                                      SpectralPIDeepONet, FNO2dElasticity,
                                      make_dataset)
    from canonical_linear import CFG, AnchoredSpectral, closed_form_readout, ClosedFormModel

    phys = PhysicsConfig()
    modes = build_modes(a.trunk)
    d = make_dataset(CFG["n_train"], CFG["n_test"], CFG["true_modes"],
                     CFG["coeff_scale"], phys,
                     seed_train=CFG["seed_train"], seed_test=CFG["seed_test"])
    te_f = d["test_f"]
    f1 = te_f[:1].clone()
    nb = min(a.batch, te_f.shape[0])

    # ---- build every model once, up front -----------------------------
    models, params, subn = {}, {}, {}
    W, b = closed_form_readout(modes, st.make_grid(phys.res), d["train_f"], d["train_u"])
    models["closed_form"] = ClosedFormModel(W, b, modes, phys).eval()
    params["closed_form"] = int(W.size + b.size)
    subn["closed_form"] = None

    for tag in ("pi_spectral_plain", "pi_spectral_anchored",
                "data_only_spectral", "fno"):
        for seed in a.seeds:
            ck = Path(a.ckpt_dir) / f"{tag}_M{a.trunk}_seed{seed}.pt"
            if not ck.exists():
                print(f"[warn] missing {ck}")
                continue
            blob = torch.load(ck, map_location="cpu")
            if tag == "fno":
                m = FNO2dElasticity(phys, modes=CFG["fno_modes"],
                                    width=CFG["fno_width"], n_layers=CFG["fno_layers"])
            elif tag == "pi_spectral_anchored":
                m = AnchoredSpectral(modes, phys, d["train_f"], d["train_u"],
                                     CFG["hidden"], CFG["depth"])
            else:
                m = SpectralPIDeepONet(modes, phys, hidden=CFG["hidden"],
                                       depth=CFG["depth"])
            m.load_state_dict(blob["state_dict"])
            m.eval()
            key = f"{tag}@{seed}"
            models[key] = m
            params[key] = sum(p.numel() for p in m.parameters())
            subn[key] = subnormal_fraction(m)

    print(f"[protocol] threads={a.threads} batch={nb} warm={a.warm} reps={a.reps} "
          f"M={a.trunk} models={len(models)} interleaved=yes")
    for k in models:
        s = subn[k]
        print(f"  {k:28s} params {params[k]:>8d}  subnormal "
              f"{'n/a' if s is None else f'{s:.3e}'}")

    out = {"protocol": dict(threads=a.threads, batch=nb, warm=a.warm, reps=a.reps,
                            trunk=a.trunk, weights="trained checkpoints",
                            inputs="held-out test fields",
                            order="round-robin, one repetition per model per pass",
                            statistic="median over repetitions; per-seed values recorded"),
           "subnormal_fraction": subn,
           "n_params": params}

    # ---- measure: every ordering x denormal-handling combination -------
    orders = (("blocked", blocked), ("interleaved", interleaved))
    if a.order != "both":
        orders = tuple(o for o in orders if o[0] == a.order)
    conds = []
    for oname, driver in orders:
        for dname, flush in (("as_is", False), ("flush_denormal", True)):
            cond = f"{oname}/{dname}"
            conds.append(cond)
            torch.set_flush_denormal(flush)
            with torch.no_grad():
                single = driver({k: (lambda m=m: m(f1)) for k, m in models.items()},
                               a.warm, a.reps)
                batch = driver({k: (lambda m=m: m(te_f[:nb])) for k, m in models.items()},
                               a.warm, a.reps)
            block = {}
            for k in models:
                block[k] = dict(single_query=single[k],
                                batched_per_sample={kk: vv / nb
                                                    for kk, vv in batch[k].items()})
                print(f"[{cond:26s}] {k:28s} single {single[k]['median_ms']:8.3f} ms  "
                      f"batched/sample {batch[k]['median_ms']/nb:8.4f} ms")
            out[cond] = block
    out["conditions"] = conds
    out["reported"] = "blocked/flush_denormal"

    # ---- aggregate every condition across seeds ------------------------
    agg = {}
    for tag in ("closed_form", "pi_spectral_plain", "pi_spectral_anchored",
                "data_only_spectral", "fno"):
        keys = [k for k in models if k == tag or k.startswith(tag + "@")]
        if not keys:
            continue
        for cond in conds:
            s_ = [out[cond][k]["single_query"]["median_ms"] for k in keys]
            b_ = [out[cond][k]["batched_per_sample"]["median_ms"] for k in keys]
            agg.setdefault(tag, {})[cond] = dict(
                n_seeds=len(keys), n_params=params[keys[0]],
                single_query_ms=dict(median=float(np.median(s_)),
                                     lo=float(min(s_)), hi=float(max(s_)),
                                     per_seed=[float(x) for x in s_]),
                batched_ms_per_sample=dict(median=float(np.median(b_)),
                                           lo=float(min(b_)), hi=float(max(b_)),
                                           per_seed=[float(x) for x in b_]))
    out["aggregate"] = agg

    for cond in conds:
        print(f"\n[aggregate, {cond}]")
        for tag, v in agg.items():
            if cond not in v:
                continue
            s_ = v[cond]["single_query_ms"]; b_ = v[cond]["batched_ms_per_sample"]
            print(f"  {tag:22s} single {s_['median']:7.3f} [{s_['lo']:.3f},{s_['hi']:.3f}]"
                  f"   batched {b_['median']:7.4f} [{b_['lo']:.4f},{b_['hi']:.4f}]")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)
    print("TIMINGLINEARV3_DONE")


if __name__ == "__main__":
    main()
