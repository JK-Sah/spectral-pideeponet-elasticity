#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nonlinear_classical_full.py

Accuracy of the finite-strain classical baselines on the full held-out test set
(397 cases, the same cases the neural operators are scored on), with the
hyper-reduction sample size chosen on validation cases only.

  * split: the training pool is divided exactly as nonlinear_train_v2.py does
    for seed 111 (20% validation).  POD and DEIM are built from the training
    part only, so the validation cases are genuinely held out;
  * Newton-FEM: iteration statistics per load step over all test cases;
  * POD-Galerkin (r=32) and POD-DEIM (r=32, m in 64..256) on validation and
    test; m is chosen as the value with no validation divergences and the
    lowest mean validation error, before the test set is looked at;
  * sample-mesh size reported as elements retained and FREE degrees of freedom
    touched, both over the right denominators.

Timing is not measured here; see timing_nonlinear_v3.py (exclusive allocation).

    python nonlinear_classical_full.py --data data/nonlinear.npz --out results_revision/nonlinear/classical_full.json
"""
import argparse, json, time
from pathlib import Path

import numpy as np

from nonlinear_fem_fast import FastNonlinearFEM
from pod_deim import build_pod, collect_trajectory_forces, reduced_galerkin_solve
from pod_deim_v2 import build_rom_v2, reduced_deim_solve_v2

# Reference Newton-FEM: the protocol the dataset was generated with -- six load
# increments, doubled to 12 and then 24 if Newton fails, absolute tolerance 1e-9,
# at most 50 iterations per increment.  Reduced models: tolerance 1e-8 on the
# reduced residual, at most 60 iterations per increment, same retry rule.
STEPS = (6, 12, 24)
FEM_TOL, FEM_MAXIT = 1e-9, 50
N_STEPS, TOL, MAXIT = 6, 1e-8, 60


def adaptive(solve_once):
    """Try 6, 12, 24 load increments; return (u, info, steps) or (None, info, None)."""
    info = {}
    for ns in STEPS:
        try:
            u, info = solve_once(ns)
        except Exception as e:          # singular tangent after an inverted element
            u, info = None, {"converged": False, "error": type(e).__name__}
        if u is not None and info.get("converged", True) and np.all(np.isfinite(u)):
            return u, info, ns
    return None, info, None


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


def summary(errs, its, n):
    ok = [e for e in errs if e is not None]
    return dict(n_cases=n, n_converged=len(ok), n_diverged=n - len(ok),
                err_mean=float(np.mean(ok)) if ok else None,
                err_std=float(np.std(ok)) if ok else None,
                err_median=float(np.median(ok)) if ok else None,
                newton_mean=float(np.mean(its)) if its else None)


def run_reduced(solver, cases, f, u_ref):
    """solver(f, n_steps) -> (u or None, info); retried with 12 and 24 increments."""
    errs, its, steps = [], [], []
    for i in cases:
        fi = f[i].astype(np.float64)
        u, info, ns = adaptive(lambda n: solver(fi, n))
        if u is None:
            errs.append(None); continue
        errs.append(rel(u, u_ref[i].astype(np.float64)))
        its.append(info["newton"]); steps.append(ns)
    run_reduced.last_steps = {str(k): steps.count(k) for k in STEPS}
    return errs, its


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/nonlinear.npz")
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--m", type=int, nargs="+", default=[64, 128, 192, 256])
    ap.add_argument("--n_traj", type=int, default=40)
    ap.add_argument("--split_seed", type=int, default=111)
    ap.add_argument("--val_frac", type=float, default=0.20)
    ap.add_argument("--n_val_eval", type=int, default=0, help="0 = all validation cases")
    ap.add_argument("--n_test_eval", type=int, default=0, help="0 = all test cases")
    ap.add_argument("--out", default="results_revision/nonlinear/classical_full.json")
    a = ap.parse_args()
    t_start = time.time()

    z = np.load(a.data)
    f_all, u_all, f_te, u_te = z["f_tr"], z["u_tr"], z["f_te"], z["u_te"]
    perm = np.random.default_rng(a.split_seed).permutation(f_all.shape[0])
    n_val = int(round(a.val_frac * f_all.shape[0]))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    val_cases = val_idx[:a.n_val_eval] if a.n_val_eval else val_idx
    test_cases = np.arange(a.n_test_eval or f_te.shape[0])
    fem = FastNonlinearFEM(f_all.shape[1]); free = fem.free
    print(f"split: train={len(tr_idx)} val={len(val_idx)} (evaluated {len(val_cases)}) "
          f"test={len(test_cases)}; mesh {fem.res-1}x{fem.res-1}, {fem.n_elem} elements, "
          f"{free.size} free dofs")

    # ---- Newton-FEM over the test set: iteration statistics ----------------
    per_step6, totals, steps_used, n_fail, d_ref, hard = [], [], [], 0, [], []
    for i in test_cases:
        f = f_te[i].astype(np.float64)
        u, info, ns = adaptive(lambda n: fem.solve(f, n_steps=n, tol=FEM_TOL, maxit=FEM_MAXIT,
                                                   collect_stats=True))
        if u is None:
            n_fail += 1; continue
        steps_used.append(ns); totals.append(info["newton_iters"])
        if ns == 6:
            per_step6.append([s["newton_iters"] for s in info["per_step"]])
        else:
            hard.append(dict(case=int(i), load_steps=ns, newton_iters=int(info["newton_iters"])))
        d_ref.append(rel(u, u_te[i].astype(np.float64)))
    per_step6 = np.array(per_step6); tot = np.array(totals)
    fem_stats = dict(n_cases=len(test_cases), n_failed=n_fail,
                     load_steps_used={str(k): int(steps_used.count(k)) for k in STEPS},
                     cases_needing_more_steps=hard,
                     iters_per_step_mean_6step=per_step6.mean(0).tolist(),
                     iters_per_step_min_6step=per_step6.min(0).tolist(),
                     iters_per_step_max_6step=per_step6.max(0).tolist(),
                     total_iters_mean=float(tot.mean()), total_iters_min=int(tot.min()),
                     total_iters_max=int(tot.max()),
                     rel_diff_to_stored_reference_max=float(np.max(d_ref)),
                     settings=dict(load_steps=list(STEPS), abs_tol=FEM_TOL,
                                   max_iter_per_step=FEM_MAXIT))
    print(f"[fem] load steps used {fem_stats['load_steps_used']}; 6-step iterations/step mean "
          f"{np.round(per_step6.mean(0), 2).tolist()} [{per_step6.min()}, {per_step6.max()}]; "
          f"total {tot.mean():.1f} [{tot.min()}, {tot.max()}]; failed {n_fail}; "
          f"max diff to stored reference {np.max(d_ref):.2e}  ({(time.time()-t_start)/60:.1f} min)")

    # ---- offline reduced models from the training part only ----------------
    S = np.asarray(u_all[tr_idx].reshape(len(tr_idx), -1)[:, free], dtype=np.float64)
    V, sv = build_pod(S, a.r); V = np.asarray(V, dtype=np.float64)
    Fs = collect_trajectory_forces(fem, V, [f_all[i].astype(np.float64)
                                            for i in tr_idx[:a.n_traj]], n_steps=N_STEPS)
    print(f"[offline] POD r={a.r} from {S.shape[0]} snapshots; DEIM force snapshots {Fs.shape}")

    gal = lambda f, n: reduced_galerkin_solve(fem, f, V, n_steps=n, tol=TOL, maxit=MAXIT)
    out = dict(split=dict(seed=a.split_seed, n_train=int(len(tr_idx)), n_val=int(len(val_idx)),
                          n_val_evaluated=int(len(val_cases)), n_test=int(len(test_cases))),
               mesh=dict(res=int(fem.res), n_elem=int(fem.n_elem), n_free=int(free.size)),
               fem=fem_stats)
    ev, iv = run_reduced(gal, val_cases, f_all, u_all)
    et, it = run_reduced(gal, test_cases, f_te, u_te)
    out["galerkin"] = dict(r=a.r, val=summary(ev, iv, len(val_cases)),
                           test=summary(et, it, len(test_cases)),
                           test_load_steps_used=run_reduced.last_steps)
    print(f"[galerkin] val {out['galerkin']['val']['err_mean']:.4f}  "
          f"test {out['galerkin']['test']['err_mean']:.4f}")

    deim = []
    for m in a.m:
        rom = build_rom_v2(fem, S, a.r, m, Fs)
        free_touched = np.intersect1d(rom["sdofs"], free).size
        solver = lambda f, n, rom=rom: reduced_deim_solve_v2(fem, f, rom, n_steps=n)
        ev, iv = run_reduced(solver, val_cases, f_all, u_all)
        rec = dict(m=m, sample_elems=rom["n_sample_elems"], n_elem=rom["n_elem_total"],
                   sample_elem_frac=rom["sample_frac"],
                   sample_dofs_all=rom["n_sample_dofs"], sample_free_dofs=int(free_touched),
                   sample_free_dof_frac=free_touched / float(free.size),
                   val=summary(ev, iv, len(val_cases)))
        deim.append((rec, solver))
        v = rec["val"]
        print(f"[deim m={m}] elems {rec['sample_elems']}/{rec['n_elem']} "
              f"({100*rec['sample_elem_frac']:.1f}%), free dofs {free_touched}/{free.size} "
              f"({100*rec['sample_free_dof_frac']:.1f}%); val err "
              f"{v['err_mean'] if v['err_mean'] is None else round(v['err_mean'], 4)} "
              f"diverged {v['n_diverged']}/{v['n_cases']}")

    # ---- choose m on validation only, then score every m on test ------------
    ranked = sorted(deim, key=lambda d: (d[0]["val"]["n_diverged"],
                                         d[0]["val"]["err_mean"] if d[0]["val"]["err_mean"]
                                         is not None else 9e9))
    m_star = ranked[0][0]["m"]
    for rec, solver in deim:
        et, it = run_reduced(solver, test_cases, f_te, u_te)
        rec["test"] = summary(et, it, len(test_cases))
        rec["test_load_steps_used"] = run_reduced.last_steps
        t = rec["test"]
        print(f"[deim m={rec['m']}] test err "
              f"{t['err_mean'] if t['err_mean'] is None else round(t['err_mean'], 4)} "
              f"diverged {t['n_diverged']}/{t['n_cases']}" + ("   <- selected" if rec["m"] == m_star else ""))
    out["deim"] = [d[0] for d in deim]
    out["deim_selected_m"] = m_star
    out["deim_selection_rule"] = ("fewest validation divergences, then lowest mean validation "
                                  "error; chosen before the test set was evaluated")
    out["minutes"] = (time.time() - t_start) / 60
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out); print("NLCLASSICAL_DONE")


if __name__ == "__main__":
    main()
