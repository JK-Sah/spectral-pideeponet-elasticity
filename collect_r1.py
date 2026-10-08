#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_r1.py

Pull the revision's result files from the cluster into results_revision/r1/
and print the quantities the manuscript and the response letter quote, each
computed from the files (mean +/- s.d. over seeds where there are seeds).

Layout: r1/ holds the residual-weight sweeps, the data-only and classical runs
(which do not depend on the weight) and the finite-strain study; r1/w/ holds
every physics-informed re-run at the validation-chosen weights (stage B).  The
weight-independent canonical files from runs/linear are copied to r1/linear_base.

    python collect_r1.py            # rsync, then summarize
    python collect_r1.py --no-sync  # summarize what is already local
"""
import argparse, glob, json, subprocess
from pathlib import Path

import numpy as np

R1 = Path("results_revision/r1")
REMOTE = "sporcsubmit.rc.rit.edu:/shared/rc/whiskers/JK/cmame_pideeponet/runs/r1/"
REMOTE_BASE = "sporcsubmit.rc.rit.edu:/shared/rc/whiskers/JK/cmame_pideeponet/runs/linear/"
W = R1 / "w"


def ms(x):
    return float(np.mean(x)), float(np.std(x))


def pct(m, s=None):
    return f"{100*m:.2f}%" + (f" ± {100*s:.2f}%" if s is not None else "")


def load(p):
    return json.load(open(p))


def routeb(field, root=R1):
    out = {}
    for f in sorted(glob.glob(str(root / "routeb" / field / "*.json"))):
        d = load(f)
        tag = Path(f).stem.rsplit("_seed", 1)[0]
        out.setdefault(tag, []).append(d)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-sync", action="store_true")
    a = ap.parse_args()
    if not a.no_sync:
        subprocess.run(["rsync", "-a", "--exclude", "*.pt", "--exclude", "*.npz.tmp",
                        REMOTE, str(R1) + "/"], check=False)
        subprocess.run(["rsync", "-a", "--exclude", "ckpt", REMOTE_BASE,
                        str(R1 / "linear_base") + "/"], check=False)

    p = R1 / "selected_weights.json"
    if p.exists():
        d = load(p)
        print(f"=== residual weights chosen on validation: W_LIN={d['w_lin']:g} W_HET={d['w_het']:g} ===")
        for name in ("manufactured", "heterogeneous"):
            for w, v in sorted(d[name].items(), key=lambda kv: float(kv[0])):
                if name == "manufactured":
                    print(f"  {name[:6]} w={float(w):<8g} val {pct(v[0], v[1])}  test {pct(v[2])}")
                else:
                    va = [a for a, _ in v]; te = [b for _, b in v]
                    print(f"  {name[:6]} w={float(w):<8g} val {pct(*ms(va))}  test {pct(*ms(te))}")

    for root in (R1, W):
        files = sorted(glob.glob(str(root / "ablation" / "*.json")))
        if files:
            print(f"=== ablations (canonical protocol) [{root}] ===")
        for f in files:
            d = load(f)
            print(f"  {d['kind']:7s} {d['value']:<8g} w {d['w_pde']:<7g} val {pct(d['val_mean'], d['val_std'])}  "
                  f"test {pct(d['test_mean'], d['test_std'])}  ep* {[r['selected_epoch'] for r in d['runs']]}")

    for M in (16, 32, 64):
        p = W / "linear" / f"canonical_M{M}.json"
        if not p.exists():
            continue
        d = load(p); b = R1 / "linear_base" / f"canonical_M{M}.json"
        agg = dict(d["aggregate"])
        if b.exists() and "data_only_spectral" in load(b)["aggregate"]:
            agg["data_only_spectral (w-independent)"] = load(b)["aggregate"]["data_only_spectral"]
        print(f"=== canonical M={M}, w={d['runs'][0]['w_pde']:g} ===")
        for k, v in agg.items():
            eps = [r["selected_epoch"] for r in d["runs"] if r["model"] == k]
            print(f"  {k:36s} {pct(v['mean'], v['std'])}  n={v['n_seeds']}  ep* {eps}")
        if "closed_form" in d:
            print(f"  closed_form                          {d['closed_form']['test_error']:.2e}")
    for M in (16, 64):
        p = W / "linear" / f"timing_linear_v3_M{M}.json"
        if p.exists():
            d = load(p); rep = d["reported"]
            print(f"=== linear timing M={M} ({rep}) ===")
            for k, v in d["aggregate"].items():
                v = v[rep]
                print(f"  {k:22s} single {v['single_query_ms']['median']:.3f} ms  "
                      f"batched {v['batched_ms_per_sample']['median']:.4f} ms  params {v['n_params']}")

    print("=== auxiliary linear experiments (data-only from r1/aux, physics-informed from r1/w/aux) ===")
    aux = {Path(f).name: f for f in glob.glob(str(R1 / "aux" / "*.json"))}
    aux.update({Path(f).name: f for f in glob.glob(str(W / "aux" / "*.json"))})
    for name in sorted(aux):
        d = load(aux[name])
        if "aggregate" not in d:
            continue
        g = d["aggregate"]
        line = (f"  {d['exp']:24s} disp {pct(g['disp_l2']['mean'], g['disp_l2']['std'])}  "
                f"strain {pct(g['strain_l2']['mean'])} stress {pct(g['stress_l2']['mean'])}  "
                f"energy {pct(g['energy_err']['mean'])} res {g['pde_mse']['mean']:.3g}")
        if "ood" in g:
            line += "  ood " + " ".join(f"K{k}:{100*v['mean']:.1f}" for k, v in g["ood"].items())
        print(line)
    for kind in ("bumps", "patch"):
        p = R1 / "aux" / f"forcing_{kind}_classical.json"
        if p.exists():
            d = load(p)
            print(f"  classical {kind}: fem {d['fem_error']}  rom {d['rom_error_by_rank']}  "
                  f"closed-form {d['closed_form_error']:.4f}  floor {d['representation_floor']:.4f}")

    for field in ("smooth", "rough"):
        rb = routeb(field)
        rb_w = routeb(field, W)
        if rb_w:   # stage B replaces the physics-informed runs; data-only runs stay
            rb = {t: r for t, r in rb.items() if "_wpde0_" in t + "_"} | rb_w
        if rb:
            print(f"=== heterogeneous ({field}) ===")
            for tag, runs in rb.items():
                e = [r["rel_l2_u"] for r in runs]; m, s = ms(e)
                print(f"  {tag:34s} n={len(e)} disp {pct(m, s)}  stress {pct(np.mean([r['rel_l2_stress'] for r in runs]))}"
                      f"  ep* {[r.get('selected_epoch') for r in runs]}  "
                      f"optimism {np.mean([r['rel_l2_u'] - (r.get('test_error_best_epoch_optimistic') or r['rel_l2_u']) for r in runs]):+.4f}")
        tp = (W if rb_w else R1) / "routeb" / f"timing_{field}.json"
        if tp.exists():
            t = load(tp)
            print(f"  timing: fem {t['fem_per_query_ms']:.3f} ms; rom "
                  + ", ".join(f"r{r['rank']}:{r['online_ms']:.3f}" for r in t["rom"])
                  + f"; svd {t.get('rom_svd_ms', float('nan')):.1f} ms")
            for k, v in t["neural"].items():
                print(f"    {k:34s} single {v['single_query_ms']:.3f} ms  batched {v['batched_ms_per_sample']:.4f} ms  params {v['n_params']}")
        rp = R1 / "refine" / f"{field}.json"
        if rp.exists():
            r = load(rp)
            print(f"  FEM refinement ({field}): {r['errors']}  regen diff {r['field_regeneration_max_abs_diff']:.1e}")

    p = W / "hetero" / "anchored.json"
    if p.exists():
        print("=== heterogeneous anchoring (stage B) ===")
        print("  " + json.dumps(load(p).get("aggregate", load(p)))[:800])

    p = R1 / "nonlinear" / "classical_full.json"
    if p.exists():
        d = load(p); fe = d["fem"]
        print("=== finite strain, classical, full test set ===")
        print(f"  newton: load steps used {fe['load_steps_used']}; 6-step per-step mean "
              f"{np.round(fe['iters_per_step_mean_6step'], 2).tolist()} "
              f"min {fe['iters_per_step_min_6step']} max {fe['iters_per_step_max_6step']}; total "
              f"{fe['total_iters_mean']:.1f} [{fe['total_iters_min']}, {fe['total_iters_max']}]; "
              f"failed {fe['n_failed']}; max diff to stored {fe['rel_diff_to_stored_reference_max']:.1e}")
        g = d["galerkin"]
        print(f"  galerkin r32: val {g['val']['err_mean']:.4f} test {g['test']['err_mean']:.4f} "
              f"(div {g['test']['n_diverged']})")
        for r in d["deim"]:
            v, t = r["val"], r["test"]
            fmt = lambda x: "--" if x is None else f"{100*x:.2f}%"
            print(f"  deim m={r['m']}: elems {r['sample_elems']}/{r['n_elem']} ({100*r['sample_elem_frac']:.1f}%) "
                  f"free dofs {r['sample_free_dofs']} ({100*r['sample_free_dof_frac']:.1f}%) "
                  f"val {fmt(v['err_mean'])} div {v['n_diverged']}/{v['n_cases']}  "
                  f"test {fmt(t['err_mean'])} div {t['n_diverged']}/{t['n_cases']}")
        print(f"  selected m = {d['deim_selected_m']}")
    p = R1 / "nonlinear" / "timing_v3.json"
    if p.exists():
        d = load(p)
        print("=== finite strain, exclusive timing ===")
        print(f"  newton {d['newton_fem']['median_ms']:.1f} ms  breakdown "
              f"{ {k: round(100*v, 1) for k, v in d['newton_fem']['breakdown_frac'].items()} }")
        print(f"  galerkin {d['pod_galerkin_r32']['median_ms']:.1f} ms; deim "
              + ", ".join(f"m{m}:{v['median_ms']:.1f}" for m, v in d["pod_deim_r32"].items()))
        for k, v in d["neural"].items():
            print(f"  {k:18s} single {v['single_query_ms']:.3f} ms  batched {v['batched_ms_per_sample']:.4f}")
        print(f"  equivalence {d['equivalence']}")
        print(f"  skfem {d['skfem_crosscheck']}")
    p = R1 / "linear" / "canonical_extras.json"
    if p.exists():
        d = load(p)
        print("=== canonical extras ===")
        for key in ("noise", "ood"):
            for tag, v in d[key].items():
                print(f"  {key:5s} {tag:22s} " + " ".join(f"{100*x:.2f}" for x in v["mean"]))


if __name__ == "__main__":
    main()
