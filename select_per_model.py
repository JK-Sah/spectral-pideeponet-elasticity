#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_per_model.py

Residual weight chosen on validation for every reported model configuration,
not only for the plain spectral branch.  A weight tuned on one model and
applied to the others handicaps the others; here each configuration gets the
weight with the lowest mean validation error over seeds 42-44, from every run
made under the same protocol (canonical protocol for the manufactured
benchmark, 4000-epoch protocol for the heterogeneous and rough ones).

Sources (all under --r1 = runs/r1 and --base = runs/linear):
  manufactured  canonical_linear.py outputs: base/canonical_M*.json (1e-4),
                w/linear/canonical_M*.json (0.03), w/sel_sweep/lin/w*/canonical_*.json;
                plain M=16 is chosen on the ablation sweep (ablation/wpde_*.json);
                capacity-matched FNO from aux/ (1e-4), w/aux/ (0.03), w/sel_sweep/fno14/w*/
  heterogeneous route_b.py outputs in hetero_wsweep_e4000/ (M=16) and w/routeb/{smooth,rough};
                hetero_anchored.py outputs in w/hetero/
An optimum at the largest weight tried, or at the smallest when that is not
zero, is reported as an edge and the script exits with status 2.

    python select_per_model.py --r1 runs/r1 --base runs/linear            # select, report
    python select_per_model.py --r1 runs/r1 --base runs/linear --assemble # and build *_sel/
"""
import argparse, glob, json, re, shutil, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SEEDS = {42, 43, 44}


def load(p):
    return json.load(open(p))


def summarize(runs):
    v = [r["val"] for r in runs]; t = [r["test"] for r in runs]
    return dict(val_mean=float(np.mean(v)), val_std=float(np.std(v)),
                test_mean=float(np.mean(t)), test_std=float(np.std(t)),
                n=len(runs), epochs=[r.get("epoch") for r in runs])


def choose(name, cands, edges):
    """cands: {w: [run, ...]} -> chosen w; only complete (3-seed) candidates count."""
    full = {w: rs for w, rs in cands.items() if {r["seed"] for r in rs} >= SEEDS}
    if not full:
        sys.exit(f"{name}: no candidate with all three seeds ({sorted(cands)})")
    stats = {w: summarize([r for r in rs if r["seed"] in SEEDS]) for w, rs in full.items()}
    w = min(stats, key=lambda k: stats[k]["val_mean"])
    grid = sorted(stats)
    if w == grid[-1] or (w == grid[0] and w > 0):
        edges.append(f"{name}: optimum {w:g} at the edge of {[f'{g:g}' for g in grid]}")
    line = "  ".join(f"{g:g}:{100*stats[g]['val_mean']:.2f}" for g in grid)
    print(f"  {name:34s} -> w={w:<7g} test {100*stats[w]['test_mean']:6.2f} +/- "
          f"{100*stats[w]['test_std']:.2f}%   val by w: {line}")
    return w, stats


# ---------------------------------------------------------------- manufactured
def linear_candidates(r1, base):
    """{(model, M): {w: [run]}} with each run's checkpoint path."""
    files = [(p, base / "ckpt") for p in sorted(base.glob("canonical_M*.json"))]
    files += [(p, r1 / "w" / "linear" / "ckpt") for p in sorted((r1 / "w" / "linear").glob("canonical_M*.json"))]
    for d in sorted((r1 / "w" / "sel_sweep" / "lin").glob("w*")):
        files += [(p, d / "ckpt") for p in sorted(d.glob("canonical_*.json"))]
    C = defaultdict(lambda: defaultdict(list))
    for p, ck in files:
        for r in load(p)["runs"]:
            if r["model"] == "data_only_spectral":
                continue
            M = 16 if r["model"] == "fno" else r["trunk"]   # the FNO has no trunk
            C[(r["model"], M)][float(r["w_pde"])].append(dict(
                seed=r["seed"], val=r["val_error"], test=r["test_error"], epoch=r["selected_epoch"],
                rec=r, src=str(p), ckpt=str(ck / f"{r['model']}_M{r['trunk']}_seed{r['seed']}.pt")))
    return C


def fno14_candidates(r1):
    C = defaultdict(list)
    for p in [r1 / "aux" / "sine_fno14.json", r1 / "w" / "aux" / "sine_fno14.json"] + \
             sorted((r1 / "w" / "sel_sweep" / "fno14").glob("w*/sine_fno14.json")):
        if p.exists():
            d = load(p)
            for r in d["runs"]:
                C[float(d["config"]["w_pde"])].append(dict(seed=r["seed"], val=r["val_error"],
                                                           test=r["disp_l2"], epoch=r["selected_epoch"],
                                                           src=str(p)))
    return C


def ablation_plain16(r1):
    C = {}
    for p in glob.glob(str(r1 / "ablation" / "wpde_*.json")):
        d = load(p)
        C[float(d["value"])] = [dict(seed=r["seed"], val=r["val_error"], test=r["test_error"],
                                     epoch=r["selected_epoch"]) for r in d["runs"]]
    return C


# ---------------------------------------------------------------- heterogeneous
def routeb_candidates(dirs):
    C = defaultdict(lambda: defaultdict(list))
    for d in dirs:
        for p in sorted(Path(d).glob("*_wpde*_seed*.json")):
            fam = p.stem.rsplit("_wpde", 1)[0]
            r = load(p)
            C[fam][float(r["w_pde"])].append(dict(seed=r["seed"], val=r["val_error"], test=r["rel_l2_u"],
                                                  epoch=r.get("selected_epoch"), src=str(p),
                                                  pt=str(p.with_suffix(".pt"))))
    return C


def anchored_candidates(d):
    C = defaultdict(lambda: defaultdict(list))
    for p in sorted(Path(d).glob("anchored*.json")):
        x = load(p); w = float(x["config"]["w_pde"])
        for r in x["runs"]:
            C["hetero_" + r["model"]][w].append(dict(seed=r["seed"], val=r["val_error"],
                                                     test=r["test_error"], epoch=r["selected_epoch"],
                                                     rec=r, src=str(p)))
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", default="runs/r1")
    ap.add_argument("--base", default="runs/linear")
    ap.add_argument("--assemble", action="store_true")
    a = ap.parse_args()
    r1, base, W = Path(a.r1), Path(a.base), Path(a.r1) / "w"
    edges, sel = [], {}

    print("manufactured benchmark (canonical protocol):")
    lin = linear_candidates(r1, base)
    w_p16, st = choose("plain M16 (ablation sweep)", ablation_plain16(r1), edges)
    sel["pi_spectral_plain_M16"] = dict(w=w_p16, **st[w_p16])
    for key in sorted(lin):
        if key == ("pi_spectral_plain", 16):
            continue
        name = f"{key[0]} M{key[1]}" if key[0] != "fno" else "fno (canonical)"
        w, st = choose(name, lin[key], edges)
        sel[f"{key[0]}_M{key[1]}"] = dict(w=w, **st[w])
    w, st = choose("capacity-matched FNO (sine)", fno14_candidates(r1), edges)
    sel["fno14"] = dict(w=w, **st[w])

    print("heterogeneous benchmark (4000 epochs):")
    sm = routeb_candidates([W / "routeb" / "smooth"])
    sw = routeb_candidates([r1 / "hetero_wsweep_e4000"])
    w, st = choose("spectral_e M16 (smooth, sweep)", sw["spectral_e_M16"], edges)
    sel["het_spectral_e_M16"] = dict(w=w, **st[w])
    for fam in sorted(sm):
        if fam == "spectral_e_M16":
            continue
        w, st = choose(f"{fam} (smooth)", {k: v for k, v in sm[fam].items()}, edges)
        sel[f"het_{fam}"] = dict(w=w, **st[w])
    ro = routeb_candidates([W / "routeb" / "rough"])
    for fam in sorted(ro):
        w, st = choose(f"{fam} (rough)", ro[fam], edges)
        sel[f"rough_{fam}"] = dict(w=w, **st[w])
    an = anchored_candidates(W / "hetero")
    for fam in sorted(an):
        w, st = choose(f"{fam} (anchoring test)", an[fam], edges)
        sel[fam] = dict(w=w, **st[w])

    out = dict(rule="lowest mean validation error over seeds 42-44, per model configuration",
               selected=sel, edges=edges)
    (W / "selected_per_model.json").write_text(json.dumps(out, indent=2))
    if edges:
        print("EDGE:\n  " + "\n  ".join(edges))
        sys.exit(2)
    print("SELECTION_OK")
    if not a.assemble:
        return

    # ---------------- manufactured: checkpoints and canonical files at the chosen weights
    L = W / "linear_sel"; (L / "ckpt").mkdir(parents=True, exist_ok=True)
    for M in (16, 32, 64):
        runs = []
        for model in ("pi_spectral_plain", "pi_spectral_anchored", "fno"):
            key = (model, 16 if model == "fno" else M)
            w = sel[f"{model}_M{key[1]}"]["w"]
            rs = [r for r in lin[key][w] if r["seed"] in SEEDS]
            seen = set()
            for r in rs:
                if r["seed"] in seen:
                    continue          # the FNO was trained once per M task; keep one copy per seed
                seen.add(r["seed"])
                shutil.copy2(r["ckpt"], L / "ckpt" / f"{model}_M{M}_seed{r['seed']}.pt")
                runs.append(dict(r["rec"], trunk=M if model != "fno" else r["rec"]["trunk"],
                                 source=r["src"]))
        b = load(base / f"canonical_M{M}.json")
        for r in b["runs"]:
            if r["model"] == "data_only_spectral":
                runs.append(dict(r, source=str(base / f"canonical_M{M}.json")))
                shutil.copy2(base / "ckpt" / f"data_only_spectral_M{M}_seed{r['seed']}.pt",
                             L / "ckpt" / f"data_only_spectral_M{M}_seed{r['seed']}.pt")
        agg = {}
        for tag in ("pi_spectral_plain", "data_only_spectral", "pi_spectral_anchored", "fno"):
            es = [r["test_error"] for r in runs if r["model"] == tag]
            agg[tag] = dict(mean=float(np.mean(es)), std=float(np.std(es)), n_seeds=len(es), values=es,
                            w_pde=float(next(r["w_pde"] for r in runs if r["model"] == tag)))
        cf = load(W / "linear" / f"canonical_M{M}.json")["closed_form"]
        (L / f"canonical_M{M}.json").write_text(json.dumps(dict(
            selection=out["rule"], runs=runs, aggregate=agg, closed_form=cf), indent=2))

    # ---------------- heterogeneous: chosen checkpoints, plus the data-only ones
    for field, cand, extra in (("smooth", dict(sm, spectral_e_M16=sw["spectral_e_M16"]),
                                ["spectral_e_M16_wpde0", "fno_e_wpde0"]),
                               ("rough", ro, [])):
        S = W / "routeb" / f"{field}_sel"
        if S.exists():
            shutil.rmtree(S)
        S.mkdir(parents=True)
        for fam, ws in cand.items():
            k = ("het_" if field == "smooth" else "rough_") + fam
            if k not in sel:
                continue
            for r in ws[sel[k]["w"]]:
                for src in (r["src"], r["pt"]):
                    if Path(src).exists():
                        shutil.copy2(src, S / Path(src).name)
        for stem in extra:
            for p in (W / "routeb" / field).glob(f"{stem}_seed*"):
                shutil.copy2(p, S / p.name)
    H = W / "hetero_sel"; H.mkdir(parents=True, exist_ok=True)
    runs = [dict(r["rec"], w_pde=sel[fam]["w"]) for fam in ("hetero_plain", "hetero_anchored")
            for r in an[fam][sel[fam]["w"]] if r["seed"] in SEEDS]
    agg = {t: dict(mean=float(np.mean([r["test_error"] for r in runs if r["model"] == t])),
                   std=float(np.std([r["test_error"] for r in runs if r["model"] == t])),
                   w_pde=sel["hetero_" + t]["w"]) for t in ("plain", "anchored")}
    (H / "anchored.json").write_text(json.dumps(dict(selection=out["rule"], runs=runs,
                                                     aggregate=agg), indent=2))
    # ---------------- auxiliary linear experiments: plain branch at its weight (w/aux, 0.03);
    # anchored and capacity-matched FNO at theirs.  Files at another weight are left out
    # here and re-run by the launcher (r1_C_aux_*.sh) into the same directory.
    A = W / "aux_sel"
    if A.exists():
        shutil.rmtree(A)
    A.mkdir(parents=True)
    w_anch, w_fno14 = sel["pi_spectral_anchored_M16"]["w"], sel["fno14"]["w"]
    for p in (W / "aux").glob("*.json"):
        d = load(p); m = d["config"]["model"]
        keep = (m == "spectral" or (m == "anchored" and d["config"]["w_pde"] == w_anch)
                or (m == "fno14" and d["config"]["w_pde"] == w_fno14))
        if keep and p.stem != "sine_fno14":
            shutil.copy2(p, A / p.name)
    shutil.copy2(fno14_candidates(r1)[w_fno14][0]["src"], A / "sine_fno14.json")
    print(f"W_ANCH={w_anch:g} W_FNO14={w_fno14:g}")
    print("ASSEMBLED")


if __name__ == "__main__":
    main()
