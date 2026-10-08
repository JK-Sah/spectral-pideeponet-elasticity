#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_forcing.py

Residual weight for each learned model on each non-sine forcing family, chosen
on that family's own validation set (lowest mean validation displacement error
over seeds 42-44), as on the heterogeneous benchmark.  A weight chosen on the
sine family and carried over handicaps a model whose optimum differs: the
anchored branch on Gaussian bumps reaches 6.4% at 1e-4 and 12.0% at 0.03.

Candidates, all aux_linear_v2.py outputs under the canonical protocol:
  r1/aux/<fam>_<model>.json            1e-4 (and <fam>_data.json = plain at w = 0)
  r1/w/aux/<fam>_<model>.json          0.03
  r1/w/aux_sel/<fam>_fno14.json        1e-4 (capacity-matched FNO, stage C)
  r1/w/sel_sweep/forcing/w*/<fam>_<model>.json
The selected file of each (family, model) is copied into r1/w/aux_sel, which is
what the manuscript reads.  Exits with status 2 if an optimum is at the largest
weight tried, or at the smallest when that is not zero.

    python select_forcing.py --r1 runs/r1 [--assemble]
"""
import argparse, glob, json, shutil, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

FAMILIES = ("bumps", "patch")
MODELS = ("pi", "anchored", "fno14")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", default="runs/r1")
    ap.add_argument("--assemble", action="store_true")
    a = ap.parse_args()
    r1 = Path(a.r1); W = r1 / "w"
    out, edges = {}, []
    for fam in FAMILIES:
        for model in MODELS:
            files = [r1 / "aux" / f"{fam}_{model}.json", W / "aux" / f"{fam}_{model}.json",
                     W / "aux_sel" / f"{fam}_{model}.json"]
            files += [Path(p) for p in glob.glob(str(W / "sel_sweep" / "forcing" / "w*" / f"{fam}_{model}.json"))]
            if model == "pi":
                files.append(r1 / "aux" / f"{fam}_data.json")     # the plain branch at w = 0
            cand = defaultdict(list)
            for p in files:
                if not p.exists():
                    continue
                d = json.load(open(p))
                w = float(d["config"]["w_pde"])
                if any(str(p) == c[0] for c in cand[w]):
                    continue
                v = [r["val_error"] for r in d["runs"]]; t = [r["disp_l2"] for r in d["runs"]]
                if len(v) < 3:
                    continue
                cand[w].append((str(p), float(np.mean(v)), float(np.mean(t)), float(np.std(t))))
            stats = {w: min(c, key=lambda x: x[1]) for w, c in cand.items()}   # duplicates: identical runs
            w = min(stats, key=lambda k: stats[k][1]); grid = sorted(stats)
            if w == grid[-1] or (w == grid[0] and w > 0):
                edges.append(f"{fam}/{model}: optimum {w:g} at the edge of {[f'{g:g}' for g in grid]}")
            print(f"  {fam:6s} {model:9s} -> w={w:<7g} test {100*stats[w][2]:6.2f} +/- {100*stats[w][3]:.2f}%   "
                  + "val by w: " + "  ".join(f"{g:g}:{100*stats[g][1]:.2f}" for g in grid))
            out[f"{fam}_{model}"] = dict(w=w, src=stats[w][0], val_mean=stats[w][1],
                                         test_mean=stats[w][2], test_std=stats[w][3],
                                         grid={str(g): stats[g][1] for g in grid})
    (W / "forcing_selected.json").write_text(json.dumps(dict(
        rule="lowest mean validation error over seeds 42-44, per model and forcing family",
        selected=out, edges=edges), indent=2))
    if edges:
        print("EDGE:\n  " + "\n  ".join(edges)); sys.exit(2)
    print("SELECTION_OK")
    if a.assemble:
        for k, v in out.items():
            shutil.copy2(v["src"], W / "aux_sel" / f"{k}.json")
        print("ASSEMBLED into", W / "aux_sel")


if __name__ == "__main__":
    main()
