#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_weights.py

Choose the residual weight of each benchmark on validation error alone: the
candidate with the lowest mean validation displacement error over seeds 42-44.
Prints the two weights and writes results_revision/r1/selected_weights.json.

  manufactured benchmark: runs/r1/ablation/wpde_*.json (canonical protocol)
  heterogeneous benchmark: spectral_e M=16, runs/r1/routeb/smooth (1e-4) and
                           runs/r1/hetero_wsweep (800 epochs), or only the
                           directory given by --het_dir (e.g. the 4000-epoch
                           sweep hetero_wsweep_e4000)

    python select_weights.py runs/r1 [--het_dir hetero_wsweep_e4000]
"""
import argparse, glob, json
from collections import defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("root", nargs="?", default="results_revision/r1")
ap.add_argument("--het_dir", default=None,
                help="heterogeneous sweep directory under root; if given, only it is used")
args = ap.parse_args()
root = Path(args.root)

lin = {}
for f in glob.glob(str(root / "ablation" / "wpde_*.json")):
    d = json.load(open(f))
    lin[d["value"]] = (d["val_mean"], d["val_std"], d["test_mean"], len(d["runs"]))

het = defaultdict(list)
het_files = (glob.glob(str(root / args.het_dir / "spectral_e_M16_wpde*_seed*.json")) if args.het_dir
             else glob.glob(str(root / "routeb" / "smooth" / "spectral_e_M16_wpde0.0001_seed*.json"))
             + glob.glob(str(root / "hetero_wsweep" / "spectral_e_M16_wpde*_seed*.json")))
for f in het_files:
    d = json.load(open(f))
    het[d["w_pde"]].append((d["val_error"], d["rel_l2_u"]))

print("manufactured benchmark (plain PI branch, canonical protocol):")
for w in sorted(lin):
    v, s, t, n = lin[w]
    print(f"  w={w:<8g} val {100*v:6.2f} +/- {100*s:4.2f}%   test {100*t:6.2f}%   seeds {n}")
w_lin = min(lin, key=lambda w: lin[w][0])
print("heterogeneous benchmark (spectral_e M=16):")
for w in sorted(het):
    v = [a for a, _ in het[w]]; t = [b for _, b in het[w]]
    print(f"  w={w:<8g} val {100*np.mean(v):6.2f} +/- {100*np.std(v):4.2f}%   test {100*np.mean(t):6.2f}%   seeds {len(v)}")
w_het = min(het, key=lambda w: np.mean([a for a, _ in het[w]]))
for name, w, cands in (("manufactured", w_lin, lin), ("heterogeneous", w_het, het)):
    if w in (min(cands), max(cands)):
        print(f"WARNING: {name} optimum is at the edge of the sweep ({w:g}); extend it")
print(f"W_LIN={w_lin:g} W_HET={w_het:g}")
json.dump(dict(w_lin=w_lin, w_het=w_het, rule="lowest mean validation error, seeds 42-44",
               het_sweep=args.het_dir or "routeb/smooth + hetero_wsweep (800 epochs)",
               manufactured={str(k): v for k, v in lin.items()},
               heterogeneous={str(k): het[k] for k in het}),
          open(root / "selected_weights.json", "w"), indent=2)
