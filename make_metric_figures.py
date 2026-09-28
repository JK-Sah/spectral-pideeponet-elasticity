#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_metric_figures.py

The two metric bar charts, rebuilt from the canonical checkpoints:

  Figure_2_main_quantitative_comparison  (Figure 3 of the compiled manuscript)
  Figure_7_fno_comparison                (Figure 8 of the compiled manuscript)

Both previously plotted the pre-revision single-seed values, which is the same
defect the comparison table carried: the displacement bar showed 0.0839, and no
canonical seed gives that. Both are now drawn from
results_revision/linear/canonical_metrics_all_M16.json, with whiskers at one
standard deviation over seeds 42, 43 and 44, so the bars and the table agree
and the seed spread is visible where two bars are close enough for it to
matter -- on strain and stress it decides whether there is a gap at all.

    python make_metric_figures.py
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path("results_revision/linear")
OUT = Path("../CompMech_submission_ready")

plt.rcParams.update({"font.size": 10, "font.family": "serif",
                     "axes.grid": True, "grid.alpha": 0.3, "lines.markersize": 7})

COLORS = {"PI-Spectral DeepONet": "#2166ac",
          "Data-only Spectral": "#d73027",
          "FNO": "#4dac26"}

METRICS = [("disp_l2", "Disp. $L^2$"), ("strain_l2", "Strain $L^2$"),
           ("stress_l2", "Stress $L^2$"), ("energy_err", "Energy"),
           ("pde_mse", "PDE MSE")]
TAGS = {"PI-Spectral DeepONet": "pi_spectral_plain",
        "Data-only Spectral": "data_only_spectral",
        "FNO": "fno"}

d = json.load(open(RES / "canonical_metrics_all_M16.json"))
models = list(TAGS)
mean = {m: [d[TAGS[m]][k]["mean"] for k, _ in METRICS] for m in models}
std = {m: [d[TAGS[m]][k]["std"] for k, _ in METRICS] for m in models}


def bars(ax, title):
    x = np.arange(len(METRICS))
    w = 0.25
    for i, m in enumerate(models):
        ax.bar(x + (i - 1) * w, mean[m], w, yerr=std[m], capsize=3,
               label=m, color=COLORS[m],
               error_kw=dict(lw=0.9, ecolor="0.25"))
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in METRICS])
    ax.set_ylabel("Error / residual (log scale)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax.grid(axis="y", which="both", ls="--", alpha=0.4)


fig, ax = plt.subplots(figsize=(8.6, 4.4))
bars(ax, "Canonical configuration, $M{=}16$: mean $\\pm$ s.d. over three seeds")
plt.tight_layout()
fig.savefig(OUT / "Figure_2_main_quantitative_comparison.pdf", bbox_inches="tight")
fig.savefig(OUT / "Fig3.eps", format="eps", bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(10.4, 4.8))
bars(ax, "Extended comparison: spectral branches and the Fourier neural operator")
plt.tight_layout()
fig.savefig(OUT / "Figure_7_fno_comparison.pdf", bbox_inches="tight")
fig.savefig(OUT / "Fig8.eps", format="eps", bbox_inches="tight")
plt.close()

print("values plotted (mean +/- s.d. over seeds 42/43/44):")
hdr = "  " + " " * 22 + "".join(f"{lab:>16s}" for _, lab in METRICS)
print(hdr)
for m in models:
    print(f"  {m:22s}" + "".join(f"{mean[m][i]:9.5f}+-{std[m][i]:.4f}"
                                 for i in range(len(METRICS))))
print("\nwhere the two closest bars sit:")
for i, (k, lab) in enumerate(METRICS):
    a, b = mean["PI-Spectral DeepONet"][i], mean["FNO"][i]
    sa, sb = std["PI-Spectral DeepONet"][i], std["FNO"][i]
    sep = abs(a - b) / max(sa + sb, 1e-30)
    print(f"  {lab:14s} PI {a:.5f}  FNO {b:.5f}   gap = {sep:5.2f} x combined s.d."
          f"{'   (overlapping)' if sep < 1 else ''}")
print("\nwrote Figure_2_main_quantitative_comparison.pdf + Fig3.eps, "
      "Figure_7_fno_comparison.pdf + Fig8.eps")
