#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py

Generate the three benchmarking figures for the reframed manuscript:
  Figure_11 : master accuracy-vs-cost Pareto (homogeneous + heterogeneous)
  Figure_12 : POD singular-value spectra (Kolmogorov n-width of the two families)
  Figure_13 : ROM rank vs error and per-query cost (the cost cliff / crossover)

Reads results_revision/{ledger.json, rom_baseline.json, rom_field.json};
neural-operator points are taken from the manuscript result tables.
Writes PDFs into the submission folder.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path("results_revision")
OUT = Path("../CompMech_submission_ready")
ledger = json.load(open(RES / "ledger.json"))
romA = json.load(open(RES / "rom_baseline.json"))
romB = json.load(open(RES / "rom_field.json"))

plt.rcParams.update({"font.size": 10, "font.family": "serif",
                     "axes.grid": True, "grid.alpha": 0.3, "lines.markersize": 7})

C_FEM, C_ROM, C_CF, C_SPEC, C_FNO = "#1b7837", "#2166ac", "#762a83", "#d6604d", "#e08214"


def rows(bench, method):
    return [r for r in ledger if r["bench"] == bench and r["method"] == method]


# ---------------------------------------------------------------- Fig 11
fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 4.2))

# -- homogeneous panel --
fem = sorted(rows("A", "fem"), key=lambda r: r["time_ms"])
axA.plot([r["time_ms"] for r in fem], [r["err"] for r in fem], "-o",
         color=C_FEM, label="FEM (mesh refine)")
rk = romA["ranks"]
axA.plot([d["online_us"]/1000 for d in rk], [d["rel_l2_u"] for d in rk], "-s",
         color=C_ROM, label="POD--Galerkin ROM (rank)")
# Canonical configuration, single-query latency (not batched throughput), so
# the learned points are directly comparable with the classical solves.
axA.plot([0.85], [3e-7], "*", color=C_CF, markersize=15, label="Closed-form LS")
# Anchored branch at two trunk capacities: it improves with capacity and at
# M=64 overtakes both general-purpose classical solvers.
axA.plot([1.08, 2.51], [0.0164, 0.0043], "-X", color="#1b9e77", markersize=10,
         label="PI-spectral, LS-anchored ($M$=16, 64)")
# Plain branch over the same capacities: it degrades.
axA.plot([1.05, 2.50], [0.0874, 0.1442], "-D", color=C_SPEC,
         label="PI-spectral, plain ($M$=16, 64)")
axA.plot([15.9], [0.1112], "P", color=C_FNO, label="FNO")
axA.plot([1.08], [0.2291], "v", color="#999999", label="Data-only spectral")
axA.annotate("$M$=64", xy=(2.51, 0.0043), xytext=(6, -12),
             textcoords="offset points", fontsize=7.5, color="#1b9e77")
axA.annotate("$M$=16", xy=(1.08, 0.0164), xytext=(-30, 4),
             textcoords="offset points", fontsize=7.5, color="#1b9e77")
axA.annotate("$M$=64", xy=(2.50, 0.1442), xytext=(6, 2),
             textcoords="offset points", fontsize=7.5, color=C_SPEC)
axA.set_xscale("log"); axA.set_yscale("log")
axA.set_xlabel("Per-query time (ms)"); axA.set_ylabel(r"Displacement rel. $L^2$ error")
axA.set_title("(a) Homogeneous, fixed operator\n(single-query latency)")
axA.legend(fontsize=6.8, loc="upper center", bbox_to_anchor=(0.52, 1.005),
           framealpha=0.93, ncol=2, columnspacing=0.8, handletextpad=0.4)
axA.set_ylim(top=60.0)

# -- heterogeneous panel --
femB = rows("B", "fem")
r29 = [r for r in femB if r["res"] == 29][0]
axB.plot([r29["time_ms"]], [r29["err"]], "o", color=C_FEM,
         label="FEM $28\\times28$ (per query)")
rb = sorted(romB["ranks"], key=lambda d: d["rank"])
axB.plot([d["online_ms"] for d in rb], [d["rom_err"] for d in rb], "-s",
         color=C_ROM, label="POD--Galerkin ROM (rank)")
for d in rb:
    if d["rank"] in (32, 128, 256):
        axB.annotate(f"r={d['rank']}", (d["online_ms"], d["rom_err"]),
                     textcoords="offset points", xytext=(4, 5), fontsize=7)
axB.plot([1.45], [0.189], "P", color=C_FNO, label="FNO + physics")
axB.plot([0.031], [0.384], "D", color=C_SPEC, label="PI-spectral DeepONet")
axB.set_xscale("log"); axB.set_yscale("log")
axB.set_xlabel("Per-query time (ms)"); axB.set_ylabel(r"Displacement rel. $L^2$ error")
axB.set_title("(b) Heterogeneous $E(\\mathbf{x})$, per-query operator")
axB.legend(fontsize=7.3, loc="lower left")

fig.tight_layout()
fig.savefig(OUT / "Figure_11_pareto.pdf")
fig.savefig(OUT / "Fig11.eps")
plt.close(fig)


# ---------------------------------------------------------------- Fig 6 (seed repeatability)
# Canonical configuration, seeds 42/43/44.  The earlier version of this figure
# showed a different (sweep-optimum) setting and is superseded.
seedm = json.load(open(RES / "linear" / "canonical_seed_metrics.json"))["per_seed"]
keys = [("disp", "Disp."), ("strain", "Strain"), ("stress", "Stress"),
        ("energy", "Energy")]
means = [100 * np.mean([seedm[s][k] for s in seedm]) for k, _ in keys]
stds = [100 * np.std([seedm[s][k] for s in seedm]) for k, _ in keys]
fig, ax = plt.subplots(figsize=(5.6, 4.0))
ax.bar([lab for _, lab in keys], means, yerr=stds, capsize=5,
       color="#2166ac", edgecolor="black", linewidth=0.6)
for i, (m, sd) in enumerate(zip(means, stds)):
    ax.text(i, m + sd + 0.35, f"{m:.2f}%", ha="center", fontsize=8.5)
ax.set_ylabel("Relative error (%)")
ax.set_title("Canonical configuration: repeatability over three seeds")
ax.set_ylim(0, max(m + sd for m, sd in zip(means, stds)) * 1.25)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(OUT / "Figure_6_seed_repeatability.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig 12
fig, ax = plt.subplots(figsize=(5.2, 4.0))
svA = romA["singular_values"]; svB = romB["singular_values"]
ax.semilogy(range(1, len(svA)+1), [s/svA[0] for s in svA], "-o", color=C_FEM,
            label="Homogeneous (fixed operator)")
ax.semilogy(range(1, len(svB)+1), [s/svB[0] for s in svB], "-s", color=C_ROM,
            label=r"Heterogeneous $E(\mathbf{x})$")
ax.axvline(32, color="#999999", ls="--", lw=1)
ax.annotate("2$\\times$16 modes", (32, 3e-3), fontsize=8, rotation=90,
            va="bottom", ha="right", color="#666666")
ax.set_xlabel("POD mode index"); ax.set_ylabel("Normalized singular value")
ax.set_title("POD spectra: Kolmogorov $n$-width")
ax.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT / "Figure_12_pod_spectrum.pdf")
fig.savefig(OUT / "Fig12.eps")
plt.close(fig)

# ---------------------------------------------------------------- Fig 13
fig, ax1 = plt.subplots(figsize=(5.6, 4.0))
rb = sorted(romB["ranks"], key=lambda d: d["rank"])
ranks = [d["rank"] for d in rb]
ax1.semilogy(ranks, [d["rom_err"] for d in rb], "-s", color=C_ROM,
             label="ROM error")
ax1.set_xlabel("POD rank $r$"); ax1.set_ylabel("Displacement rel. $L^2$ error", color=C_ROM)
ax1.tick_params(axis="y", labelcolor=C_ROM)

ax2 = ax1.twinx(); ax2.grid(False)
ax2.plot(ranks, [d["online_ms"] for d in rb], "-^", color=C_FNO,
         label="ROM time/query")
femB_ms = [r for r in rows("B", "fem") if r["res"] == 29][0]["time_ms"]
ax2.axhline(femB_ms, color=C_FEM, ls="--", lw=1.3, label=f"FEM/query ({femB_ms:.0f} ms)")
ax2.axhline(0.03, color=C_SPEC, ls=":", lw=1.3, label="Surrogate inference (0.03 ms)")
ax2.set_ylabel("Per-query time (ms)", color=C_FNO)
ax2.set_yscale("log"); ax2.tick_params(axis="y", labelcolor=C_FNO)
ax2.annotate("cost cliff\n(overtakes FEM)", xy=(256, 64),
             xytext=(0.40, 0.82), textcoords=ax2.transAxes,
             fontsize=8, color="#333333", ha="center", va="center",
             arrowprops=dict(arrowstyle="->", color="#333333"))

l1, la1 = ax1.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, la1 + la2, fontsize=7.2, loc="center",
           bbox_to_anchor=(0.63, 0.30), framealpha=0.92)
ax1.set_title("Heterogeneous ROM: accuracy and cost vs rank")
fig.tight_layout()
fig.savefig(OUT / "Figure_13_rom_cliff.pdf")
fig.savefig(OUT / "Fig13.eps")
plt.close(fig)

# ---------------------------------------------------------------- Fig 14 (crossover)
# Smooth vs rough E(x): the Pareto frontier flips. On the smooth field the ROM
# dominates and the FNO is off-frontier; on the rough field the ROM's cost
# cliffs and the FNO moves onto the frontier.
romR = json.load(open(RES / "rough" / "rom_field.json"))
fig, (axs, axr) = plt.subplots(1, 2, figsize=(9.4, 4.2), sharey=True)

def pareto_panel(ax, rom, fem_ms, fem_err, fno, spec, title):
    rb = sorted(rom["ranks"], key=lambda d: d["rank"])
    ax.plot([d["online_ms"] for d in rb], [d["rom_err"] for d in rb], "-s",
            color=C_ROM, label="POD--Galerkin ROM (rank)")
    ax.plot([fem_ms], [fem_err], "o", color=C_FEM, markersize=9,
            label="FEM $28\\times28$ (per query)")
    ax.plot([fno[0]], [fno[1]], "P", color=C_FNO, markersize=11, label="FNO")
    ax.plot([spec[0]], [spec[1]], "D", color=C_SPEC, markersize=9,
            label="Spectral DeepONet")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Per-query time (ms)"); ax.set_title(title)

pareto_panel(axs, romB, 10.8, 0.0086, (1.45, 0.189), (0.031, 0.384),
             "(a) Smooth $E(\\mathbf{x})$: ROM dominates")
pareto_panel(axr, romR, 10.3, 0.0736, (1.45, 0.2448), (0.031, 0.7109),
             "(b) Rough $E(\\mathbf{x})$: FNO reaches the frontier")
axs.set_ylabel(r"Displacement rel. $L^2$ error")
axr.legend(fontsize=7.6, loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "Figure_14_crossover.pdf")
fig.savefig(OUT / "Fig14.eps")
plt.close(fig)

# ---------------------------------------------------------------- Fig 15 (nonlinear)
# Finite-strain hyperelasticity: the neural operator dominates the reduced-order
# models -- the flip from the linear case.
nl = json.load(open(RES / "nonlinear" / "ledger.json"))
fig, ax = plt.subplots(figsize=(5.8, 4.4))
style = {
    "POD-Galerkin(r=32)": ("POD--Galerkin", C_ROM, "s"),
    "POD-DEIM(r=32,m=128)": ("POD--DEIM (hyper-reduced)", "#2166ac", "^"),
    "FNO": ("FNO", C_FNO, "P"),
    "Spectral": ("Spectral DeepONet", C_SPEC, "D"),
}
for key, (lab, col, mk) in style.items():
    d = nl[key]
    ax.plot([d["ms"]], [d["err"]], mk, color=col, markersize=12, label=lab)
# Newton-FEM is exact (err=0): draw as a reference line at its cost.
ax.axvline(nl["Newton-FEM"]["ms"], color=C_FEM, ls="--", lw=1.4)
errs_all = [nl[k]["err"] for k in style]
ax.annotate("Newton-FEM (reference, %.0f ms)" % nl["Newton-FEM"]["ms"],
            xy=(nl["Newton-FEM"]["ms"], np.sqrt(min(errs_all) * max(errs_all))),
            fontsize=8, color=C_FEM, ha="right", va="center", rotation=90,
            xytext=(-4, 0), textcoords="offset points")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Per-query time (ms)")
ax.set_ylabel(r"Displacement rel. $L^2$ error")
ax.set_title("Finite-strain hyperelasticity: a learned operator\n"
             "reaches the accuracy-cost frontier")
ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
ax.annotate("", xy=(nl["FNO"]["ms"], nl["FNO"]["err"]),
            xytext=(nl["POD-Galerkin(r=32)"]["ms"], nl["POD-Galerkin(r=32)"]["err"]),
            arrowprops=dict(arrowstyle="<->", color="#777777", lw=1.1))
ax.text(np.sqrt(nl["FNO"]["ms"] * nl["POD-Galerkin(r=32)"]["ms"]),
        nl["FNO"]["err"] * 0.90, "~8x", fontsize=9, color="#555555",
        ha="center", va="top")
ax.text(0.02, 0.03, "matched accuracy, ~8x lower cost per query",
        transform=ax.transAxes, fontsize=8, color="#555555", ha="left")
fig.tight_layout()
fig.savefig(OUT / "Figure_15_nonlinear.pdf")
fig.savefig(OUT / "Fig15.eps")
plt.close(fig)

print("wrote Figure_6_seed_repeatability.pdf, Figure_11_pareto.pdf, Figure_12_pod_spectrum.pdf, "
      "Figure_13_rom_cliff.pdf, Figure_14_crossover.pdf, Figure_15_nonlinear.pdf")
