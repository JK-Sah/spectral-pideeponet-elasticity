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
axA.plot([0.01], [4e-6], "*", color=C_CF, markersize=15, label="Closed-form LS")
axA.plot([0.01], [0.0839], "D", color=C_SPEC, label="PI-spectral DeepONet")
axA.plot([0.77], [0.1274], "P", color=C_FNO, label="FNO")
axA.plot([0.01], [0.2299], "v", color="#999999", label="Data-only spectral")
axA.set_xscale("log"); axA.set_yscale("log")
axA.set_xlabel("Per-query time (ms)"); axA.set_ylabel(r"Displacement rel. $L^2$ error")
axA.set_title("(a) Homogeneous, fixed operator")
axA.legend(fontsize=7.3, loc="lower left")

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
axB.plot([0.03], [0.189], "P", color=C_FNO, label="FNO + physics")
axB.plot([0.03], [0.384], "D", color=C_SPEC, label="PI-spectral DeepONet")
axB.set_xscale("log"); axB.set_yscale("log")
axB.set_xlabel("Per-query time (ms)"); axB.set_ylabel(r"Displacement rel. $L^2$ error")
axB.set_title("(b) Heterogeneous $E(\\mathbf{x})$, per-query operator")
axB.legend(fontsize=7.3, loc="lower left")

fig.tight_layout()
fig.savefig(OUT / "Figure_11_pareto.pdf")
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
ax2.annotate("cost cliff\n(overtakes FEM)", (256, 69),
             textcoords="offset points", xytext=(-70, -6), fontsize=8,
             color="#333333", arrowprops=dict(arrowstyle="->", color="#333333"))

l1, la1 = ax1.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, la1 + la2, fontsize=7.6, loc="center left")
ax1.set_title("Heterogeneous ROM: accuracy and cost vs rank")
fig.tight_layout()
fig.savefig(OUT / "Figure_13_rom_cliff.pdf")
plt.close(fig)

print("wrote Figure_11_pareto.pdf, Figure_12_pod_spectrum.pdf, Figure_13_rom_cliff.pdf")
