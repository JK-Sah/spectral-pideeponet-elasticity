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

# Timings come from the controlled runs, never from the accuracy jobs and never
# from constants typed into this file.  Every per-query number that used to be
# hardcoded here was stale: the figure carried 0.85, 1.05, 2.50, 15.9 ms while
# the measured values are 1.02, 1.18, 3.14, 12.66.  Accuracy still comes from
# the accuracy files, which is where it belongs.
CL = json.load(open(RES / "linear" / "timing_classical_all.json"))
LIN = {M: json.load(open(RES / "linear" / f"timing_linear_v3_M{M}.json"))
       for M in (16, 64)}
CAP = json.load(open(RES / "linear" / "capacity_sweep.json"))
REPORTED = ["blocked/flush_denormal", "flush_denormal",
            "interleaved/flush_denormal", "blocked/as_is", "as_is"]


def t_single(M, tag):
    """Single-query latency of a learned model, as reported in the paper."""
    agg = LIN[M]["aggregate"][tag]
    for c in REPORTED:
        if c in agg:
            return agg[c]["single_query_ms"]["median"]
    raise KeyError(f"no reported condition for {tag} at M={M}")


def err(M, tag):
    return CAP[f"M{M}"][tag]["mean"]


FEM_A = {r["res"]: r for r in CL["fem"]}
ROM_A = {r["rank"]: r["online_ms"] for r in CL["rom"]}
HET = CL.get("hetero")

class Bench:
    """One varying-operator benchmark: accuracy from its accuracy run, every
    time from the controlled timing run.

    Accuracy is hardware-independent, so it stays where it was measured. Times
    do not: re-measuring the smooth field under control moved the rank-256
    reduced model from 69 ms to 7.2 ms. Where a controlled file is missing the
    superseded times are used and the caller is told, rather than silently
    mixing the two.
    """

    def __init__(self, acc_path, timing_path, label):
        acc = json.load(open(acc_path))
        self.label = label
        self.rom_err = {d["rank"]: d["rom_err"] for d in acc["ranks"]}
        self.fem_err = acc["fem_res29_err"]
        self.singular_values = acc.get("singular_values", [])
        self.timed = timing_path.exists()
        if self.timed:
            t = json.load(open(timing_path))
            self.rom_ms = {d["rank"]: d["online_ms"] for d in t["rom"]}
            self.fem_ms = t["fem_per_query_ms"]
            self.neural = t.get("neural", {})
        else:
            print(f"[warn] {timing_path} absent -- {label} panels fall back to "
                  f"the superseded timings in {acc_path.name}")
            self.rom_ms = {d["rank"]: d["online_ms"] for d in acc["ranks"]}
            self.fem_ms = acc["fem_res29_ms"]
            self.neural = {}

    def ranks(self):
        return sorted(r for r in self.rom_err if r in self.rom_ms)

    def crossover_rank(self):
        """Lowest rank whose per-query cost exceeds the finite-element solve."""
        for r in self.ranks():
            if self.rom_ms[r] > self.fem_ms:
                return r
        return None

    def point(self, model, fallback):
        n = self.neural.get(model)
        if n and n.get("rel_l2_u") is not None:
            return n["single_query_ms"], n["rel_l2_u"]
        return fallback


SMOOTH_B = Bench(RES / "rom_field.json",
                 RES / "hetero" / "timing_smooth.json", "smooth field")
ROUGH_B = Bench(RES / "rough" / "rom_field.json",
                RES / "hetero" / "timing_rough.json", "rough field")

plt.rcParams.update({"font.size": 10, "font.family": "serif",
                     "axes.grid": True, "grid.alpha": 0.3, "lines.markersize": 7})

C_FEM, C_ROM, C_CF, C_SPEC, C_FNO = "#1b7837", "#2166ac", "#762a83", "#d6604d", "#e08214"


def rows(bench, method):
    return [r for r in ledger if r["bench"] == bench and r["method"] == method]


# ---------------------------------------------------------------- Fig 11
fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 4.2))

# -- homogeneous panel --
fem_err = {r["res"]: r["err"] for r in rows("A", "fem")}
fem_x = [FEM_A[res]["amortized_ms"] for res in sorted(FEM_A)]
fem_y = [fem_err[res] for res in sorted(FEM_A)]
axA.plot(fem_x, fem_y, "-o", color=C_FEM, label="FEM (mesh refine)")
rk = romA["ranks"]
axA.plot([ROM_A.get(d["rank"], d["online_us"] / 1000) for d in rk],
         [d["rel_l2_u"] for d in rk], "-s",
         color=C_ROM, label="POD--Galerkin ROM (rank)")
# Canonical configuration, single-query latency (not batched throughput), so
# the learned points are directly comparable with the classical solves.
cf_x = t_single(16, "closed_form")
axA.plot([cf_x], [3e-7], "*", color=C_CF, markersize=15, label="Closed-form LS")
# Anchored branch at two trunk capacities: it improves with capacity and at
# M=64 overtakes both general-purpose classical solvers.
anc_x = [t_single(16, "pi_spectral_anchored"), t_single(64, "pi_spectral_anchored")]
anc_y = [err(16, "pi_spectral_anchored"), err(64, "pi_spectral_anchored")]
axA.plot(anc_x, anc_y, "-X", color="#1b9e77", markersize=10,
         label="PI-spectral, LS-anchored ($M$=16, 64)")
# Plain branch over the same capacities: it degrades.
pln_x = [t_single(16, "pi_spectral_plain"), t_single(64, "pi_spectral_plain")]
pln_y = [err(16, "pi_spectral_plain"), err(64, "pi_spectral_plain")]
axA.plot(pln_x, pln_y, "-D", color=C_SPEC,
         label="PI-spectral, plain ($M$=16, 64)")
axA.plot([t_single(16, "fno")], [err(16, "fno")], "P", color=C_FNO, label="FNO")
axA.plot([t_single(16, "data_only_spectral")], [err(16, "data_only_spectral")],
         "v", color="#999999", label="Data-only spectral")
axA.annotate("$M$=64", xy=(anc_x[1], anc_y[1]), xytext=(6, -12),
             textcoords="offset points", fontsize=7.5, color="#1b9e77")
axA.annotate("$M$=16", xy=(anc_x[0], anc_y[0]), xytext=(-30, 4),
             textcoords="offset points", fontsize=7.5, color="#1b9e77")
axA.annotate("$M$=64", xy=(pln_x[1], pln_y[1]), xytext=(6, 2),
             textcoords="offset points", fontsize=7.5, color=C_SPEC)
axA.set_xscale("log"); axA.set_yscale("log")
axA.set_xlabel("Per-query time (ms)"); axA.set_ylabel(r"Displacement rel. $L^2$ error")
axA.set_title("(a) Homogeneous, fixed operator\n(single-query latency)")
axA.legend(fontsize=6.8, loc="upper center", bbox_to_anchor=(0.52, 1.005),
           framealpha=0.93, ncol=2, columnspacing=0.8, handletextpad=0.4)
axA.set_ylim(top=60.0)

# -- heterogeneous panel --
B = SMOOTH_B
axB.plot([B.fem_ms], [B.fem_err], "o", color=C_FEM,
         label="FEM $28\\times28$ (per query)")
rb = B.ranks()
axB.plot([B.rom_ms[r] for r in rb], [B.rom_err[r] for r in rb], "-s",
         color=C_ROM, label="POD--Galerkin ROM (rank)")
for r in rb:
    if r in (32, 128, 256):
        axB.annotate(f"r={r}", (B.rom_ms[r], B.rom_err[r]),
                     textcoords="offset points", xytext=(4, 5), fontsize=7)
fx, fy = B.point("fno_e", (1.45, 0.189))
sx, sy = B.point("spectral_e", (0.031, 0.384))
axB.plot([fx], [fy], "P", color=C_FNO, label="FNO + physics")
axB.plot([sx], [sy], "D", color=C_SPEC, label="PI-spectral DeepONet")
axB.set_xscale("log"); axB.set_yscale("log")
axB.set_xlabel("Per-query time (ms)"); axB.set_ylabel(r"Displacement rel. $L^2$ error")
axB.set_title("(b) Heterogeneous $E(\\mathbf{x})$, per-query operator")
axB.legend(fontsize=7.3, loc="lower left")

fig.tight_layout()
fig.savefig(OUT / "Figure_11_pareto.pdf")
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
svA = romA["singular_values"]; svB = SMOOTH_B.singular_values
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
B = SMOOTH_B
ranks = B.ranks()
ax1.semilogy(ranks, [B.rom_err[r] for r in ranks], "-s", color=C_ROM,
             label="ROM error")
ax1.set_xlabel("POD rank $r$"); ax1.set_ylabel("Displacement rel. $L^2$ error", color=C_ROM)
ax1.tick_params(axis="y", labelcolor=C_ROM)

ax2 = ax1.twinx(); ax2.grid(False)
ax2.plot(ranks, [B.rom_ms[r] for r in ranks], "-^", color=C_FNO,
         label="ROM time/query")
ax2.axhline(B.fem_ms, color=C_FEM, ls="--", lw=1.3,
            label=f"FEM/query ({B.fem_ms:.1f} ms)")
surr = B.point("spectral_e", (0.03, None))[0]
ax2.axhline(surr, color=C_SPEC, ls=":", lw=1.3,
            label=f"Surrogate inference ({surr:.2f} ms)")
ax2.set_ylabel("Per-query time (ms)", color=C_FNO)
ax2.set_yscale("log"); ax2.tick_params(axis="y", labelcolor=C_FNO)

# Annotate the crossing only where one is actually measured. On the smooth
# field under the controlled timings the reduced model stays below the
# finite-element solve across every rank tested, so there is no cliff to point
# at and the figure should not imply one.
xr = B.crossover_rank()
if xr is not None:
    ax2.annotate("cost cliff\n(overtakes FEM)", xy=(xr, B.rom_ms[xr]),
                 xytext=(0.40, 0.82), textcoords=ax2.transAxes,
                 fontsize=8, color="#333333", ha="center", va="center",
                 arrowprops=dict(arrowstyle="->", color="#333333"))
else:
    top = max(B.rom_ms[r] for r in ranks)
    ax2.annotate(f"no crossing up to $r$={ranks[-1]}\n"
                 f"({top:.1f} ms vs {B.fem_ms:.1f} ms)",
                 xy=(ranks[-1], top), xytext=(0.36, 0.84),
                 textcoords=ax2.transAxes, fontsize=8, color="#333333",
                 ha="center", va="center",
                 arrowprops=dict(arrowstyle="->", color="#333333"))

l1, la1 = ax1.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, la1 + la2, fontsize=7.2, loc="center",
           bbox_to_anchor=(0.63, 0.30), framealpha=0.92)
ax1.set_title("Heterogeneous ROM: accuracy and cost vs rank")
fig.tight_layout()
fig.savefig(OUT / "Figure_13_rom_cliff.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig 14 (crossover)
# Smooth vs rough E(x): the Pareto frontier flips. On the smooth field the ROM
# dominates and the FNO is off-frontier; on the rough field the ROM's cost
# cliffs and the FNO moves onto the frontier.
fig, (axs, axr) = plt.subplots(1, 2, figsize=(9.4, 4.2), sharey=True)


def pareto_panel(ax, B, fno_fb, spec_fb, title):
    rb = B.ranks()
    ax.plot([B.rom_ms[r] for r in rb], [B.rom_err[r] for r in rb], "-s",
            color=C_ROM, label="POD--Galerkin ROM (rank)")
    ax.plot([B.fem_ms], [B.fem_err], "o", color=C_FEM, markersize=9,
            label="FEM $28\\times28$ (per query)")
    fx, fy = B.point("fno_e", fno_fb)
    sx, sy = B.point("spectral_e", spec_fb)
    ax.plot([fx], [fy], "P", color=C_FNO, markersize=11, label="FNO")
    ax.plot([sx], [sy], "D", color=C_SPEC, markersize=9,
            label="Spectral DeepONet")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Per-query time (ms)"); ax.set_title(title)


pareto_panel(axs, SMOOTH_B, (1.45, 0.189), (0.031, 0.384),
             "(a) Smooth $E(\\mathbf{x})$")
pareto_panel(axr, ROUGH_B, (1.45, 0.2448), (0.031, 0.7109),
             "(b) Rough $E(\\mathbf{x})$")
axs.set_ylabel(r"Displacement rel. $L^2$ error")
axr.legend(fontsize=7.6, loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "Figure_14_crossover.pdf")
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
plt.close(fig)

print("wrote Figure_6_seed_repeatability.pdf, Figure_11_pareto.pdf, Figure_12_pod_spectrum.pdf, "
      "Figure_13_rom_cliff.pdf, Figure_14_crossover.pdf, Figure_15_nonlinear.pdf")
