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
LS = RES / "r1" / "w" / "linear_sel"      # per-model selected checkpoints, one timing job
CL = json.load(open(LS / "timing_classical_all.json"))
LIN = {M: json.load(open(LS / f"timing_linear_v3_M{M}.json"))
       for M in (16, 64)}
CAP = json.load(open(LS / "capacity_sweep.json"))
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

    def point(self, prefix):
        """Single-query cost and error of one trained configuration.

        prefix names the checkpoint family, e.g. "fno_e_wpde0.01"; the error is
        the mean over its seeds and the latency the median, as in the ledgers.
        """
        runs = [v for k, v in self.neural.items() if k.startswith(prefix + "_seed")]
        if not runs:
            raise KeyError(f"{self.label}: no checkpoint matching {prefix!r} in "
                           f"{sorted(self.neural)} -- refusing to fall back "
                           f"silently to a hardcoded point")
        return (float(np.median([v["single_query_ms"] for v in runs])),
                float(np.mean([v["rel_l2_u"] for v in runs])))


# Residual weights chosen on validation for each model (select_per_model.py).
SEL = json.load(open(RES / "r1" / "w" / "selected_per_model.json"))["selected"]


def tag(model, key):
    return f"{model}_wpde{SEL[key]['w']:g}"


# best spectral trunk on the smooth field: lowest validation error among the
# trunk sizes, each at its own weight
BEST_M = min((k for k in SEL if k.startswith("het_spectral_e_M")),
             key=lambda k: SEL[k]["val_mean"]).split("_M")[1]
SMOOTH_FNO, SMOOTH_SPEC = tag("fno_e", "het_fno_e"), tag(f"spectral_e_M{BEST_M}", f"het_spectral_e_M{BEST_M}")
ROUGH_FNO, ROUGH_SPEC = tag("fno_e", "rough_fno_e"), tag("spectral_e_M64", "rough_spectral_e_M64")


SMOOTH_B = Bench(RES / "rom_field.json",
                 RES / "r1" / "w" / "routeb" / "timing_smooth.json", "smooth field")
ROUGH_B = Bench(RES / "rough" / "rom_field.json",
                RES / "r1" / "w" / "routeb" / "timing_rough.json", "rough field")

plt.rcParams.update({"font.size": 10, "font.family": "serif",
                     "axes.grid": True, "grid.alpha": 0.3, "lines.markersize": 7})

C_FEM, C_ROM, C_CF, C_SPEC, C_FNO = "#1b7837", "#2166ac", "#762a83", "#d6604d", "#e08214"


def plain_log(axis):
    """Ticks at 1, 2, 5 x 10^k with plain labels, for log axes spanning about a
    decade, where the default minor labels (3x10^0, 4x10^0, ...) collide."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


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
fx, fy = B.point(SMOOTH_FNO)
sx, sy = B.point(SMOOTH_SPEC)
axB.plot([fx], [fy], "P", color=C_FNO, label="FNO + physics")
axB.plot([sx], [sy], "D", color=C_SPEC, label=f"PI-spectral DeepONet ($M$={BEST_M})")
axB.set_xscale("log"); axB.set_yscale("log")
axB.set_xlabel("Per-query time (ms)"); axB.set_ylabel(r"Displacement rel. $L^2$ error")
axB.set_title("(b) Heterogeneous $E(\\mathbf{x})$, per-query operator")
plain_log(axB.xaxis)
axB.legend(fontsize=7.3, loc="center right", bbox_to_anchor=(1.0, 0.42))

fig.tight_layout()
fig.savefig(OUT / "Figure_11_pareto.pdf")
plt.close(fig)


# ---------------------------------------------------------------- Fig 6 (seed repeatability)
# Canonical configuration, seeds 42/43/44.  The earlier version of this figure
# showed a different (sweep-optimum) setting and is superseded.
seedm = json.load(open(LS / "canonical_seed_metrics.json"))["per_seed"]
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
surr = B.point(SMOOTH_SPEC)[0]
ax2.axhline(surr, color=C_SPEC, ls=":", lw=1.3,
            label=f"Spectral DeepONet, $M$={BEST_M} ({surr:.2f} ms)")
ax2.set_ylabel("Per-query time (ms)", color=C_FNO)
ax2.set_yscale("log"); ax2.tick_params(axis="y", labelcolor=C_FNO)
plain_log(ax2.yaxis)

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
# Smooth vs rough E(x). On the smooth field the ROM dominates; on the rough
# field its rank, and with it its cost, passes the finite-element solve, but no
# learned operator takes the frontier as a single query.
fig, (axs, axr) = plt.subplots(1, 2, figsize=(9.4, 4.2), sharey=True)


def pareto_panel(ax, B, fno, spec, spec_label, title, legend=dict(loc="lower left")):
    rb = B.ranks()
    ax.plot([B.rom_ms[r] for r in rb], [B.rom_err[r] for r in rb], "-s",
            color=C_ROM, label="POD--Galerkin ROM (rank)")
    ax.plot([B.fem_ms], [B.fem_err], "o", color=C_FEM, markersize=9,
            label="FEM $28\\times28$ (per query)")
    fx, fy = B.point(fno)
    sx, sy = B.point(spec)
    ax.plot([fx], [fy], "P", color=C_FNO, markersize=11, label="FNO")
    ax.plot([sx], [sy], "D", color=C_SPEC, markersize=9, label=spec_label)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Single-query time (ms)"); ax.set_title(title)
    plain_log(ax.xaxis)
    ax.legend(fontsize=7.6, **legend)


pareto_panel(axs, SMOOTH_B, SMOOTH_FNO, SMOOTH_SPEC, f"Spectral DeepONet ($M$={BEST_M})",
             "(a) Smooth $E(\\mathbf{x})$", dict(loc="center right", bbox_to_anchor=(1.0, 0.42)))
pareto_panel(axr, ROUGH_B, ROUGH_FNO, ROUGH_SPEC, "Spectral DeepONet ($M$=64)",
             "(b) Rough $E(\\mathbf{x})$")
axs.set_ylabel(r"Displacement rel. $L^2$ error")
fig.tight_layout()
fig.savefig(OUT / "Figure_14_crossover.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig 15 (nonlinear)
# Finite-strain hyperelasticity, built from the measurement files by
# fig15_nonlinear.py (which also writes results_revision/nonlinear/ledger.json).
import fig15_nonlinear
fig15_nonlinear.plot(fig15_nonlinear.build_ledger(), OUT, (C_FEM, C_ROM, C_SPEC, C_FNO))

print("wrote Figure_6_seed_repeatability.pdf, Figure_11_pareto.pdf, Figure_12_pod_spectrum.pdf, "
      "Figure_13_rom_cliff.pdf, Figure_14_crossover.pdf, Figure_15_nonlinear.pdf")
