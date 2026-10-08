#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig15_nonlinear.py

Finite-strain ledger and Figure 15, built from the measurement files only:
  accuracy   results_revision/r1/nonlinear/classical_full.json  (all 397 test cases)
             results_revision/r1/nonlinear/neural_long/*.json   (3 seeds, 4000 epochs)
  cost       results_revision/r1/nonlinear/timing_v3.json       (one exclusive allocation)

Writes results_revision/nonlinear/ledger.json and Figure_15_nonlinear.pdf.
make_figures.py calls this for Figure 15; run it directly to rebuild only this
figure:

    python fig15_nonlinear.py
"""
import glob, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path("results_revision")
N_CORPUS = 1987     # Newton solves behind the training pool, shared by every data-driven method


def build_ledger():
    nl = RES / "r1" / "nonlinear"
    cl = json.load(open(nl / "classical_full.json"))
    t = json.load(open(nl / "timing_v3.json"))
    m = cl["deim_selected_m"]
    deim = next(d for d in cl["deim"] if d["m"] == m)
    fem_ms = t["newton_fem"]["median_ms"]
    off = t["offline_s"]
    corpus_s = N_CORPUS * fem_ms / 1e3

    def neural(kind):
        runs = [json.load(open(f)) for f in sorted(glob.glob(str(nl / "neural_long" / f"{kind}_seed*.json")))]
        tim = [t["neural"][f"{kind}_seed{r['seed']}"] for r in runs]
        return dict(err=float(np.mean([r["test_error_heldout"] for r in runs])),
                    err_std=float(np.std([r["test_error_heldout"] for r in runs])),
                    ms=float(np.median([x["single_query_ms"] for x in tim])),
                    ms_range=[float(min(x["single_query_ms"] for x in tim)),
                              float(max(x["single_query_ms"] for x in tim))],
                    ms_batched_per_sample=float(np.median([x["batched_ms_per_sample"] for x in tim])),
                    offline_s=float(np.mean([r["train_minutes"] for r in runs]) * 60),
                    params=int(runs[0]["n_params"]))

    L = {
        "Newton-FEM": dict(err=0.0, ms=fem_ms, breakdown=t["newton_fem"]["breakdown_frac"]),
        "POD-Galerkin(r=32)": dict(err=cl["galerkin"]["test"]["err_mean"],
                                   ms=t["pod_galerkin_r32"]["median_ms"], offline_s=off["pod_svd"]),
        f"POD-DEIM(r=32,m={m})": dict(err=deim["test"]["err_mean"],
                                      diverged=f"{deim['test']['n_diverged']}/{deim['test']['n_cases']}",
                                      ms=t["pod_deim_r32"][str(m)]["median_ms"],
                                      offline_s=off["pod_svd"] + off["deim_force_snapshots"]
                                      + off[f"deim_build_m{m}"]),
        "FNO": neural("fno"),
        "Spectral": neural("spectral"),
    }
    for k, v in L.items():
        if k != "Newton-FEM":
            v["break_even_queries"] = (corpus_s + v["offline_s"]) / ((fem_ms - v["ms"]) / 1e3)
    L["_meta"] = dict(deim_selected_m=m, corpus_newton_solves=N_CORPUS, corpus_s=corpus_s,
                      sources=["r1/nonlinear/classical_full.json", "r1/nonlinear/neural_long/",
                               "r1/nonlinear/timing_v3.json"])
    (RES / "nonlinear").mkdir(exist_ok=True)
    (RES / "nonlinear" / "ledger.json").write_text(json.dumps(L, indent=2))
    return L


def plot(L, out, colors):
    C_FEM, C_ROM, C_SPEC, C_FNO = colors
    deim_key = next(k for k in L if k.startswith("POD-DEIM"))
    m = L["_meta"]["deim_selected_m"]
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    style = {
        "POD-Galerkin(r=32)": ("POD--Galerkin ($r=32$)", C_ROM, "s"),
        deim_key: (f"POD--DEIM ($r=32$, $m={m}$)", "#2166ac", "^"),
        "FNO": ("FNO", C_FNO, "P"),
        "Spectral": ("Spectral DeepONet", C_SPEC, "D"),
    }
    for key, (lab, col, mk) in style.items():
        d = L[key]
        ax.plot([d["ms"]], [d["err"]], mk, color=col, markersize=12, label=lab)
    fem = L["Newton-FEM"]["ms"]
    ax.axvline(fem, color=C_FEM, ls="--", lw=1.4)
    errs = [L[k]["err"] for k in style]
    ax.annotate("Newton-FEM (reference, %.0f ms)" % fem,
                xy=(fem, np.sqrt(min(errs) * max(errs))), fontsize=8, color=C_FEM,
                ha="right", va="center", rotation=90, xytext=(-4, 0), textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log")
    ms = [L[k]["ms"] for k in style]
    ax.set_xlim(min(ms) * 0.5, fem * 1.8)
    ax.set_ylim(min(errs) * 0.8, max(errs) * 1.3)
    ax.set_xlabel("Single-query time (ms)")
    ax.set_ylabel(r"Displacement rel. $L^2$ error")
    ax.set_title("Finite-strain hyperelasticity: a learned operator\n"
                 "reaches the accuracy-cost frontier")
    ax.legend(fontsize=8, loc="upper center", framealpha=0.95)
    g, f = L["POD-Galerkin(r=32)"], L["FNO"]
    ratio = g["ms"] / f["ms"]
    ax.annotate("", xy=(f["ms"], f["err"]), xytext=(g["ms"], g["err"]),
                arrowprops=dict(arrowstyle="<->", color="#777777", lw=1.1))
    ax.text(np.sqrt(f["ms"] * g["ms"]), np.sqrt(f["err"] * g["err"]) * 0.97, f"{ratio:.1f}x",
            fontsize=9, color="#555555", ha="center", va="top")
    ax.text(0.02, 0.03, f"FNO {100*f['err']:.2f}% vs POD-Galerkin {100*g['err']:.2f}%, "
                        f"{ratio:.1f}x lower cost per query",
            transform=ax.transAxes, fontsize=8, color="#555555", ha="left")
    fig.tight_layout()
    fig.savefig(Path(out) / "Figure_15_nonlinear.pdf")
    plt.close(fig)


def main(out="../CompMech_submission_ready", colors=("#1b7837", "#2166ac", "#d6604d", "#e08214")):
    plt.rcParams.update({"font.size": 10, "font.family": "serif",
                         "axes.grid": True, "grid.alpha": 0.3, "lines.markersize": 7})
    L = build_ledger()
    plot(L, out, colors)
    for k, v in L.items():
        if k != "_meta":
            print(f"  {k:22s} err {v['err']:.4f}  {v['ms']:8.2f} ms"
                  + (f"  break-even {v['break_even_queries']:.0f}" if "break_even_queries" in v else ""))
    print("wrote Figure_15_nonlinear.pdf and results_revision/nonlinear/ledger.json")


if __name__ == "__main__":
    main()
