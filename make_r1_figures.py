#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_r1_figures.py

Figures 3, 4, 5, 9 and 10, redrawn from the revision's measurement files so that
every one of them describes the canonical, validation-selected models:

  Fig. 3/4  displacement and stress fields of the canonical physics-informed
            spectral DeepONet (M=16, seed 42) on the first test sample;
  Fig. 5    the three ablations under the canonical protocol (mean +/- s.d. over
            seeds 42-44; test error, with the validation error that selected it);
  Fig. 9    noise robustness of the canonical checkpoints;
  Fig. 10   out-of-distribution forcing for the canonical checkpoints.

Reads results_revision/r1/ablation/wpde_*.json (residual-weight sweep),
r1/w/ablation/*.json (trunk and size sweeps at the selected weight),
r1/w/aux_sel/sine_t24_pi.json, and the per-model selected checkpoints' extras
in r1/w/linear_sel/canonical_extras{.json,.snapshot.npz}.
Raises rather than falling back to hard-coded values.
"""
import glob, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from cmame_extended_study import PhysicsConfig, fd_strain_stress

RES = Path("results_revision")
R1 = RES / "r1"
OUT = Path("../CompMech_submission_ready")
OUT.mkdir(parents=True, exist_ok=True)   # the manuscript folder; created in a fresh checkout
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "savefig.bbox": "tight",
                     "savefig.dpi": 300, "font.family": "serif"})
COL = {"pi_spectral_plain": "#1f5aa6", "data_only_spectral": "#c0392b", "fno": "#27864a",
       "pi_spectral_anchored": "#7b3fa0"}
LAB = {"pi_spectral_plain": "PI-spectral DeepONet", "data_only_spectral": "Data-only spectral",
       "fno": "FNO", "pi_spectral_anchored": "PI-spectral, LS-anchored"}


# ---------------------------------------------------------------- Fig 5
def fig5():
    """Residual-weight sweep from r1/ablation; trunk and training-size sweeps at
    the selected weight (0.03) from r1/w/ablation, with the canonical point
    (M = 16, 3000 samples) taken from the 0.03 point of the weight sweep and
    M = 24 from the out-of-distribution ablation (same protocol and weight)."""
    W_SEL = 0.03
    rows = {"wpde": [], "trunk": [], "ntrain": []}
    for f in glob.glob(str(R1 / "ablation" / "wpde_*.json")):
        d = json.load(open(f))
        rows["wpde"].append((d["value"], d["test_mean"], d["test_std"], d["val_mean"]))
    for f in glob.glob(str(R1 / "w" / "ablation" / "*.json")):
        d = json.load(open(f))
        rows[d["kind"]].append((d["value"], d["test_mean"], d["test_std"], d["val_mean"]))
    canon = next(r for r in rows["wpde"] if abs(r[0] - W_SEL) < 1e-12)
    rows["trunk"].append((16,) + canon[1:]); rows["ntrain"].append((3000,) + canon[1:])
    t24 = json.load(open(R1 / "w" / "aux_sel" / "sine_t24_pi.json"))
    g = t24["aggregate"]
    rows["trunk"].append((24, g["disp_l2"]["mean"], g["disp_l2"]["std"], g["val_error"]["mean"]))
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.0))
    spec = (("wpde", r"Residual weight $w_\mathrm{PDE}$", True, W_SEL),
            ("trunk", r"Trunk capacity $M$", False, 16),
            ("ntrain", "Training samples", True, 3000))
    for ax, (k, xl, logx, cv) in zip(axs, spec):
        r = sorted(rows[k])
        if not r:
            raise SystemExit(f"no ablation results for {k}")
        x = np.array([a[0] for a in r]); m = np.array([a[1] for a in r]) * 100
        sd = np.array([a[2] for a in r]) * 100; v = np.array([a[3] for a in r]) * 100
        ax.errorbar(x, m, yerr=sd, marker="o", ms=4, capsize=3, color=COL["pi_spectral_plain"],
                    label="test (mean $\\pm$ s.d.)")
        ax.plot(x, v, ls="--", marker="s", ms=3, color="0.45", label="validation")
        ax.axvline(cv, color="0.75", lw=0.8, zorder=0)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xl); ax.set_yscale("log")
        from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel("Displacement error (%)")
    fig.subplots_adjust(wspace=0.28)
    axs[0].legend(fontsize=7.5, frameon=False)
    axs[0].set_title("(a) residual weight"); axs[1].set_title("(b) trunk capacity")
    axs[2].set_title("(c) training-set size")
    fig.savefig(OUT / "Figure_5_ablation_studies.pdf"); plt.close(fig)


# ---------------------------------------------------------------- Fig 9, 10
def fig9_10():
    ex = json.load(open(R1 / "w" / "linear_sel" / "canonical_extras.json"))
    for key, fn, xs, xl in (("noise", "Figure_9_noise_robustness.pdf",
                             [100 * s for s in ex["noise_sigmas"]],
                             "Body-force noise level (% of signal s.d.)"),
                            ("ood", "Figure_10_ood_generalization.pdf", ex["ood_K"],
                             "Forcing modes $K$ in the test set")):
        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        for tag in ("pi_spectral_anchored", "pi_spectral_plain", "fno", "data_only_spectral"):
            m = 100 * np.array(ex[key][tag]["mean"]); s = 100 * np.array(ex[key][tag]["std"])
            ax.plot(xs, m, marker="o", ms=3.5, color=COL[tag], label=LAB[tag])
            ax.fill_between(xs, m - s, m + s, color=COL[tag], alpha=0.15, lw=0)
        if key == "ood":
            ax.axvline(16, color="0.5", ls="--", lw=0.8)
        ax.set_yscale("log"); ax.set_xlabel(xl); ax.set_ylabel("Displacement error (%)")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7, frameon=False)
        fig.savefig(OUT / fn); plt.close(fig)


# ---------------------------------------------------------------- Fig 3, 4
def fig3_4():
    z = np.load(R1 / "w" / "linear_sel" / "canonical_extras.snapshot.npz")
    ut, up = z["u_true"], z["u_pred_pi_spectral_plain"]
    fig, axs = plt.subplots(2, 3, figsize=(9.6, 5.6))
    for r, (c, name) in enumerate(((0, "u_x"), (1, "u_y"))):
        vmax = np.abs(ut[..., c]).max()
        for col, (fld, title, cmap, lim) in enumerate((
                (ut[..., c], f"exact ${name}$", "RdBu_r", vmax),
                (up[..., c], f"predicted ${name}$", "RdBu_r", vmax),
                (np.abs(up[..., c] - ut[..., c]), f"$|$error$|$ in ${name}$", "magma", None))):
            ax = axs[r, col]
            im = ax.imshow(fld, origin="lower", extent=(0, 1, 0, 1), cmap=cmap,
                           vmin=-lim if lim else None, vmax=lim)
            ax.set_title(title); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "Figure_3_displacement_field_comparison.png", dpi=300); plt.close(fig)

    phys = PhysicsConfig()
    t = lambda a: torch.tensor(a[None], dtype=torch.float32)
    _, s_true = fd_strain_stress(t(ut), phys); _, s_pred = fd_strain_stress(t(up), phys)
    s_true, s_pred = s_true[0].numpy(), s_pred[0].numpy()
    names = (r"\sigma_{xx}", r"\sigma_{yy}", r"\sigma_{xy}")
    fig, axs = plt.subplots(2, 3, figsize=(9.6, 5.6))
    for c in range(3):
        vmax = np.abs(s_true[..., c]).max()
        for r, (fld, lab) in enumerate(((s_true[..., c], "exact"), (s_pred[..., c], "predicted"))):
            ax = axs[r, c]
            im = ax.imshow(fld, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax)
            ax.set_title(f"{lab} ${names[c]}$"); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "Figure_4_predicted_stress_components.png", dpi=300); plt.close(fig)


if __name__ == "__main__":
    fig5(); fig9_10(); fig3_4()
    print("wrote Figures 3, 4, 5, 9, 10")
