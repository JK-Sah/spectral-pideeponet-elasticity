#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_linear_sel_summaries.py

Derived files the figure scripts read, rebuilt from the per-model selected
results in results_revision/r1/w/linear_sel (select_per_model.py --assemble):

  capacity_sweep.json         per trunk capacity and model: mean/std displacement
                              error over seeds 42-44, residual weight, parameters
  canonical_seed_metrics.json per-seed metrics of the plain physics-informed
                              branch at M = 16 (Table 6, Fig. 6)

    python build_linear_sel_summaries.py
"""
import json
from pathlib import Path

L = Path("results_revision/r1/w/linear_sel")

cap = {}
for M in (16, 32, 64):
    d = json.load(open(L / f"canonical_M{M}.json"))
    cap[f"M{M}"] = {}
    for tag, a in d["aggregate"].items():
        r = next(r for r in d["runs"] if r["model"] == tag)
        cap[f"M{M}"][tag] = dict(mean=a["mean"], std=a["std"], w_pde=a["w_pde"],
                                 n_params=r["n_params"])
    cap[f"M{M}"]["closed_form"] = dict(mean=d["closed_form"]["test_error"], std=0.0,
                                       n_params=d["closed_form"]["n_params"])
(L / "capacity_sweep.json").write_text(json.dumps(cap, indent=2))

m = json.load(open(L / "canonical_metrics_all_M16.json"))["pi_spectral_plain"]["runs"]
seed = {str(r["seed"]): dict(disp=r["disp_l2"], strain=r["strain_l2"], stress=r["stress_l2"],
                             energy=r["energy_err"], pde_mse=r["pde_mse"]) for r in m}
(L / "canonical_seed_metrics.json").write_text(json.dumps(dict(
    config="canonical, trunk M=16, plain physics-informed branch at its validation-chosen "
           "residual weight, seeds 42/43/44", per_seed=seed), indent=2))
print("wrote", L / "capacity_sweep.json", "and", L / "canonical_seed_metrics.json")
