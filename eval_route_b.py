#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_route_b.py

Capstone evaluation for a trained Route-B surrogate checkpoint:

  1. Field accuracy vs the res-113 FEM ground truth (displacement, stress).
  2. Uncertainty-propagation demonstration: treating the random E(x) ensemble
     as the source of uncertainty, compare the surrogate's predicted
     distribution of a quantity of interest (total strain energy Pi and peak
     von Mises stress) against the Monte-Carlo FEM reference distribution --
     ensemble mean/std and per-sample correlation.
  3. Resource ledger row: parameter count, on-disk model size, inference
     throughput, and peak memory.

    python eval_route_b.py --ckpt runs/route_b/m64/spectral_e_wpde0.0001_seed111.pt \
        --data data/hetero_field.npz
"""

import argparse
import json
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from route_b import (
    build_model, load_data, hetero_strain_stress, strain_energy_field,
)
from cmame_extended_study import PhysicsConfig, get_device, relative_l2


def von_mises_max(u, E, h):
    """Peak in-plane von Mises stress per sample (variable-coefficient)."""
    _, sig = hetero_strain_stress(u, E, h)
    sxx, syy, sxy = sig[..., 0], sig[..., 1], sig[..., 2]
    vm = torch.sqrt(torch.clamp(sxx**2 - sxx*syy + syy**2 + 3*sxy**2, min=0.0))
    return vm.amax(dim=(1, 2))                      # [N]


def dist_stats(pred, true):
    """Compare two per-sample QoI vectors: ensemble mean/std + correlation."""
    p, t = pred.numpy(), true.numpy()
    corr = float(np.corrcoef(p, t)[0, 1])
    return {
        "ref_mean": float(t.mean()), "sur_mean": float(p.mean()),
        "ref_std": float(t.std()), "sur_std": float(p.std()),
        "mean_rel_err": float(abs(p.mean() - t.mean()) / (abs(t.mean()) + 1e-12)),
        "std_rel_err": float(abs(p.std() - t.std()) / (abs(t.std()) + 1e-12)),
        "per_sample_corr": corr,
        "per_sample_mae_rel": float(np.mean(np.abs(p - t)) / (np.mean(np.abs(t)) + 1e-12)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/hetero_field.npz")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = get_device(args.device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    phys = PhysicsConfig(res=a["res"], nu=a["nu"])
    model = build_model(a["model"], phys, Namespace(**a)).to(device)
    model.load_state_dict(ck["state"]); model.eval()
    h = phys.h

    _, _, _, f_te, E_te, u_te = load_data(args.data)

    # ---- inference (throughput + peak memory) ----
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    preds = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, f_te.shape[0], 128):
            preds.append(model(f_te[i:i+128].to(device), E_te[i:i+128].to(device)).cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    up = torch.cat(preds, 0)
    thr = f_te.shape[0] / dt
    peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else float("nan")

    # ---- field accuracy ----
    eps_p, sig_p = hetero_strain_stress(up, E_te, h)
    eps_t, sig_t = hetero_strain_stress(u_te, E_te, h)
    acc = {
        "rel_l2_u": relative_l2(up, u_te),
        "rel_l2_stress": relative_l2(sig_p, sig_t),
    }

    # ---- UQ propagation: QoI distributions ----
    Pi_pred = strain_energy_field(up, E_te, h)
    Pi_true = strain_energy_field(u_te, E_te, h)
    vm_pred = von_mises_max(up, E_te, h)
    vm_true = von_mises_max(u_te, E_te, h)
    uq = {"strain_energy": dist_stats(Pi_pred, Pi_true),
          "peak_von_mises": dist_stats(vm_pred, vm_true)}

    n_params = sum(p.numel() for p in model.parameters())
    row = {
        "ckpt": args.ckpt, "model": a["model"], "modes": a.get("modes"),
        "cos_modes": a.get("cos_modes"), "w_pde": a.get("w_pde"),
        "accuracy": acc,
        "uq": uq,
        "resources": {
            "n_params": n_params,
            "disk_MB_fp32": n_params * 4 / 1e6,
            "throughput_samples_per_s": thr,
            "ms_per_sample": 1000.0 / thr,
            "peak_gpu_mem_MB": peak_mb,
            "device": device.type,
        },
    }
    print(json.dumps(row, indent=2))
    out = args.out or (Path(args.ckpt).with_suffix("").as_posix() + "_eval.json")
    json.dump(row, open(out, "w"), indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
