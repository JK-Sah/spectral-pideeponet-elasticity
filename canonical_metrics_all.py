#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
canonical_metrics_all.py

The full metric suite for every model of the linear benchmark, from the saved
canonical checkpoints, so that the comparison table and the timing table
describe the same trained weights.

Why this is needed.  The comparison table's derived quantities -- strain,
stress, strain energy and the residual -- were still the ones measured before
the revision, from runs whose training size and schedule differed from the
canonical configuration.  Its caption nonetheless attributed them to the
canonical configuration.  The displacement column made this visible: the
canonical seeds give 9.24 / 8.53 / 8.45 per cent and the table reported 8.39,
which is none of them.  Rather than relabel the table, the numbers are
remeasured here.

Nothing is retrained.  Every figure comes from the checkpoints that
canonical_linear.py saved and that timing_linear_v3.py times, evaluated on the
same held-out test set, so accuracy and cost in the paper refer to one set of
weights.  Reported per model: the mean and standard deviation over the
initialization seeds, and each seed's value.

    python canonical_metrics_all.py --ckpt_dir runs/linear/ckpt --trunk 16
"""

import argparse, json
from pathlib import Path
import numpy as np
import torch

import cmame_extended_study as st
from cmame_extended_study import (PhysicsConfig, build_modes, SpectralPIDeepONet,
                                  FNO2dElasticity, make_dataset, evaluate_model,
                                  get_device)
from canonical_linear import (CFG, AnchoredSpectral, closed_form_readout,
                              ClosedFormModel)

TAGS = ("pi_spectral_plain", "pi_spectral_anchored", "data_only_spectral", "fno")


def boundary_linf(model, f, dev, bs=100):
    """Largest |u| predicted on the domain boundary; an exact-BC trunk gives 0."""
    model.eval()
    worst = 0.0
    with torch.no_grad():
        for i in range(0, f.shape[0], bs):
            p = model(f[i:i + bs].to(dev)).cpu().numpy()
            edges = np.concatenate([np.abs(p[:, 0, :, :]).ravel(),
                                    np.abs(p[:, -1, :, :]).ravel(),
                                    np.abs(p[:, :, 0, :]).ravel(),
                                    np.abs(p[:, :, -1, :]).ravel()])
            worst = max(worst, float(edges.max()))
    return worst


def agg(rows, key):
    v = [r[key] for r in rows]
    return dict(mean=float(np.mean(v)), std=float(np.std(v)),
                per_seed=[float(x) for x in v])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="runs/linear/ckpt")
    ap.add_argument("--trunk", type=int, default=16)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default="results_revision/linear/canonical_metrics_all.json")
    a = ap.parse_args()

    dev = get_device()
    phys = PhysicsConfig()
    modes = build_modes(a.trunk)
    d = make_dataset(CFG["n_train"], CFG["n_test"], CFG["true_modes"],
                     CFG["coeff_scale"], phys,
                     seed_train=CFG["seed_train"], seed_test=CFG["seed_test"])
    te_f, te_u = d["test_f"], d["test_u"]
    print(f"[setup] device={dev} trunk={a.trunk} n_test={te_f.shape[0]}")

    out = {"config": dict(CFG), "trunk": a.trunk,
           "source": "saved canonical checkpoints; no retraining",
           "test_seed": CFG["seed_test"]}

    # ---- closed-form read-out, refit from the same training data --------
    W, b = closed_form_readout(modes, st.make_grid(phys.res), d["train_f"], d["train_u"])
    cf = ClosedFormModel(W, b, modes, phys).to(dev).eval()
    m_cf = evaluate_model(cf, te_f, te_u, phys, dev)
    m_cf["boundary_linf"] = boundary_linf(cf, te_f, dev)
    m_cf["n_params"] = int(W.size + b.size)
    out["closed_form"] = m_cf
    print(f"[closed_form         ] disp {m_cf['disp_l2']:.6f}  strain {m_cf['strain_l2']:.6f}  "
          f"stress {m_cf['stress_l2']:.6f}  energy {m_cf['energy_err']:.6f}  "
          f"pde {m_cf['pde_mse']:.6e}  bnd {m_cf['boundary_linf']:.3e}")

    # ---- every trained model, from its checkpoint -----------------------
    for tag in TAGS:
        rows = []
        for seed in a.seeds:
            ck = Path(a.ckpt_dir) / f"{tag}_M{a.trunk}_seed{seed}.pt"
            if not ck.exists():
                print(f"[warn] missing {ck}")
                continue
            blob = torch.load(ck, map_location="cpu")
            if tag == "fno":
                m = FNO2dElasticity(phys, modes=CFG["fno_modes"],
                                    width=CFG["fno_width"], n_layers=CFG["fno_layers"])
            elif tag == "pi_spectral_anchored":
                m = AnchoredSpectral(modes, phys, d["train_f"], d["train_u"],
                                     CFG["hidden"], CFG["depth"])
            else:
                m = SpectralPIDeepONet(modes, phys, hidden=CFG["hidden"],
                                       depth=CFG["depth"])
            m.load_state_dict(blob["state_dict"])
            m = m.to(dev).eval()
            r = evaluate_model(m, te_f, te_u, phys, dev)
            r["boundary_linf"] = boundary_linf(m, te_f, dev)
            r["seed"] = seed
            r["selected_epoch"] = blob.get("selected_epoch", blob.get("epoch"))
            r["n_params"] = int(sum(p.numel() for p in m.parameters()))
            rows.append(r)
            print(f"[{tag:20s}] seed {seed}  disp {r['disp_l2']:.6f}  "
                  f"strain {r['strain_l2']:.6f}  stress {r['stress_l2']:.6f}  "
                  f"energy {r['energy_err']:.6f}  pde {r['pde_mse']:.6e}  "
                  f"bnd {r['boundary_linf']:.3e}")
        if not rows:
            continue
        out[tag] = dict(runs=rows, n_params=rows[0]["n_params"],
                        n_seeds=len(rows),
                        **{k: agg(rows, k) for k in ("disp_l2", "strain_l2",
                                                     "stress_l2", "energy_err",
                                                     "pde_mse", "boundary_linf")})

    print("\n[summary: mean +/- std over seeds]")
    hdr = f"{'model':22s} {'disp':>18s} {'strain':>18s} {'stress':>18s} {'energy':>18s} {'pde_mse':>14s}"
    print(hdr)
    for tag in TAGS:
        if tag not in out:
            continue
        v = out[tag]
        cells = "".join(
            f" {v[k]['mean']:9.6f}+-{v[k]['std']:.6f}"
            for k in ("disp_l2", "strain_l2", "stress_l2", "energy_err"))
        print(f"{tag:22s}{cells} {v['pde_mse']['mean']:14.6e}")

    # ---- the physics-informed vs data-only reduction the text quotes ----
    if "pi_spectral_plain" in out and "data_only_spectral" in out:
        red = {}
        for k in ("disp_l2", "strain_l2", "stress_l2", "energy_err", "pde_mse"):
            pi = out["pi_spectral_plain"][k]["mean"]
            do = out["data_only_spectral"][k]["mean"]
            red[k] = dict(pi=pi, data_only=do,
                          reduction_pct=100.0 * (pi - do) / do)
        out["pi_vs_data_only"] = red
        print("\n[physics-informed vs data-only, mean over seeds]")
        for k, v in red.items():
            print(f"  {k:12s} {v['data_only']:.6f} -> {v['pi']:.6f}  "
                  f"({v['reduction_pct']:+.1f}%)")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("\nwrote", a.out)
    print("CANONICALMETRICSALL_DONE")


if __name__ == "__main__":
    main()
