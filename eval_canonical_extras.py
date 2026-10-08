#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_canonical_extras.py

Noise robustness, out-of-distribution forcing, and the field snapshots of
Figs. 3-4, all evaluated from the saved canonical checkpoints (trunk M=16,
seeds 42-44) -- the same trained weights as the headline tables.  Nothing is
retrained, so these figures now describe the canonical models instead of an
earlier, differently configured run.

  noise: f + sigma * std(f) * eta, eta ~ N(0, I) with a fixed seed,
         sigma in {0, 0.01, 0.02, 0.05, 0.10, 0.20}
  OOD:   test sets drawn with K in {16, 20, 24, 28, 32} sine modes (seed 999)

    python eval_canonical_extras.py --ckpt_dir runs/linear/ckpt --out results_revision/linear/canonical_extras.json
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

from cmame_extended_study import (PhysicsConfig, build_modes, SpectralPIDeepONet,
                                  FNO2dElasticity, make_dataset, relative_l2)
from canonical_linear import CFG, AnchoredSpectral

SIGMAS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
KS = [16, 20, 24, 28, 32]
MODELS = ["pi_spectral_plain", "data_only_spectral", "fno", "pi_spectral_anchored"]


def build(tag, phys, tr):
    modes = build_modes(16)
    if tag == "fno":
        return FNO2dElasticity(phys, modes=CFG["fno_modes"], width=CFG["fno_width"],
                               n_layers=CFG["fno_layers"])
    if tag == "pi_spectral_anchored":
        return AnchoredSpectral(modes, phys, tr[0], tr[1], CFG["hidden"], CFG["depth"])
    return SpectralPIDeepONet(modes, phys, hidden=CFG["hidden"], depth=CFG["depth"])


@torch.no_grad()
def predict(m, f, bs=100):
    m.eval()
    return torch.cat([m(f[i:i + bs]) for i in range(0, f.shape[0], bs)], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--snapshot_index", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    phys = PhysicsConfig()
    d = make_dataset(CFG["n_train"], CFG["n_test"], CFG["true_modes"], CFG["coeff_scale"],
                     phys, seed_train=CFG["seed_train"], seed_test=CFG["seed_test"])
    tr = (d["train_f"], d["train_u"]); f_te, u_te = d["test_f"], d["test_u"]
    g = torch.Generator().manual_seed(0)
    eta = torch.randn(f_te.shape, generator=g)
    f_std = f_te.std().item()
    ood = {K: make_dataset(1, CFG["n_test"], K, CFG["coeff_scale"], phys,
                           seed_train=1, seed_test=CFG["seed_test"]) for K in KS}

    res = {"noise_sigmas": SIGMAS, "ood_K": KS, "noise": {}, "ood": {}}
    snap = {"u_true": u_te[a.snapshot_index].numpy(), "f": f_te[a.snapshot_index].numpy()}
    for tag in MODELS:
        nz, oo = [], []
        for seed in a.seeds:
            m = build(tag, phys, tr)
            ck = torch.load(Path(a.ckpt_dir) / f"{tag}_M16_seed{seed}.pt", map_location="cpu")
            m.load_state_dict(ck["state_dict"])
            nz.append([relative_l2(predict(m, f_te + s * f_std * eta), u_te) for s in SIGMAS])
            oo.append([relative_l2(predict(m, ood[K]["test_f"]), ood[K]["test_u"]) for K in KS])
            if seed == a.seeds[0]:
                snap[f"u_pred_{tag}"] = predict(m, f_te[a.snapshot_index:a.snapshot_index + 1])[0].numpy()
        nz, oo = np.array(nz), np.array(oo)
        res["noise"][tag] = dict(mean=nz.mean(0).tolist(), std=nz.std(0).tolist())
        res["ood"][tag] = dict(mean=oo.mean(0).tolist(), std=oo.std(0).tolist())
        print(f"[{tag}] noise {np.round(nz.mean(0), 4).tolist()}  ood {np.round(oo.mean(0), 4).tolist()}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    np.savez_compressed(Path(a.out).with_suffix(".snapshot.npz"), **snap)
    print("EXTRAS_DONE")


if __name__ == "__main__":
    main()
