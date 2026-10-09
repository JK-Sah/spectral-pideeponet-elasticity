#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aux_linear_v2.py

The linear experiments outside the headline comparison -- Poisson-ratio sweep,
the residual-scaling check near incompressibility, the trunk-capacity
out-of-distribution ablation, the capacity-matched FNO, and the non-sine
forcing families -- re-run under the canonical protocol of canonical_linear.py.

What changed relative to revision_chunk_runner.py, which produced the earlier
numbers:
  * checkpoint selected on a validation set, the test set evaluated once
    (the earlier runner kept the epoch with the lowest TEST error);
  * the canonical configuration throughout -- 3000 training pairs, Adam 1e-3,
    weight decay 1e-6, step schedule halving at the midpoint, 800 epochs
    (600 for the FNO), residual weight 1e-4 -- instead of a 1000-sample,
    differently scheduled setting;
  * three initialization seeds (42, 43, 44) instead of one.

    python aux_linear_v2.py --exp sine_nu0.49_pi --out_dir results_revision/aux
    python aux_linear_v2.py --list
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch

import cmame_extended_study as st
from cmame_extended_study import (PhysicsConfig, build_modes, SpectralPIDeepONet,
                                  FNO2dElasticity, make_dataset, set_seed, get_device,
                                  evaluate_model, count_params)
from canonical_linear import CFG, AnchoredSpectral, evaluate

SEEDS = [42, 43, 44]
OOD_K = [16, 20, 24, 28, 32]


def experiments(w=None, w_anchored=None, w_fno=None):
    """w: residual weight of the plain physics-informed branch (default: canonical);
    w_anchored, w_fno: weights of the anchored branch and of the capacity-matched
    FNO, each chosen on validation for that model (default: w)."""
    w = CFG["w_pde"] if w is None else w
    wa = w if w_anchored is None else w_anchored
    wf = w if w_fno is None else w_fno
    lam0, mu0 = PhysicsConfig(nu=0.30).lame
    w_sres = w * (lam0 + 2 * mu0) ** 2   # scaled residual: equal to the plain one at nu = 0.3
    E = {}
    for nu in (0.30, 0.40, 0.45, 0.49, 0.499):
        for kind, wk in (("pi", w), ("data", 0.0)):
            E[f"sine_nu{nu}_{kind}"] = dict(data="sine", nu=nu, model="spectral", w_pde=wk,
                                           trunk=16, ood=(nu == 0.30))
    E["sine_nu0.499_anchored"] = dict(data="sine", nu=0.499, model="anchored", w_pde=wa, trunk=16)
    for nu in (0.49, 0.499):   # residual divided by (lambda + 2 mu)
        E[f"sine_nu{nu}_sres"] = dict(data="sine", nu=nu, model="spectral", w_pde=w_sres,
                                      trunk=16, scale_res=True)
    for kind, wk in (("pi", w), ("data", 0.0)):
        E[f"sine_t24_{kind}"] = dict(data="sine", nu=0.30, model="spectral", w_pde=wk,
                                     trunk=24, ood=True)
    E["sine_fno14"] = dict(data="sine", nu=0.30, model="fno14", w_pde=wf)
    for fam in ("bumps", "patch"):
        for kind, model, wk in (("pi", "spectral", w), ("data", "spectral", 0.0),
                                ("anchored", "anchored", wa), ("fno14", "fno14", wf)):
            E[f"{fam}_{kind}"] = dict(data=fam, nu=0.30, model=model, w_pde=wk, trunk=16)
    return E


def fno_matched_width(phys, modes=14, target=123_680):
    best = None
    for w in range(4, 65):
        p = count_params(FNO2dElasticity(phys, modes=modes, width=w))
        if best is None or abs(p - target) < best[1]:
            best = (w, abs(p - target))
    return best[0]


def load_data(cfg, phys, data_dir):
    if cfg["data"] == "sine":
        d = make_dataset(CFG["n_train"], CFG["n_test"], CFG["true_modes"], CFG["coeff_scale"],
                         phys, seed_train=CFG["seed_train"], seed_test=CFG["seed_test"])
        v = make_dataset(CFG["n_val"], 1, CFG["true_modes"], CFG["coeff_scale"], phys,
                         seed_train=CFG["seed_val"], seed_test=12345)
        return ((d["train_f"], d["train_u"]), (v["train_f"], v["train_u"]),
                (d["test_f"], d["test_u"]))
    z = np.load(Path(data_dir) / f"forcing_{cfg['data']}_v2.npz")
    t = lambda k: torch.tensor(z[k])
    return (t("f_tr"), t("u_tr")), (t("f_va"), t("u_va")), (t("f_te"), t("u_te"))


def train(model, tr, va, phys, dev, seed, w_pde, epochs, scale_res=False):
    """canonical_linear.train, with optional residual scaling by (lambda+2mu)."""
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 2),
                                          gamma=CFG["sched_gamma"])
    lam, mu = phys.lame
    scale = (lam + 2.0 * mu) if scale_res else 1.0
    f_tr, u_tr = tr[0].to(dev), tr[1].to(dev)
    n = f_tr.shape[0]
    best, best_ep, best_state = 9e9, -1, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        for b in range(0, n, CFG["batch"]):
            idx = perm[b:b + CFG["batch"]]
            fb = f_tr[idx]; pred = model(fb)
            loss = CFG["w_data"] * torch.mean((pred - u_tr[idx]) ** 2)
            if w_pde > 0:
                r = (st.fd_elasticity_operator(pred, phys) + fb[:, 1:-1, 1:-1, :]) / scale
                loss = loss + w_pde * torch.mean(r ** 2)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sch.step()
        if ep % CFG["eval_every"] == 0 or ep == epochs:
            e = evaluate(model, va[0], va[1], dev)
            if e < best:
                best, best_ep = e, ep
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best, best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="results_revision/aux")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=None, help="smoke tests only")
    ap.add_argument("--w_pde", type=float, default=None,
                    help="residual weight of the plain physics-informed branch (default: canonical)")
    ap.add_argument("--w_anchored", type=float, default=None,
                    help="residual weight of the anchored branch (default: --w_pde)")
    ap.add_argument("--w_fno", type=float, default=None,
                    help="residual weight of the capacity-matched FNO (default: --w_pde)")
    a = ap.parse_args()
    E = experiments(a.w_pde, a.w_anchored, a.w_fno)
    if a.list:
        print("\n".join(E)); return
    cfg = E[a.exp]
    dev = get_device(a.device)
    phys = PhysicsConfig(nu=cfg["nu"])
    tr, va, te = load_data(cfg, phys, a.data_dir)
    print(f"[{a.exp}] {cfg}  train={tr[0].shape[0]} val={va[0].shape[0]} "
          f"test={te[0].shape[0]} device={dev}")

    runs = []
    for seed in a.seeds:
        set_seed(seed)
        if cfg["model"] == "spectral":
            m = SpectralPIDeepONet(build_modes(cfg["trunk"]), phys, hidden=CFG["hidden"],
                                   depth=CFG["depth"])
        elif cfg["model"] == "anchored":
            m = AnchoredSpectral(build_modes(cfg["trunk"]), phys, tr[0], tr[1],
                                 CFG["hidden"], CFG["depth"])
        else:
            m = FNO2dElasticity(phys, modes=14, width=fno_matched_width(phys))
        m = m.to(dev)
        epochs = a.epochs or (CFG["epochs_fno"] if cfg["model"] == "fno14" else CFG["epochs"])
        t0 = time.time()
        m, e_val, ep = train(m, tr, va, phys, dev, seed, cfg["w_pde"], epochs,
                             cfg.get("scale_res", False))
        met = evaluate_model(m, te[0], te[1], phys, dev)
        rec = dict(seed=seed, val_error=e_val, selected_epoch=ep, n_params=count_params(m),
                   minutes=(time.time() - t0) / 60, **{k: float(v) for k, v in met.items()
                                                       if k != "infer_ms"})
        if cfg.get("ood"):
            rec["ood"] = {}
            for K in OOD_K:
                d = make_dataset(1, CFG["n_test"], K, CFG["coeff_scale"], phys,
                                 seed_train=1, seed_test=CFG["seed_test"])
                rec["ood"][str(K)] = evaluate(m, d["test_f"], d["test_u"], dev)
        runs.append(rec)
        print(f"  seed {seed}: val={e_val:.5f} test disp={rec['disp_l2']:.5f} "
              f"stress={rec['stress_l2']:.5f} ep*={ep} ({rec['minutes']:.1f} min)"
              + (f" ood={rec['ood']}" if "ood" in rec else ""))

    agg = {}
    for k in ("val_error", "disp_l2", "strain_l2", "stress_l2", "energy_err", "pde_mse"):
        v = [r[k] for r in runs]
        agg[k] = dict(mean=float(np.mean(v)), std=float(np.std(v)))
    if cfg.get("ood"):
        agg["ood"] = {K: dict(mean=float(np.mean([r["ood"][K] for r in runs])),
                              std=float(np.std([r["ood"][K] for r in runs])))
                      for K in runs[0]["ood"]}
    out = dict(exp=a.exp, config=cfg,
               protocol=dict(CFG, selection="min validation displacement error",
                             test_evaluations=1, seeds=a.seeds),
               runs=runs, aggregate=agg)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    Path(a.out_dir, f"{a.exp}.json").write_text(json.dumps(out, indent=2))
    print(f"[AGG {a.exp}] disp {agg['disp_l2']['mean']:.5f} +/- {agg['disp_l2']['std']:.5f}")
    print("AUX_DONE")


if __name__ == "__main__":
    main()
