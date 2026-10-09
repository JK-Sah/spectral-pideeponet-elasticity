#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hetero_anchored.py

Does the least-squares anchor still help when the operator varies?

On the manufactured benchmark, initializing the spectral branch at the
closed-form least-squares optimum takes it from 8.7% to 0.43% and past both
classical baselines, which is why the manuscript attributes the plain
branch's deficit to optimization rather than representation.  That claim was
tested only where a single closed-form map exists, because the operator is
the same for every sample.  Here it is not: each sample carries its own
modulus field E(x), so an anchor must be fitted jointly over the forcing
features and the material features, and the linear-optimality that makes the
construction work on the manufactured family no longer holds.

This runs the test rather than assuming the answer.  The two branches are
identical apart from where optimization starts, and both use the held-out
protocol: a validation split carved from the training pool, checkpoint
selection on validation only, and the test set evaluated once at the end.

    python hetero_anchored.py --data data/hetero_field.npz --seeds 42 43 44
"""

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from cmame_extended_study import (PhysicsConfig, build_modes, make_grid,
                                  project_onto_sine, set_seed, get_device)
from route_b import (SpectralPIDeepONetE, build_cos_modes, load_data,
                     evaluate, hetero_residual_mse, project_onto_cosine)


class AnchoredSpectralE(SpectralPIDeepONetE):
    """SpectralPIDeepONetE whose branch starts at the closed-form optimum.

    The linear path is the least-squares map from the concatenated forcing and
    material features to the trunk coefficients; the MLP output layer is
    zero-initialized so training begins exactly at that map.  This is the same
    construction used on the manufactured benchmark, generalized to features
    that include E(x).
    """

    def __init__(self, modes, cos_modes, phys, f_tr, E_tr, u_tr,
                 hidden=192, depth=4, ridge=1e-8):
        super().__init__(modes, cos_modes, phys, hidden, depth)
        D = 2 * self.M + self.P
        self.lin = nn.Linear(D, 2 * self.M)
        last = [m for m in self.branch.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)

        with torch.no_grad():
            feat_f = project_onto_sine(f_tr, modes, self.grid)
            logE = torch.log(E_tr.clamp_min(1e-6))
            feat_E = project_onto_cosine(logE, cos_modes, self.grid)
            X = torch.cat([feat_f, feat_E], dim=-1).numpy().astype(np.float64)
            M = self.M
            Yx = project_onto_sine(u_tr[..., 0:1].repeat(1, 1, 1, 2), modes,
                                   self.grid).numpy()[:, :M]
            Yy = project_onto_sine(u_tr[..., 1:2].repeat(1, 1, 1, 2), modes,
                                   self.grid).numpy()[:, :M]
            Y = np.concatenate([Yx, Yy], axis=1).astype(np.float64)
            Xb = np.hstack([X, np.ones((X.shape[0], 1))])
            A = Xb.T @ Xb + ridge * np.eye(Xb.shape[1])
            try:
                Wb = np.linalg.solve(A, Xb.T @ Y)
                if not np.all(np.isfinite(Wb)):
                    raise np.linalg.LinAlgError("non-finite normal-equation solve")
            except np.linalg.LinAlgError:
                # rank-deficient features: fall back to a least-norm solution
                Wb = np.linalg.lstsq(Xb, Y, rcond=None)[0]
            self.lin.weight.copy_(torch.tensor(Wb[:-1].T, dtype=torch.float32))
            self.lin.bias.copy_(torch.tensor(Wb[-1], dtype=torch.float32))

    def forward(self, f_grid, E):
        feat_f = project_onto_sine(f_grid, self.modes, self.grid)
        logE = torch.log(E.clamp_min(1e-6))
        feat_E = project_onto_cosine(logE, self.cos_modes, self.grid)
        feat = torch.cat([feat_f, feat_E], dim=-1)
        coeff = self.branch(feat) + self.lin(feat)
        cx, cy = coeff[:, :self.M], coeff[:, self.M:]
        ux = torch.einsum("bm,ijm->bij", cx, self.phi)
        uy = torch.einsum("bm,ijm->bij", cy, self.phi)
        return torch.stack([ux, uy], dim=-1)


def train_one(model, tr, va, phys, dev, args, seed):
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 2),
                                          gamma=0.5)
    f_tr, E_tr, u_tr = (x.to(dev) for x in tr)
    n = f_tr.shape[0]
    best, best_ep, best_state = 9e9, -1, None
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        for b in range(0, n, args.batch):
            idx = perm[b:b + args.batch]
            pred = model(f_tr[idx], E_tr[idx])
            loss = torch.mean((pred - u_tr[idx]) ** 2)
            if args.w_pde > 0:
                loss = loss + args.w_pde * hetero_residual_mse(
                    pred, f_tr[idx], E_tr[idx], phys.h)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sch.step()
        if ep % args.eval_every == 0 or ep == args.epochs:
            m = evaluate(model, va[0], va[1], va[2], phys, dev)
            e = float(m["rel_l2_u"])
            if e < best:
                best, best_ep = e, ep
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best, best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/hetero_field.npz")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--cos_modes", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w_pde", type=float, default=1e-4)
    ap.add_argument("--eval_every", type=int, default=20)
    ap.add_argument("--val_frac", type=float, default=0.20)
    ap.add_argument("--split_seed", type=int, default=42,
                    help="seed of the train/validation split, fixed across initialization seeds")
    ap.add_argument("--res", type=int, default=29)
    ap.add_argument("--nu", type=float, default=0.30)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results_revision/hetero/anchored.json")
    a = ap.parse_args()

    dev = get_device(a.device)
    phys = PhysicsConfig(res=a.res, nu=a.nu)
    modes, cos_modes = build_modes(a.modes), build_cos_modes(a.cos_modes)
    f_tr, E_tr, u_tr, f_te, E_te, u_te = load_data(a.data)

    g = np.random.default_rng(a.split_seed)
    perm = g.permutation(f_tr.shape[0])
    nval = int(round(a.val_frac * f_tr.shape[0]))
    vi, ti = perm[:nval], perm[nval:]
    tr = (f_tr[ti], E_tr[ti], u_tr[ti])
    va = (f_tr[vi], E_tr[vi], u_tr[vi])
    print(f"[split] train={len(ti)} val={len(vi)} test={f_te.shape[0]}")

    out = {"config": vars(a), "runs": []}
    for seed in a.seeds:
        for tag in ("plain", "anchored"):
            set_seed(seed)
            if tag == "anchored":
                m = AnchoredSpectralE(modes, cos_modes, phys, tr[0], tr[1], tr[2],
                                      a.hidden, a.depth).to(dev)
            else:
                m = SpectralPIDeepONetE(modes, cos_modes, phys, a.hidden,
                                        a.depth).to(dev)
            npar = sum(p.numel() for p in m.parameters())
            t0 = time.time()
            m, e_val, ep = train_one(m, tr, va, phys, dev, a, seed)
            met = evaluate(m, f_te, E_te, u_te, phys, dev)
            rec = dict(model=tag, seed=seed, val_error=e_val, selected_epoch=int(ep),
                       test_error=float(met["rel_l2_u"]), n_params=int(npar),
                       minutes=(time.time() - t0) / 60)
            out["runs"].append(rec)
            print(f"[{tag:9s} s{seed}] val={e_val:.4f} test={rec['test_error']:.4f} "
                  f"ep*={ep} params={npar} ({rec['minutes']:.1f} min)")

    for tag in ("plain", "anchored"):
        es = [r["test_error"] for r in out["runs"] if r["model"] == tag]
        if es:
            out.setdefault("aggregate", {})[tag] = dict(
                mean=float(np.mean(es)), std=float(np.std(es)), values=es)
            print(f"[AGG {tag:9s}] {np.mean(es):.4f} +/- {np.std(es):.4f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)
    print("HETEROANCHORED_DONE")


if __name__ == "__main__":
    main()
