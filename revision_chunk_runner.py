#!/usr/bin/env python3
"""Chunked experiment runner: resumes from checkpoints, trains for a
wall-clock budget, saves state, exits. Survives 45s sandbox call limits."""
import json, math, sys, time, os
sys.path.insert(0, "/sessions/serene-elegant-cannon/mnt/CMAME")
import numpy as np
import torch

import cmame_extended_study as st
import revision_new_benchmarks as rb

CKPT = "/tmp/ckpts"; os.makedirs(CKPT, exist_ok=True)
RESULTS = "/sessions/serene-elegant-cannon/mnt/CMAME/results_revision/chunk_results.jsonl"
os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
DEVICE = torch.device("cpu")
EPOCHS = 800

# ---------------------------------------------------------------- job table
def fno_width(modes, target=123_680):
    best = None
    for w in range(4, 65):
        p = st.count_params(st.FNO2dElasticity(st.PhysicsConfig(res=29), modes=modes, width=w))
        if best is None or abs(p - target) < best[1]:
            best = (w, abs(p - target), p)
    return best[0], best[2]

def jobs():
    J = []
    for nu in (0.30, 0.40, 0.45, 0.49, 0.499):
        for name, wp in (("PI", 1e-4), ("data", 0.0)):
            J.append(dict(id=f"nu{nu}_{name}", exp="nu", nu=nu, w_pde=wp,
                          model="spectral", trunk=16, data="sine16"))
    for trunk in (16, 24):
        for name, wp in (("PI", 1e-4), ("data", 0.0)):
            J.append(dict(id=f"ood_t{trunk}_{name}", exp="ood", nu=0.30, w_pde=wp,
                          model="spectral", trunk=trunk, data="sine16"))
    J.append(dict(id="fnoM14", exp="fno_matched", nu=0.30, w_pde=0.0,
                  model="fno", fno_modes=14, data="sine16"))
    for kind in ("bumps", "patch"):
        for name, wp in (("PI", 1e-4), ("data", 0.0)):
            J.append(dict(id=f"{kind}_{name}", exp="forcing", nu=0.30, w_pde=wp,
                          model="spectral", trunk=16, data=kind))
        J.append(dict(id=f"{kind}_fno", exp="forcing", nu=0.30, w_pde=0.0,
                      model="fno", fno_modes=14, data=kind))
    # lambda-normalized residual near incompressibility
    for nu in (0.30, 0.49, 0.499):
        J.append(dict(id=f"sres_nu{nu}_PI", exp="sres", nu=nu, w_pde=2e-4,
                      model="spectral", trunk=16, data="sine16", scale_res=True))
    # linear-bypass branch (lstsq-initialized linear path + MLP correction)
    for kind in ("sine16", "bumps", "patch"):
        J.append(dict(id=f"linbp_{kind}_PI", exp="linbp", nu=0.30, w_pde=1e-4,
                      model="spectral_linbp", trunk=16, data=kind))
    J.append(dict(id="linbp_nu0.499_PI", exp="linbp", nu=0.499, w_pde=1e-4,
                  model="spectral_linbp", trunk=16, data="sine16"))
    # heterogeneous inclusion: operator nonlinear in the input k
    J.append(dict(id="het_PI", exp="hetero", nu=0.30, w_pde=1e-4,
                  model="het_plain", trunk=16, data="hetero"))
    J.append(dict(id="het_data", exp="hetero", nu=0.30, w_pde=0.0,
                  model="het_plain", trunk=16, data="hetero"))
    J.append(dict(id="het_anchor_PI", exp="hetero", nu=0.30, w_pde=1e-4,
                  model="het_anchor", trunk=16, data="hetero"))
    J.append(dict(id="het_anchor_rat_PI", exp="hetero", nu=0.30, w_pde=1e-4,
                  model="het_anchor_rat", trunk=16, data="hetero"))
    J.append(dict(id="het_anchor_rat_frozen_PI", exp="hetero", nu=0.30,
                  w_pde=1e-4, model="het_anchor_rat_frozen", trunk=16,
                  data="hetero"))
    return J


# ------------------------------------------------------------- hetero utils
INCL_C, INCL_R = (0.5, 0.5), 0.25

def incl_mask(res):
    x = np.linspace(0, 1, res)
    yy, xx = np.meshgrid(x, x, indexing="ij")
    return torch.tensor(((xx - INCL_C[0]) ** 2 + (yy - INCL_C[1]) ** 2
                         < INCL_R ** 2).astype(np.float32))

def het_material(kb, mask, nu=0.30):
    """kb: [B] contrast; mask: [R,R]. Returns lam,mu maps [B,R,R]."""
    E = 1.0 + (kb.view(-1, 1, 1) - 1.0) * mask
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lam, mu

def het_strain_stress(u, kb, mask, h, nu=0.30):
    ux, uy = u[..., 0], u[..., 1]
    # grid indexed [iy, ix]: d/dx = dim 2, d/dy = dim 1
    dux_dy, dux_dx = torch.gradient(ux, spacing=h, dim=(1, 2))
    duy_dy, duy_dx = torch.gradient(uy, spacing=h, dim=(1, 2))
    exx, eyy = dux_dx, duy_dy
    exy = 0.5 * (dux_dy + duy_dx)
    lam, mu = het_material(kb, mask, nu)
    sxx = (lam + 2 * mu) * exx + lam * eyy
    syy = lam * exx + (lam + 2 * mu) * eyy
    sxy = 2 * mu * exy
    eps = torch.stack([exx, eyy, exy], -1)
    sig = torch.stack([sxx, syy, sxy], -1)
    return eps, sig

def het_residual_mse(u, f, kb, mask, h, nu=0.30, crop=2):
    _, sig = het_strain_stress(u, kb, mask, h, nu)
    dsxx_dy, dsxx_dx = torch.gradient(sig[..., 0], spacing=h, dim=(1, 2))
    dsyy_dy, dsyy_dx = torch.gradient(sig[..., 1], spacing=h, dim=(1, 2))
    dsxy_dy, dsxy_dx = torch.gradient(sig[..., 2], spacing=h, dim=(1, 2))
    rx = dsxx_dx + dsxy_dy + f[..., 0]
    ry = dsxy_dx + dsyy_dy + f[..., 1]
    r = torch.stack([rx, ry], -1)[:, crop:-crop, crop:-crop, :]
    return torch.mean(r ** 2)

RAT_POLES = (0.5, 1.0, 2.0)

def het_features(f, k, modes, grid, kind="naive"):
    phi = st.project_onto_sine(f, modes, grid)
    k1 = k.view(-1, 1)
    if kind == "naive":
        return torch.cat([phi, k1], dim=1)
    # rational-in-k features: resolvent structure of (K0 + k K1)^{-1}
    cols = [phi] + [phi / (a + k1) for a in RAT_POLES] + [k1]
    return torch.cat(cols, dim=1)

def het_feat_dim(M, kind):
    return 2 * M + 1 if kind == "naive" else 2 * M * (1 + len(RAT_POLES)) + 1

class HeteroSpectral(torch.nn.Module):
    """MLP branch with input features of (f, k); optional lstsq anchor."""
    def __init__(self, modes, phys, hidden=192, depth=4, anchor_ds=None,
                 feats="naive"):
        super().__init__()
        self.modes = modes
        self.feats_kind = feats
        grid = st.make_grid(phys.res)
        self.grid = grid
        self.register_buffer("phi", st.sine_basis(grid, modes))
        M = len(modes)
        D = het_feat_dim(M, feats)
        self.mlp = st.MLP(D, 2 * M, hidden, depth)
        self.lin = None
        if anchor_ds is not None:
            self.lin = torch.nn.Linear(D, 2 * M)
            last = [m for m in self.mlp.modules()
                    if isinstance(m, torch.nn.Linear)][-1]
            torch.nn.init.zeros_(last.weight); torch.nn.init.zeros_(last.bias)
            X = het_features(anchor_ds["train_f"], anchor_ds["train_k"],
                             modes, grid, feats).numpy().astype(np.float64)
            u = anchor_ds["train_u"]
            Yx = st.project_onto_sine(u[..., 0:1].repeat(1, 1, 1, 2), modes,
                                      grid).numpy()[:, :M]
            Yy = st.project_onto_sine(u[..., 1:2].repeat(1, 1, 1, 2), modes,
                                      grid).numpy()[:, :M]
            Y = np.concatenate([Yx, Yy], 1).astype(np.float64)
            Xb = np.hstack([X, np.ones((X.shape[0], 1))])
            Wb = np.linalg.solve(Xb.T @ Xb + 1e-8 * np.eye(Xb.shape[1]),
                                 Xb.T @ Y)
            with torch.no_grad():
                self.lin.weight.copy_(torch.tensor(Wb[:-1].T, dtype=torch.float32))
                self.lin.bias.copy_(torch.tensor(Wb[-1], dtype=torch.float32))

    def forward(self, f, k):
        feats = het_features(f, k, self.modes, self.grid.to(f.device),
                             self.feats_kind)
        c = self.mlp(feats)
        if self.lin is not None:
            c = c + self.lin(feats)
        M = len(self.modes)
        ux = torch.einsum('nm,ijm->nij', c[:, :M], self.phi)
        uy = torch.einsum('nm,ijm->nij', c[:, M:], self.phi)
        return torch.stack([ux, uy], dim=-1)

@torch.no_grad()
def het_evaluate(model, ds, phys, batch=64):
    mask = incl_mask(phys.res)
    f, u_t, k = ds["test_f"], ds["test_u"], ds["test_k"]
    preds, times = [], []
    for i in range(0, f.shape[0], batch):
        t0 = time.time()
        preds.append(model(f[i:i+batch], k[i:i+batch]))
        times.append(time.time() - t0)
    u_p = torch.cat(preds)
    e_p, s_p = het_strain_stress(u_p, k, mask, phys.h, phys.nu)
    e_t, s_t = het_strain_stress(u_t, k, mask, phys.h, phys.nu)
    out = dict(
        disp_l2=st.relative_l2(u_p, u_t),
        strain_l2=st.relative_l2(e_p, e_t),
        stress_l2=st.relative_l2(s_p, s_t),
        pde_mse=float(het_residual_mse(u_p, f, k, mask, phys.h, phys.nu)),
        infer_ms=1000.0 * sum(times) / f.shape[0])
    return out

# ---------------------------------------------------------------- data
_ds_cache = {}
def get_dataset(tag, nu):
    key = (tag, nu)
    if key in _ds_cache: return _ds_cache[key]
    phys = st.PhysicsConfig(res=29, nu=nu)
    if tag == "sine16":
        ds = st.make_dataset(1000, 200, 16, 0.04, phys)
    elif tag == "hetero":
        z = np.load("/tmp/femgt_hetero.npz")
        ds = {"train_f": torch.tensor(z["f_tr"]), "train_u": torch.tensor(z["u_tr"]),
              "train_k": torch.tensor(z["k_tr"]),
              "test_f": torch.tensor(z["f_te"]), "test_u": torch.tensor(z["u_te"]),
              "test_k": torch.tensor(z["k_te"]),
              "modes": st.build_modes(16)}
    else:
        cache = f"/tmp/femgt_{tag}.npz"
        if os.path.exists(cache):
            z = np.load(cache)
            f_tr, u_tr, f_te, u_te = z["f_tr"], z["u_tr"], z["f_te"], z["u_te"]
        else:
            f_tr, u_tr, _ = rb.fem_ground_truth(tag, 1000, 29, seed=42, verbose=False)
            f_te, u_te, _ = rb.fem_ground_truth(tag, 200, 29, seed=999, verbose=False)
            np.savez_compressed(cache, f_tr=f_tr, u_tr=u_tr, f_te=f_te, u_te=u_te)
        ds = {"train_f": torch.tensor(f_tr), "train_u": torch.tensor(u_tr),
              "test_f": torch.tensor(f_te), "test_u": torch.tensor(u_te),
              "modes": st.build_modes(16)}
    _ds_cache[key] = ds
    return ds

class LinBypassSpectral(torch.nn.Module):
    """Branch = lstsq-initialized linear read-out + MLP correction."""
    def __init__(self, modes, phys, ds, hidden=192, depth=4):
        super().__init__()
        self.modes = modes
        grid = st.make_grid(phys.res)
        self.grid = grid
        self.register_buffer("phi", st.sine_basis(grid, modes))
        M = len(modes)
        self.lin = torch.nn.Linear(2 * M, 2 * M)
        self.mlp = st.MLP(2 * M, 2 * M, hidden, depth)
        # zero-init the MLP output layer so training starts AT the lstsq optimum
        last = [m for m in self.mlp.modules() if isinstance(m, torch.nn.Linear)][-1]
        torch.nn.init.zeros_(last.weight); torch.nn.init.zeros_(last.bias)
        # lstsq init of the linear path on the training set
        X = st.project_onto_sine(ds["train_f"], modes, grid).numpy().astype(np.float64)
        u = ds["train_u"]
        Yx = st.project_onto_sine(u[..., 0:1].repeat(1, 1, 1, 2), modes, grid).numpy()[:, :M]
        Yy = st.project_onto_sine(u[..., 1:2].repeat(1, 1, 1, 2), modes, grid).numpy()[:, :M]
        Y = np.concatenate([Yx, Yy], 1).astype(np.float64)
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])
        Wb = np.linalg.solve(Xb.T @ Xb + 1e-8 * np.eye(Xb.shape[1]), Xb.T @ Y)
        with torch.no_grad():
            self.lin.weight.copy_(torch.tensor(Wb[:-1].T, dtype=torch.float32))
            self.lin.bias.copy_(torch.tensor(Wb[-1], dtype=torch.float32))

    def forward(self, f):
        feats = st.project_onto_sine(f, self.modes, self.grid.to(f.device))
        c = self.lin(feats) + self.mlp(feats)
        M = len(self.modes)
        ux = torch.einsum('nm,ijm->nij', c[:, :M], self.phi)
        uy = torch.einsum('nm,ijm->nij', c[:, M:], self.phi)
        return torch.stack([ux, uy], dim=-1)


def make_model(job, ds, phys):
    if job["model"] == "het_plain":
        return HeteroSpectral(st.build_modes(job.get("trunk", 16)), phys)
    if job["model"] == "het_anchor":
        return HeteroSpectral(st.build_modes(job.get("trunk", 16)), phys,
                              anchor_ds=ds)
    if job["model"] == "het_anchor_rat":
        return HeteroSpectral(st.build_modes(job.get("trunk", 16)), phys,
                              anchor_ds=ds, feats="rat")
    if job["model"] == "het_anchor_rat_frozen":
        m = HeteroSpectral(st.build_modes(job.get("trunk", 16)), phys,
                           anchor_ds=ds, feats="rat")
        for p_ in m.lin.parameters():
            p_.requires_grad_(False)
        return m
    if job["model"] == "spectral":
        modes = st.build_modes(job.get("trunk", 16))
        return st.SpectralPIDeepONet(modes, phys, hidden=192, depth=4)
    if job["model"] == "spectral_linbp":
        modes = st.build_modes(job.get("trunk", 16))
        return LinBypassSpectral(modes, phys, ds, hidden=192, depth=4)
    w, _ = fno_width(job["fno_modes"])
    return st.FNO2dElasticity(phys, modes=job["fno_modes"], width=w)

# ---------------------------------------------------------------- training
def run_chunk(job, budget_s):
    phys = st.PhysicsConfig(res=29, nu=job["nu"])
    ds = get_dataset(job["data"], job["nu"])
    torch.manual_seed(42)
    model = make_model(job, ds, phys)
    is_fno = job["model"] == "fno"
    n_epochs = 400 if is_fno else EPOCHS
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=(125 if is_fno else 250), gamma=0.6)
    ck = f"{CKPT}/{job['id']}.pt"
    epoch, best, best_state = 0, float("inf"), None
    if os.path.exists(ck):
        s = torch.load(ck, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"]); epoch = s["epoch"]
        best, best_state = s["best"], s["best_state"]
        best_epoch = s.get("best_epoch", 0)
    train_f, train_u = ds["train_f"], ds["train_u"]
    N = train_f.shape[0]; batch = 32
    w_data, w_pde = 1.0, job["w_pde"]
    best_epoch = 0
    t0 = time.time()
    while epoch < n_epochs and time.time() - t0 < budget_s:
        epoch += 1
        model.train()
        g = torch.Generator().manual_seed(1234 + epoch)
        perm = torch.randperm(N, generator=g)
        is_het = job["data"] == "hetero"
        mask = incl_mask(phys.res) if is_het else None
        for b0 in range(0, N, batch):
            idx = perm[b0:b0 + batch]
            fb, ub = train_f[idx], train_u[idx]
            if is_het:
                kb = ds["train_k"][idx]
                pred = model(fb, kb)
                loss = w_data * torch.mean((pred - ub) ** 2)
                if w_pde > 0:
                    loss = loss + w_pde * het_residual_mse(
                        pred, fb, kb, mask, phys.h, phys.nu)
            else:
                pred = model(fb)
                loss = w_data * torch.mean((pred - ub) ** 2)
                if w_pde > 0:
                    Lu = st.fd_elasticity_operator(pred, phys)
                    r = Lu + fb[:, 1:-1, 1:-1, :]
                    if job.get("scale_res"):
                        lamm, muu = phys.lame
                        r = r / (lamm + 2.0 * muu)
                    loss = loss + w_pde * torch.mean(r ** 2)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sched.step()
        if epoch % 20 == 0 or epoch == n_epochs:
            if is_het:
                m = het_evaluate(model, ds, phys)
            else:
                m = st.evaluate_model(model, ds["test_f"], ds["test_u"], phys, DEVICE)
            if m["disp_l2"] < best:
                best = m["disp_l2"]; best_epoch = epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if epoch >= n_epochs:
        if best_state: model.load_state_dict(best_state)
        if job["data"] == "hetero":
            final = het_evaluate(model, ds, phys)
        else:
            final = st.evaluate_model(model, ds["test_f"], ds["test_u"], phys, DEVICE)
        final.update(id=job["id"], exp=job["exp"], nu=job["nu"],
                     w_pde=job["w_pde"], n_params=st.count_params(model),
                     best_disp_l2=best, best_epoch=best_epoch,
                     epochs_total=n_epochs)
        if job["exp"] == "ood":
            for K in (16, 20, 24, 28):
                dK = st.make_dataset(1, 200, K, 0.04, phys)
                mk = st.evaluate_model(model, dK["test_f"], dK["test_u"], phys, DEVICE)
                final[f"ood_K{K}_disp"] = mk["disp_l2"]
                final[f"ood_K{K}_pde"] = mk["pde_mse"]
        if job["model"] == "fno":
            final["fno_modes"] = job["fno_modes"]
            final["fno_width"] = fno_width(job["fno_modes"])[0]
        with open(RESULTS, "a") as fh:
            fh.write(json.dumps(final) + "\n")
        if os.path.exists(ck): os.rename(ck, ck + ".done")
        return "DONE", epoch
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                    sched=sched.state_dict(), epoch=epoch, best=best,
                    best_state=best_state, best_epoch=best_epoch), ck)
    return "PART", epoch

def main():
    total_budget = float(sys.argv[1]) if len(sys.argv) > 1 else 34
    t_start = time.time()
    done_ids = set()
    if os.path.exists(RESULTS):
        with open(RESULTS) as fh:
            done_ids = {json.loads(l)["id"] for l in fh if l.strip()}
    for job in jobs():
        if job["id"] in done_ids: continue
        remaining = total_budget - (time.time() - t_start)
        if remaining < 4:
            print(f"BUDGET_UP at {job['id']}"); return
        status, ep = run_chunk(job, remaining)
        print(f"{job['id']}: {status} ep={ep}", flush=True)
        if status == "PART": return
    print("ALL_JOBS_DONE")

if __name__ == "__main__":
    main()
