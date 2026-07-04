import sys, json
sys.path.insert(0, "/sessions/serene-elegant-cannon/mnt/CMAME")
sys.path.insert(0, "/sessions/serene-elegant-cannon/mnt/outputs")
import numpy as np, torch
import cmame_extended_study as st
from chunk_runner import het_strain_stress, het_residual_mse, incl_mask

z = np.load("/tmp/femgt_hetero.npz")
phys = st.PhysicsConfig(res=29)
modes = st.build_modes(16)
grid = st.make_grid(29)
mask = incl_mask(29)

def feats(f, k, kind):
    phi = st.project_onto_sine(torch.tensor(f), modes, grid).numpy().astype(np.float64)
    k = k.astype(np.float64).reshape(-1,1)
    if kind == "naive":   return np.hstack([phi, k])
    if kind == "bilin":   return np.hstack([phi, k*phi, k])
    if kind == "quad":    return np.hstack([phi, k*phi, (k**2)*phi, k, k**2])
    if kind == "rat":     # rational features 1/(a+k) style via few poles
        cols = [phi]
        for a in (0.5, 1.0, 2.0): cols.append(phi/(a+k))
        cols.append(k)
        return np.hstack(cols)
def targets(u):
    u = torch.tensor(u)
    Yx = st.project_onto_sine(u[...,0:1].repeat(1,1,1,2), modes, grid).numpy()[:, :16]
    Yy = st.project_onto_sine(u[...,1:2].repeat(1,1,1,2), modes, grid).numpy()[:, :16]
    return np.concatenate([Yx, Yy], 1).astype(np.float64)

Phi = torch.tensor(st.sine_basis(grid, modes))
def recon(c):
    c = torch.tensor(c, dtype=torch.float32)
    ux = torch.einsum('nm,ijm->nij', c[:, :16], Phi)
    uy = torch.einsum('nm,ijm->nij', c[:, 16:], Phi)
    return torch.stack([ux, uy], -1)

Ytr = targets(z["u_tr"])
u_te = torch.tensor(z["u_te"]); k_te = torch.tensor(z["k_te"])
f_te = torch.tensor(z["f_te"])
rows = []
for kind in ("naive", "bilin", "quad", "rat"):
    X = feats(z["f_tr"], z["k_tr"], kind)
    Xb = np.hstack([X, np.ones((X.shape[0],1))])
    Wb = np.linalg.solve(Xb.T@Xb + 1e-8*np.eye(Xb.shape[1]), Xb.T@Ytr)
    Xt = feats(z["f_te"], z["k_te"], kind)
    C = np.hstack([Xt, np.ones((Xt.shape[0],1))]) @ Wb
    u_p = recon(C)
    e_p, s_p = het_strain_stress(u_p, k_te, mask, phys.h)
    e_t, s_t = het_strain_stress(u_te, k_te, mask, phys.h)
    r = dict(id=f"hetlin_{kind}", exp="hetero_linear",
             disp_l2=st.relative_l2(u_p, u_te),
             strain_l2=st.relative_l2(e_p, e_t),
             stress_l2=st.relative_l2(s_p, s_t),
             pde_mse=float(het_residual_mse(u_p, f_te, k_te, mask, phys.h)),
             n_feats=X.shape[1])
    print({k2: (round(v,4) if isinstance(v,float) else v) for k2,v in r.items()})
    rows.append(r)
# representation floor for hetero test set
c_opt = targets(z["u_te"])
u_fl = recon(c_opt)
print("repr floor disp:", round(st.relative_l2(u_fl, u_te), 4))
with open("/sessions/serene-elegant-cannon/mnt/CMAME/results_revision/chunk_results.jsonl","a") as fh:
    for r in rows: fh.write(json.dumps(r)+"\n")
