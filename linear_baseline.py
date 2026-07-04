"""Closed-form linear least-squares read-out from the same branch features.
The target operator is linear, so this is the natural optimal-linear baseline."""
import sys, json
sys.path.insert(0, "/sessions/serene-elegant-cannon/mnt/CMAME")
import numpy as np, torch
import cmame_extended_study as st

class LinModel(torch.nn.Module):
    def __init__(self, W, b, modes, phys):
        super().__init__()
        self.W = torch.tensor(W, dtype=torch.float32)
        self.b = torch.tensor(b, dtype=torch.float32)
        self.modes = modes
        grid = st.make_grid(phys.res)
        self.register_buffer("phi", st.sine_basis(grid, modes))
        self.grid = grid
    def forward(self, f):
        feats = st.project_onto_sine(f, self.modes, self.grid.to(f.device))
        c = feats @ self.W.T + self.b
        M = len(self.modes)
        cx, cy = c[:, :M], c[:, M:]
        ux = torch.einsum('nm,ijm->nij', cx, self.phi)
        uy = torch.einsum('nm,ijm->nij', cy, self.phi)
        return torch.stack([ux, uy], dim=-1)

def fit_eval(name, ds, phys, n_train=None, ridge=1e-8):
    modes = ds["modes"]
    tf, tu = ds["train_f"], ds["train_u"]
    if n_train: tf, tu = tf[:n_train], tu[:n_train]
    grid = st.make_grid(phys.res)
    X = st.project_onto_sine(tf, modes, grid).numpy().astype(np.float64)
    # targets: optimal sine coefficients of u
    Yx = st.project_onto_sine(tu[..., 0:1].repeat(1,1,1,2), modes, grid).numpy()[:, :len(modes)]
    Yy = st.project_onto_sine(tu[..., 1:2].repeat(1,1,1,2), modes, grid).numpy()[:, :len(modes)]
    Y = np.concatenate([Yx, Yy], axis=1).astype(np.float64)
    Xb = np.hstack([X, np.ones((X.shape[0],1))])
    A = Xb.T @ Xb + ridge*np.eye(Xb.shape[1])
    Wb = np.linalg.solve(A, Xb.T @ Y)
    W, b = Wb[:-1].T, Wb[-1]
    m = LinModel(W, b, modes, phys)
    res = st.evaluate_model(m, ds["test_f"], ds["test_u"], phys, torch.device("cpu"))
    out = {"id": name, "exp": "linear_baseline", **{k: float(v) for k,v in res.items()}}
    print(name, {k: round(v,4) for k,v in res.items() if k!='infer_ms'})
    return out

phys = st.PhysicsConfig(res=29)
rows = []
ds = st.make_dataset(1000, 200, 16, 0.04, phys)
rows.append(fit_eval("lin_sine16_n1000", ds, phys))
rows.append(fit_eval("lin_sine16_n100", ds, phys, n_train=100))
for kind in ("bumps", "patch"):
    z = np.load(f"/tmp/femgt_{kind}.npz")
    d = {"train_f": torch.tensor(z["f_tr"]), "train_u": torch.tensor(z["u_tr"]),
         "test_f": torch.tensor(z["f_te"]), "test_u": torch.tensor(z["u_te"]),
         "modes": st.build_modes(16)}
    rows.append(fit_eval(f"lin_{kind}_n1000", d, phys))
with open("/sessions/serene-elegant-cannon/mnt/CMAME/results_revision/chunk_results.jsonl","a") as fh:
    for r in rows: fh.write(json.dumps(r)+"\n")
