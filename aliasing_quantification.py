"""Quantify R2's aliasing concern: fraction of body-force energy outside the
sine-sine span used for branch features, and conditioning of coeffs->features map."""
import math, numpy as np

def build_modes(max_modes):
    pairs = [(p, q) for p in range(1, max_modes+1) for q in range(1, max_modes+1)]
    pairs.sort(key=lambda pq: (pq[0]**2 + pq[1]**2, pq))
    return pairs[:max_modes]  # matches ordered low-freq-first selection

def main(n_modes=16, res=29, n_samples=200, scale=0.04, seed=999, E=1.0, nu=0.3):
    lam = E*nu/((1+nu)*(1-2*nu)); mu = E/(2*(1+nu))
    modes = build_modes(n_modes)
    x = np.linspace(0, 1, res); y = np.linspace(0, 1, res)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    rng = np.random.default_rng(seed)
    M = len(modes)
    ax = np.zeros((n_samples, M)); ay = np.zeros((n_samples, M))
    for k, (p, q) in enumerate(modes):
        d = 1.0/(p**2+q**2)
        ax[:,k] = rng.normal(0, scale*d, n_samples)
        ay[:,k] = rng.normal(0, scale*d, n_samples)
    fx = np.zeros((n_samples,res,res)); fy = np.zeros_like(fx)
    Phi = np.zeros((res,res,M))
    for k,(p,q) in enumerate(modes):
        P, Q = p*math.pi, q*math.pi
        sx, cx = np.sin(P*xx), np.cos(P*xx)
        sy, cy = np.sin(Q*yy), np.cos(Q*yy)
        phi = sx*sy; Phi[:,:,k] = phi
        phi_xx = -(P**2)*phi; phi_yy = -(Q**2)*phi; phi_xy = P*Q*cx*cy
        a = ax[:,k][:,None,None]; b = ay[:,k][:,None,None]
        # f = -[mu lap u + (lam+mu) grad div u]
        fx += -(mu*(phi_xx+phi_yy)*a + (lam+mu)*(phi_xx*a + phi_xy*b))
        fy += -(mu*(phi_xx+phi_yy)*b + (lam+mu)*(phi_xy*a + phi_yy*b))
    # project f onto sine span (trapezoid quadrature as in paper: factor 4)
    w = np.ones(res); w[0]=w[-1]=0.5; W = np.outer(w,w)/(res-1)**2
    def proj_coeffs(f):  # [n,res,res] -> [n,M]
        return 4.0*np.einsum('nij,ijm->nm', f*W, Phi)
    def recon(c): return np.einsum('nm,ijm->nij', c, Phi)
    loss = []
    for f in (fx, fy):
        c = proj_coeffs(f); r = f - recon(c)
        # discrete L2 norms with same quadrature
        num = np.einsum('nij,ij->n', r**2, W); den = np.einsum('nij,ij->n', f**2, W)
        loss.append(np.sqrt(num/den))
    loss = np.stack(loss)
    print(f"n_modes={n_modes}, res={res}: unrepresentable fraction of ||f|| "
          f"(rel L2 of projection residual)")
    print(f"  mean {loss.mean():.4f}  median {np.median(loss):.4f}  "
          f"min {loss.min():.4f}  max {loss.max():.4f}")
    # conditioning of linear map (ax,ay) -> branch features
    Afeat = np.zeros((2*M, 2*M))
    for k in range(M):
        e = np.zeros((1,M)); e[0,k] = 1.0
        # unit ax_k
        fxk = np.zeros((1,res,res)); fyk = np.zeros_like(fxk)
        p,q = modes[k]; P,Q = p*math.pi, q*math.pi
        phi = np.sin(P*xx)*np.sin(Q*yy); phi_xx=-(P**2)*phi; phi_yy=-(Q**2)*phi
        phi_xy = P*Q*np.cos(P*xx)*np.cos(Q*yy)
        fxk[0] = -(mu*(phi_xx+phi_yy) + (lam+mu)*phi_xx); fyk[0] = -((lam+mu)*phi_xy)
        Afeat[:M,k] = proj_coeffs(fxk)[0]; Afeat[M:,k] = proj_coeffs(fyk)[0]
        # unit ay_k
        fxk[0] = -((lam+mu)*phi_xy); fyk[0] = -(mu*(phi_xx+phi_yy) + (lam+mu)*phi_yy)
        Afeat[:M,M+k] = proj_coeffs(fxk)[0]; Afeat[M:,M+k] = proj_coeffs(fyk)[0]
    s = np.linalg.svd(Afeat, compute_uv=False)
    print(f"  coeffs->features linear map: cond = {s[0]/s[-1]:.2f} "
          f"(invertible => features still determine the exact solution)")

for nm in (12, 16):
    main(n_modes=nm)
