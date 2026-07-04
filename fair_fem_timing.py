"""Fair FEM timing: assemble + factorize ONCE, reuse for many RHS (R2's point)."""
import time, numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def q4_element_stiffness(hx, hy, lam, mu):
    gp = np.array([-1.0/np.sqrt(3), 1.0/np.sqrt(3)]); gw = np.array([1.0, 1.0])
    C = np.array([[lam+2*mu, lam, 0],[lam, lam+2*mu, 0],[0,0,mu]])
    ke = np.zeros((8,8))
    for xi, wi in zip(gp, gw):
        for eta, wj in zip(gp, gw):
            dN_dxi = np.array([-(1-eta),(1-eta),(1+eta),-(1+eta)])/4.0
            dN_deta = np.array([-(1-xi),-(1+xi),(1+xi),(1-xi)])/4.0
            dN_dx = (2.0/hx)*dN_dxi; dN_dy = (2.0/hy)*dN_deta
            detJ = (hx/2.0)*(hy/2.0)
            B = np.zeros((3,8))
            for k in range(4):
                B[0,2*k]=dN_dx[k]; B[1,2*k+1]=dN_dy[k]
                B[2,2*k]=dN_dy[k]; B[2,2*k+1]=dN_dx[k]
            ke += detJ*wi*wj*(B.T@C@B)
    return ke

def run(res=29, E=1.0, nu=0.3, n_samples=200, seed=0):
    lam = E*nu/((1+nu)*(1-2*nu)); mu = E/(2*(1+nu))
    nx = ny = res-1; hx = hy = 1.0/nx
    n_nodes = res*res; n_dof = 2*n_nodes
    # assembly (timed once)
    t0 = time.perf_counter()
    ke = q4_element_stiffness(hx, hy, lam, mu)
    rows, cols, vals = [], [], []
    for ey in range(ny):
        for ex in range(nx):
            n1 = ey*res+ex; n2 = n1+1; n3 = (ey+1)*res+ex+1; n4 = (ey+1)*res+ex
            dofs = [d for n in (n1,n2,n3,n4) for d in (2*n, 2*n+1)]
            for i, di in enumerate(dofs):
                for j, dj in enumerate(dofs):
                    rows.append(di); cols.append(dj); vals.append(ke[i,j])
    K = sp.csr_matrix((vals,(rows,cols)), shape=(n_dof,n_dof))
    bc = set()
    for i in range(res):
        for n in (i, (res-1)*res+i, i*res, i*res+res-1):
            bc.update([2*n, 2*n+1])
    free = np.array([i for i in range(n_dof) if i not in bc])
    K_free = K[np.ix_(free, free)].tocsc()
    t_asm = (time.perf_counter()-t0)*1000
    # factorization (timed once)
    t0 = time.perf_counter(); lu = spla.splu(K_free)
    t_fac = (time.perf_counter()-t0)*1000
    # per-RHS solves with reused factorization
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((n_samples, free.size))
    lu.solve(F[0])  # warm-up
    t0 = time.perf_counter()
    for i in range(n_samples): lu.solve(F[i])
    t_rhs = (time.perf_counter()-t0)*1000/n_samples
    # old style: full spsolve per sample
    spla.spsolve(K_free, F[0])
    t0 = time.perf_counter()
    for i in range(min(n_samples,50)): spla.spsolve(K_free, F[i])
    t_sps = (time.perf_counter()-t0)*1000/min(n_samples,50)
    return dict(res=res, n_dof=int(n_dof), assembly_ms=t_asm, factor_ms=t_fac,
                per_rhs_ms=t_rhs, spsolve_per_sample_ms=t_sps)

if __name__ == "__main__":
    nn_ms = 0.00833270518342033
    print(f"{'res':>5}{'n_dof':>8}{'asm_ms':>10}{'fac_ms':>10}{'perRHS_ms':>11}{'spsolve_ms':>12}{'amortized200':>14}{'speedup_fair':>13}")
    for res in (29, 57, 113):
        r = run(res)
        amort = (r['assembly_ms']+r['factor_ms'])/200 + r['per_rhs_ms']
        print(f"{r['res']:>5}{r['n_dof']:>8}{r['assembly_ms']:>10.2f}{r['factor_ms']:>10.3f}"
              f"{r['per_rhs_ms']:>11.4f}{r['spsolve_per_sample_ms']:>12.3f}{amort:>14.4f}{amort/nn_ms:>13.1f}")
