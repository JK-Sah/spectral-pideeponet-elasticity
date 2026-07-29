#!/usr/bin/env python3
"""Time the nonlinear neural operators' inference on the same CPU as the ledger."""
import time, torch
from cmame_extended_study import (PhysicsConfig, build_modes,
                                  SpectralPIDeepONet, FNO2dElasticity)
torch.set_num_threads(8)   # match cluster/nonlinear_ledger.sh --cpus-per-task=8
phys = PhysicsConfig(res=29, nu=0.30)

def ms_per_sample(model, f, warm=3, rep=10, batch=128):
    model.eval(); N = f.shape[0]
    with torch.no_grad():
        for _ in range(warm): model(f[:batch])
        t0 = time.perf_counter()
        for _ in range(rep):
            for i in range(0, N, batch): model(f[i:i+batch])
    return (time.perf_counter() - t0) / rep / N * 1000

f = torch.randn(400, 29, 29, 2)
sp = SpectralPIDeepONet(build_modes(64), phys, 256, 4)
fno = FNO2dElasticity(phys, modes=12, width=32, n_layers=4)
print("device cpu, threads", torch.get_num_threads())
print("nonlinear Spectral: %.4f ms/sample  params %d"
      % (ms_per_sample(sp, f), sum(p.numel() for p in sp.parameters())))
print("nonlinear FNO:      %.4f ms/sample  params %d"
      % (ms_per_sample(fno, f), sum(p.numel() for p in fno.parameters())))
print("NLTIMING_DONE")
