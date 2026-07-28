#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
time_neural_cpu.py

Time neural-operator inference on the SAME CPU as the classical baselines
in ledger.py, so the accuracy-cost comparison is hardware-consistent.
Forward-pass cost is weight-independent, so fresh-initialized models are
timed on random inputs of the working-grid shape.
"""
import time, resource, torch
from cmame_extended_study import (PhysicsConfig, build_modes,
                                  SpectralPIDeepONet, FNO2dElasticity)
from route_b import SpectralPIDeepONetE, FNOEfield, build_cos_modes

torch.set_num_threads(4)   # match cluster/ledger.sh --cpus-per-task=4


def ms_per_sample(model, inputs, n_warm=3, n_rep=8, batch=128):
    model.eval()
    N = inputs[0].shape[0]
    with torch.no_grad():
        for _ in range(n_warm):
            model(*[x[:batch] for x in inputs])
        t0 = time.perf_counter()
        for _ in range(n_rep):
            for i in range(0, N, batch):
                model(*[x[i:i+batch] for x in inputs])
        dt = (time.perf_counter() - t0) / n_rep / N
    return dt * 1000.0


def npar(m):
    return sum(p.numel() for p in m.parameters())


phys = PhysicsConfig(res=29, nu=0.30)
print("device: cpu, torch threads =", torch.get_num_threads())

# Homogeneous benchmark models (200 test samples)
fA = torch.randn(200, 29, 29, 2)
spA = SpectralPIDeepONet(build_modes(16), phys, 192, 4)
fnoA = FNO2dElasticity(phys, modes=12, width=20, n_layers=4)
print(f"A  PI-spectral   {ms_per_sample(spA,[fA]):.4f} ms/sample  params {npar(spA)}")
print(f"A  FNO           {ms_per_sample(fnoA,[fA]):.4f} ms/sample  params {npar(fnoA)}")

# Heterogeneous benchmark models (400 test samples), inputs (f, E)
fB = torch.randn(400, 29, 29, 2)
EB = torch.rand(400, 29, 29) + 0.5
spB = SpectralPIDeepONetE(build_modes(16), build_cos_modes(16), phys, 192, 4)
fnoB = FNOEfield(phys, modes=12, width=32)
print(f"B  spectral_e     {ms_per_sample(spB,[fB,EB]):.4f} ms/sample  params {npar(spB)}")
print(f"B  fno_e          {ms_per_sample(fnoB,[fB,EB]):.4f} ms/sample  params {npar(fnoB)}")

print(f"peak_rss_MB {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f}")
print("DONE")
