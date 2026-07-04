#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
continuum_elasticity_pideeponet_publication_study.py

What this script produces:
    1. Manufactured 2D linear-elasticity datasets on [0,1]^2 with u=0 on boundary.
    2. Physics-informed spectral DeepONet model.
    3. Non-physics/data-only ablation.
    4. MLP-trunk DeepONet baseline with boundary mollifier.
    5. CNN/U-Net surrogate baseline.
    6. Ablation studies:
        - number of sine modes,
        - number of training samples,
        - PDE-loss weight,
        - data-only vs physics-informed training.
    7. Continuum-mechanics validation metrics:
        - displacement relative L2 error,
        - strain relative L2 error,
        - stress relative L2 error,
        - total strain-energy relative error,
        - PDE residual mean-square error,
        - inference time.
    8. Publication-style plots and CSV tables.

Run examples:
    Quick test:
        python continuum_elasticity_pideeponet_publication_study.py --study quick --device cpu

    More serious run:
        python continuum_elasticity_pideeponet_publication_study.py --study publication --device cuda

Outputs:
    ./results_elasticity_publication/
        figures/
        tables/
        checkpoints/
        logs/

"""

import os
import time
import json
import math
import argparse
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    raise ImportError(
        "PyTorch is required. Install it with: pip install torch"
    ) from exc


# -------------------------------------------------------------------------
# 1. Reproducibility and device
# -------------------------------------------------------------------------

def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


# -------------------------------------------------------------------------
# 2. Configuration
# -------------------------------------------------------------------------

@dataclass
class PhysicsConfig:
    res: int = 29
    E: float = 1.0
    nu: float = 0.30
    plane: str = "strain"  # "strain" or "stress"

    @property
    def h(self) -> float:
        return 1.0 / (self.res - 1)

    @property
    def lame(self) -> Tuple[float, float]:
        """
        Returns lambda, mu.
        Plane strain uses standard 3D isotropic Lamé constants.
        Plane stress uses effective lambda = E*nu/(1-nu^2).
        """
        E, nu = self.E, self.nu
        mu = E / (2.0 * (1.0 + nu))
        if self.plane.lower() == "stress":
            lam = E * nu / (1.0 - nu ** 2)
        else:
            lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return lam, mu


@dataclass
class DataConfig:
    n_train: int = 300
    n_test: int = 60
    true_modes: int = 12
    coeff_scale: float = 0.04
    seed_train: int = 42
    seed_test: int = 999


@dataclass
class TrainConfig:
    epochs: int = 500
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-6
    scheduler_step: int = 250
    scheduler_gamma: float = 0.6
    w_data: float = 1.0
    w_pde: float = 1e-4
    patience: int = 100000  # disabled by default
    eval_every: int = 10


@dataclass
class ModelConfig:
    model_name: str = "spectral_pideeponet"
    n_modes_model: int = 12
    hidden: int = 128
    depth: int = 3


# -------------------------------------------------------------------------
# 3. Continuum mechanics: grid, modes, manufactured solution, body force
# -------------------------------------------------------------------------

def make_grid(res: int, device: Optional[torch.device] = None) -> torch.Tensor:
    xs = torch.linspace(0.0, 1.0, res, device=device)
    ys = torch.linspace(0.0, 1.0, res, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1)  # [res,res,2], last = (x,y)


def build_modes(max_modes: int) -> List[Tuple[int, int]]:
    """
    Ordered smooth sine modes. Low frequencies first.
    """
    candidates = []
    max_pq = max(8, int(np.ceil(np.sqrt(max_modes))) + 4)
    for p in range(1, max_pq + 1):
        for q in range(1, max_pq + 1):
            candidates.append((p, q))
    candidates = sorted(candidates, key=lambda t: (t[0] ** 2 + t[1] ** 2, t[0], t[1]))
    return candidates[:max_modes]


def sine_basis(grid: torch.Tensor, modes: List[Tuple[int, int]]) -> torch.Tensor:
    """
    Returns Phi: [res,res,M], Phi_k = sin(p*pi*x) sin(q*pi*y).
    """
    x = grid[..., 0:1]
    y = grid[..., 1:2]
    feats = []
    for p, q in modes:
        feats.append(torch.sin(p * math.pi * x) * torch.sin(q * math.pi * y))
    return torch.cat(feats, dim=-1)


def generate_coefficients(n: int, modes: List[Tuple[int, int]], scale: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Random coefficients for ux and uy. Higher modes are damped so the fields remain smooth.
    """
    rng = np.random.default_rng(seed)
    M = len(modes)
    ax = np.zeros((n, M), dtype=np.float32)
    ay = np.zeros((n, M), dtype=np.float32)

    for k, (p, q) in enumerate(modes):
        damping = 1.0 / (p ** 2 + q ** 2)
        ax[:, k] = rng.normal(0.0, scale * damping, size=n)
        ay[:, k] = rng.normal(0.0, scale * damping, size=n)

    return ax, ay


def analytical_solution_and_force(
    coeff_x: np.ndarray,
    coeff_y: np.ndarray,
    modes: List[Tuple[int, int]],
    phys: PhysicsConfig,
) -> Dict[str, np.ndarray]:
    """
    Manufactured solution:
        ux = sum a_k sin(p*pi*x) sin(q*pi*y)
        uy = sum b_k sin(p*pi*x) sin(q*pi*y)

    Body force:
        f = -[ mu Laplacian(u) + (lambda+mu) grad(div u) ]

    All derivatives are analytical, evaluated on the grid.
    """
    lam, mu = phys.lame
    res = phys.res

    x = np.linspace(0.0, 1.0, res, dtype=np.float64)
    y = np.linspace(0.0, 1.0, res, dtype=np.float64)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    n, M = coeff_x.shape

    ux = np.zeros((n, res, res), dtype=np.float64)
    uy = np.zeros((n, res, res), dtype=np.float64)

    ux_x = np.zeros_like(ux)
    ux_y = np.zeros_like(ux)
    uy_x = np.zeros_like(ux)
    uy_y = np.zeros_like(ux)

    ux_xx = np.zeros_like(ux)
    ux_yy = np.zeros_like(ux)
    uy_xx = np.zeros_like(ux)
    uy_yy = np.zeros_like(ux)

    ux_xy = np.zeros_like(ux)
    uy_xy = np.zeros_like(ux)

    for k, (p, q) in enumerate(modes):
        P = p * math.pi
        Q = q * math.pi
        sx = np.sin(P * xx)
        cx = np.cos(P * xx)
        sy = np.sin(Q * yy)
        cy = np.cos(Q * yy)

        phi = sx * sy
        phi_x = P * cx * sy
        phi_y = Q * sx * cy
        phi_xx = -(P ** 2) * phi
        phi_yy = -(Q ** 2) * phi
        phi_xy = P * Q * cx * cy

        ax = coeff_x[:, k][:, None, None]
        ay = coeff_y[:, k][:, None, None]

        ux += ax * phi
        uy += ay * phi

        ux_x += ax * phi_x
        ux_y += ax * phi_y
        uy_x += ay * phi_x
        uy_y += ay * phi_y

        ux_xx += ax * phi_xx
        ux_yy += ax * phi_yy
        uy_xx += ay * phi_xx
        uy_yy += ay * phi_yy

        ux_xy += ax * phi_xy
        uy_xy += ay * phi_xy

    lap_ux = ux_xx + ux_yy
    lap_uy = uy_xx + uy_yy

    div_x = ux_xx + uy_xy
    div_y = ux_xy + uy_yy

    fx = -(mu * lap_ux + (lam + mu) * div_x)
    fy = -(mu * lap_uy + (lam + mu) * div_y)

    eps_xx = ux_x
    eps_yy = uy_y
    eps_xy = 0.5 * (ux_y + uy_x)

    sig_xx = (lam + 2.0 * mu) * eps_xx + lam * eps_yy
    sig_yy = lam * eps_xx + (lam + 2.0 * mu) * eps_yy
    sig_xy = 2.0 * mu * eps_xy

    u = np.stack([ux, uy], axis=-1).astype(np.float32)
    f = np.stack([fx, fy], axis=-1).astype(np.float32)
    eps = np.stack([eps_xx, eps_yy, eps_xy], axis=-1).astype(np.float32)
    sig = np.stack([sig_xx, sig_yy, sig_xy], axis=-1).astype(np.float32)

    return {"u": u, "f": f, "eps": eps, "sig": sig}


def make_dataset(data_cfg: DataConfig, phys: PhysicsConfig) -> Dict[str, torch.Tensor]:
    modes = build_modes(data_cfg.true_modes)

    ax_train, ay_train = generate_coefficients(data_cfg.n_train, modes, data_cfg.coeff_scale, data_cfg.seed_train)
    ax_test, ay_test = generate_coefficients(data_cfg.n_test, modes, data_cfg.coeff_scale, data_cfg.seed_test)

    train = analytical_solution_and_force(ax_train, ay_train, modes, phys)
    test = analytical_solution_and_force(ax_test, ay_test, modes, phys)

    out = {
        "train_f": torch.tensor(train["f"], dtype=torch.float32),
        "train_u": torch.tensor(train["u"], dtype=torch.float32),
        "train_eps": torch.tensor(train["eps"], dtype=torch.float32),
        "train_sig": torch.tensor(train["sig"], dtype=torch.float32),
        "test_f": torch.tensor(test["f"], dtype=torch.float32),
        "test_u": torch.tensor(test["u"], dtype=torch.float32),
        "test_eps": torch.tensor(test["eps"], dtype=torch.float32),
        "test_sig": torch.tensor(test["sig"], dtype=torch.float32),
        "modes_true": modes,
        "coeff_train_x": torch.tensor(ax_train, dtype=torch.float32),
        "coeff_train_y": torch.tensor(ay_train, dtype=torch.float32),
        "coeff_test_x": torch.tensor(ax_test, dtype=torch.float32),
        "coeff_test_y": torch.tensor(ay_test, dtype=torch.float32),
    }
    return out


# -------------------------------------------------------------------------
# 4. Feature extraction and finite-difference operators
# -------------------------------------------------------------------------

def project_onto_sine(f_grid: torch.Tensor, modes: List[Tuple[int, int]], grid: torch.Tensor) -> torch.Tensor:
    """
    Projects the 2-component input force field onto sine modes.
    f_grid: [B,res,res,2]
    returns [B, 2*M] coefficients-like features.
    """
    phi = sine_basis(grid, modes).to(f_grid.device)  # [res,res,M]
    # approximate integral over [0,1]^2; normalization for sine-sine basis is 4
    # because int_0^1 sin^2(p*pi*x) dx = 1/2 in each direction.
    res = f_grid.shape[1]
    h = 1.0 / (res - 1)
    wx = torch.ones(res, device=f_grid.device)
    wy = torch.ones(res, device=f_grid.device)
    wx[0] = wx[-1] = 0.5
    wy[0] = wy[-1] = 0.5
    w2 = (wy[:, None] * wx[None, :])[None, :, :, None] * h * h

    fx = f_grid[..., 0:1]
    fy = f_grid[..., 1:2]
    phi_b = phi[None, :, :, :]  # [1,res,res,M]

    cx = 4.0 * torch.sum(w2 * fx * phi_b, dim=(1, 2))
    cy = 4.0 * torch.sum(w2 * fy * phi_b, dim=(1, 2))
    return torch.cat([cx, cy], dim=-1)


def fd_first_derivatives(u: torch.Tensor, h: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Central differences on interior points.
    u: [B,res,res,2]
    returns ux_x, ux_y, uy_x, uy_y each [B,res-2,res-2]
    """
    ux = u[..., 0]
    uy = u[..., 1]

    ux_x = (ux[:, 1:-1, 2:] - ux[:, 1:-1, :-2]) / (2.0 * h)
    ux_y = (ux[:, 2:, 1:-1] - ux[:, :-2, 1:-1]) / (2.0 * h)
    uy_x = (uy[:, 1:-1, 2:] - uy[:, 1:-1, :-2]) / (2.0 * h)
    uy_y = (uy[:, 2:, 1:-1] - uy[:, :-2, 1:-1]) / (2.0 * h)
    return ux_x, ux_y, uy_x, uy_y


def fd_strain_stress(u: torch.Tensor, phys: PhysicsConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes strain and stress on interior grid.
    Returns:
        eps: [B,res-2,res-2,3] = eps_xx, eps_yy, eps_xy
        sig: [B,res-2,res-2,3] = sig_xx, sig_yy, sig_xy
    """
    lam, mu = phys.lame
    ux_x, ux_y, uy_x, uy_y = fd_first_derivatives(u, phys.h)
    eps_xx = ux_x
    eps_yy = uy_y
    eps_xy = 0.5 * (ux_y + uy_x)

    sig_xx = (lam + 2.0 * mu) * eps_xx + lam * eps_yy
    sig_yy = lam * eps_xx + (lam + 2.0 * mu) * eps_yy
    sig_xy = 2.0 * mu * eps_xy

    eps = torch.stack([eps_xx, eps_yy, eps_xy], dim=-1)
    sig = torch.stack([sig_xx, sig_yy, sig_xy], dim=-1)
    return eps, sig


def fd_elasticity_operator(u: torch.Tensor, phys: PhysicsConfig) -> torch.Tensor:
    """
    Navier-Cauchy operator:
        L(u)_x = (lambda+2mu) u_x,xx + mu u_x,yy + (lambda+mu) u_y,xy
        L(u)_y = mu u_y,xx + (lambda+2mu) u_y,yy + (lambda+mu) u_x,xy

    u: [B,res,res,2]
    returns [B,res-2,res-2,2]
    """
    lam, mu = phys.lame
    h = phys.h
    ux = u[..., 0]
    uy = u[..., 1]

    ux_xx = (ux[:, 1:-1, 2:] - 2.0 * ux[:, 1:-1, 1:-1] + ux[:, 1:-1, :-2]) / (h ** 2)
    ux_yy = (ux[:, 2:, 1:-1] - 2.0 * ux[:, 1:-1, 1:-1] + ux[:, :-2, 1:-1]) / (h ** 2)

    uy_xx = (uy[:, 1:-1, 2:] - 2.0 * uy[:, 1:-1, 1:-1] + uy[:, 1:-1, :-2]) / (h ** 2)
    uy_yy = (uy[:, 2:, 1:-1] - 2.0 * uy[:, 1:-1, 1:-1] + uy[:, :-2, 1:-1]) / (h ** 2)

    ux_xy = (
        ux[:, 2:, 2:] - ux[:, 2:, :-2] - ux[:, :-2, 2:] + ux[:, :-2, :-2]
    ) / (4.0 * h ** 2)
    uy_xy = (
        uy[:, 2:, 2:] - uy[:, 2:, :-2] - uy[:, :-2, 2:] + uy[:, :-2, :-2]
    ) / (4.0 * h ** 2)

    Lx = (lam + 2.0 * mu) * ux_xx + mu * ux_yy + (lam + mu) * uy_xy
    Ly = mu * uy_xx + (lam + 2.0 * mu) * uy_yy + (lam + mu) * ux_xy
    return torch.stack([Lx, Ly], dim=-1)


def strain_energy(u: torch.Tensor, phys: PhysicsConfig) -> torch.Tensor:
    """
    Total strain energy:
        Pi = 1/2 ∫ sigma:epsilon dOmega
           = 1/2 ∫ (sig_xx eps_xx + sig_yy eps_yy + 2 sig_xy eps_xy) dOmega
    evaluated on interior grid.
    Returns [B].
    """
    eps, sig = fd_strain_stress(u, phys)
    density = 0.5 * (sig[..., 0] * eps[..., 0] + sig[..., 1] * eps[..., 1] + 2.0 * sig[..., 2] * eps[..., 2])
    return torch.sum(density, dim=(1, 2)) * (phys.h ** 2)


# -------------------------------------------------------------------------
# 5. Models
# -------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, depth: int, activation: str = "silu"):
        super().__init__()
        acts = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }
        act = acts[activation.lower()]
        layers = []
        last = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(last, hidden))
            layers.append(act())
            last = hidden
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SpectralPIDeepONet(nn.Module):
    """
    Branch network predicts sine coefficients for ux and uy.
    Boundary condition u=0 on all sides is exact because sine basis vanishes on boundary.
    """
    def __init__(self, modes: List[Tuple[int, int]], phys: PhysicsConfig, hidden: int = 128, depth: int = 3):
        super().__init__()
        self.modes = modes
        self.phys = phys
        self.M = len(modes)
        self.branch = MLP(in_dim=2*self.M, out_dim=2*self.M, hidden=hidden, depth=depth, activation="silu")

        grid = make_grid(phys.res)
        self.register_buffer("grid", grid)
        self.register_buffer("phi", sine_basis(grid, modes))  # [res,res,M]

    def forward(self, f_grid: torch.Tensor) -> torch.Tensor:
        B = f_grid.shape[0]
        features = project_onto_sine(f_grid, self.modes, self.grid.to(f_grid.device))
        coeff = self.branch(features)  # [B,2M]
        cx = coeff[:, :self.M]
        cy = coeff[:, self.M:]
        phi = self.phi.to(f_grid.device)
        ux = torch.einsum("bm,ijm->bij", cx, phi)
        uy = torch.einsum("bm,ijm->bij", cy, phi)
        return torch.stack([ux, uy], dim=-1)


class MLPTrunkDeepONet(nn.Module):
    """
    Standard DeepONet-like baseline:
        branch encodes force features,
        trunk is learned coordinate MLP,
        output is dot product.
    Homogeneous Dirichlet BC is enforced with x(1-x)y(1-y) mollifier.
    """
    def __init__(self, modes: List[Tuple[int, int]], phys: PhysicsConfig, hidden: int = 128, depth: int = 3, latent: int = 64):
        super().__init__()
        self.modes = modes
        self.phys = phys
        self.latent = latent

        self.branch = MLP(in_dim=2*len(modes), out_dim=2*latent, hidden=hidden, depth=depth, activation="silu")
        self.trunk = MLP(in_dim=2, out_dim=latent, hidden=hidden, depth=depth, activation="silu")

        grid = make_grid(phys.res)
        self.register_buffer("grid", grid)

    def forward(self, f_grid: torch.Tensor) -> torch.Tensor:
        B = f_grid.shape[0]
        grid = self.grid.to(f_grid.device)
        coords = grid.reshape(-1, 2)
        features = project_onto_sine(f_grid, self.modes, grid)

        branch = self.branch(features)
        bx = branch[:, :self.latent]
        by = branch[:, self.latent:]

        trunk = self.trunk(coords)  # [N,latent]

        ux = torch.matmul(bx, trunk.T).reshape(B, self.phys.res, self.phys.res)
        uy = torch.matmul(by, trunk.T).reshape(B, self.phys.res, self.phys.res)

        x = grid[..., 0]
        y = grid[..., 1]
        mollifier = x * (1.0 - x) * y * (1.0 - y)
        ux = ux * mollifier
        uy = uy * mollifier
        return torch.stack([ux, uy], dim=-1)


class SmallUNet(nn.Module):
    """
    CNN/U-Net surrogate baseline mapping force image [fx,fy] -> displacement image [ux,uy].
    Boundary condition is softly enforced by multiplying output with x(1-x)y(1-y).
    """
    def __init__(self, phys: PhysicsConfig, width: int = 32):
        super().__init__()
        self.phys = phys

        self.enc1 = nn.Sequential(nn.Conv2d(2, width, 3, padding=1), nn.SiLU(), nn.Conv2d(width, width, 3, padding=1), nn.SiLU())
        self.enc2 = nn.Sequential(nn.Conv2d(width, 2*width, 3, padding=1), nn.SiLU(), nn.Conv2d(2*width, 2*width, 3, padding=1), nn.SiLU())
        self.enc3 = nn.Sequential(nn.Conv2d(2*width, 4*width, 3, padding=1), nn.SiLU(), nn.Conv2d(4*width, 4*width, 3, padding=1), nn.SiLU())

        self.pool = nn.AvgPool2d(2)
        self.up2 = nn.ConvTranspose2d(4*width, 2*width, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(4*width, 2*width, 3, padding=1), nn.SiLU(), nn.Conv2d(2*width, 2*width, 3, padding=1), nn.SiLU())
        self.up1 = nn.ConvTranspose2d(2*width, width, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(2*width, width, 3, padding=1), nn.SiLU(), nn.Conv2d(width, width, 3, padding=1), nn.SiLU())
        self.out = nn.Conv2d(width, 2, 1)

        grid = make_grid(phys.res)
        mollifier = grid[..., 0] * (1.0 - grid[..., 0]) * grid[..., 1] * (1.0 - grid[..., 1])
        self.register_buffer("mollifier", mollifier[None, None, :, :])

    def forward(self, f_grid: torch.Tensor) -> torch.Tensor:
        # f_grid [B,res,res,2] -> [B,2,res,res]
        x = f_grid.permute(0, 3, 1, 2)

        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)

        u2 = self.up2(e3)
        # handle odd resolution by center/pad interpolation
        if u2.shape[-2:] != e2.shape[-2:]:
            u2 = F.interpolate(u2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        if u1.shape[-2:] != e1.shape[-2:]:
            u1 = F.interpolate(u1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        out = self.out(d1) * self.mollifier.to(f_grid.device)
        return out.permute(0, 2, 3, 1)


def make_model(model_cfg: ModelConfig, phys: PhysicsConfig) -> nn.Module:
    modes = build_modes(model_cfg.n_modes_model)
    if model_cfg.model_name == "spectral_pideeponet":
        return SpectralPIDeepONet(modes, phys, hidden=model_cfg.hidden, depth=model_cfg.depth)
    if model_cfg.model_name == "mlp_trunk_deeponet":
        return MLPTrunkDeepONet(modes, phys, hidden=model_cfg.hidden, depth=model_cfg.depth, latent=max(32, model_cfg.hidden // 2))
    if model_cfg.model_name == "unet_surrogate":
        return SmallUNet(phys, width=max(16, model_cfg.hidden // 4))
    raise ValueError(f"Unknown model_name: {model_cfg.model_name}")


# -------------------------------------------------------------------------
# 6. Losses and metrics
# -------------------------------------------------------------------------

def relative_l2(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    num = torch.linalg.norm((pred - true).reshape(pred.shape[0], -1), dim=1)
    den = torch.linalg.norm(true.reshape(true.shape[0], -1), dim=1) + eps
    return num / den


def pde_residual_loss(u_pred: torch.Tensor, f_grid: torch.Tensor, phys: PhysicsConfig) -> torch.Tensor:
    Lu = fd_elasticity_operator(u_pred, phys)
    f_int = f_grid[:, 1:-1, 1:-1, :]
    residual = Lu + f_int
    return torch.mean(residual ** 2)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    f: torch.Tensor,
    u_true: torch.Tensor,
    phys: PhysicsConfig,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    model.eval()

    preds = []
    t0 = time.time()
    for i in range(0, f.shape[0], batch_size):
        fb = f[i:i+batch_size].to(device)
        preds.append(model(fb).cpu())
    elapsed = time.time() - t0

    u_pred = torch.cat(preds, dim=0)

    err_u = relative_l2(u_pred, u_true).mean().item()

    eps_pred, sig_pred = fd_strain_stress(u_pred, phys)
    eps_true, sig_true = fd_strain_stress(u_true, phys)

    err_eps = relative_l2(eps_pred, eps_true).mean().item()
    err_sig = relative_l2(sig_pred, sig_true).mean().item()

    energy_pred = strain_energy(u_pred, phys)
    energy_true = strain_energy(u_true, phys)
    err_energy = (torch.abs(energy_pred - energy_true) / (torch.abs(energy_true) + 1e-12)).mean().item()

    pde_mse = pde_residual_loss(u_pred, f, phys).item()

    return {
        "rel_l2_u": err_u,
        "rel_l2_strain": err_eps,
        "rel_l2_stress": err_sig,
        "rel_energy_error": err_energy,
        "pde_mse": pde_mse,
        "inference_time_total_s": elapsed,
        "inference_time_per_sample_ms": 1000.0 * elapsed / f.shape[0],
    }


# -------------------------------------------------------------------------
# 7. Training
# -------------------------------------------------------------------------

def train_one_model(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    phys: PhysicsConfig,
    dataset: Dict[str, torch.Tensor],
    device: torch.device,
    out_dir: Path,
    run_name: str,
) -> Tuple[nn.Module, pd.DataFrame, Dict[str, float]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    model = make_model(model_cfg, phys).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=train_cfg.scheduler_step, gamma=train_cfg.scheduler_gamma)

    train_f = dataset["train_f"].to(device)
    train_u = dataset["train_u"].to(device)
    test_f = dataset["test_f"]
    test_u = dataset["test_u"]

    n_train = train_f.shape[0]
    best_err = float("inf")
    best_epoch = -1
    best_path = ckpt_dir / f"{run_name}_best.pt"
    history = []

    start = time.time()
    epochs_no_improve = 0

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        loss_epoch = 0.0
        data_epoch = 0.0
        pde_epoch = 0.0
        nb = 0

        for b0 in range(0, n_train, train_cfg.batch_size):
            idx = perm[b0:b0 + train_cfg.batch_size]
            fb = train_f[idx]
            ub = train_u[idx]

            pred = model(fb)

            loss_data = torch.mean((pred - ub) ** 2)
            loss_pde = pde_residual_loss(pred, fb, phys)
            loss = train_cfg.w_data * loss_data + train_cfg.w_pde * loss_pde

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            loss_epoch += loss.item()
            data_epoch += loss_data.item()
            pde_epoch += loss_pde.item()
            nb += 1

        sched.step()

        if epoch % train_cfg.eval_every == 0 or epoch == 1 or epoch == train_cfg.epochs:
            metrics = evaluate_model(model, test_f, test_u, phys, batch_size=64, device=device)
            test_err = metrics["rel_l2_u"]
            elapsed = time.time() - start

            row = {
                "run_name": run_name,
                "epoch": epoch,
                "wall_time_s": elapsed,
                "lr": sched.get_last_lr()[0],
                "total_loss": loss_epoch / nb,
                "data_loss": data_epoch / nb,
                "pde_loss": pde_epoch / nb,
                **metrics,
            }
            history.append(row)

            if test_err < best_err:
                best_err = test_err
                best_epoch = epoch
                torch.save({
                    "model_state": model.state_dict(),
                    "model_cfg": asdict(model_cfg),
                    "train_cfg": asdict(train_cfg),
                    "phys": asdict(phys),
                    "best_epoch": best_epoch,
                    "best_rel_l2_u": best_err,
                }, best_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += train_cfg.eval_every

            print(
                f"[{run_name}] epoch {epoch:5d}/{train_cfg.epochs} "
                f"loss={loss_epoch/nb:.3e} data={data_epoch/nb:.3e} pde={pde_epoch/nb:.3e} "
                f"test_u={test_err:.4e} best={best_err:.4e} time={elapsed:.1f}s"
            )

            if epochs_no_improve >= train_cfg.patience:
                print(f"[{run_name}] early stopping at epoch {epoch}")
                break

    # Load best checkpoint before final evaluation.
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    final_metrics = evaluate_model(model, test_f, test_u, phys, batch_size=64, device=device)
    final_metrics["best_epoch"] = best_epoch
    final_metrics["best_rel_l2_u"] = best_err
    final_metrics["n_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "logs" / f"{run_name}_history.csv", index=False)
    return model, hist_df, final_metrics


# -------------------------------------------------------------------------
# 8. Plotting
# -------------------------------------------------------------------------

def ensure_dirs(root: Path) -> Dict[str, Path]:
    paths = {
        "root": root,
        "figures": root / "figures",
        "tables": root / "tables",
        "checkpoints": root / "checkpoints",
        "logs": root / "logs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def save_training_plots(hist: pd.DataFrame, fig_dir: Path, run_name: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.semilogy(hist["epoch"], hist["total_loss"], label="total loss")
    plt.semilogy(hist["epoch"], hist["data_loss"], label="data loss")
    plt.semilogy(hist["epoch"], hist["pde_loss"], label="PDE loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"{run_name}_training_loss.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(hist["wall_time_s"], hist["rel_l2_u"], label=r"displacement relative $L^2$")
    plt.xlabel("Wall-clock time (s)")
    plt.ylabel(r"Relative $L^2$ error")
    plt.title("Test Error vs Training Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"{run_name}_test_error_vs_time.png", dpi=300)
    plt.close()


@torch.no_grad()
def save_field_plots(
    model: nn.Module,
    dataset: Dict[str, torch.Tensor],
    phys: PhysicsConfig,
    fig_dir: Path,
    run_name: str,
    device: torch.device,
    sample_idx: int = 0,
) -> None:
    model.eval()
    f = dataset["test_f"][sample_idx:sample_idx+1].to(device)
    u_true = dataset["test_u"][sample_idx:sample_idx+1]
    u_pred = model(f).cpu()

    # Displacement comparison.
    ux_true = u_true[0, :, :, 0].numpy()
    uy_true = u_true[0, :, :, 1].numpy()
    ux_pred = u_pred[0, :, :, 0].numpy()
    uy_pred = u_pred[0, :, :, 1].numpy()
    err_x = np.abs(ux_true - ux_pred)
    err_y = np.abs(uy_true - uy_pred)

    fields = [ux_true, ux_pred, err_x, uy_true, uy_pred, err_y]
    titles = [
        r"$u_x$ true", r"$u_x$ pred", r"$|u_x^{true}-u_x^{pred}|$",
        r"$u_y$ true", r"$u_y$ pred", r"$|u_y^{true}-u_y^{pred}|$",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, im_data, title in zip(axes.ravel(), fields, titles):
        im = ax.imshow(im_data, origin="lower", extent=[0, 1, 0, 1], aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("2D Linear Elasticity: Displacement Field Comparison")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{run_name}_field_comparison.png", dpi=300)
    plt.close()

    # Stress components from prediction.
    _, sig_pred = fd_strain_stress(u_pred, phys)
    sig = sig_pred[0].numpy()
    stress_fields = [sig[..., 0], sig[..., 1], sig[..., 2]]
    stress_titles = [r"$\sigma_{xx}$", r"$\sigma_{yy}$", r"$\sigma_{xy}$"]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, im_data, title in zip(axes.ravel(), stress_fields, stress_titles):
        im = ax.imshow(im_data, origin="lower", extent=[phys.h, 1-phys.h, phys.h, 1-phys.h], aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Stress Components Computed from Predicted Displacement")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{run_name}_stress_components.png", dpi=300)
    plt.close()

    # Body force and displacement sample.
    f_cpu = f.cpu()
    fx = f_cpu[0, :, :, 0].numpy()
    fy = f_cpu[0, :, :, 1].numpy()

    sample_fields = [fx, fy, ux_true, uy_true]
    sample_titles = [r"$f_x$", r"$f_y$", r"$u_x$", r"$u_y$"]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, im_data, title in zip(axes.ravel(), sample_fields, sample_titles):
        im = ax.imshow(im_data, origin="lower", extent=[0, 1, 0, 1], aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Representative Test Sample")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{run_name}_test_sample.png", dpi=300)
    plt.close()


def plot_summary_bars(results: pd.DataFrame, fig_dir: Path, metric: str, filename: str, title: str) -> None:
    df = results.copy()
    if "run_name" not in df.columns:
        return
    df = df.sort_values(metric)

    plt.figure(figsize=(max(7, 0.45 * len(df)), 4))
    plt.bar(df["run_name"], df[metric])
    plt.ylabel(metric)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / filename, dpi=300)
    plt.close()


def plot_ablation(df: pd.DataFrame, xcol: str, ycol: str, groupcol: Optional[str], fig_dir: Path, filename: str, title: str) -> None:
    plt.figure(figsize=(6, 4))
    if groupcol and groupcol in df.columns:
        for key, g in df.groupby(groupcol):
            g = g.sort_values(xcol)
            plt.plot(g[xcol], g[ycol], marker="o", label=str(key))
        plt.legend()
    else:
        g = df.sort_values(xcol)
        plt.plot(g[xcol], g[ycol], marker="o")
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fig_dir / filename, dpi=300)
    plt.close()


# -------------------------------------------------------------------------
# 9. Study definitions
# -------------------------------------------------------------------------

def make_quick_runs() -> List[Tuple[str, ModelConfig, TrainConfig, DataConfig]]:
    runs = []

    base_data = DataConfig(n_train=300, n_test=60, true_modes=12, coeff_scale=0.04)
    base_train = TrainConfig(epochs=250, batch_size=16, w_data=1.0, w_pde=1e-4, eval_every=10)

    runs.append((
        "spectral_PI_modes12_n300_wpde1e-4",
        ModelConfig("spectral_pideeponet", n_modes_model=12, hidden=128, depth=3),
        base_train,
        base_data,
    ))
    runs.append((
        "spectral_dataonly_modes12_n300",
        ModelConfig("spectral_pideeponet", n_modes_model=12, hidden=128, depth=3),
        TrainConfig(epochs=250, batch_size=16, w_data=1.0, w_pde=0.0, eval_every=10),
        base_data,
    ))
    runs.append((
        "mlp_trunk_PI_modes12_n300",
        ModelConfig("mlp_trunk_deeponet", n_modes_model=12, hidden=128, depth=3),
        base_train,
        base_data,
    ))
    runs.append((
        "unet_surrogate_n300",
        ModelConfig("unet_surrogate", n_modes_model=12, hidden=128, depth=3),
        base_train,
        base_data,
    ))

    return runs


def make_publication_runs() -> List[Tuple[str, ModelConfig, TrainConfig, DataConfig]]:
    runs = []

    # Baseline comparison.
    base_data = DataConfig(n_train=1000, n_test=200, true_modes=16, coeff_scale=0.04)
    base_train = TrainConfig(epochs=800, batch_size=32, w_data=1.0, w_pde=1e-4, eval_every=10)

    runs.append(("spectral_PI_main", ModelConfig("spectral_pideeponet", 16, 192, 4), base_train, base_data))
    runs.append(("spectral_dataonly", ModelConfig("spectral_pideeponet", 16, 192, 4), TrainConfig(epochs=800, batch_size=32, w_data=1.0, w_pde=0.0, eval_every=10), base_data))
    runs.append(("spectral_phys_heavy", ModelConfig("spectral_pideeponet", 16, 192, 4), TrainConfig(epochs=800, batch_size=32, w_data=1.0, w_pde=1e-3, eval_every=10), base_data))
    runs.append(("mlp_trunk_PI", ModelConfig("mlp_trunk_deeponet", 16, 192, 4), base_train, base_data))
    runs.append(("unet_surrogate", ModelConfig("unet_surrogate", 16, 128, 3), base_train, base_data))

    # Sine-mode ablation.
    for M in [4, 8, 12, 16, 24]:
        runs.append((
            f"ablation_modes_{M}",
            ModelConfig("spectral_pideeponet", M, 192, 4),
            TrainConfig(epochs=600, batch_size=32, w_data=1.0, w_pde=1e-4, eval_every=10),
            base_data,
        ))

    # Training-sample ablation.
    for ntr in [100, 300, 1000, 3000]:
        runs.append((
            f"ablation_ntrain_{ntr}",
            ModelConfig("spectral_pideeponet", 16, 192, 4),
            TrainConfig(epochs=600, batch_size=32, w_data=1.0, w_pde=1e-4, eval_every=10),
            DataConfig(n_train=ntr, n_test=200, true_modes=16, coeff_scale=0.04),
        ))

    # PDE weight ablation.
    for wpde in [0.0, 1e-6, 1e-5, 1e-4, 1e-3]:
        runs.append((
            f"ablation_wpde_{wpde:g}",
            ModelConfig("spectral_pideeponet", 16, 192, 4),
            TrainConfig(epochs=600, batch_size=32, w_data=1.0, w_pde=wpde, eval_every=10),
            base_data,
        ))

    return runs


# -------------------------------------------------------------------------
# 10. Main
# -------------------------------------------------------------------------

def run_study(args) -> None:
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    root = Path(args.out_dir)
    paths = ensure_dirs(root)

    phys = PhysicsConfig(res=args.res, E=args.E, nu=args.nu, plane=args.plane)
    lam, mu = phys.lame

    print("Physics configuration:")
    print(json.dumps(asdict(phys), indent=2))
    print(f"Lamé parameters: lambda={lam:.6f}, mu={mu:.6f}")

    if args.study == "quick":
        runs = make_quick_runs()
    elif args.study == "publication":
        runs = make_publication_runs()
    elif args.study == "single":
        data_cfg = DataConfig(n_train=args.n_train, n_test=args.n_test, true_modes=args.true_modes, coeff_scale=args.coeff_scale)
        train_cfg = TrainConfig(
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, w_data=args.w_data, w_pde=args.w_pde,
            eval_every=args.eval_every,
        )
        model_cfg = ModelConfig(args.model_name, args.n_modes_model, args.hidden, args.depth)
        runs = [(args.run_name, model_cfg, train_cfg, data_cfg)]
    else:
        raise ValueError("--study must be quick, publication, or single")

    all_results = []
    all_histories = []

    dataset_cache: Dict[str, Dict[str, torch.Tensor]] = {}

    for run_name, model_cfg, train_cfg, data_cfg in runs:
        print("\n" + "=" * 80)
        print(f"Run: {run_name}")
        print("=" * 80)

        data_key = json.dumps(asdict(data_cfg), sort_keys=True)
        if data_key not in dataset_cache:
            dataset_cache[data_key] = make_dataset(data_cfg, phys)
        dataset = dataset_cache[data_key]

        # Move logs dir exists.
        (paths["logs"]).mkdir(exist_ok=True, parents=True)

        model, hist, metrics = train_one_model(
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            phys=phys,
            dataset=dataset,
            device=device,
            out_dir=root,
            run_name=run_name,
        )

        # Add metadata.
        row = {
            "run_name": run_name,
            **asdict(model_cfg),
            **{f"train_{k}": v for k, v in asdict(train_cfg).items()},
            **{f"data_{k}": v for k, v in asdict(data_cfg).items()},
            **metrics,
        }
        all_results.append(row)

        hist = hist.copy()
        hist["run_name"] = run_name
        all_histories.append(hist)

        save_training_plots(hist, paths["figures"], run_name)
        save_field_plots(model, dataset, phys, paths["figures"], run_name, device=device, sample_idx=0)

        # Save interim results after each run.
        pd.DataFrame(all_results).to_csv(paths["tables"] / "results_summary_interim.csv", index=False)
        pd.concat(all_histories, ignore_index=True).to_csv(paths["tables"] / "training_histories_interim.csv", index=False)

    results_df = pd.DataFrame(all_results)
    histories_df = pd.concat(all_histories, ignore_index=True) if all_histories else pd.DataFrame()

    results_df.to_csv(paths["tables"] / "results_summary.csv", index=False)
    histories_df.to_csv(paths["tables"] / "training_histories.csv", index=False)

    # Baseline and metric plots.
    plot_summary_bars(results_df, paths["figures"], "rel_l2_u", "summary_displacement_error.png", "Displacement Relative L2 Error")
    plot_summary_bars(results_df, paths["figures"], "rel_l2_stress", "summary_stress_error.png", "Stress Relative L2 Error")
    plot_summary_bars(results_df, paths["figures"], "rel_energy_error", "summary_energy_error.png", "Strain-Energy Relative Error")
    plot_summary_bars(results_df, paths["figures"], "inference_time_per_sample_ms", "summary_inference_time.png", "Inference Time per Sample")

    # Ablation plots.
    if "ablation_modes" in " ".join(results_df["run_name"].tolist()):
        mode_rows = results_df[results_df["run_name"].str.contains("ablation_modes")].copy()
        if not mode_rows.empty:
            mode_rows["n_modes"] = mode_rows["run_name"].str.extract(r"ablation_modes_(\d+)").astype(int)
            plot_ablation(mode_rows, "n_modes", "rel_l2_u", None, paths["figures"], "ablation_modes_error.png", "Effect of Number of Sine Modes")

    ntrain_rows = results_df[results_df["run_name"].str.contains("ablation_ntrain")].copy()
    if not ntrain_rows.empty:
        ntrain_rows["n_train"] = ntrain_rows["run_name"].str.extract(r"ablation_ntrain_(\d+)").astype(int)
        plot_ablation(ntrain_rows, "n_train", "rel_l2_u", None, paths["figures"], "ablation_ntrain_error.png", "Effect of Training Sample Size")

    wpde_rows = results_df[results_df["run_name"].str.contains("ablation_wpde")].copy()
    if not wpde_rows.empty:
        wpde_rows["w_pde"] = wpde_rows["run_name"].str.extract(r"ablation_wpde_([0-9eE\.-]+)").astype(float)
        plot_ablation(wpde_rows, "w_pde", "rel_l2_u", None, paths["figures"], "ablation_wpde_error.png", "Effect of PDE-Loss Weight")

    # Save configuration.
    with open(root / "study_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "physics": asdict(phys),
            "lambda": lam,
            "mu": mu,
            "n_runs": len(runs),
        }, f, indent=2)

    print("\nStudy complete.")
    print(f"Results saved to: {root.resolve()}")
    print(f"Main summary table: {paths['tables'] / 'results_summary.csv'}")
    print(f"Figures directory: {paths['figures']}")


def parse_args():
    p = argparse.ArgumentParser(description="Continuum-mechanics publication-style PIDeepONet study for 2D linear elasticity.")

    p.add_argument("--study", type=str, default="quick", choices=["quick", "publication", "single"])
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out_dir", type=str, default="results_elasticity_publication")

    # Physics.
    p.add_argument("--res", type=int, default=29)
    p.add_argument("--E", type=float, default=1.0)
    p.add_argument("--nu", type=float, default=0.30)
    p.add_argument("--plane", type=str, default="strain", choices=["strain", "stress"])

    # Single-run data config.
    p.add_argument("--n_train", type=int, default=300)
    p.add_argument("--n_test", type=int, default=60)
    p.add_argument("--true_modes", type=int, default=12)
    p.add_argument("--coeff_scale", type=float, default=0.04)

    # Single-run training config.
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--w_data", type=float, default=1.0)
    p.add_argument("--w_pde", type=float, default=1e-4)
    p.add_argument("--eval_every", type=int, default=10)

    # Single-run model config.
    p.add_argument("--model_name", type=str, default="spectral_pideeponet",
                   choices=["spectral_pideeponet", "mlp_trunk_deeponet", "unet_surrogate"])
    p.add_argument("--n_modes_model", type=int, default=12)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--run_name", type=str, default="single_run")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_study(args)
