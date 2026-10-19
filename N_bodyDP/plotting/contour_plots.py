import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..bayesian import (
    FIDUCIAL,
    SIGNAL_PARAMETERS,
    CoarseGrainedLikelihood,
    PriorSet,
    generate_mock,
    signal_block_geometry,
)
from ..constants import PhaseTransitionParameters
from ..gravitational_waves import ConfinementTransition, LisaDetector


def snr_map(alphas=None, betas=None, detector=None):
    alphas = np.geomspace(0.03, 0.3, 24) if alphas is None else alphas
    betas = np.geomspace(30.0, 400.0, 24) if betas is None else betas
    det = detector or LisaDetector()
    grid = np.zeros((len(betas), len(alphas)))
    for i, b in enumerate(betas):
        for j, a in enumerate(alphas):
            tr = ConfinementTransition(PhaseTransitionParameters(alpha_pt=float(a), beta_over_h=float(b)))
            grid[i, j] = det.snr(tr)
    return alphas, betas, grid


def plot_snr_map(alphas, betas, grid, path=None):
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    a, b = np.meshgrid(alphas, betas)
    cs = ax.contourf(a, b, grid, levels=20)
    fig.colorbar(cs, ax=ax, label="SNR")
    ax.contour(a, b, grid, levels=[5.0, 8.5], colors=["white", "red"])
    ax.plot([0.1], [100.0], "k*", ms=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("alpha_PT")
    ax.set_ylabel("beta/H")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    return fig


def diagnostic_maps(log_alphas=None, log_betas=None):
    la = np.linspace(-1.7, -0.5, 10) if log_alphas is None else log_alphas
    lb = np.linspace(1.4, 2.6, 10) if log_betas is None else log_betas
    keys = ("sigma1", "sigma2", "rho", "one_minus_rho2", "kappa_deg", "area", "sqrt_lambda_perp")
    maps = {k: np.full((len(lb), len(la)), np.nan) for k in keys}
    for i, b in enumerate(lb):
        for j, a in enumerate(la):
            truth = dict(FIDUCIAL)
            truth["log10_alpha_pt"] = float(a)
            truth["log10_beta_over_h"] = float(b)
            prior = PriorSet(free=SIGNAL_PARAMETERS, fixed=truth)
            like = CoarseGrainedLikelihood(generate_mock(truth, noiseless=True), prior)
            try:
                g = signal_block_geometry(like, np.array([float(a), float(b)]))
            except np.linalg.LinAlgError:
                continue
            for k in keys:
                maps[k][i, j] = g[k]
    return la, lb, maps


def plot_diagnostic_maps(la, lb, maps, path=None):
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0))
    panels = (
        ("one_minus_rho2", "1 - rho^2", False),
        ("kappa_deg", "log10 kappa_deg", True),
        ("sigma1", "sigma(log10 alpha_PT)", False),
        ("sigma2", "sigma(log10 beta/H)", False),
    )
    for ax, (key, label, log) in zip(axes.ravel(), panels):
        z = np.log10(maps[key]) if log else maps[key]
        im = ax.contourf(la, lb, z, levels=20)
        fig.colorbar(im, ax=ax, label=label)
        ax.plot([FIDUCIAL["log10_alpha_pt"]], [FIDUCIAL["log10_beta_over_h"]], "w*", ms=10)
        ax.set_xlabel("log10 alpha_PT")
        ax.set_ylabel("log10 beta/H")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    return fig
