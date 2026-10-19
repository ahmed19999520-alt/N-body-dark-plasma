import math

import numpy as np

from .likelihood import CoarseGrainedLikelihood

RECONSTRUCTION_THRESHOLD_DEX = math.log10(1.25)


def posterior_geometry(samples: np.ndarray, weights: np.ndarray = None) -> dict[str, float]:
    samples = np.asarray(samples)
    if weights is None:
        weights = np.ones(samples.shape[0])
    w = weights / weights.sum()
    mean = w @ samples
    d = samples - mean
    cov = (d * w[:, None]).T @ d
    return geometry_from_covariance(cov, mean)


_KEYS = ("sigma1", "sigma2", "rho", "one_minus_rho2", "lambda_par", "lambda_perp", "kappa_deg", "area", "sqrt_lambda_perp")


def geometry_from_covariance(cov: np.ndarray, mean: np.ndarray = None) -> dict[str, float]:
    vals, vecs = np.linalg.eigh(cov)
    if vals[0] <= 0.0 or cov[0, 0] <= 0.0 or cov[1, 1] <= 0.0:
        out = {k: float("nan") for k in _KEYS}
        out["reconstructed"] = False
        return out
    lam_perp, lam_par = float(vals[0]), float(vals[1])
    s1, s2 = math.sqrt(cov[0, 0]), math.sqrt(cov[1, 1])
    rho = cov[0, 1] / (s1 * s2)
    out = {
        "sigma1": s1,
        "sigma2": s2,
        "rho": float(rho),
        "one_minus_rho2": float(1.0 - rho ** 2),
        "lambda_par": lam_par,
        "lambda_perp": lam_perp,
        "kappa_deg": lam_par / lam_perp,
        "area": math.sqrt(lam_par * lam_perp),
        "sqrt_lambda_perp": math.sqrt(lam_perp),
        "reconstructed": math.sqrt(lam_perp) < RECONSTRUCTION_THRESHOLD_DEX,
    }
    if mean is not None:
        out["mean1"], out["mean2"] = float(mean[0]), float(mean[1])
    return out


def fractional_uncertainty_dex(sigma_dex: float) -> float:
    return 10.0 ** sigma_dex - 1.0


def reconstruction_offset(median_log10: float, injected_log10: float) -> float:
    return abs(median_log10 - injected_log10)


def numerical_hessian(fn, x0: np.ndarray, steps: np.ndarray) -> np.ndarray:
    n = len(x0)
    h = np.zeros((n, n))
    f0 = fn(x0)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = steps[i]
        h[i, i] = (fn(x0 + ei) - 2.0 * f0 + fn(x0 - ei)) / steps[i] ** 2
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = steps[j]
            h[i, j] = h[j, i] = (
                fn(x0 + ei + ej) - fn(x0 + ei - ej) - fn(x0 - ei + ej) + fn(x0 - ei - ej)
            ) / (4.0 * steps[i] * steps[j])
    return h


def laplace_covariance(like: CoarseGrainedLikelihood, theta0: np.ndarray, step: float = 1.0e-3) -> np.ndarray:
    steps = np.full(len(theta0), step)
    hess = numerical_hessian(like.log_likelihood, theta0, steps)
    return np.linalg.inv(-hess)


def signal_block_geometry(like: CoarseGrainedLikelihood, theta0: np.ndarray, signal_names=("log10_alpha_pt", "log10_beta_over_h")) -> dict[str, float]:
    cov = laplace_covariance(like, theta0)
    idx = [like.prior.free.index(n) for n in signal_names]
    sub = cov[np.ix_(idx, idx)]
    return geometry_from_covariance(sub, theta0[idx])
