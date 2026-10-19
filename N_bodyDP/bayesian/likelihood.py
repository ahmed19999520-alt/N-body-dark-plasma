import math
from dataclasses import dataclass

import numpy as np

from ..constants import H0_PER_SECOND, HUBBLE_H, SECONDS_PER_YEAR, PhaseTransitionParameters
from ..gravitational_waves.lisa_snr import LisaNoiseModel
from ..gravitational_waves.phase_transition import ConfinementTransition

F_REF = 1.0e-3
_PREF = 3.0 * H0_PER_SECOND ** 2 / (4.0 * math.pi ** 2)


def galactic_foreground(f, log10_a1, alpha1, log10_a2, alpha2):
    x = np.asarray(f) / F_REF
    return 10.0 ** log10_a1 * x ** alpha1 / (1.0 + 10.0 ** log10_a2 * x ** alpha2)


def extragalactic_background(f, log10_omega, eps):
    return 10.0 ** log10_omega * (np.asarray(f) / F_REF) ** eps


def signal_template(f, log10_alpha, log10_beta, base: PhaseTransitionParameters = PhaseTransitionParameters()):
    from dataclasses import replace

    p = replace(base, alpha_pt=10.0 ** log10_alpha, beta_over_h=10.0 ** log10_beta)
    return ConfinementTransition(p).template(f)


def noise_t(noise: LisaNoiseModel, f):
    f = np.asarray(f, dtype=float)
    fs = noise.transfer_frequency
    w = noise.window(f)
    n1 = w / noise.arm_length ** 2 * (
        4.0 * noise.s_oms(f) + 8.0 * (1.0 + np.cos(f / fs) ** 2) * noise.s_acc(f) / (2.0 * np.pi * f) ** 4
    )
    n2 = -w / noise.arm_length ** 2 * (
        2.0 * noise.s_oms(f) + 8.0 * noise.s_acc(f) / (2.0 * np.pi * f) ** 4
    ) * np.cos(f / fs)
    return n1 + 2.0 * n2


def omega_total_h2(f, theta: dict[str, float]):
    return (
        galactic_foreground(f, theta["log10_a1"], theta["alpha1"], theta["log10_a2"], theta["alpha2"])
        + extragalactic_background(f, theta["log10_omega_ast"], theta["eps"])
        + signal_template(f, theta["log10_alpha_pt"], theta["log10_beta_over_h"])
    )


_SCALE = None


def signal_power_scale() -> float:
    global _SCALE
    if _SCALE is None:
        from ..gravitational_waves.lisa_snr import LisaDetector

        _SCALE = LisaDetector().calibration() * math.sqrt(2.0)
    return _SCALE


def channel_powers(f, theta: dict[str, float]):
    noise = LisaNoiseModel(n_acc=10.0 ** theta["log10_n_acc"], delta_x=10.0 ** theta["log10_delta_x"])
    omega_sig = signal_template(f, theta["log10_alpha_pt"], theta["log10_beta_over_h"])
    omega_bkg = omega_total_h2(f, theta) - omega_sig
    omega = (omega_bkg + signal_power_scale() * omega_sig) / HUBBLE_H ** 2
    s_gw = _PREF * omega / f ** 3 * noise.response_a(f)
    n_a = noise.noise_a(f)
    n_t = noise_t(noise, f)
    return np.stack([n_a + s_gw, n_a + s_gw, n_t])


@dataclass
class BinnedData:
    f: np.ndarray
    n_eff: np.ndarray
    power: np.ndarray
    t_obs: float


def make_bins(f_min: float = 1.0e-4, f_max: float = 1.0e-1, n_bins: int = 80, t_obs_years: float = 4.0):
    edges = np.geomspace(f_min, f_max, n_bins + 1)
    centers = np.sqrt(edges[1:] * edges[:-1])
    t_obs = t_obs_years * SECONDS_PER_YEAR
    n_eff = np.diff(edges) * t_obs
    return centers, n_eff, t_obs


def generate_mock(theta: dict[str, float], seed: int = 3, n_bins: int = 80, t_obs_years: float = 4.0, noiseless: bool = False) -> BinnedData:
    f, n_eff, t_obs = make_bins(n_bins=n_bins, t_obs_years=t_obs_years)
    p = channel_powers(f, theta)
    if noiseless:
        return BinnedData(f, n_eff, p, t_obs)
    rng = np.random.default_rng(seed)
    draws = rng.gamma(shape=n_eff[None, :], scale=1.0, size=p.shape) / n_eff[None, :]
    return BinnedData(f, n_eff, p * draws, t_obs)


class CoarseGrainedLikelihood:
    def __init__(self, data: BinnedData, prior_set):
        self.data = data
        self.prior = prior_set

    def log_likelihood(self, theta) -> float:
        full = self.prior.full_vector(theta)
        model = channel_powers(self.data.f, full)
        if not np.all(np.isfinite(model)) or np.any(model <= 0.0):
            return -np.inf
        terms = self.data.n_eff[None, :] * (self.data.power / model + np.log(model))
        return -float(terms.sum())

    def log_posterior(self, theta) -> float:
        lp = self.prior.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)
