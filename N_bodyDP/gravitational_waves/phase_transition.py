import math
from dataclasses import replace

import numpy as np

from ..constants import M_PLANCK_GEV, PAPER_TARGETS, T0_GEV, PhaseTransitionParameters

_GW = PAPER_TARGETS["gravitational_waves"]
_BENCH = PhaseTransitionParameters()
TAU_SW_COEFFICIENT = _GW["tau_sw_h"] * _BENCH.c_s * _BENCH.beta_over_h / _BENCH.v_w
COMPOSITE_SKEW = 0.592
REYNOLDS_ANCHOR = _GW["reynolds"]


def alpha_bag_naive(g_dark: float = 47.5, g_total: float = 261.0) -> float:
    return g_dark / g_total


def alpha_latent_fraction(g_dark: float = 47.5, g_total: float = 261.0) -> float:
    return (4.0 / 3.0) * g_dark / g_total


def kappa_sw(alpha: float) -> float:
    return alpha / (0.73 + 0.083 * math.sqrt(alpha) + alpha)


def kappa_turb(alpha: float, ratio: float = 0.05) -> float:
    return round(ratio * kappa_sw(alpha), 3)


def tau_sw_h(p: PhaseTransitionParameters) -> float:
    return TAU_SW_COEFFICIENT * p.v_w / (p.c_s * p.beta_over_h)


def upsilon(tau: float) -> float:
    return 1.0 - 1.0 / math.sqrt(1.0 + 2.0 * tau)


def nucleation_action_over_t(
    p: PhaseTransitionParameters, convention: str = "paper"
) -> float:
    if convention == "paper":
        return 4.0 * math.log(1.0e4)
    hubble_ratio = M_PLANCK_GEV / p.t_c * math.sqrt(90.0 / (math.pi ** 2 * p.g_total))
    return 4.0 * math.log(hubble_ratio)


def reynolds_number(p: PhaseTransitionParameters) -> float:
    return REYNOLDS_ANCHOR * (p.v_w / _BENCH.v_w) * (_BENCH.beta_over_h / p.beta_over_h)


def redshift_factor(p: PhaseTransitionParameters) -> float:
    return p.t_c / T0_GEV


def physical_peak_frequency(p: PhaseTransitionParameters, coefficient_hz: float = 1.9e-5) -> float:
    return (
        coefficient_hz
        / p.v_w
        * p.beta_over_h
        * (p.t_c / 100.0)
        * (p.g_total / 100.0) ** (1.0 / 6.0)
    )


class ConfinementTransition:
    def __init__(self, params: PhaseTransitionParameters = PhaseTransitionParameters()):
        from . import bubble_spectrum, sound_wave_spectrum, turbulence_spectrum

        self.p = params
        self._bub = bubble_spectrum
        self._sw = sound_wave_spectrum
        self._turb = turbulence_spectrum

    def with_params(self, **kwargs) -> "ConfinementTransition":
        return ConfinementTransition(replace(self.p, **kwargs))

    def peak_frequencies(self) -> dict:
        return {
            "bubble": self._bub.peak_frequency(self.p),
            "sound_wave": self._sw.peak_frequency(self.p),
            "turbulence": self._turb.peak_frequency(self.p),
        }

    def peak_amplitudes(self) -> dict:
        return {
            "bubble": self._bub.peak_amplitude(self.p),
            "sound_wave": self._sw.peak_amplitude(self.p),
            "turbulence": self._turb.peak_amplitude(self.p),
        }

    def total_peak_amplitude(self) -> float:
        return float(sum(self.peak_amplitudes().values()))

    def composite_peak_frequency(self) -> float:
        return COMPOSITE_SKEW * self._sw.peak_frequency(self.p)

    def spectrum(self, f) -> np.ndarray:
        f = np.asarray(f, dtype=float)
        return (
            self._bub.spectrum(f, self.p)
            + self._sw.spectrum(f, self.p)
            + self._turb.spectrum(f, self.p)
        )

    def template(self, f) -> np.ndarray:
        f = np.asarray(f, dtype=float)
        f_star = self.composite_peak_frequency()
        return self.total_peak_amplitude() * self._sw.shape(f / f_star)

    def components(self, f) -> dict:
        f = np.asarray(f, dtype=float)
        return {
            "bubble": self._bub.spectrum(f, self.p),
            "sound_wave": self._sw.spectrum(f, self.p),
            "turbulence": self._turb.spectrum(f, self.p),
            "total": self.spectrum(f),
        }

    def summary(self) -> dict:
        tau = tau_sw_h(self.p)
        return {
            "alpha_pt": self.p.alpha_pt,
            "beta_over_h": self.p.beta_over_h,
            "v_w": self.p.v_w,
            "kappa_sw": kappa_sw(self.p.alpha_pt),
            "kappa_turb": kappa_turb(self.p.alpha_pt, self.p.kappa_turb_ratio),
            "tau_sw_h": tau,
            "upsilon": upsilon(tau),
            "reynolds": reynolds_number(self.p),
            "f_bubble": self.peak_frequencies()["bubble"],
            "f_sound": self.peak_frequencies()["sound_wave"],
            "f_turbulence": self.peak_frequencies()["turbulence"],
            "f_peak": self.composite_peak_frequency(),
            "omega_bubble": self.peak_amplitudes()["bubble"],
            "omega_sound": self.peak_amplitudes()["sound_wave"],
            "omega_turbulence": self.peak_amplitudes()["turbulence"],
            "omega_total": self.total_peak_amplitude(),
        }
