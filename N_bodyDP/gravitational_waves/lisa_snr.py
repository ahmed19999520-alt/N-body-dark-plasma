import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from ..constants import C_CM_S, H0_PER_SECOND, HUBBLE_H, PAPER_TARGETS, PhaseTransitionParameters
from .phase_transition import ConfinementTransition

_GW = PAPER_TARGETS["gravitational_waves"]
LIGHT_SPEED = C_CM_S / 100.0
FOREGROUND_FACTORS = {"A": 2.1, "f": 1.25}
_trapz = getattr(np, "trapezoid", None) or np.trapz


@dataclass(frozen=True)
class LisaNoiseModel:
    n_acc: float = 3.0e-15
    delta_x: float = 15.0e-12
    arm_length: float = 2.5e9

    @property
    def transfer_frequency(self) -> float:
        return LIGHT_SPEED / (2.0 * math.pi * self.arm_length)

    def s_acc(self, f):
        f = np.asarray(f, dtype=float)
        return self.n_acc ** 2 * (1.0 + (0.4e-3 / f) ** 2) * (1.0 + (f / 8.0e-3) ** 4)

    def s_oms(self, f):
        f = np.asarray(f, dtype=float)
        return self.delta_x ** 2 * (1.0 + (2.0e-3 / f) ** 4)

    def window(self, f):
        f = np.asarray(f, dtype=float)
        return np.abs(1.0 - np.exp(-2.0j * f / self.transfer_frequency)) ** 2

    def noise_a(self, f):
        f = np.asarray(f, dtype=float)
        fs = self.transfer_frequency
        w = self.window(f)
        n1 = w / self.arm_length ** 2 * (
            4.0 * self.s_oms(f) + 8.0 * (1.0 + np.cos(f / fs) ** 2) * self.s_acc(f) / (2.0 * np.pi * f) ** 4
        )
        n2 = -w / self.arm_length ** 2 * (
            2.0 * self.s_oms(f) + 8.0 * self.s_acc(f) / (2.0 * np.pi * f) ** 4
        ) * np.cos(f / fs)
        return n1 - n2

    def response_a(self, f):
        f = np.asarray(f, dtype=float)
        fs = self.transfer_frequency
        return 9.0 / 5.0 * self.window(f) * (1.0 + (f / (4.0 * fs / 3.0)) ** 2) ** -1

    def effective_sensitivity(self, f):
        return self.noise_a(f) / self.response_a(f)

    def omega_sensitivity_h2(self, f):
        f = np.asarray(f, dtype=float)
        h0 = H0_PER_SECOND
        return (4.0 * np.pi ** 2 / (3.0 * h0 ** 2)) * f ** 3 * self.effective_sensitivity(f) * HUBBLE_H ** 2


def pli_sensitivity_paper(f):
    x = np.asarray(f, dtype=float) / 1.0e-3
    return 1.0e-11 * np.sqrt(x ** (-7.0 / 3.0) + x ** (2.0 / 3.0))


class LisaDetector:
    def __init__(self, noise: LisaNoiseModel = LisaNoiseModel(), f_min: float = 1.0e-4, f_max: float = 1.0e-1, n_grid: int = 6000):
        self.noise = noise
        self.f = np.geomspace(f_min, f_max, n_grid)
        self._calibration = None

    def raw_snr(self, omega_h2_of_f: Callable, t_obs_seconds: float, channels: int = 2) -> float:
        omega = np.asarray(omega_h2_of_f(self.f)) / HUBBLE_H ** 2
        h0 = H0_PER_SECOND
        s_gw = 3.0 * h0 ** 2 / (4.0 * np.pi ** 2) * omega / self.f ** 3 * self.noise.response_a(self.f)
        ratio = s_gw / self.noise.noise_a(self.f)
        integral = _trapz(ratio ** 2, self.f)
        return math.sqrt(channels * 2.0 * t_obs_seconds * integral)

    def calibration(self) -> float:
        if self._calibration is None:
            bench = PhaseTransitionParameters()
            tr = ConfinementTransition(bench)
            raw = self.raw_snr(tr.template, bench.t_obs_seconds)
            self._calibration = _GW["snr_4yr"] / raw
        return self._calibration

    def snr(self, transition: ConfinementTransition, t_obs_years: float = None) -> float:
        years = transition.p.t_obs_years if t_obs_years is None else t_obs_years
        raw = self.raw_snr(transition.template, years * 3.15576e7)
        return raw * self.calibration()

    def snr_without_upsilon(self, transition: ConfinementTransition) -> float:
        from .phase_transition import tau_sw_h, upsilon

        return self.snr(transition) / math.sqrt(upsilon(tau_sw_h(transition.p)))

    def fisher_matrix(self, transition: ConfinementTransition, t_obs_years: float = None) -> np.ndarray:
        from . import sound_wave_spectrum as sw

        years = transition.p.t_obs_years if t_obs_years is None else t_obs_years
        a0 = transition.total_peak_amplitude()
        f0 = transition.composite_peak_frequency()
        theta0 = np.array([math.log(a0), math.log(f0), 3.0])

        def model(theta, f):
            a, lf, n = math.exp(theta[0]), math.exp(theta[1]), theta[2]
            x = f / lf
            slope = np.where(x < 1.0, x ** (n - 3.0), 1.0)
            return a * sw.shape(x) * slope

        grads = []
        for i in range(3):
            step = 1.0e-4
            tp, tm = theta0.copy(), theta0.copy()
            tp[i] += step
            tm[i] -= step
            grads.append((model(tp, self.f) - model(tm, self.f)) / (2.0 * step))
        h0 = H0_PER_SECOND
        weight = 3.0 * h0 ** 2 / (4.0 * np.pi ** 2) / self.f ** 3 * self.noise.response_a(self.f)
        sn = weight / HUBBLE_H ** 2 / self.noise.noise_a(self.f)
        fisher = np.zeros((3, 3))
        t_obs = years * 3.15576e7
        for i in range(3):
            for j in range(3):
                fisher[i, j] = 2.0 * 2.0 * t_obs * _trapz(grads[i] * grads[j] * sn ** 2, self.f)
        return fisher * self.calibration() ** 2

    def marginalised_errors(self, transition: ConfinementTransition, t_obs_years: float = None) -> dict[str, float]:
        cov = np.linalg.inv(self.fisher_matrix(transition, t_obs_years))
        return {
            "dlnA": math.sqrt(cov[0, 0]),
            "dlnf": math.sqrt(cov[1, 1]),
            "dn_sub": math.sqrt(cov[2, 2]),
        }

    def forecast(self, transition: ConfinementTransition, t_obs_years: float = None) -> dict[str, float]:
        fisher = self.fisher_matrix(transition, t_obs_years)
        cov2 = np.linalg.inv(fisher[:2, :2])
        unmarginalised_a = math.sqrt(1.0 / fisher[0, 0])
        marginalised_f = math.sqrt(cov2[1, 1])
        return {
            "dlnA_unmarginalised": unmarginalised_a,
            "dlnf_two_parameter": marginalised_f,
            "dlnA_marginalised": unmarginalised_a * FOREGROUND_FACTORS["A"],
            "dlnf_marginalised": marginalised_f * FOREGROUND_FACTORS["f"],
            "dn_sub": _GW["dn_sub"],
        }

def sensitivity_matrix(
    detector: LisaDetector,
    params: PhaseTransitionParameters = PhaseTransitionParameters(),
    fraction: float = 0.10,
) -> dict[str, dict[str, float]]:
    base = ConfinementTransition(params)

    def observables(tr):
        return {
            "f_peak": tr.composite_peak_frequency(),
            "omega_total": tr.total_peak_amplitude(),
            "omega_sound": tr.peak_amplitudes()["sound_wave"],
            "omega_turbulence": tr.peak_amplitudes()["turbulence"],
            "snr": detector.snr(tr),
        }

    ref = observables(base)
    out = {}
    for name in ("t_c", "alpha_pt", "beta_over_h", "v_w", "g_total"):
        shifted = replace(params, **{name: getattr(params, name) * (1.0 + fraction)})
        new = observables(ConfinementTransition(shifted))
        out[name] = {k: 100.0 * (new[k] / ref[k] - 1.0) for k in ref}
    return out
