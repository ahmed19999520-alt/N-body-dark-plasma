import numpy as np

from ..constants import PhaseTransitionParameters
from .phase_transition import kappa_turb

PREFACTOR_HZ = 2.7e-10
H_OBS_HZ = 2.4e-21
_PEAK_NORM = None


def peak_frequency(p: PhaseTransitionParameters) -> float:
    return (
        PREFACTOR_HZ
        / p.v_w
        * p.beta_over_h
        * (p.t_c / 1.0e10)
        * (p.g_total / 100.0) ** (1.0 / 6.0)
    )


def peak_amplitude(p: PhaseTransitionParameters) -> float:
    a = p.alpha_pt
    k = kappa_turb(a, p.kappa_turb_ratio)
    return (
        3.35e-4
        * (1.0 / p.beta_over_h)
        * (k * a / (1.0 + a)) ** 1.5
        * (100.0 / p.g_total) ** (1.0 / 3.0)
        * p.v_w
    )


def _raw_shape(x, f_t):
    x = np.asarray(x, dtype=float)
    return x ** 3 / (1.0 + x) ** (11.0 / 3.0) / (1.0 + 8.0 * np.pi * x * f_t / H_OBS_HZ)


def shape(x, f_t=1.0e-3):
    global _PEAK_NORM
    if _PEAK_NORM is None:
        grid = np.geomspace(1.0e-2, 1.0e3, 40001)
        _PEAK_NORM = float(_raw_shape(grid, f_t).max())
    return _raw_shape(x, f_t) / _PEAK_NORM


def spectrum(f, p: PhaseTransitionParameters):
    f = np.asarray(f, dtype=float)
    f_t = peak_frequency(p)
    return peak_amplitude(p) * shape(f / f_t, f_t)
