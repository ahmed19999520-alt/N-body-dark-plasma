import numpy as np

from ..constants import PhaseTransitionParameters
from .phase_transition import kappa_sw, tau_sw_h, upsilon

PREFACTOR_HZ = 1.9e-10


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
    k = kappa_sw(a)
    ups = upsilon(tau_sw_h(p))
    return (
        3.24e-1
        * (1.0 / p.beta_over_h)
        * (k * a / (1.0 + a)) ** 2
        * (100.0 / p.g_total) ** (1.0 / 3.0)
        * p.v_w
        * ups
    )


def shape(x):
    x = np.asarray(x, dtype=float)
    return x ** 3 * (7.0 / (4.0 + 3.0 * x ** 2)) ** 3.5


def spectrum(f, p: PhaseTransitionParameters):
    f = np.asarray(f, dtype=float)
    return peak_amplitude(p) * shape(f / peak_frequency(p))
