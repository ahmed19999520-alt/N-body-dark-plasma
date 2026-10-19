import numpy as np

from ..constants import PhaseTransitionParameters

_PEAK_NORM = None


def peak_frequency(p: PhaseTransitionParameters) -> float:
    return (
        2.63e-9
        * 0.62
        / (1.8 - 0.1 * p.v_w + p.v_w ** 2)
        * p.beta_over_h
        * (p.t_c / 1.0e10)
        * (p.g_total / 100.0) ** (1.0 / 6.0)
    )


def peak_amplitude(p: PhaseTransitionParameters) -> float:
    a = p.alpha_pt
    return (
        5.27e-2
        * (1.0 / p.beta_over_h) ** 2
        * (p.kappa_phi * a / (1.0 + a)) ** 2
        * (100.0 / p.g_total) ** (1.0 / 3.0)
        * 0.11 * p.v_w ** 3 / (0.42 + p.v_w ** 2)
    )


def _raw_shape(x):
    x = np.asarray(x, dtype=float)
    return 3.8 * x ** 2.8 / (1.0 + 2.8 * x ** 3.8)


def shape(x):
    global _PEAK_NORM
    if _PEAK_NORM is None:
        grid = np.geomspace(1.0e-2, 1.0e2, 20001)
        _PEAK_NORM = float(_raw_shape(grid).max())
    return _raw_shape(x) / _PEAK_NORM


def spectrum(f, p: PhaseTransitionParameters):
    f = np.asarray(f, dtype=float)
    return peak_amplitude(p) * shape(f / peak_frequency(p))
