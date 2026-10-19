import math

import numpy as np
from scipy.optimize import brentq

from ..constants import (
    GEV_INV2_TO_CM3_S,
    M_PLANCK_GEV,
    OMEGA_TO_YIELD,
)


def effective_dof(model, x: float) -> float:
    return float(np.sum(model.dof * (1.0 + model.delta) ** 1.5 * np.exp(-x * model.delta)))


def sigma_eff(model, x: float) -> float:
    geff = effective_dof(model, x)
    w = model.dof * (1.0 + model.delta) ** 1.5 * np.exp(-x * model.delta)
    return float(w @ model.sigma_matrix() @ w / geff ** 2)


def _hubble_over_neq_sigma(model, x: float, g_star: float) -> float:
    m0 = model.params.M0
    h = math.sqrt(math.pi ** 2 * g_star / 90.0) * m0 ** 2 / (M_PLANCK_GEV * x ** 2)
    geff = effective_dof(model, x)
    n_eq = geff * (m0 ** 3) * (1.0 / (2.0 * math.pi * x)) ** 1.5 * math.exp(-x)
    return n_eq * sigma_eff(model, x) / h


def solve_freeze_out(model, g_star: float = 106.75, bracket=(5.0, 80.0)) -> float:
    f = lambda x: math.log(_hubble_over_neq_sigma(model, x, g_star))
    return float(brentq(f, bracket[0], bracket[1]))


def omega_lee_weinberg(sigma_eff_cm3_s: float, x_f: float, g_star: float = 106.75) -> float:
    sigma_gev2 = sigma_eff_cm3_s / GEV_INV2_TO_CM3_S
    return 1.07e9 / (M_PLANCK_GEV * math.sqrt(g_star)) * x_f / sigma_gev2


def omega_from_yield(y_inf: float, m_chi: float) -> float:
    return OMEGA_TO_YIELD * m_chi * y_inf


def yield_from_omega(omega_h2: float, m_chi: float) -> float:
    return omega_h2 / (OMEGA_TO_YIELD * m_chi)
