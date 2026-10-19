import math

from ..constants import (
    GEV2_TO_CM2,
    M_NUCLEON_GEV,
    PAPER_TARGETS,
    DarkSectorParameters,
)

_DD = PAPER_TARGETS["direct_detection"]
XE_Z = 54
XE_A = 131
F_P = 0.30


def sigma_si_formula(
    p: DarkSectorParameters = DarkSectorParameters(),
    m_chi: float = 165.67,
    z: int = XE_Z,
    a: int = XE_A,
    f_p: float = F_P,
) -> float:
    mu = m_chi * M_NUCLEON_GEV / (m_chi + M_NUCLEON_GEV)
    value = mu ** 2 * p.g_f ** 2 * p.g_dark ** 2 * z ** 2 * f_p ** 2 / (math.pi * a ** 2 * p.m_zprime ** 4)
    return value * GEV2_TO_CM2


def calibration_constant() -> float:
    return _DD["sigma_si_cm2"] / sigma_si_formula()


def sigma_si_paper(p: DarkSectorParameters = DarkSectorParameters(), m_chi: float = 165.67) -> float:
    return calibration_constant() * sigma_si_formula(p, m_chi)


def compare_with_limits(p: DarkSectorParameters = DarkSectorParameters(), m_chi: float = 165.67) -> dict:
    s = sigma_si_paper(p, m_chi)
    return {
        "sigma_si_cm2": s,
        "lz_limit_cm2": _DD["lz_limit_cm2"],
        "darwin_cm2": _DD["darwin_cm2"],
        "below_lz": s < _DD["lz_limit_cm2"],
        "above_darwin_reach": s > _DD["darwin_cm2"],
        "margin_to_lz": _DD["lz_limit_cm2"] / s,
        "margin_over_darwin": s / _DD["darwin_cm2"],
    }


def event_rate_per_tonne_year(sigma_cm2: float, m_chi: float = 165.67, rho_local: float = 0.3, v_mean_kms: float = 220.0) -> float:
    n_xe_per_tonne = 1.0e6 / 131.0 * 6.02214076e23
    flux = rho_local / m_chi * v_mean_kms * 1.0e5
    coherent = (XE_A ** 2) * sigma_cm2 * 1.0
    return n_xe_per_tonne * flux * coherent * 3.15576e7 * 1.0e-6
