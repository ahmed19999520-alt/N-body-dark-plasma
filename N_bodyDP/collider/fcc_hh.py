import math

from ..constants import GEV2_TO_CM2, PAPER_TARGETS, DarkSectorParameters

_C = PAPER_TARGETS["collider"]
K_QCD = 1.3
LUMI_PER_YEAR_FB = 100.0
LUMI_TOTAL_FB = 30.0e3


def sigma_pp_formula(
    s_hat: float = (2.0 * 165.67 * 1.5) ** 2,
    p: DarkSectorParameters = DarkSectorParameters(),
    m_chi: float = 165.67,
    parton_luminosity: float = 1.0,
) -> float:
    if s_hat <= 4.0 * m_chi ** 2:
        return 0.0
    kinematic = (1.0 + 2.0 * m_chi ** 2 / s_hat) * math.sqrt(1.0 - 4.0 * m_chi ** 2 / s_hat)
    value = p.g_f ** 2 * p.g_dark ** 2 / (12.0 * math.pi * s_hat) * kinematic * K_QCD * parton_luminosity
    return value * GEV2_TO_CM2 * 1.0e39


def sigma_pp_chi0_pair() -> float:
    return _C["sigma_pp_chi0chi0_fb"]


def events_per_year(sigma_fb: float = None, lumi_per_year_fb: float = LUMI_PER_YEAR_FB) -> float:
    s = _C["sigma_pp_chi0chi0_fb"] if sigma_fb is None else sigma_fb
    return s * lumi_per_year_fb


def total_events(sigma_fb: float, lumi_fb: float = LUMI_TOTAL_FB) -> float:
    return sigma_fb * lumi_fb
