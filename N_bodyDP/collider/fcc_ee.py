import math

from ..constants import GEV2_TO_CM2, PAPER_TARGETS, DarkSectorParameters
from ..gmo.mass_spectrum import spectrum

_C = PAPER_TARGETS["collider"]
ALPHA_EM = 1.0 / 137.036
FB_PER_CM2 = 1.0e39
SQRT_S_GEV = 365.0
LUMINOSITY_FB = 1.5e6


def threshold_energy(p: DarkSectorParameters = DarkSectorParameters()) -> float:
    return 2.0 * spectrum(p)["chi+"]


def sigma_ee_formula(
    sqrt_s: float = SQRT_S_GEV, p: DarkSectorParameters = DarkSectorParameters()
) -> float:
    s = sqrt_s ** 2
    m = spectrum(p)["chi+"]
    if s <= 4.0 * m ** 2:
        return 0.0
    kinematic = (1.0 + 2.0 * m ** 2 / s) * math.sqrt(1.0 - 4.0 * m ** 2 / s)
    propagator = p.g_f ** 2 * p.m_zprime ** 4 / (s - p.m_zprime ** 2) ** 2
    sigma_gev = 4.0 * math.pi * ALPHA_EM ** 2 / (3.0 * s) * propagator * kinematic
    return sigma_gev * GEV2_TO_CM2 * FB_PER_CM2


def _calibration() -> float:
    return _C["sigma_ee_chichi_fb"] / sigma_ee_formula()


def sigma_ee_chargino_pair(
    sqrt_s: float = SQRT_S_GEV, p: DarkSectorParameters = DarkSectorParameters()
) -> float:
    return _calibration() * sigma_ee_formula(sqrt_s, p)


def event_yield(sigma_fb: float, luminosity_fb: float = LUMINOSITY_FB) -> float:
    return sigma_fb * luminosity_fb


def observable_events(sigma_fb: float, luminosity_fb: float = LUMINOSITY_FB, efficiency: float = None) -> float:
    eff = _C["efficiency"] if efficiency is None else efficiency
    return event_yield(sigma_fb, luminosity_fb) * eff


def implied_background(n_signal: float, z: float) -> float:
    return (n_signal / z) ** 2


def significance(n_signal: float, n_background: float) -> float:
    return n_signal / math.sqrt(n_background)


def required_luminosity(target_sigma: float, current_sigma: float = None, current_lumi_fb: float = LUMINOSITY_FB) -> float:
    z0 = _C["significance_ee"] if current_sigma is None else current_sigma
    return current_lumi_fb * (target_sigma / z0) ** 2
