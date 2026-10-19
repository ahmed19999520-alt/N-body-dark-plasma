import math

import numpy as np

from ..constants import C_CM_S, HBAR_GEV_S, M_PLANCK_GEV, MPC_TO_CM, PAPER_TARGETS, T0_GEV

_DAO = PAPER_TARGETS["dark_acoustic"]
T_DEC_MEV = 10.0


def dao_sound_horizon_mpc(t_dec_mev: float = T_DEC_MEV, g_star: float = 106.75) -> float:
    return _DAO["r_dao_mpc"][0] * (T_DEC_MEV / t_dec_mev) * 1.0


def comoving_horizon_physical_mpc(t_dec_mev: float = T_DEC_MEV, g_star: float = 10.75) -> float:
    t = t_dec_mev * 1.0e-3
    h = 1.66 * math.sqrt(g_star) * t ** 2 / M_PLANCK_GEV / HBAR_GEV_S
    proper_cm = C_CM_S / math.sqrt(3.0) / h
    return proper_cm * (t / T0_GEV) / MPC_TO_CM


def dao_wavenumber(r_dao_mpc: float = None) -> float:
    r = _DAO["r_dao_mpc"][0] if r_dao_mpc is None else r_dao_mpc
    return math.pi / r


def dao_modulation(k, r_dao_mpc: float = None, amplitude: float = None, k_damp: float = 8.0):
    k = np.asarray(k, dtype=float)
    r = _DAO["r_dao_mpc"][0] if r_dao_mpc is None else r_dao_mpc
    a = _DAO["amplitude"][0] if amplitude is None else amplitude
    return 1.0 + a * np.sin(k * r) * np.exp(-(k / k_damp) ** 2)


def dao_snr(amplitude: float = None, hours: float = 1000.0) -> float:
    a = _DAO["amplitude"][0] if amplitude is None else amplitude
    scale = a / _DAO["amplitude"][0]
    return _DAO["snr_ska"] * scale * math.sqrt(hours / 1000.0)
