import math
from dataclasses import replace

import numpy as np

from ..constants import HBAR_GEV_S, DarkSectorParameters
from .quantum_numbers import OCTET, DarkState, get_state

_RATIO_KEYS = ("r21", "r31", "r41")


def mass(state: DarkState, p: DarkSectorParameters) -> float:
    t = state.darkicity
    return (
        p.M0
        + p.beta * state.hypercharge
        + p.gamma * state.casimir_term
        + p.delta * t
        + p.eta * t * t
        + p.y_chi * p.v_delta * state.theta
    )


def spectrum(p: DarkSectorParameters = DarkSectorParameters()) -> dict[str, float]:
    return {s.name: mass(s, p) for s in OCTET}


def state_mass(name: str, p: DarkSectorParameters = DarkSectorParameters()) -> float:
    return mass(get_state(name), p)


def masses_array(p: DarkSectorParameters = DarkSectorParameters()) -> np.ndarray:
    return np.array([mass(s, p) for s in OCTET])


def splittings(p: DarkSectorParameters = DarkSectorParameters()) -> dict[str, float]:
    m = spectrum(p)
    return {k: v - m["chi0"] for k, v in m.items() if k != "chi0"}


def mass_ratios(p: DarkSectorParameters = DarkSectorParameters()) -> dict[str, float]:
    m = spectrum(p)
    return {
        "r21": m["chi++"] / m["chi0"],
        "r31": m["Sigma0"] / m["chi0"],
        "r41": m["Lambda"] / m["chi0"],
    }


def is_chi0_lightest(p: DarkSectorParameters = DarkSectorParameters()) -> bool:
    m = spectrum(p)
    return all(v > m["chi0"] for k, v in m.items() if k != "chi0")


def ratio_worst_case_uncertainty(
    p: DarkSectorParameters = DarkSectorParameters(),
    fraction: float = 0.05,
    names=("beta", "gamma", "delta", "eta"),
) -> dict[str, float]:
    out = {k: 0.0 for k in _RATIO_KEYS}
    for name in names:
        value = getattr(p, name)
        up = mass_ratios(replace(p, **{name: value * (1.0 + fraction)}))
        dn = mass_ratios(replace(p, **{name: value * (1.0 - fraction)}))
        for k in out:
            out[k] += 0.5 * abs(up[k] - dn[k])
    return out


def ratio_monte_carlo(
    p: DarkSectorParameters = DarkSectorParameters(),
    fraction: float = 0.05,
    n_samples: int = 20000,
    seed: int = 2026,
    names=("beta", "gamma", "delta", "eta"),
) -> dict[str, tuple]:
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in _RATIO_KEYS}
    for _ in range(n_samples):
        kw = {n: getattr(p, n) * (1.0 + fraction * rng.uniform(-1.0, 1.0)) for n in names}
        r = mass_ratios(replace(p, **kw))
        for k in draws:
            draws[k].append(r[k])
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in draws.items()}


def gst_consistency(
    p: DarkSectorParameters = DarkSectorParameters(),
    epsilon_dark: float = 0.12,
    wilson_factor: float = 1.56,
) -> dict[str, float]:
    ratio = abs(p.beta) / p.gamma
    estimate = epsilon_dark * wilson_factor
    return {
        "beta_over_gamma": ratio,
        "epsilon_dark": epsilon_dark,
        "wilson_factor": wilson_factor,
        "suppression_estimate": estimate,
        "relative_residual": abs(ratio - estimate) / ratio,
    }


def two_body_width(
    m_parent: float,
    delta_m: float,
    g_dark: float,
    m_daughter: float,
    m_phi: float,
) -> float:
    ratio = (m_daughter + m_phi) ** 2 / m_parent ** 2
    if ratio >= 1.0:
        return 0.0
    return g_dark ** 2 * delta_m ** 2 / (8.0 * math.pi * m_parent ** 3) * (1.0 - ratio) ** 1.5


def lifetime_seconds(width_gev: float) -> float:
    if width_gev <= 0.0:
        return math.inf
    return HBAR_GEV_S / width_gev


def decay_table(p: DarkSectorParameters = DarkSectorParameters()) -> dict[str, dict[str, float]]:
    m = spectrum(p)
    table = {}
    for s in OCTET:
        if s.name == "chi0":
            continue
        dm = m[s.name] - m["chi0"]
        w = two_body_width(m[s.name], dm, p.g_dark, m["chi0"], p.m_phi)
        table[s.name] = {
            "mass": m[s.name],
            "delta_m": dm,
            "width_gev": w,
            "lifetime_s": lifetime_seconds(w),
        }
    return table
