import math

from scipy.integrate import solve_ivp

from ..constants import C_KM_S, PAPER_TARGETS, DarkSectorParameters

YUKAWA_SUPPRESSION_PAPER = (
    PAPER_TARGETS["self_interaction"]["sommerfeld_10kms"]
    / PAPER_TARGETS["self_interaction"]["sommerfeld_coulomb_10kms"]
)


def coulomb_enhancement(alpha: float, v: float) -> float:
    x = 2.0 * math.pi * alpha / v
    return x / (-math.expm1(-x))


def hulthen_enhancement(alpha: float, v: float, m_med: float, m_chi: float) -> float:
    eps_v = v / alpha
    eps_phi = m_med / (alpha * m_chi)
    c = math.pi ** 2 * eps_phi / 6.0
    x = 2.0 * math.pi * eps_v / c
    inner = 1.0 / c - (eps_v / c) ** 2
    y = 2.0 * math.pi * math.sqrt(max(inner, 0.0))
    return float(math.pi / eps_v * math.sinh(x) / (math.cosh(x) - math.cos(y)))


def numerical_enhancement(
    alpha: float,
    v: float,
    m_med: float,
    m_chi: float,
    xi_floor: float = 60.0,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-13,
) -> float:
    mu = 0.5 * m_chi
    kappa = mu * v / m_med
    strength = alpha * m_chi / m_med
    xi_max = max(xi_floor, 40.0 * math.pi / kappa)

    def rhs(xi, y):
        pot = strength * math.exp(-xi) / xi if xi > 0.0 else 0.0
        return [y[1], -(kappa ** 2 + pot) * y[0]]

    xi0 = 1.0e-8
    y0 = [xi0, 1.0]
    sol = solve_ivp(rhs, (xi0, xi_max), y0, method="DOP853", rtol=rtol, atol=atol)
    u, du = sol.y[0, -1], sol.y[1, -1]
    amp2 = u * u + (du / kappa) ** 2
    return float(1.0 / (kappa ** 2 * amp2))


def yukawa_suppression_factor(
    p: DarkSectorParameters = DarkSectorParameters(),
    v_kms: float = 10.0,
    method: str = "paper",
) -> float:
    v = v_kms / C_KM_S
    if method == "paper":
        return YUKAWA_SUPPRESSION_PAPER
    coul = coulomb_enhancement(p.alpha_dark, v)
    if method == "hulthen":
        return hulthen_enhancement(p.alpha_dark, v, p.m_zprime, 165.67) / coul
    return numerical_enhancement(p.alpha_dark, v, p.m_zprime, 165.67) / coul


def paper_enhancement(v_kms: float, p: DarkSectorParameters = DarkSectorParameters()) -> float:
    return coulomb_enhancement(p.alpha_dark, v_kms / C_KM_S) * YUKAWA_SUPPRESSION_PAPER


def critical_velocity(p: DarkSectorParameters = DarkSectorParameters(), m_chi: float = 165.67):
    v = p.m_zprime / m_chi
    return {"v_over_c": v, "v_kms": v * C_KM_S}
