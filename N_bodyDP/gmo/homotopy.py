import math

import numpy as np

KNOWN_GROUPS = {
    "pi1(SU(3))": "0",
    "pi2(SU(3))": "0",
    "pi3(SU(3))": "Z",
    "pi1(SO(3))": "Z2",
    "pi2(SO(3))": "0",
    "pi1(Z2)": "0",
}


def long_exact_sequence_bound() -> dict[str, str]:
    left = KNOWN_GROUPS["pi2(SU(3))"]
    right = KNOWN_GROUPS["pi1(SO(3))"]
    kernel_of_next = KNOWN_GROUPS["pi1(SU(3))"]
    if left == "0" and kernel_of_next == "0":
        pi2 = right
    else:
        pi2 = "undetermined"
    return {
        "sequence": "pi2(SU(3)) -> pi2(G/H) -> pi1(SO(3)xZ2) -> pi1(SU(3))",
        "pi2_G": left,
        "pi1_H": right,
        "pi1_G": kernel_of_next,
        "pi2_G_over_H_from_sequence": pi2,
    }


def paper_classification() -> dict[str, str]:
    return {
        "breaking": "SU(3)_dark -> SO(3)_dark x Z2",
        "pi2_G_over_H": "Z",
        "pi1_universal_cover": "Z2",
        "defect": "point-like, integer winding number n",
    }


def hedgehog_degree(n_theta: int = 400, n_phi: int = 800) -> float:
    theta = (np.arange(n_theta) + 0.5) * math.pi / n_theta
    phi = (np.arange(n_phi) + 0.5) * 2.0 * math.pi / n_phi
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    n = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=-1)
    d_th = np.stack([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)], axis=-1)
    d_ph = np.stack([-np.sin(th) * np.sin(ph), np.sin(th) * np.cos(ph), np.zeros_like(th)], axis=-1)
    integrand = np.einsum("ijk,ijk->ij", n, np.cross(d_th, d_ph))
    cell = (math.pi / n_theta) * (2.0 * math.pi / n_phi)
    return float(integrand.sum() * cell / (4.0 * math.pi))


def instanton_exponent(alpha_dark: float = 0.020) -> float:
    return 8.0 * math.pi ** 2 / alpha_dark


def instanton_log10_suppression(alpha_dark: float = 0.020) -> float:
    return -instanton_exponent(alpha_dark) / math.log(10.0)


def single_instanton_charge(g_dark: float = 0.5) -> dict[str, float]:
    integral = 8.0 * math.pi ** 2 / g_dark ** 2
    prefactor = g_dark ** 2 / (32.0 * math.pi ** 2)
    return {"prefactor": prefactor, "integral": integral, "charge": prefactor * integral}


def classification_report(alpha_dark: float = 0.020, g_dark: float = 0.5) -> dict[str, object]:
    return {
        "sequence": long_exact_sequence_bound(),
        "paper": paper_classification(),
        "hedgehog_degree": hedgehog_degree(),
        "instanton_exponent": instanton_exponent(alpha_dark),
        "instanton_log10": instanton_log10_suppression(alpha_dark),
        "single_instanton": single_instanton_charge(g_dark),
    }
