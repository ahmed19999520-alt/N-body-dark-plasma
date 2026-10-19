
from ..constants import DarkSectorParameters
from ..gmo.mass_spectrum import spectrum
from .fcc_ee import event_yield, implied_background

_PUBLISHED_ROWS = (
    ("ee_to_chi+chi-", 365.0, 1.1e-3, 1.5e6, 1650, 5.3),
    ("ee_to_chi0chi0_gamma", 365.0, 2.3e-5, 1.5e6, 35, 2.1),
    ("pp_to_chi0chi0_jet", 1.0e5, 0.80, 30.0e3, 24000, 12.0),
    ("pp_to_chi+chi-_jet", 1.0e5, 0.12, 30.0e3, 3600, 7.0),
    ("pp_to_KK_graviton_to_chi0chi0", 1.0e5, 0.30, 30.0e3, 9000, 9.0),
)


def lepton_endpoint(delta_m: float, m_pi: float = 0.0) -> float:
    return 0.5 * delta_m * (1.0 - m_pi ** 2 / delta_m ** 2)


def boosted_endpoint(delta_m: float, boost_factor: float = 0.93 / 0.85) -> float:
    return lepton_endpoint(delta_m) * boost_factor


def ratio_precision_from_endpoint(
    delta_e: float = 0.1,
    p: DarkSectorParameters = DarkSectorParameters(),
) -> dict[str, float]:
    m0 = spectrum(p)["chi0"]
    d_split = 2.0 * delta_e
    relative = d_split / m0
    return {
        "d_split_gev": d_split,
        "dr_over_r_direct": relative,
        "dr_over_r_published": 0.003,
    }


def signal_table() -> list[dict[str, float]]:
    rows = []
    for name, sqrt_s, sigma, lumi, n_pub, z in _PUBLISHED_ROWS:
        n = event_yield(sigma, lumi)
        rows.append(
            {
                "channel": name,
                "sqrt_s_gev": sqrt_s,
                "sigma_fb": sigma,
                "luminosity_fb": lumi,
                "n_signal": n,
                "n_signal_published": n_pub,
                "n_consistent": abs(n - n_pub) / n_pub < 0.02,
                "significance": z,
                "implied_background": implied_background(n, z),
            }
        )
    return rows
