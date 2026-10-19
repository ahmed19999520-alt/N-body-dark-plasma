import math
from dataclasses import dataclass

HBAR_GEV_S = 6.582119569e-25
C_CM_S = 2.99792458e10
C_KM_S = 2.99792458e5
GEV2_TO_CM2 = 3.8937966e-28
GEV_INV2_TO_CM3_S = GEV2_TO_CM2 * C_CM_S
GEV_INV_TO_CM = 1.97326980e-14
GEV_TO_GRAM = 1.78266192e-24
MPC_TO_CM = 3.0856775814913673e24
SECONDS_PER_YEAR = 3.15576e7
M_PLANCK_GEV = 1.22e19
M_NUCLEON_GEV = 0.939
HUBBLE_H = 0.732
H0_PER_SECOND = 2.373e-18
OMEGA_DM_H2_PLANCK = 0.1200
OMEGA_DM_H2_PLANCK_ERR = 0.0012
ENTROPY_TODAY_CM3 = 2891.2
RHO_CRIT_H2_GEV_CM3 = 1.05371e-5
OMEGA_TO_YIELD = ENTROPY_TODAY_CM3 / RHO_CRIT_H2_GEV_CM3
T0_GEV = 2.3e-13
PI = math.pi


@dataclass(frozen=True)
class DarkSectorParameters:
    M0: float = 208.67
    beta: float = -15.0
    gamma: float = 8.0
    delta: float = 5.0
    eta: float = 8.0
    v_delta: float = 17.0
    y_chi: float = 0.10
    g_dark: float = 0.5
    alpha_dark: float = 0.020
    g_f: float = 1.0e-3
    m_zprime: float = 9.7
    m_phi: float = 0.005
    delta_neff: float = 0.09

    @property
    def triplet_splitting(self) -> float:
        return self.y_chi * self.v_delta


@dataclass(frozen=True)
class PhaseTransitionParameters:
    alpha_pt: float = 0.10
    beta_over_h: float = 100.0
    v_w: float = 0.8
    t_c: float = 1.0e15
    g_dark: float = 47.5
    g_total: float = 261.0
    kappa_phi: float = 0.05
    kappa_turb_ratio: float = 0.05
    t_obs_years: float = 4.0
    c_s: float = 1.0 / math.sqrt(3.0)

    @property
    def t_obs_seconds(self) -> float:
        return self.t_obs_years * SECONDS_PER_YEAR


PAPER_TARGETS = {
    "masses_gev": {
        "chi0": 165.67,
        "chi+": 167.37,
        "chi-": 168.67,
        "chi++": 180.37,
        "Sigma0": 192.67,
        "Sigma-": 195.67,
        "Sigma+": 207.37,
        "Lambda": 208.67,
    },
    "splittings_gev": {
        "chi+": 1.70,
        "chi-": 3.00,
        "chi++": 14.70,
        "Sigma0": 27.00,
        "Sigma-": 30.00,
        "Sigma+": 41.70,
        "Lambda": 43.00,
    },
    "ratios": {"r21": (1.089, 0.005), "r31": (1.163, 0.006), "r41": (1.260, 0.008)},
    "relic": {
        "omega_h2": 0.120,
        "x_f": 26.97,
        "g_eff": 5.09,
        "sigma_eff_v_cm3_s": 3.0e-26,
        "g_star": 106.75,
        "y_inf": 1.2e-9,
    },
    "self_interaction": {
        "sommerfeld_coulomb_10kms": 3770.0,
        "yukawa_suppression": 0.52,
        "sommerfeld_10kms": 1964.0,
        "v_crit_c": 0.059,
        "table": {
            "ultra_faint_dwarfs": (5.0, 0.62, 0.11, 1.0),
            "dwarf_spheroidals": (10.0, 0.47, 0.09, 1.0),
            "milky_way_satellites": (75.0, 0.10, 0.03, 0.5),
            "galaxy_groups": (300.0, 0.026, 0.007, 0.1),
            "galaxy_clusters": (1000.0, 0.0059, 0.0011, 0.47),
            "bullet_cluster": (1000.0, 0.0044, 0.0009, 0.47),
        },
    },
    "direct_detection": {"sigma_si_cm2": 3.1e-48, "lz_limit_cm2": 5.0e-47, "darwin_cm2": 3.0e-49},
    "gravitational_waves": {
        "f_bub_hz": 8.1e-3,
        "f_sw_hz": 2.77e-3,
        "f_turb_hz": 3.95e-3,
        "f_peak_hz": 1.65e-3,
        "omega_bub": 4.2e-12,
        "omega_sw": 1.0e-8,
        "omega_turb": 2.48e-11,
        "omega_total": 1.0e-8,
        "kappa_sw": 0.117,
        "tau_sw_h": 0.052,
        "upsilon": 0.047,
        "alpha_naive": 0.18,
        "reynolds": 80.0,
        "snr_4yr": 8.5,
        "snr_2yr": 6.0,
        "snr_naive_no_upsilon": 39.0,
        "dfstar_over_fstar": 0.15,
        "domega_over_omega": 0.25,
        "dn_sub": 0.30,
    },
    "dark_acoustic": {"r_dao_mpc": (2.3, 0.4), "k_dao_h_mpc": 1.4, "amplitude": (1.8e-3, 0.5e-3), "snr_ska": 7.2},
    "collider": {
        "sigma_ee_chichi_fb": 1.1e-3,
        "n_events_ee": 1650,
        "efficiency": 0.30,
        "n_obs_ee": 500,
        "significance_ee": 5.3,
        "sigma_pp_chi0chi0_fb": 0.8,
        "n_pp_per_year": 80,
    },
}
