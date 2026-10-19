import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from N_bodyDP.constants import PAPER_TARGETS
from N_bodyDP.gravitational_waves import LisaNoiseModel, pli_sensitivity_paper

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


def write(path, header, rows):
    with open(os.path.join(ROOT, path), "w") as fh:
        fh.write(header + "\n")
        fh.writelines(" ".join(f"{x:.8e}" if isinstance(x, float) else str(x) for x in r) + "\n" for r in rows)


def main():
    table = PAPER_TARGETS["self_interaction"]["table"]
    write(
        "observational/dwarf_galaxies.dat",
        "# system v_kms sigma_over_m_model sigma_over_m_err upper_limit_cm2_per_g",
        [(k, *v) for k, v in table.items() if k in ("ultra_faint_dwarfs", "dwarf_spheroidals", "milky_way_satellites")],
    )
    write(
        "observational/bullet_cluster.dat",
        "# system v_kms sigma_over_m_model sigma_over_m_err upper_limit_cm2_per_g",
        [(k, *v) for k, v in table.items() if k in ("galaxy_groups", "galaxy_clusters", "bullet_cluster")],
    )
    write(
        "observational/planck_2018.dat",
        "# parameter value uncertainty",
        [
            ("omega_c_h2", 0.1200, 0.0012),
            ("omega_b_h2", 0.02237, 0.00015),
            ("H0_km_s_Mpc", 67.4, 0.5),
            ("n_s", 0.9649, 0.0042),
            ("sigma_8", 0.811, 0.006),
            ("delta_N_eff_upper_95", 0.17, 0.0),
        ],
    )
    f = np.geomspace(1.0e-4, 1.0e-1, 400)
    noise = LisaNoiseModel()
    write(
        "sensitivity/lisa_sensitivity.dat",
        "# f_hz omega_sens_h2_tdi omega_pli_h2_four_year",
        [(float(a), float(b), float(c)) for a, b, c in zip(f, noise.omega_sensitivity_h2(f), pli_sensitivity_paper(f))],
    )
    dd = PAPER_TARGETS["direct_detection"]
    write(
        "sensitivity/darwin_sensitivity.dat",
        "# experiment mass_gev sigma_si_cm2",
        [
            ("LZ_first_result", 36.0, 9.2e-48),
            ("LZ_2024_at_benchmark", 165.67, dd["lz_limit_cm2"]),
            ("DARWIN_projected", 100.0, dd["darwin_cm2"]),
            ("model_prediction", 165.67, dd["sigma_si_cm2"]),
        ],
    )
    c = PAPER_TARGETS["collider"]
    write(
        "sensitivity/fcc.dat",
        "# machine sqrt_s_gev luminosity_fb sigma_fb n_events",
        [
            ("FCC_ee", 365.0, 1.5e6, c["sigma_ee_chichi_fb"], c["n_events_ee"]),
            ("FCC_hh_total", 1.0e5, 30.0e3, c["sigma_pp_chi0chi0_fb"], 24000),
            ("FCC_hh_per_year", 1.0e5, 100.0, c["sigma_pp_chi0chi0_fb"], c["n_pp_per_year"]),
        ],
    )


if __name__ == "__main__":
    main()
