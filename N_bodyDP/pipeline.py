import csv
import json
import os

import numpy as np

from .constants import PAPER_TARGETS, PhaseTransitionParameters
from .verification import render_markdown, run_all_checks, summarise


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, complex):
            return [o.real, o.imag]
        return super().default(o)


def _dump(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, cls=_Encoder)


def octet_stage(out_dir: str) -> dict:
    from .gmo import (
        classification_report,
        decay_table,
        gst_consistency,
        mass_ratios,
        ratio_monte_carlo,
        ratio_worst_case_uncertainty,
        spectrum,
        splittings,
    )

    res = {
        "masses": spectrum(),
        "splittings": splittings(),
        "ratios": mass_ratios(),
        "ratio_worst_case": ratio_worst_case_uncertainty(),
        "ratio_monte_carlo": ratio_monte_carlo(n_samples=4000),
        "decays": decay_table(),
        "gst": gst_consistency(),
        "homotopy": classification_report(),
    }
    _dump(os.path.join(out_dir, "octet.json"), res)
    with open(os.path.join(out_dir, "table_octet.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["state", "mass_gev", "delta_m_gev"])
        m = res["masses"]
        for k, v in m.items():
            w.writerow([k, f"{v:.4f}", f"{v - m['chi0']:.4f}"])
    return res


def relic_stage(out_dir: str, quick: bool) -> dict:
    from .boltzmann import (
        BoltzmannSolver,
        CoannihilationModel,
        calibrate_to_relic,
        effective_dof,
        omega_lee_weinberg,
        sigma_eff,
        solve_freeze_out,
    )
    from .constants import GEV_INV2_TO_CM3_S

    R = PAPER_TARGETS["relic"]
    base = CoannihilationModel().with_target_sigma_eff(R["sigma_eff_v_cm3_s"], R["x_f"])
    res = {
        "g_eff": effective_dof(base, R["x_f"]),
        "x_f": solve_freeze_out(base),
        "sigma_eff_cm3_s": sigma_eff(base, R["x_f"]) * GEV_INV2_TO_CM3_S,
        "omega_lee_weinberg": omega_lee_weinberg(R["sigma_eff_v_cm3_s"], R["x_f"]),
    }
    model = base if quick else calibrate_to_relic(base)
    sol = BoltzmannSolver(model).solve(n_eval=300)
    res.update({"omega_h2": sol.omega_h2, "y_inf": sol.y_inf, "success": sol.success, "x_freeze_departure": sol.x_freeze})
    _dump(os.path.join(out_dir, "relic.json"), res)
    np.savez(os.path.join(out_dir, "relic_yields.npz"), x=sol.x, y=sol.y, y_eq=sol.y_eq)
    return res


def self_interaction_stage(out_dir: str) -> dict:
    from .self_interaction import constraint_table, critical_velocity

    res = {"table": constraint_table(), "critical_velocity": critical_velocity()}
    _dump(os.path.join(out_dir, "self_interaction.json"), res)
    with open(os.path.join(out_dir, "table_self_interaction.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(list(res["table"][0].keys()))
        for r in res["table"]:
            w.writerow(list(r.values()))
    return res


def gw_stage(out_dir: str) -> dict:
    from .gravitational_waves import ConfinementTransition, LisaDetector, sensitivity_matrix

    tr = ConfinementTransition(PhaseTransitionParameters())
    det = LisaDetector()
    res = {
        "summary": tr.summary(),
        "snr_4yr": det.snr(tr),
        "snr_2yr": det.snr(tr, 2.0),
        "snr_without_upsilon": det.snr_without_upsilon(tr),
        "forecast": det.forecast(tr),
        "sensitivity_matrix": sensitivity_matrix(det),
    }
    _dump(os.path.join(out_dir, "gravitational_waves.json"), res)
    return res


def other_stage(out_dir: str) -> dict:
    from .collider import events_per_year, observable_events, sigma_ee_chargino_pair, signal_table
    from .dark_acoustic import dao_snr, dao_sound_horizon_mpc, dao_wavenumber
    from .direct_detection import compare_with_limits

    res = {
        "collider": {
            "sigma_ee_fb": sigma_ee_chargino_pair(),
            "n_observed_ee": observable_events(sigma_ee_chargino_pair()),
            "n_pp_per_year": events_per_year(),
            "table": signal_table(),
        },
        "dark_acoustic": {"r_dao_mpc": dao_sound_horizon_mpc(), "k_dao": dao_wavenumber(), "snr": dao_snr()},
        "direct_detection": compare_with_limits(),
    }
    _dump(os.path.join(out_dir, "other.json"), res)
    return res


def nbody_stage(out_dir: str) -> dict:
    from .nbody import (
        SidmScatterer,
        barnes_hut_accelerations,
        direct_accelerations,
        direct_potential_energy,
        kinetic_energy,
        plummer_sphere,
    )
    from .self_interaction import SelfInteractionModel

    pos, vel, m = plummer_sphere(800, 1.0e9, 1.0)
    a = direct_accelerations(pos, m, 0.05)
    b = barnes_hut_accelerations(pos, m, 0.05, theta=0.5)
    err = np.linalg.norm(a - b, axis=1) / np.linalg.norm(a, axis=1)
    model = SelfInteractionModel()
    sc = SidmScatterer(lambda v: model.sigma_over_m(v) * 1.0e3, m[0], 0.05)
    new = sc.step(pos, vel, 5.0)
    res = {
        "bh_median_error": float(np.median(err)),
        "bh_max_error": float(err.max()),
        "virial_ratio": 2.0 * kinetic_energy(vel, m) / abs(direct_potential_energy(pos, m, 0.05)),
        "sidm_scatterings": sc.n_scatter,
        "sidm_energy_change": float(kinetic_energy(new, m) / kinetic_energy(vel, m) - 1.0),
    }
    _dump(os.path.join(out_dir, "nbody.json"), res)
    return res


def bayesian_stage(out_dir: str, n_live: int) -> dict:
    from .bayesian import posterior_geometry, reconstruct
    from .plotting import corner_plot

    like, result, truth = reconstruct(n_live=n_live, seed=3)
    geo = posterior_geometry(result.samples, result.weights)
    res = {
        "names": list(result.names),
        "mean": result.mean(),
        "std": result.std(),
        "log_evidence": result.log_evidence,
        "geometry": geo,
        "truth": {k: truth[k] for k in result.names},
    }
    _dump(os.path.join(out_dir, "bayesian.json"), res)
    corner_plot(result, truth, os.path.join(out_dir, "fig_corner.png"))
    return res


def figures_stage(out_dir: str, relic: dict = None, quick: bool = False) -> None:
    from .plotting import (
        boltzmann_yield_plot,
        dao_plot,
        diagnostic_maps,
        gw_spectrum_plot,
        lisa_sensitivity_plot,
        octet_weight_diagram,
        plot_diagnostic_maps,
        plot_snr_map,
        self_interaction_plot,
        snr_map,
    )

    octet_weight_diagram(os.path.join(out_dir, "fig_octet.png"))
    gw_spectrum_plot(path=os.path.join(out_dir, "fig_gw_spectrum.png"))
    self_interaction_plot(os.path.join(out_dir, "fig_self_interaction.png"))
    dao_plot(os.path.join(out_dir, "fig_dao.png"))
    lisa_sensitivity_plot(os.path.join(out_dir, "fig_lisa.png"))
    data = np.load(os.path.join(out_dir, "relic_yields.npz")) if relic is not None else None
    if data is not None:
        class _R:
            pass

        r = _R()
        r.x, r.y, r.y_eq = data["x"], data["y"], data["y_eq"]
        boltzmann_yield_plot(r, path=os.path.join(out_dir, "fig_yields.png"))
    n = 8 if quick else 16
    a, b, g = snr_map(np.geomspace(0.03, 0.3, n), np.geomspace(30.0, 400.0, n))
    plot_snr_map(a, b, g, os.path.join(out_dir, "fig_snr_map.png"))
    la, lb, maps = diagnostic_maps(np.linspace(-1.7, -0.5, n), np.linspace(1.4, 2.6, n))
    plot_diagnostic_maps(la, lb, maps, os.path.join(out_dir, "fig_reconstruction_maps.png"))


def verification_stage(out_dir: str, include_calibrated: bool) -> dict:
    checks = run_all_checks(include_calibrated)
    _dump(os.path.join(out_dir, "verification.json"), [c.as_row() for c in checks])
    with open(os.path.join(out_dir, "verification.md"), "w") as fh:
        fh.write(render_markdown(checks))
    return summarise(checks)


def run(output: str, quick: bool = False, n_live: int = 300, skip_bayesian: bool = False) -> dict:
    os.makedirs(output, exist_ok=True)
    out = {}
    out["octet"] = octet_stage(output)
    out["relic"] = relic_stage(output, quick)
    out["self_interaction"] = self_interaction_stage(output)
    out["gravitational_waves"] = gw_stage(output)
    out["other"] = other_stage(output)
    out["nbody"] = nbody_stage(output)
    if not (quick or skip_bayesian):
        out["bayesian"] = bayesian_stage(output, n_live)
    figures_stage(output, out["relic"], quick)
    out["verification"] = verification_stage(output, not quick)
    return out
