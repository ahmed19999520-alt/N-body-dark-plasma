from dataclasses import asdict, dataclass

from .constants import (
    GEV_INV2_TO_CM3_S,
    PAPER_TARGETS,
    PhaseTransitionParameters,
)


@dataclass
class Check:
    group: str
    name: str
    published: float
    value: float
    tolerance: float
    basis: str

    @property
    def deviation(self) -> float:
        if self.published == 0.0:
            return abs(self.value)
        return abs(self.value - self.published) / abs(self.published)

    @property
    def status(self) -> str:
        return "match" if self.deviation <= self.tolerance else "differs"

    def as_row(self) -> dict[str, object]:
        row = asdict(self)
        row["deviation"] = self.deviation
        row["status"] = self.status
        return row


def _gmo_checks() -> list[Check]:
    from .gmo import (
        decay_table,
        hedgehog_degree,
        instanton_exponent,
        long_exact_sequence_bound,
        mass_ratios,
        ratio_worst_case_uncertainty,
        single_instanton_charge,
        spectrum,
        splittings,
    )

    out = []
    m = spectrum()
    for k, v in PAPER_TARGETS["masses_gev"].items():
        out.append(Check("octet", f"mass_{k}", v, m[k], 1e-6, "direct"))
    s = splittings()
    for k, v in PAPER_TARGETS["splittings_gev"].items():
        out.append(Check("octet", f"splitting_{k}", v, s[k], 1e-6, "direct"))
    r = mass_ratios()
    u = ratio_worst_case_uncertainty()
    for k, (v, e) in PAPER_TARGETS["ratios"].items():
        out.append(Check("octet", f"ratio_{k}", v, r[k], 5e-4, "direct"))
        out.append(Check("octet", f"ratio_{k}_uncertainty", e, u[k], 0.10, "direct"))
    d = decay_table()
    out.append(Check("octet", "width_chi_plus_gev", 2.3e-8, d["chi+"]["width_gev"], 0.5, "formula"))
    out.append(Check("octet", "lifetime_chi_plus_s", 2.9e-17, d["chi+"]["lifetime_s"], 0.5, "formula"))
    out.append(Check("homotopy", "hedgehog_degree", 1.0, hedgehog_degree(), 1e-3, "direct"))
    out.append(Check("homotopy", "instanton_exponent", 3948.0, instanton_exponent(), 1e-3, "direct"))
    out.append(Check("homotopy", "single_instanton_charge", 1.0, single_instanton_charge()["charge"], 0.01, "formula"))
    out.append(Check("homotopy", "pi2_group_order_from_sequence", 0.0, 2.0 if long_exact_sequence_bound()["pi2_G_over_H_from_sequence"] == "Z2" else 0.0, 0.0, "formula"))
    return out


def _relic_checks(include_calibrated: bool) -> list[Check]:
    from .boltzmann import (
        CoannihilationModel,
        effective_dof,
        omega_lee_weinberg,
        sigma_eff,
        solve_freeze_out,
        yield_from_omega,
    )

    R = PAPER_TARGETS["relic"]
    raw = CoannihilationModel()
    model = raw.with_target_sigma_eff(R["sigma_eff_v_cm3_s"], R["x_f"])
    out = [
        Check("relic", "sigma_v_zprime_formula_cm3_s", R["sigma_eff_v_cm3_s"], raw.sigma0_cm3_s(), 0.5, "formula"),
        Check("relic", "sigma_eff_v_cm3_s", R["sigma_eff_v_cm3_s"], sigma_eff(model, R["x_f"]) * GEV_INV2_TO_CM3_S, 1e-6, "anchored"),
        Check("relic", "g_eff", R["g_eff"], effective_dof(model, R["x_f"]), 0.10, "direct"),
        Check("relic", "x_f", R["x_f"], solve_freeze_out(model), 0.10, "direct"),
        Check("relic", "omega_lee_weinberg", R["omega_h2"], omega_lee_weinberg(R["sigma_eff_v_cm3_s"], R["x_f"]), 0.10, "formula"),
        Check("relic", "y_inf_for_omega_0p12", R["y_inf"], yield_from_omega(R["omega_h2"], 165.67), 0.10, "formula"),
    ]
    if include_calibrated:
        from .boltzmann import BoltzmannSolver, calibrate_to_relic

        cal = calibrate_to_relic(model)
        res = BoltzmannSolver(cal).solve(n_eval=200)
        out.append(Check("relic", "omega_h2_calibrated_solver", R["omega_h2"], res.omega_h2, 0.01, "anchored"))
    return out


def _si_checks() -> list[Check]:
    from .constants import C_KM_S
    from .self_interaction import (
        SelfInteractionModel,
        coulomb_enhancement,
        numerical_enhancement,
        paper_enhancement,
    )

    S = PAPER_TARGETS["self_interaction"]
    out = [
        Check("self_interaction", "sommerfeld_coulomb_10kms", S["sommerfeld_coulomb_10kms"], coulomb_enhancement(0.02, 10.0 / C_KM_S), 5e-3, "direct"),
        Check("self_interaction", "sommerfeld_yukawa_10kms", S["sommerfeld_10kms"], paper_enhancement(10.0), 5e-3, "anchored"),
        Check("self_interaction", "sommerfeld_yukawa_numerical", S["sommerfeld_10kms"], numerical_enhancement(0.02, 10.0 / C_KM_S, 9.7, 165.67), 0.5, "formula"),
    ]
    model = SelfInteractionModel()
    for name, (v, s, e, lim) in S["table"].items():
        out.append(Check("self_interaction", f"sigma_over_m_{name}", s, float(model.sigma_over_m(v)), e / s, "anchored"))
    return out


def _gw_checks() -> list[Check]:
    from .gravitational_waves import (
        ConfinementTransition,
        LisaDetector,
        alpha_bag_naive,
        kappa_sw,
        tau_sw_h,
        upsilon,
    )

    G = PAPER_TARGETS["gravitational_waves"]
    p = PhaseTransitionParameters()
    tr = ConfinementTransition(p)
    det = LisaDetector()
    f = tr.peak_frequencies()
    a = tr.peak_amplitudes()
    fc = det.forecast(tr)
    return [
        Check("gravitational_waves", "f_bubble_hz", G["f_bub_hz"], f["bubble"], 0.02, "direct"),
        Check("gravitational_waves", "f_sound_hz", G["f_sw_hz"], f["sound_wave"], 0.02, "anchored"),
        Check("gravitational_waves", "f_turbulence_hz", G["f_turb_hz"], f["turbulence"], 0.02, "anchored"),
        Check("gravitational_waves", "f_peak_hz", G["f_peak_hz"], tr.composite_peak_frequency(), 0.02, "anchored"),
        Check("gravitational_waves", "omega_bubble", G["omega_bub"], a["bubble"], 0.05, "direct"),
        Check("gravitational_waves", "omega_sound", G["omega_sw"], a["sound_wave"], 0.05, "direct"),
        Check("gravitational_waves", "omega_turbulence", G["omega_turb"], a["turbulence"], 0.05, "direct"),
        Check("gravitational_waves", "omega_total", G["omega_total"], tr.total_peak_amplitude(), 0.05, "direct"),
        Check("gravitational_waves", "kappa_sw", G["kappa_sw"], kappa_sw(p.alpha_pt), 0.01, "direct"),
        Check("gravitational_waves", "tau_sw_h", G["tau_sw_h"], tau_sw_h(p), 1e-9, "anchored"),
        Check("gravitational_waves", "upsilon", G["upsilon"], upsilon(tau_sw_h(p)), 0.05, "direct"),
        Check("gravitational_waves", "alpha_naive", G["alpha_naive"], alpha_bag_naive(), 0.03, "direct"),
        Check("gravitational_waves", "snr_4yr", G["snr_4yr"], det.snr(tr), 1e-6, "anchored"),
        Check("gravitational_waves", "snr_2yr", G["snr_2yr"], det.snr(tr, 2.0), 0.02, "direct"),
        Check("gravitational_waves", "snr_without_upsilon", G["snr_naive_no_upsilon"], det.snr_without_upsilon(tr), 0.03, "direct"),
        Check("gravitational_waves", "dlnf", G["dfstar_over_fstar"], fc["dlnf_marginalised"], 0.05, "anchored"),
        Check("gravitational_waves", "dlnA", G["domega_over_omega"], fc["dlnA_marginalised"], 0.05, "anchored"),
    ]


def _misc_checks() -> list[Check]:
    from .collider import (
        events_per_year,
        observable_events,
        sigma_ee_chargino_pair,
        threshold_energy,
    )
    from .dark_acoustic import dao_snr, dao_sound_horizon_mpc, dao_wavenumber
    from .direct_detection import sigma_si_paper

    C = PAPER_TARGETS["collider"]
    D = PAPER_TARGETS["dark_acoustic"]
    return [
        Check("collider", "threshold_gev", 334.74, threshold_energy(), 1e-6, "direct"),
        Check("collider", "sigma_ee_fb", C["sigma_ee_chichi_fb"], sigma_ee_chargino_pair(), 1e-9, "anchored"),
        Check("collider", "n_observed_ee", C["n_obs_ee"], observable_events(sigma_ee_chargino_pair()), 0.01, "direct"),
        Check("collider", "n_pp_per_year", C["n_pp_per_year"], events_per_year(), 1e-9, "direct"),
        Check("direct_detection", "sigma_si_cm2", PAPER_TARGETS["direct_detection"]["sigma_si_cm2"], sigma_si_paper(), 1e-9, "anchored"),
        Check("dark_acoustic", "r_dao_mpc", D["r_dao_mpc"][0], dao_sound_horizon_mpc(), 1e-9, "anchored"),
        Check("dark_acoustic", "k_dao", D["k_dao_h_mpc"], dao_wavenumber(), 0.05, "direct"),
        Check("dark_acoustic", "snr_ska", D["snr_ska"], dao_snr(), 1e-9, "anchored"),
    ]


def run_all_checks(include_calibrated_relic: bool = False) -> list[Check]:
    checks = []
    checks += _gmo_checks()
    checks += _relic_checks(include_calibrated_relic)
    checks += _si_checks()
    checks += _gw_checks()
    checks += _misc_checks()
    return checks


def summarise(checks: list[Check]) -> dict[str, int]:
    out = {"total": len(checks), "match": 0, "differs": 0}
    for c in checks:
        out[c.status] += 1
    return out


def render_markdown(checks: list[Check]) -> str:
    lines = ["| group | quantity | published | library | deviation | basis | status |", "|---|---|---|---|---|---|---|"]
    for c in checks:
        lines.append(
            f"| {c.group} | {c.name} | {c.published:.6g} | {c.value:.6g} | {c.deviation:.3g} | {c.basis} | {c.status} |"
        )
    s = summarise(checks)
    lines.append("")
    lines.append(f"total {s['total']}, match {s['match']}, differs {s['differs']}")
    return "\n".join(lines)
