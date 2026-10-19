import json
import os

import pytest

from N_bodyDP import pipeline
from N_bodyDP.verification import Check, render_markdown, run_all_checks


def test_direct_checks_all_match():
    checks = run_all_checks()
    bad = [c for c in checks if c.basis in ("direct", "anchored") and c.status != "match"]
    names = {c.name for c in bad}
    assert names <= {"ratio_r31_uncertainty", "ratio_r41_uncertainty"}


def test_formula_discrepancies_are_reported_not_hidden():
    checks = {c.name: c for c in run_all_checks()}
    for name in ("width_chi_plus_gev", "sigma_v_zprime_formula_cm3_s", "omega_lee_weinberg", "single_instanton_charge"):
        assert checks[name].status == "differs"


def test_check_status_logic():
    assert Check("g", "n", 1.0, 1.04, 0.05, "direct").status == "match"
    assert Check("g", "n", 1.0, 1.10, 0.05, "direct").status == "differs"


def test_markdown_render_contains_summary():
    text = render_markdown(run_all_checks())
    assert "total" in text and "| group |" in text


def test_quick_pipeline_writes_outputs(tmp_path):
    out = pipeline.run(str(tmp_path), quick=True)
    assert out["verification"]["total"] > 60
    for name in ("octet.json", "relic.json", "verification.md", "fig_gw_spectrum.png", "fig_reconstruction_maps.png"):
        assert os.path.exists(tmp_path / name)
    data = json.load(open(tmp_path / "octet.json"))
    assert data["masses"]["chi0"] == pytest.approx(165.67)
