import pytest

from N_bodyDP.constants import C_KM_S
from N_bodyDP.self_interaction import (
    SelfInteractionModel,
    constraint_table,
    coulomb_enhancement,
    critical_velocity,
    hulthen_enhancement,
    numerical_enhancement,
    paper_enhancement,
)


def test_coulomb_enhancement_matches_published():
    s = coulomb_enhancement(0.020, 10.0 / C_KM_S)
    assert s == pytest.approx(3770.0, rel=2e-3)


def test_paper_enhancement_matches_published():
    assert paper_enhancement(10.0) == pytest.approx(1964.0, rel=2e-3)


def test_critical_velocity_matches_published():
    assert critical_velocity()["v_over_c"] == pytest.approx(0.059, abs=1e-3)


def test_table_within_published_errors_and_limits():
    for row in constraint_table():
        assert row["within_published_error"]
        assert row["below_limit"]


def test_model_is_monotonic_decreasing():
    m = SelfInteractionModel()
    assert m.sigma_over_m(5.0) > m.sigma_over_m(10.0) > m.sigma_over_m(300.0) > m.sigma_over_m(1000.0)


def test_numerical_and_hulthen_agree():
    a = numerical_enhancement(0.02, 1.0e-3, 9.7, 165.67)
    b = hulthen_enhancement(0.02, 1.0e-3, 9.7, 165.67)
    assert a == pytest.approx(b, rel=0.05)
