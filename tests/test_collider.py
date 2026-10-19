import pytest

from N_bodyDP.collider import (
    boosted_endpoint,
    events_per_year,
    implied_background,
    lepton_endpoint,
    observable_events,
    required_luminosity,
    sigma_ee_chargino_pair,
    signal_table,
    significance,
    threshold_energy,
)
from N_bodyDP.constants import PAPER_TARGETS

C = PAPER_TARGETS["collider"]


def test_threshold_and_cross_section():
    assert threshold_energy() == pytest.approx(334.74, abs=1e-6)
    assert sigma_ee_chargino_pair() == pytest.approx(C["sigma_ee_chichi_fb"])


def test_event_counts():
    assert observable_events(sigma_ee_chargino_pair()) == pytest.approx(C["n_obs_ee"], rel=0.01)
    assert events_per_year() == pytest.approx(C["n_pp_per_year"])


def test_table_event_numbers_consistent():
    for row in signal_table():
        assert row["n_consistent"]


def test_significance_roundtrip():
    b = implied_background(500.0, 5.3)
    assert significance(500.0, b) == pytest.approx(5.3)
    assert required_luminosity(5.3) == pytest.approx(1.5e6)


def test_endpoints():
    assert lepton_endpoint(1.70) == pytest.approx(0.85)
    assert boosted_endpoint(1.70) == pytest.approx(0.93, abs=1e-3)
