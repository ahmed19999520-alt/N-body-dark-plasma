import math

import pytest

from N_bodyDP.constants import PAPER_TARGETS, DarkSectorParameters
from N_bodyDP.gmo import (
    decay_table,
    hedgehog_degree,
    instanton_exponent,
    instanton_log10_suppression,
    is_chi0_lightest,
    long_exact_sequence_bound,
    mass_ratios,
    ratio_worst_case_uncertainty,
    spectrum,
    splittings,
)


def test_masses_match_published_table():
    m = spectrum()
    for name, value in PAPER_TARGETS["masses_gev"].items():
        assert m[name] == pytest.approx(value, abs=1e-6)


def test_splittings_match_published_table():
    s = splittings()
    for name, value in PAPER_TARGETS["splittings_gev"].items():
        assert s[name] == pytest.approx(value, abs=1e-6)


def test_ratios_round_to_published_values():
    r = mass_ratios()
    for key, (value, _) in PAPER_TARGETS["ratios"].items():
        assert round(r[key], 3) == value


def test_chi0_is_lightest():
    assert is_chi0_lightest()


def test_worst_case_uncertainty_r21():
    u = ratio_worst_case_uncertainty()
    assert u["r21"] == pytest.approx(0.005, abs=1e-3)


def test_triplet_splitting_value():
    assert DarkSectorParameters().triplet_splitting == pytest.approx(1.70)


def test_hedgehog_degree_is_unity():
    assert hedgehog_degree() == pytest.approx(1.0, abs=1e-4)


def test_instanton_suppression():
    assert instanton_exponent() == pytest.approx(3947.84, abs=0.01)
    assert instanton_log10_suppression() == pytest.approx(-1714.5, abs=0.1)


def test_sequence_bound_is_reported():
    assert long_exact_sequence_bound()["pi1_H"] == "Z2"


def test_decay_table_positive_widths():
    for row in decay_table().values():
        assert row["width_gev"] > 0.0
        assert math.isfinite(row["lifetime_s"])
