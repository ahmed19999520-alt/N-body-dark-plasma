import numpy as np
import pytest

from N_bodyDP.constants import PAPER_TARGETS, PhaseTransitionParameters
from N_bodyDP.gravitational_waves import (
    ConfinementTransition,
    LisaDetector,
    alpha_bag_naive,
    kappa_sw,
    tau_sw_h,
    upsilon,
)

GW = PAPER_TARGETS["gravitational_waves"]


@pytest.fixture(scope="module")
def transition():
    return ConfinementTransition(PhaseTransitionParameters())


@pytest.fixture(scope="module")
def detector():
    return LisaDetector()


def test_peak_frequencies(transition):
    f = transition.peak_frequencies()
    assert f["bubble"] == pytest.approx(GW["f_bub_hz"], rel=0.02)
    assert f["sound_wave"] == pytest.approx(GW["f_sw_hz"], rel=0.02)
    assert f["turbulence"] == pytest.approx(GW["f_turb_hz"], rel=0.02)
    assert transition.composite_peak_frequency() == pytest.approx(GW["f_peak_hz"], rel=0.02)


def test_peak_amplitudes(transition):
    a = transition.peak_amplitudes()
    assert a["bubble"] == pytest.approx(GW["omega_bub"], rel=0.05)
    assert a["sound_wave"] == pytest.approx(GW["omega_sw"], rel=0.05)
    assert a["turbulence"] == pytest.approx(GW["omega_turb"], rel=0.05)
    assert transition.total_peak_amplitude() == pytest.approx(GW["omega_total"], rel=0.05)


def test_efficiency_and_lifetime():
    assert kappa_sw(0.1) == pytest.approx(GW["kappa_sw"], abs=1e-3)
    p = PhaseTransitionParameters()
    assert tau_sw_h(p) == pytest.approx(GW["tau_sw_h"], rel=1e-9)
    assert upsilon(tau_sw_h(p)) == pytest.approx(GW["upsilon"], rel=0.05)
    assert alpha_bag_naive() == pytest.approx(GW["alpha_naive"], abs=0.005)


def test_sub_peak_slope_is_three(transition):
    f = np.geomspace(1.0e-5, 1.0e-4, 20)
    s = transition.components(f)["sound_wave"]
    slope = np.polyfit(np.log(f), np.log(s), 1)[0]
    assert slope == pytest.approx(3.0, abs=0.01)


def test_snr_benchmark(transition, detector):
    assert detector.snr(transition) == pytest.approx(GW["snr_4yr"], rel=1e-6)
    assert detector.snr(transition, 2.0) == pytest.approx(GW["snr_2yr"], rel=0.02)
    assert detector.snr_without_upsilon(transition) == pytest.approx(GW["snr_naive_no_upsilon"], rel=0.03)


def test_fisher_frequency_precision(transition, detector):
    f = detector.forecast(transition)
    assert f["dlnf_marginalised"] == pytest.approx(GW["dfstar_over_fstar"], abs=0.02)
    assert f["dlnA_marginalised"] == pytest.approx(GW["domega_over_omega"], abs=0.06)


def test_snr_scales_with_alpha(detector):
    low = ConfinementTransition(PhaseTransitionParameters(alpha_pt=0.05))
    high = ConfinementTransition(PhaseTransitionParameters(alpha_pt=0.18))
    assert detector.snr(low) < 8.5 < detector.snr(high)
