import pytest

from N_bodyDP.constants import PAPER_TARGETS
from N_bodyDP.dark_acoustic import dao_snr, dao_sound_horizon_mpc, dao_wavenumber
from N_bodyDP.direct_detection import compare_with_limits, sigma_si_paper


def test_dao_scale_and_wavenumber():
    assert dao_sound_horizon_mpc() == pytest.approx(2.3)
    assert dao_wavenumber() == pytest.approx(1.4, abs=0.05)
    assert dao_snr() == pytest.approx(7.2)


def test_direct_detection_value_and_window():
    assert sigma_si_paper() == pytest.approx(PAPER_TARGETS["direct_detection"]["sigma_si_cm2"])
    c = compare_with_limits()
    assert c["below_lz"] and c["above_darwin_reach"]
