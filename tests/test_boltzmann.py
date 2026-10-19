import pytest

from N_bodyDP.boltzmann import (
    BoltzmannSolver,
    CoannihilationModel,
    effective_dof,
    omega_from_yield,
    omega_lee_weinberg,
    sigma_eff,
    solve_freeze_out,
    yield_from_omega,
)
from N_bodyDP.constants import GEV_INV2_TO_CM3_S


def test_sigma_eff_calibration_hits_target():
    m = CoannihilationModel().with_target_sigma_eff(3.0e-26, 27.0)
    assert sigma_eff(m, 27.0) * GEV_INV2_TO_CM3_S == pytest.approx(3.0e-26, rel=1e-9)


def test_effective_dof_decreases_with_x():
    m = CoannihilationModel()
    assert effective_dof(m, 10.0) > effective_dof(m, 40.0) > 2.0


def test_freeze_out_in_expected_window():
    m = CoannihilationModel().with_target_sigma_eff(3.0e-26, 27.0)
    assert 20.0 < solve_freeze_out(m) < 35.0


def test_lee_weinberg_scaling():
    a = omega_lee_weinberg(3.0e-26, 27.0)
    b = omega_lee_weinberg(6.0e-26, 27.0)
    assert a == pytest.approx(2.0 * b)


def test_yield_omega_roundtrip():
    y = yield_from_omega(0.12, 165.67)
    assert omega_from_yield(y, 165.67) == pytest.approx(0.12)


def test_solver_converges_and_conserves_positivity():
    m = CoannihilationModel().with_target_sigma_eff(3.0e-26, 27.0)
    res = BoltzmannSolver(m).solve(n_eval=200)
    assert res.success
    assert (res.y[0] >= -1e-15).all()
    assert res.y_inf > 0.0
