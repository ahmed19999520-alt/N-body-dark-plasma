import numpy as np
import pytest

from N_bodyDP.bayesian import (
    FIDUCIAL,
    RECONSTRUCTION_THRESHOLD_DEX,
    SIGNAL_PARAMETERS,
    CoarseGrainedLikelihood,
    PriorSet,
    fractional_uncertainty_dex,
    generate_mock,
    geometry_from_covariance,
    posterior_geometry,
    run_nested,
    signal_block_geometry,
)


@pytest.fixture(scope="module")
def setup():
    truth = dict(FIDUCIAL)
    prior = PriorSet(free=SIGNAL_PARAMETERS, fixed=truth)
    noiseless = CoarseGrainedLikelihood(generate_mock(truth, noiseless=True), prior)
    noisy = CoarseGrainedLikelihood(generate_mock(truth, seed=3), prior)
    return prior, noiseless, noisy


def test_prior_roundtrip():
    prior = PriorSet()
    u = np.full(prior.ndim, 0.37)
    assert np.allclose(prior.inverse(prior.transform(u)), u)
    assert np.isfinite(prior.log_prior(prior.fiducial_vector()))
    assert prior.log_prior(prior.hi + 1.0) == -np.inf


def test_likelihood_peaks_at_truth_for_noiseless_data(setup):
    prior, noiseless, _ = setup
    th0 = prior.fiducial_vector()
    base = noiseless.log_likelihood(th0)
    for shift in ([0.05, 0.0], [-0.05, 0.0], [0.0, 0.05], [0.0, -0.05]):
        assert noiseless.log_likelihood(th0 + np.array(shift)) < base


def test_geometry_identities():
    cov = np.array([[0.04, 0.03], [0.03, 0.09]])
    g = geometry_from_covariance(cov)
    assert g["area"] == pytest.approx(np.sqrt(np.linalg.det(cov)))
    assert g["rho"] == pytest.approx(0.03 / np.sqrt(0.04 * 0.09))
    assert g["kappa_deg"] >= 1.0


def test_threshold_value():
    assert RECONSTRUCTION_THRESHOLD_DEX == pytest.approx(0.0969, abs=1e-4)
    assert fractional_uncertainty_dex(RECONSTRUCTION_THRESHOLD_DEX) == pytest.approx(0.25)


def test_laplace_geometry_is_finite(setup):
    prior, noiseless, _ = setup
    g = signal_block_geometry(noiseless, prior.fiducial_vector())
    assert g["sigma1"] > 0 and g["sigma2"] > 0
    assert -1.0 < g["rho"] < 1.0


def test_nested_sampling_recovers_injection(setup):
    prior, _, noisy = setup
    res = run_nested(noisy, n_live=250, dlogz=0.1, seed=2)
    mean, std = res.mean(), res.std()
    truth = prior.fiducial_vector()
    assert np.all(np.abs(mean - truth) < 3.0 * std)
    assert abs(res.weights.sum() - 1.0) < 1e-9
    geo = posterior_geometry(res.samples, res.weights)
    assert geo["area"] > 0
