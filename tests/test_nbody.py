import numpy as np
import pytest

from N_bodyDP.nbody import (
    SidmScatterer,
    TwoStreamPIC,
    angular_momentum,
    barnes_hut_accelerations,
    direct_accelerations,
    direct_potential_energy,
    half_mass_radius,
    kinetic_energy,
    leapfrog,
    plummer_sphere,
    sigma_over_m_to_internal,
)
from N_bodyDP.self_interaction import SelfInteractionModel


def test_barnes_hut_matches_direct():
    pos, vel, m = plummer_sphere(500, 1.0e9, 1.0)
    a = direct_accelerations(pos, m, 0.05)
    b = barnes_hut_accelerations(pos, m, 0.05, theta=0.5)
    err = np.linalg.norm(a - b, axis=1) / np.linalg.norm(a, axis=1)
    assert np.median(err) < 5.0e-3
    assert err.max() < 5.0e-2


def test_barnes_hut_theta_zero_limit_is_exact():
    pos, vel, m = plummer_sphere(200, 1.0e9, 1.0)
    a = direct_accelerations(pos, m, 0.05)
    b = barnes_hut_accelerations(pos, m, 0.05, theta=1.0e-6)
    assert np.allclose(a, b, rtol=1e-8, atol=1e-12)


def test_leapfrog_energy_conservation():
    pos, vel, m = plummer_sphere(150, 1.0e9, 1.0)
    fn = lambda x: direct_accelerations(x, m, 0.1)
    e0 = kinetic_energy(vel, m) + direct_potential_energy(pos, m, 0.1)
    p1, v1 = leapfrog(pos, vel, fn, 0.0005, 400)
    e1 = kinetic_energy(v1, m) + direct_potential_energy(p1, m, 0.1)
    assert abs(e1 / e0 - 1.0) < 2.0e-3
    l0 = angular_momentum(pos, vel, m)
    l1 = angular_momentum(p1, v1, m)
    assert np.linalg.norm(l1 - l0) < 1.0e-3 * (np.linalg.norm(l0) + np.sum(m * np.linalg.norm(pos, axis=1) * np.linalg.norm(vel, axis=1)))


def test_plummer_virial_state():
    pos, vel, m = plummer_sphere(2000, 1.0e9, 1.0)
    k = kinetic_energy(vel, m)
    w = direct_potential_energy(pos, m, 0.01)
    assert abs(2.0 * k / abs(w) - 1.0) < 0.15
    assert half_mass_radius(pos, m) == pytest.approx(1.305, rel=0.15)


def test_sidm_conserves_momentum_and_energy():
    pos, vel, m = plummer_sphere(3000, 1.0e9, 0.5)
    model = SelfInteractionModel()
    sc = SidmScatterer(lambda v: model.sigma_over_m(v) * 1.0e3, m[0], 0.05)
    new = sc.step(pos, vel, 5.0)
    assert sc.n_scatter > 0
    assert np.allclose(new.sum(axis=0), vel.sum(axis=0), atol=1e-6 * np.abs(vel).sum())
    assert kinetic_energy(new, m) == pytest.approx(kinetic_energy(vel, m), rel=1e-10)


def test_unit_conversion():
    assert sigma_over_m_to_internal(1.0) == pytest.approx(2.089e-10, rel=1e-3)


def test_pic_conserves_energy_and_grows():
    pic = TwoStreamPIC(n_particles=20000, n_cells=64)
    h = pic.run(500, 0.05)
    total = h.sum(axis=1)
    assert total.max() / total[0] < 1.01
    assert h[:, 0].max() / h[0, 0] > 100.0
