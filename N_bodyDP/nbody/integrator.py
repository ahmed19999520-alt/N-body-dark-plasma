import numpy as np


def leapfrog(pos, vel, accel_fn, dt: float, n_steps: int, callback=None, interval: int = 1):
    pos = pos.copy()
    vel = vel.copy()
    acc = accel_fn(pos)
    for step in range(1, n_steps + 1):
        vel += 0.5 * dt * acc
        pos += dt * vel
        acc = accel_fn(pos)
        vel += 0.5 * dt * acc
        if callback is not None and step % interval == 0:
            callback(step, pos, vel)
    return pos, vel


def kinetic_energy(vel, mass) -> float:
    return 0.5 * float(np.sum(mass * np.einsum("ij,ij->i", vel, vel)))


def angular_momentum(pos, vel, mass):
    return np.sum(mass[:, None] * np.cross(pos, vel), axis=0)


def half_mass_radius(pos, mass) -> float:
    com = np.average(pos, axis=0, weights=mass)
    r = np.linalg.norm(pos - com, axis=1)
    order = np.argsort(r)
    cum = np.cumsum(mass[order])
    return float(r[order][np.searchsorted(cum, 0.5 * cum[-1])])


def density_profile(pos, mass, n_bins: int = 20, r_min: float = None, r_max: float = None):
    com = np.average(pos, axis=0, weights=mass)
    r = np.linalg.norm(pos - com, axis=1)
    lo = r[r > 0].min() if r_min is None else r_min
    hi = r.max() if r_max is None else r_max
    edges = np.geomspace(lo, hi, n_bins + 1)
    hist, _ = np.histogram(r, bins=edges, weights=mass)
    vol = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    centers = np.sqrt(edges[1:] * edges[:-1])
    return centers, hist / vol
