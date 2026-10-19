import math

import numpy as np

from .units import G_KPC


def plummer_sphere(n: int, total_mass: float, scale_radius: float, seed: int = 7, g: float = G_KPC):
    rng = np.random.default_rng(seed)
    u = rng.uniform(1.0e-6, 1.0 - 1.0e-6, n)
    r = scale_radius / np.sqrt(u ** (-2.0 / 3.0) - 1.0)
    cos_t = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * math.pi, n)
    sin_t = np.sqrt(1.0 - cos_t ** 2)
    pos = np.stack([r * sin_t * np.cos(phi), r * sin_t * np.sin(phi), r * cos_t], axis=1)
    v_esc = np.sqrt(2.0 * g * total_mass / scale_radius) * (1.0 + (r / scale_radius) ** 2) ** -0.25
    q = np.empty(n)
    filled = 0
    while filled < n:
        x = rng.uniform(0.0, 1.0, 2 * (n - filled))
        y = rng.uniform(0.0, 0.1, 2 * (n - filled))
        ok = x[y < x ** 2 * (1.0 - x ** 2) ** 3.5]
        take = min(ok.size, n - filled)
        q[filled:filled + take] = ok[:take]
        filled += take
    speed = q * v_esc
    cos_v = rng.uniform(-1.0, 1.0, n)
    phi_v = rng.uniform(0.0, 2.0 * math.pi, n)
    sin_v = np.sqrt(1.0 - cos_v ** 2)
    vel = np.stack([speed * sin_v * np.cos(phi_v), speed * sin_v * np.sin(phi_v), speed * cos_v], axis=1)
    mass = np.full(n, total_mass / n)
    pos -= np.average(pos, axis=0, weights=mass)
    vel -= np.average(vel, axis=0, weights=mass)
    return pos, vel, mass


def uniform_sphere(n: int, total_mass: float, radius: float, seed: int = 7):
    rng = np.random.default_rng(seed)
    r = radius * rng.uniform(0.0, 1.0, n) ** (1.0 / 3.0)
    cos_t = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * math.pi, n)
    sin_t = np.sqrt(1.0 - cos_t ** 2)
    pos = np.stack([r * sin_t * np.cos(phi), r * sin_t * np.sin(phi), r * cos_t], axis=1)
    return pos, np.zeros((n, 3)), np.full(n, total_mass / n)
