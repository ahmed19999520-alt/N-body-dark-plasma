import math

import numpy as np
from scipy.spatial import cKDTree

from .units import sigma_over_m_to_internal


class SidmScatterer:
    def __init__(self, sigma_over_m_fn, particle_mass: float, kernel_radius: float, seed: int = 11):
        self.fn = sigma_over_m_fn
        self.m = particle_mass
        self.h = kernel_radius
        self.rng = np.random.default_rng(seed)
        self.n_scatter = 0

    def step(self, pos, vel, dt: float):
        tree = cKDTree(pos)
        pairs = tree.query_pairs(self.h, output_type="ndarray")
        if pairs.size == 0:
            return vel
        i, j = pairs[:, 0], pairs[:, 1]
        dv = vel[i] - vel[j]
        vrel = np.linalg.norm(dv, axis=1)
        sigma = sigma_over_m_to_internal(np.asarray(self.fn(np.maximum(vrel, 1.0e-3))))
        prob = sigma * self.m * vrel * dt / (4.0 / 3.0 * math.pi * self.h ** 3)
        hit = self.rng.uniform(size=prob.size) < prob
        used = np.zeros(pos.shape[0], dtype=bool)
        out = vel.copy()
        for k in np.where(hit)[0]:
            a, b = i[k], j[k]
            if used[a] or used[b]:
                continue
            used[a] = used[b] = True
            cm = 0.5 * (vel[a] + vel[b])
            c = self.rng.uniform(-1.0, 1.0)
            s = math.sqrt(1.0 - c * c)
            phi = self.rng.uniform(0.0, 2.0 * math.pi)
            n = np.array([s * math.cos(phi), s * math.sin(phi), c])
            out[a] = cm + 0.5 * vrel[k] * n
            out[b] = cm - 0.5 * vrel[k] * n
            self.n_scatter += 1
        return out
