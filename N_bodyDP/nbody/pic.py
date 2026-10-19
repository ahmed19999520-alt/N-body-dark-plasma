import numpy as np


class TwoStreamPIC:
    def __init__(
        self,
        n_particles: int = 40000,
        n_cells: int = 128,
        box_length: float = 2.0 * np.pi / 0.29,
        drift_velocity: float = 3.0,
        thermal_velocity: float = 0.2,
        perturbation: float = 1.0e-3,
        seed: int = 5,
    ):
        rng = np.random.default_rng(seed)
        self.n = n_particles
        self.nc = n_cells
        self.length = box_length
        self.dx = box_length / n_cells
        self.x = rng.uniform(0.0, box_length, n_particles)
        sign = np.where(np.arange(n_particles) % 2 == 0, 1.0, -1.0)
        self.v = sign * drift_velocity + thermal_velocity * rng.standard_normal(n_particles)
        self.x = (self.x + perturbation * np.cos(2.0 * np.pi * self.x / box_length)) % box_length
        self.charge_weight = box_length / n_particles
        self.k = 2.0 * np.pi * np.fft.fftfreq(n_cells, d=self.dx)
        self.field = np.zeros(n_cells)

    def _deposit(self):
        s = self.x / self.dx
        i0 = np.floor(s).astype(int) % self.nc
        frac = s - np.floor(s)
        i1 = (i0 + 1) % self.nc
        rho = np.bincount(i0, weights=1.0 - frac, minlength=self.nc) + np.bincount(i1, weights=frac, minlength=self.nc)
        return rho * self.charge_weight / self.dx, i0, i1, frac

    def _solve(self, rho):
        rho_k = np.fft.fft(rho - rho.mean())
        e_k = np.zeros_like(rho_k)
        nz = self.k != 0.0
        e_k[nz] = -1j * rho_k[nz] / self.k[nz]
        return np.real(np.fft.ifft(e_k))

    def step(self, dt: float):
        rho, i0, i1, frac = self._deposit()
        self.field = self._solve(-rho)
        e_p = self.field[i0] * (1.0 - frac) + self.field[i1] * frac
        self.v += -dt * e_p
        self.x = (self.x + dt * self.v) % self.length

    def field_energy(self) -> float:
        return 0.5 * float(np.sum(self.field ** 2) * self.dx)

    def kinetic_energy(self) -> float:
        return 0.5 * float(np.sum(self.v ** 2) * self.charge_weight)

    def run(self, n_steps: int, dt: float = 0.05):
        history = np.zeros((n_steps, 2))
        for s in range(n_steps):
            self.step(dt)
            history[s] = (self.field_energy(), self.kinetic_energy())
        return history
