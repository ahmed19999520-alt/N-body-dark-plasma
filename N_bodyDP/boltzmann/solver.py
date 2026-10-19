import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import kve

from ..constants import M_PLANCK_GEV, OMEGA_DM_H2_PLANCK
from .freeze_out import omega_from_yield


@dataclass
class BoltzmannResult:
    x: np.ndarray
    y: np.ndarray
    y_eq: np.ndarray
    y_inf: float
    omega_h2: float
    x_freeze: float
    success: bool
    n_steps: int


class BoltzmannSolver:
    def __init__(self, model, g_star: float = 106.75, g_star_s: float | None = None):
        self.model = model
        self.g_star = g_star
        self.g_star_s = g_star if g_star_s is None else g_star_s
        self.m0 = model.params.M0
        self.mass_ratio = model.masses / self.m0
        self.dof = model.dof
        self.sigma = model.sigma_matrix()
        self.gamma = model.decay_widths()
        self.n = model.n
        hub = math.sqrt(math.pi ** 2 * g_star / 90.0)
        self.c_ann = (2.0 * math.pi ** 2 / 45.0) * self.g_star_s * self.m0 * M_PLANCK_GEV / hub
        self.c_dec = M_PLANCK_GEV / (hub * self.m0 ** 2)
        self.decay_rate = self.c_dec * self.gamma

    def y_eq(self, x: float) -> np.ndarray:
        xi = x * self.mass_ratio
        return (45.0 / (4.0 * math.pi ** 4)) * self.dof / self.g_star_s * xi ** 2 * kve(2, xi) * np.exp(-xi)

    def rhs(self, x: float, y: np.ndarray) -> np.ndarray:
        ye = self.y_eq(x)
        ann = (self.c_ann / x ** 2) * (y * (self.sigma @ y) - ye * (self.sigma @ ye))
        dy = -ann
        loss = self.decay_rate * x * y
        dy -= loss
        dy[0] += loss[1:].sum()
        return dy

    def jacobian(self, x: float, y: np.ndarray) -> np.ndarray:
        k = self.c_ann / x ** 2
        jac = -k * (np.diag(self.sigma @ y) + y[:, None] * self.sigma)
        d = self.decay_rate * x
        for i in range(1, self.n):
            jac[i, i] -= d[i]
            jac[0, i] += d[i]
        return jac

    def solve(
        self,
        x_init: float = 1.0,
        x_fin: float = 1.0e3,
        rtol: float = 1.0e-9,
        atol: float = 1.0e-12,
        n_eval: int = 600,
        method: str = "Radau",
    ) -> BoltzmannResult:
        y0 = self.y_eq(x_init)
        x_eval = np.geomspace(x_init, x_fin, n_eval)
        sol = solve_ivp(
            self.rhs,
            (x_init, x_fin),
            y0,
            method=method,
            jac=self.jacobian,
            t_eval=x_eval,
            rtol=rtol,
            atol=atol,
        )
        y_eq = np.array([self.y_eq(x) for x in sol.t]).T
        y_inf = float(sol.y[:, -1].sum())
        ratio = sol.y[0] / np.maximum(y_eq[0], 1e-300)
        idx = np.where(ratio > 1.1)[0]
        x_freeze = float(sol.t[idx[0]]) if idx.size else float("nan")
        return BoltzmannResult(
            x=sol.t,
            y=sol.y,
            y_eq=y_eq,
            y_inf=y_inf,
            omega_h2=omega_from_yield(y_inf, self.model.masses[0]),
            x_freeze=x_freeze,
            success=bool(sol.success),
            n_steps=int(sol.t.size),
        )


def calibrate_to_relic(
    model,
    target: float = OMEGA_DM_H2_PLANCK,
    bracket=(1.0e-3, 1.0e3),
    **solver_kwargs,
):
    def residual(log_factor: float) -> float:
        m = model.scaled(math.exp(log_factor))
        res = BoltzmannSolver(m, **solver_kwargs).solve(n_eval=200)
        return math.log(res.omega_h2 / target)

    root = brentq(residual, math.log(bracket[0]), math.log(bracket[1]), xtol=1e-6)
    return model.scaled(math.exp(root))
