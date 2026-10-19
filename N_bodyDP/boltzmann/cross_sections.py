import math
from dataclasses import dataclass, field

import numpy as np

from ..constants import GEV_INV2_TO_CM3_S, DarkSectorParameters
from ..gmo.mass_spectrum import masses_array, two_body_width
from ..gmo.quantum_numbers import OCTET

FAMILY = np.array([0, 0, 0, 0, 1, 1, 1, 2])


def zprime_annihilation_formula(
    m_chi: float,
    m_zprime: float,
    g_dark: float,
    g_f: float,
    m_f: float = 0.0,
    s: float | None = None,
) -> float:
    if s is None:
        s = 4.0 * m_chi ** 2
    phase = math.sqrt(max(1.0 - 4.0 * m_f ** 2 / s, 0.0))
    return (
        g_f ** 2 * g_dark ** 2 / (16.0 * math.pi)
        * m_chi ** 2 / (m_chi ** 2 + m_zprime ** 2) ** 2
        * phase
    )


@dataclass
class CoannihilationModel:
    params: DarkSectorParameters = field(default_factory=DarkSectorParameters)
    dof: np.ndarray | None = None
    same_family_weight: float = 1.0
    cross_family_weight: float = 0.05
    sigma0_gev2: float = 0.0

    def __post_init__(self):
        self.masses = masses_array(self.params)
        self.n = len(OCTET)
        if self.dof is None:
            self.dof = np.full(self.n, 2.0)
        self.delta = (self.masses - self.masses[0]) / self.params.M0
        same = FAMILY[:, None] == FAMILY[None, :]
        self.weights = np.where(same, self.same_family_weight, self.cross_family_weight)
        if self.sigma0_gev2 == 0.0:
            self.sigma0_gev2 = zprime_annihilation_formula(
                self.masses[0], self.params.m_zprime, self.params.g_dark, self.params.g_f
            )

    def sigma_matrix(self) -> np.ndarray:
        return self.sigma0_gev2 * self.weights

    def decay_widths(self) -> np.ndarray:
        widths = np.zeros(self.n)
        for i in range(1, self.n):
            widths[i] = two_body_width(
                self.masses[i],
                self.masses[i] - self.masses[0],
                self.params.g_dark,
                self.masses[0],
                self.params.m_phi,
            )
        return widths

    def sigma0_cm3_s(self) -> float:
        return self.sigma0_gev2 * GEV_INV2_TO_CM3_S

    def scaled(self, factor: float) -> "CoannihilationModel":
        return CoannihilationModel(
            params=self.params,
            dof=self.dof.copy(),
            same_family_weight=self.same_family_weight,
            cross_family_weight=self.cross_family_weight,
            sigma0_gev2=self.sigma0_gev2 * factor,
        )

    def with_target_sigma_eff(self, target_cm3_s: float, x: float) -> "CoannihilationModel":
        from .freeze_out import sigma_eff

        current = sigma_eff(self, x) * GEV_INV2_TO_CM3_S
        return self.scaled(target_cm3_s / current)
