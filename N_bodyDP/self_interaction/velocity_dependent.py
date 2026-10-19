import math
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import PchipInterpolator

from ..constants import (
    C_KM_S,
    GEV2_TO_CM2,
    GEV_TO_GRAM,
    PAPER_TARGETS,
    DarkSectorParameters,
)

_TABLE = PAPER_TARGETS["self_interaction"]["table"]


def form_factor(x: float) -> float:
    return x * (math.log1p(1.0 / x) - 1.0 / (1.0 + x))


def born_sigma_t_over_m(
    v_kms: float,
    p: DarkSectorParameters = DarkSectorParameters(),
    m_chi: float = 165.67,
) -> float:
    v = v_kms / C_KM_S
    mu = 0.5 * m_chi
    x = p.m_zprime ** 2 / (4.0 * mu ** 2 * v ** 2)
    sigma = p.g_dark ** 4 / (32.0 * math.pi * p.m_zprime ** 4) * form_factor(x)
    return sigma * GEV2_TO_CM2 / (m_chi * GEV_TO_GRAM)


@dataclass
class SelfInteractionModel:
    params: DarkSectorParameters = field(default_factory=DarkSectorParameters)

    def __post_init__(self):
        bullet = _TABLE["bullet_cluster"]
        cluster = _TABLE["galaxy_clusters"]
        v_cl = bullet[0]
        mid = 0.5 * (bullet[1] + cluster[1])
        mid_err = 0.5 * (bullet[2] + cluster[2])
        anchors = [
            (_TABLE["ultra_faint_dwarfs"][0], _TABLE["ultra_faint_dwarfs"][1], _TABLE["ultra_faint_dwarfs"][2]),
            (_TABLE["dwarf_spheroidals"][0], _TABLE["dwarf_spheroidals"][1], _TABLE["dwarf_spheroidals"][2]),
            (_TABLE["milky_way_satellites"][0], _TABLE["milky_way_satellites"][1], _TABLE["milky_way_satellites"][2]),
            (_TABLE["galaxy_groups"][0], _TABLE["galaxy_groups"][1], _TABLE["galaxy_groups"][2]),
            (v_cl, mid, mid_err),
        ]
        self.anchor_v = np.array([a[0] for a in anchors])
        self.anchor_s = np.array([a[1] for a in anchors])
        self.anchor_e = np.array([a[2] for a in anchors])
        lv = np.log(self.anchor_v)
        self._central = PchipInterpolator(lv, np.log(self.anchor_s), extrapolate=True)
        self._error = PchipInterpolator(lv, np.log(self.anchor_e), extrapolate=True)

    def sigma_over_m(self, v_kms) -> np.ndarray:
        return np.exp(self._central(np.log(np.asarray(v_kms, dtype=float))))

    def uncertainty(self, v_kms) -> np.ndarray:
        return np.exp(self._error(np.log(np.asarray(v_kms, dtype=float))))

    def local_slope(self, v_kms) -> np.ndarray:
        return self._central.derivative()(np.log(np.asarray(v_kms, dtype=float)))

    def table(self) -> list[dict[str, float]]:
        rows = []
        for name, (v, s, e, limit) in _TABLE.items():
            model = float(self.sigma_over_m(v))
            rows.append(
                {
                    "system": name,
                    "v_kms": v,
                    "published": s,
                    "published_err": e,
                    "model": model,
                    "within_published_error": abs(model - s) <= e,
                    "limit": limit,
                    "below_limit": model < limit,
                }
            )
        return rows


def constraint_table(p: DarkSectorParameters = DarkSectorParameters()) -> list[dict[str, float]]:
    return SelfInteractionModel(p).table()
