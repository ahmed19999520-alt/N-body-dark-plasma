from dataclasses import dataclass

import numpy as np

PARAMETER_NAMES = (
    "log10_n_acc",
    "log10_delta_x",
    "log10_a1",
    "alpha1",
    "log10_a2",
    "alpha2",
    "log10_omega_ast",
    "eps",
    "log10_alpha_pt",
    "log10_beta_over_h",
)

FIDUCIAL = {
    "log10_n_acc": -14.523,
    "log10_delta_x": -11.097,
    "log10_a1": -15.4,
    "alpha1": -5.7,
    "log10_a2": -6.32,
    "alpha2": -6.2,
    "log10_omega_ast": -11.0,
    "eps": 0.67,
    "log10_alpha_pt": -1.0,
    "log10_beta_over_h": 2.0,
}

BOUNDS = {
    "log10_n_acc": (-16.0, -13.7),
    "log10_delta_x": (-13.0, -10.7),
    "log10_a1": (-18.0, -5.0),
    "alpha1": (-15.0, -3.0),
    "log10_a2": (-10.0, 5.0),
    "alpha2": (-10.0, -1.0),
    "log10_omega_ast": (-15.0, -8.0),
    "eps": (0.0, 1.0),
    "log10_alpha_pt": (-2.0, 0.0),
    "log10_beta_over_h": (1.0, 3.0),
}

SIGNAL_PARAMETERS = ("log10_alpha_pt", "log10_beta_over_h")


@dataclass
class PriorSet:
    free: tuple[str, ...] = PARAMETER_NAMES
    fixed: dict[str, float] = None

    def __post_init__(self):
        base = dict(FIDUCIAL)
        if self.fixed:
            base.update(self.fixed)
        self.fixed = {k: base[k] for k in PARAMETER_NAMES if k not in self.free}
        self.lo = np.array([BOUNDS[k][0] for k in self.free])
        self.hi = np.array([BOUNDS[k][1] for k in self.free])

    @property
    def ndim(self) -> int:
        return len(self.free)

    def transform(self, u):
        return self.lo + np.asarray(u) * (self.hi - self.lo)

    def inverse(self, theta):
        return (np.asarray(theta) - self.lo) / (self.hi - self.lo)

    def log_prior(self, theta) -> float:
        theta = np.asarray(theta)
        if np.any(theta < self.lo) or np.any(theta > self.hi):
            return -np.inf
        return -float(np.sum(np.log(self.hi - self.lo)))

    def sample(self, n: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        return self.lo + rng.uniform(size=(n, self.ndim)) * (self.hi - self.lo)

    def full_vector(self, theta) -> dict[str, float]:
        out = dict(self.fixed)
        out.update({k: float(v) for k, v in zip(self.free, theta)})
        return out

    def fiducial_vector(self):
        return np.array([FIDUCIAL[k] for k in self.free])
