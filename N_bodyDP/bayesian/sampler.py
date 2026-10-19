from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .likelihood import CoarseGrainedLikelihood, generate_mock
from .priors import FIDUCIAL, SIGNAL_PARAMETERS, PriorSet


@dataclass
class NestedResult:
    samples: np.ndarray
    weights: np.ndarray
    log_evidence: float
    log_evidence_err: float
    names: Sequence[str]

    def mean(self) -> np.ndarray:
        return self.weights @ self.samples

    def std(self) -> np.ndarray:
        m = self.mean()
        return np.sqrt(self.weights @ (self.samples - m) ** 2)

    def column(self, name: str) -> np.ndarray:
        return self.samples[:, list(self.names).index(name)]

    def quantiles(self, name: str, q=(0.16, 0.5, 0.84)) -> np.ndarray:
        col = self.column(name)
        order = np.argsort(col)
        cum = np.cumsum(self.weights[order])
        return np.interp(q, cum, col[order])


def run_nested(
    like: CoarseGrainedLikelihood,
    n_live: int = 400,
    dlogz: float = 0.1,
    seed: int = 1,
    sample: str = "rwalk",
) -> NestedResult:
    import dynesty

    prior = like.prior
    rng = np.random.default_rng(seed)
    sampler = dynesty.NestedSampler(
        like.log_likelihood,
        prior.transform,
        prior.ndim,
        nlive=n_live,
        sample=sample,
        rstate=rng,
    )
    sampler.run_nested(dlogz=dlogz, print_progress=False)
    res = sampler.results
    weights = np.exp(res.logwt - res.logz[-1])
    weights /= weights.sum()
    return NestedResult(
        samples=np.asarray(res.samples),
        weights=weights,
        log_evidence=float(res.logz[-1]),
        log_evidence_err=float(res.logzerr[-1]),
        names=prior.free,
    )


def reconstruct(
    injected: dict[str, float] = None,
    free: Sequence[str] = SIGNAL_PARAMETERS,
    n_live: int = 400,
    seed: int = 3,
    t_obs_years: float = 4.0,
    noiseless: bool = False,
):
    truth = dict(FIDUCIAL)
    if injected:
        truth.update(injected)
    data = generate_mock(truth, seed=seed, t_obs_years=t_obs_years, noiseless=noiseless)
    prior = PriorSet(free=tuple(free), fixed=truth)
    like = CoarseGrainedLikelihood(data, prior)
    return like, run_nested(like, n_live=n_live, seed=seed), truth
