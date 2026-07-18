"""Population initialization operators for NSGA-II / NSGA-III."""

from __future__ import annotations

import numpy as np
from pymoo.core.sampling import Sampling


class UniformWeightSampling(Sampling):
    """Initialize population uniformly in ``[-sigma, sigma]``."""

    def __init__(self, sigma: float = 0.1) -> None:
        super().__init__()
        if sigma <= 0:
            raise ValueError(f"init_sigma must be > 0, got {sigma}")
        self.sigma = float(sigma)
        self.low = -self.sigma
        self.high = self.sigma

    def _do(self, problem, n_samples, **kwargs):
        return np.random.uniform(self.low, self.high, size=(n_samples, problem.n_var))


class GaussianWeightSampling(Sampling):
    """Initialize population from ``N(0, sigma)``."""

    def __init__(self, sigma: float = 0.1) -> None:
        super().__init__()
        if sigma <= 0:
            raise ValueError(f"init_sigma must be > 0, got {sigma}")
        self.sigma = float(sigma)

    def _do(self, problem, n_samples, **kwargs):
        return np.random.normal(0.0, self.sigma, size=(n_samples, problem.n_var))


class MixedWeightSampling(Sampling):
    """Half uniform ``[-sigma, sigma]``, half Gaussian ``N(0, sigma)``."""

    def __init__(self, sigma: float = 0.1) -> None:
        super().__init__()
        if sigma <= 0:
            raise ValueError(f"init_sigma must be > 0, got {sigma}")
        self.sigma = float(sigma)
        self.low = -self.sigma
        self.high = self.sigma

    def _do(self, problem, n_samples, **kwargs):
        n_uniform = n_samples // 2
        n_gaussian = n_samples - n_uniform
        u = np.random.uniform(self.low, self.high, size=(n_uniform, problem.n_var))
        g = np.random.normal(0.0, self.sigma, size=(n_gaussian, problem.n_var))
        X = np.vstack([u, g])
        np.random.shuffle(X)
        return X


class ZeroWeightSampling(Sampling):
    """Initialize every individual at the zero vector (``init_sigma`` unused)."""

    def __init__(self, sigma: float = 0.1) -> None:
        super().__init__()
        self.sigma = float(sigma)

    def _do(self, problem, n_samples, **kwargs):
        return np.zeros((n_samples, problem.n_var), dtype=np.float64)


def get_population_init(name: str, init_sigma: float = 0.1) -> Sampling:
    name = name.lower()
    if name == "uniform":
        return UniformWeightSampling(sigma=init_sigma)
    if name == "gaussian":
        return GaussianWeightSampling(sigma=init_sigma)
    if name in {"both", "mixed"}:
        return MixedWeightSampling(sigma=init_sigma)
    if name in {"zeros", "zero"}:
        return ZeroWeightSampling(sigma=init_sigma)
    raise ValueError(
        f"Unknown population init: {name!r}. "
        "Use 'uniform', 'gaussian', 'both', or 'zeros'."
    )
