"""Population initialization operators for NSGA-II / NSGA-III."""

from __future__ import annotations

import numpy as np
from pymoo.core.sampling import Sampling


INIT_NAMES = ("uniform", "gaussian", "both", "zeros", "kaiming")


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


class PytorchDefaultSampling(Sampling):
    """One individual at PyTorch default ``theta0``; rest ``theta0 + N(0, sigma)``.

    ``theta0`` is taken from the constructor arg, else ``problem.theta0``.
    """

    def __init__(self, sigma: float = 0.1, theta0: np.ndarray | None = None) -> None:
        super().__init__()
        if sigma <= 0:
            raise ValueError(f"init_sigma must be > 0, got {sigma}")
        self.sigma = float(sigma)
        self.theta0 = (
            None if theta0 is None else np.asarray(theta0, dtype=np.float64).ravel()
        )

    def _resolve_theta0(self, problem) -> np.ndarray:
        if self.theta0 is not None:
            theta0 = self.theta0
        else:
            theta0 = getattr(problem, "theta0", None)
            if theta0 is None:
                raise RuntimeError(
                    "kaiming init needs problem.theta0 (model default / Kaiming weights)."
                )
            theta0 = np.asarray(theta0, dtype=np.float64).ravel()
        if theta0.size != int(problem.n_var):
            raise ValueError(
                f"theta0 length {theta0.size} != n_var {problem.n_var}"
            )
        return theta0

    def _do(self, problem, n_samples, **kwargs):
        theta0 = self._resolve_theta0(problem)
        X = np.empty((n_samples, problem.n_var), dtype=np.float64)
        X[0] = theta0
        if n_samples > 1:
            X[1:] = theta0 + np.random.normal(
                0.0, self.sigma, size=(n_samples - 1, problem.n_var)
            )
        return X


def get_population_init(
    name: str,
    init_sigma: float = 0.1,
    theta0: np.ndarray | None = None,
) -> Sampling:
    name = name.lower()
    if name == "uniform":
        return UniformWeightSampling(sigma=init_sigma)
    if name == "gaussian":
        return GaussianWeightSampling(sigma=init_sigma)
    if name in {"both", "mixed"}:
        return MixedWeightSampling(sigma=init_sigma)
    if name in {"zeros", "zero"}:
        return ZeroWeightSampling(sigma=init_sigma)
    if name in {"kaiming", "he", "theta0", "pytorch", "torch", "default"}:
        return PytorchDefaultSampling(sigma=init_sigma, theta0=theta0)
    raise ValueError(
        f"Unknown population init: {name!r}. "
        "Use 'uniform', 'gaussian', 'both', 'zeros', or 'kaiming'."
    )
