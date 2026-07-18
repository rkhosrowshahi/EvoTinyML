"""Custom mutation operators for weight-space evolution."""

from __future__ import annotations

import numpy as np
from pymoo.core.mutation import Mutation
from pymoo.core.variable import Real, get
from pymoo.util import default_random_state


def fan_ins_from_shapes(shapes: list[tuple[int, ...]]) -> list[float]:
    """He-style fan-in per parameter tensor.

    Conv2d ``(out_c, in_c, kH, kW)`` → ``in_c * kH * kW``.
    Linear ``(out_f, in_f)`` → ``in_f``.
    Bias ``(n,)`` → fan-in of the preceding weight tensor.
    """
    fan_ins: list[float] = []
    last_weight_fan_in = 1.0
    for shape in shapes:
        ndim = len(shape)
        if ndim == 4:
            fan_in = float(shape[1] * shape[2] * shape[3])
            last_weight_fan_in = fan_in
        elif ndim == 2:
            fan_in = float(shape[1])
            last_weight_fan_in = fan_in
        elif ndim == 1:
            fan_in = last_weight_fan_in
        else:
            fan_in = float(np.prod(shape))
        fan_ins.append(max(fan_in, 1.0))
    return fan_ins


def layer_sigma_vector(
    shapes: list[tuple[int, ...]],
    base_sigma: float,
    n_var: int | None = None,
) -> np.ndarray:
    """Per-variable σ with He fan-in scaling, mean-normalized to ``base_sigma``.

    ``σ_i = base_sigma * (1/√fan_in_ℓ(i)) / mean_j(1/√fan_in_ℓ(j))`` so the
    average mutation std matches absolute Gaussian with the same ``base_sigma``.
    """
    fan_ins = fan_ins_from_shapes(shapes)
    scales: list[float] = []
    for shape, fan_in in zip(shapes, fan_ins):
        n = int(np.prod(shape))
        scales.extend([1.0 / np.sqrt(fan_in)] * n)
    scale_vec = np.asarray(scales, dtype=float)
    if n_var is not None and scale_vec.size != n_var:
        raise ValueError(
            f"Layer shapes cover {scale_vec.size} vars but problem has n_var={n_var}"
        )
    mean_scale = float(scale_vec.mean())
    if mean_scale <= 0:
        raise ValueError("layer scale mean must be > 0")
    return base_sigma * (scale_vec / mean_scale)


class AbsoluteGaussianMutation(Mutation):
    """Add ``N(0, sigma)`` noise to selected variables (absolute scale).

    Unlike pymoo's ``GM`` (which scales sigma by ``xu - xl``), this uses an
    absolute standard deviation in decision-variable units — natural for
    neural network weight vectors.
    """

    def __init__(
        self,
        sigma: float = 0.1,
        prob: float = 1.0,
        prob_var: float | None = None,
        clip_to_bounds: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(prob=prob, prob_var=prob_var, **kwargs)
        if sigma <= 0:
            raise ValueError(f"mutation_sigma must be > 0, got {sigma}")
        self.sigma = Real(sigma, bounds=(1e-8, 10.0), strict=(0.0, None))
        self.clip_to_bounds = clip_to_bounds

    @default_random_state
    def _do(self, problem, X, random_state=None, **kwargs):
        X = np.asarray(X, dtype=float)
        n, n_var = X.shape

        sigma = get(self.sigma, size=n)
        prob_var = self.get_prob_var(problem, size=n)

        Xp = X.copy()
        mut = random_state.random((n, n_var)) < prob_var[:, None]
        noise = random_state.normal(0.0, 1.0, size=(n, n_var)) * sigma[:, None]
        Xp[mut] = X[mut] + noise[mut]

        if self.clip_to_bounds and problem.xl is not None and problem.xu is not None:
            xl = np.asarray(problem.xl, dtype=float)
            xu = np.asarray(problem.xu, dtype=float)
            Xp = np.clip(Xp, xl, xu)

        return Xp


class LayerWiseGaussianMutation(Mutation):
    """Gaussian mutation with He-style per-layer σ, mean-normalized to ``sigma``.

    Requires ``problem._param_shapes`` (as on ``WeightOptimizationProblem``), or
    an explicit ``param_shapes`` list at construction time.
    """

    def __init__(
        self,
        sigma: float = 0.1,
        prob: float = 1.0,
        prob_var: float | None = None,
        param_shapes: list[tuple[int, ...]] | None = None,
        clip_to_bounds: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(prob=prob, prob_var=prob_var, **kwargs)
        if sigma <= 0:
            raise ValueError(f"mutation_sigma must be > 0, got {sigma}")
        self.sigma = float(sigma)
        self.param_shapes = param_shapes
        self.clip_to_bounds = clip_to_bounds
        self._sigma_vec: np.ndarray | None = None
        if param_shapes is not None:
            self._sigma_vec = layer_sigma_vector(param_shapes, self.sigma)

    def _ensure_sigma_vec(self, problem) -> np.ndarray:
        if self._sigma_vec is not None:
            if self._sigma_vec.size != problem.n_var:
                raise ValueError(
                    f"Cached sigma vec has {self._sigma_vec.size} entries, "
                    f"problem n_var={problem.n_var}"
                )
            return self._sigma_vec

        shapes = self.param_shapes or getattr(problem, "_param_shapes", None)
        if not shapes:
            raise ValueError(
                "LayerWiseGaussianMutation needs param_shapes or problem._param_shapes"
            )
        self.param_shapes = list(shapes)
        self._sigma_vec = layer_sigma_vector(self.param_shapes, self.sigma, problem.n_var)
        return self._sigma_vec

    @default_random_state
    def _do(self, problem, X, random_state=None, **kwargs):
        X = np.asarray(X, dtype=float)
        n, n_var = X.shape
        sigma_vec = self._ensure_sigma_vec(problem)
        prob_var = self.get_prob_var(problem, size=n)

        Xp = X.copy()
        mut = random_state.random((n, n_var)) < prob_var[:, None]
        noise = random_state.normal(0.0, 1.0, size=(n, n_var)) * sigma_vec[None, :]
        Xp[mut] = X[mut] + noise[mut]

        if self.clip_to_bounds and problem.xl is not None and problem.xu is not None:
            xl = np.asarray(problem.xl, dtype=float)
            xu = np.asarray(problem.xu, dtype=float)
            Xp = np.clip(Xp, xl, xu)

        return Xp
