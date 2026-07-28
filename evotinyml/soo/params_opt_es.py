"""Mean-update ES variants that pass ``state.mean`` into Optax ``update``.

Upstream OpenES / SNES / xNES / ASEBO call ``optimizer.update(grad, opt_state)``
without ``params``. Optax transforms that need the current parameters
(``adamw``, ``add_decayed_weights`` / ``--es-optim-wd``) then fail. These
subclasses keep upstream ask/tell math and only fix the Optax call site.

Also defines ``SparseOpenES``: OpenES with per-dimension noise masking before
antithetic sampling; and ``AdaOpenES``: OpenES with a self-adapted
per-parameter sampling std (see its docstring for the derivation).
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from evosax.algorithms.distribution_based.open_es import Open_ES as _Open_ES
from evosax.algorithms.distribution_based.open_es import State as _OpenESState
from evosax.algorithms.distribution_based.snes import SNES as _SNES
from evosax.algorithms.distribution_based.xnes import xNES as _xNES
from evosax.core.fitness_shaping import centered_rank_fitness_shaping_fn
from evosax.types import Fitness, Population, Solution
from flax import struct

from evosax.algorithms.distribution_based.base import (
    DistributionBasedAlgorithm,
    Params as _DistBaseParams,
    metrics_fn,
)


class OpenES(_Open_ES):
    """OpenES with Optax updates that receive the current mean."""

    def _tell(self, key, population: Population, fitness: Fitness, state, params):
        grad = jnp.dot(fitness, (population - state.mean) / state.std) / (
            self.population_size * state.std
        )
        updates, opt_state = self.optimizer.update(
            grad, state.opt_state, state.mean
        )
        mean = optax.apply_updates(state.mean, updates)
        return state.replace(
            mean=mean,
            std=self.std_schedule(state.generation_counter),
            opt_state=opt_state,
        )


class SparseOpenES(OpenES):
    """OpenES with sparse isotropic noise (mask before antithetic sampling).

    Each half-population row draws ``z+ ~ N(0, I)``, then zeros coordinates with
    probability ``mask_prob`` before mirroring to ``-z+``. Shared support keeps
    antithetic pairs matched; inactive dims stay at the mean for that sample.
    """

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        mask_prob: float = 0.2,
        use_antithetic_sampling: bool = True,
        optimizer: optax.GradientTransformation = optax.sgd(learning_rate=1e-3),
        std_schedule: Callable = optax.constant_schedule(1.0),
        fitness_shaping_fn: Callable = centered_rank_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        if not (0.0 <= float(mask_prob) < 1.0):
            raise ValueError(f"mask_prob must be in [0, 1), got {mask_prob}")
        if not use_antithetic_sampling:
            raise ValueError("SparseOpenES requires antithetic sampling.")
        super().__init__(
            population_size,
            solution,
            use_antithetic_sampling=True,
            optimizer=optimizer,
            std_schedule=std_schedule,
            fitness_shaping_fn=fitness_shaping_fn,
            metrics_fn=metrics_fn,
        )
        self.mask_prob = float(mask_prob)

    def _ask(self, key: jax.Array, state, params) -> tuple[Population, object]:
        key_z, key_m = jax.random.split(key)
        half = self.population_size // 2
        z_plus = jax.random.normal(key_z, (half, self.num_dims))
        keep = jax.random.bernoulli(key_m, 1.0 - self.mask_prob, z_plus.shape)
        z_plus = z_plus * keep.astype(z_plus.dtype)
        z = jnp.concatenate([z_plus, -z_plus], axis=0)
        population = state.mean + state.std * z
        return population, state


class SNES(_SNES):
    """SNES with Optax updates that receive the current mean."""

    def _tell(self, key, population: Population, fitness: Fitness, state, params):
        z = (population - state.mean) / state.std
        grad_mean = -state.std * jnp.dot(fitness, z)
        updates, opt_state = self.optimizer.update(
            grad_mean, state.opt_state, state.mean
        )
        mean = optax.apply_updates(state.mean, updates)
        grad_std = jnp.dot(fitness, z**2 - 1)
        std = state.std * jnp.exp(0.5 * state.lr_std * grad_std)
        return state.replace(mean=mean, std=std, opt_state=opt_state)


class xNES(_xNES):
    """xNES with Optax updates that receive the current mean."""

    def _tell(self, key, population: Population, fitness: Fitness, state, params):
        import jax

        grad_mean = -state.std * state.B @ jnp.dot(fitness, state.z)
        updates, opt_state = self.optimizer.update(
            grad_mean, state.opt_state, state.mean
        )
        mean = optax.apply_updates(state.mean, updates)
        grad_M = jnp.einsum(
            "i,ijk->jk",
            fitness,
            jax.vmap(jnp.outer)(state.z, state.z) - jnp.eye(self.num_dims),
        )
        grad_std = jnp.trace(grad_M) / self.num_dims
        std = state.std * jnp.exp(0.5 * state.lr_std * grad_std)
        grad_B = grad_M - grad_std * jnp.eye(self.num_dims)
        B = state.B * jnp.exp(0.5 * params.lr_B * grad_B)
        return state.replace(mean=mean, std=std, opt_state=opt_state, B=B)


@struct.dataclass
class AdaOpenESParams(_DistBaseParams):
    std_init: float
    sigma_lr_init: float
    sigma_min: float
    sigma_max: float


class AdaOpenES(DistributionBasedAlgorithm):
    """OpenES with a self-adapted per-parameter (diagonal) sampling std.

    Standard OpenES draws antithetic noise from a single scalar σ, hand-tuned
    or hand-scheduled for the whole parameter vector. This variant keeps
    OpenES's antithetic sampling, centered-rank fitness shaping, and Optax
    mean update unchanged, but gives every parameter its own σ_j that adapts
    online instead of following a fixed schedule.

    The σ update reuses the same log-derivative ("score function") trick
    OpenES already uses for the mean gradient, applied to the per-coordinate
    log-std of a separable Gaussian search distribution (the same
    natural-gradient form SNES uses, e.g. Wierstra et al. 2014,
    https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf):

        z = (population - mean) / std
        grad_mean_j    = E[fitness * z_j] / std_j
        grad_logstd_j  = E[fitness * (z_j^2 - 1)]

    Both gradients are computed from the *same* population/fitness batch
    already spent on the mean update, so per-parameter σ-adaptation costs no
    extra function evaluations. Since ``fitness`` here is centered-rank
    shaped (worse = larger, matching OpenES's convention), both updates are
    applied as gradient *descent* — the mean step through the Optax
    optimizer, the σ step as a direct log-normal multiplicative update
    clipped to ``[sigma_min, sigma_max]`` each generation to guard against
    runaway growth/collapse in any single dimension (the per-parameter
    estimate is noisier than the shared-σ case, since ``population_size``
    samples now estimate ``num_dims`` independent step sizes instead of one).

    ``sigma_lr_init`` defaults to SNES's dimension-scaled rate
    ``(3 + ln(d)) / (5 * sqrt(d))`` so the adaptation speed needs no manual
    tuning as model size changes.
    """

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        use_antithetic_sampling: bool = True,
        optimizer: optax.GradientTransformation = optax.sgd(learning_rate=1e-3),
        fitness_shaping_fn: Callable = centered_rank_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize AdaOpenES."""
        assert population_size % 2 == 0, "Population size must be even."
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)
        self.optimizer = optimizer
        self.use_antithetic_sampling = use_antithetic_sampling

    @property
    def _default_params(self) -> AdaOpenESParams:
        sigma_lr_init = (3.0 + jnp.log(self.num_dims)) / (5.0 * jnp.sqrt(self.num_dims))
        return AdaOpenESParams(
            std_init=1.0,
            sigma_lr_init=sigma_lr_init,
            sigma_min=1e-2,
            sigma_max=1e2,
        )

    def _init(self, key: jax.Array, params: AdaOpenESParams) -> _OpenESState:
        return _OpenESState(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=params.std_init * jnp.ones(self.num_dims),
            opt_state=self.optimizer.init(jnp.zeros(self.num_dims)),
            best_solution=jnp.full((self.num_dims,), jnp.nan),
            best_fitness=jnp.inf,
            generation_counter=0,
        )

    def _ask(
        self, key: jax.Array, state: _OpenESState, params: AdaOpenESParams
    ) -> tuple[Population, _OpenESState]:
        if self.use_antithetic_sampling:
            z_plus = jax.random.normal(key, (self.population_size // 2, self.num_dims))
            z = jnp.concatenate([z_plus, -z_plus])
        else:
            z = jax.random.normal(key, (self.population_size, self.num_dims))
        population = state.mean + state.std * z
        return population, state

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: _OpenESState,
        params: AdaOpenESParams,
    ) -> _OpenESState:
        z = (population - state.mean) / state.std

        grad_mean = jnp.dot(fitness, z) / (self.population_size * state.std)
        updates, opt_state = self.optimizer.update(
            grad_mean, state.opt_state, state.mean
        )
        mean = optax.apply_updates(state.mean, updates)

        grad_log_std = jnp.dot(fitness, z**2 - 1.0) / self.population_size
        std = jnp.clip(
            state.std * jnp.exp(-params.sigma_lr_init * grad_log_std),
            params.sigma_min,
            params.sigma_max,
        )

        return state.replace(mean=mean, std=std, opt_state=opt_state)
