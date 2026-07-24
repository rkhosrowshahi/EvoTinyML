"""Mean-update ES variants that pass ``state.mean`` into Optax ``update``.

Upstream OpenES / SNES / xNES / ASEBO call ``optimizer.update(grad, opt_state)``
without ``params``. Optax transforms that need the current parameters
(``adamw``, ``add_decayed_weights`` / ``--es-optim-wd``) then fail. These
subclasses keep upstream ask/tell math and only fix the Optax call site.

Also defines ``SparseOpenES``: OpenES with per-dimension noise masking before
antithetic sampling.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from evosax.algorithms.distribution_based.open_es import Open_ES as _Open_ES
from evosax.algorithms.distribution_based.snes import SNES as _SNES
from evosax.algorithms.distribution_based.xnes import xNES as _xNES
from evosax.core.fitness_shaping import centered_rank_fitness_shaping_fn
from evosax.types import Fitness, Population, Solution

from evosax.algorithms.distribution_based.base import metrics_fn


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
