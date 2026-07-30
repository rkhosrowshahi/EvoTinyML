"""PSO with init seeding fix (evosax PR #109) and velocity clamping.

Upstream evosax 0.2.0 ``PSO`` leaves ``population_best`` as NaN and
``fitness_best`` as ``+∞`` after ``init``. The first ``ask`` then always picks
particle 0 as gbest (``argmin`` over all ``+∞``), and the first ``tell``
overwrites personal bests with post-move positions — discarding the evaluated
initial population (including a true optimum that is not particle 0).

This subclass applies https://github.com/RobertTLange/evosax/pull/109:
seed archive + personal bests from the initial eval, and drop the broken
NaN fitness fallback in ``_ask``.

Additionally clamps velocity to ``[-v_max, v_max]`` and uses per-dimension
``r1``/``r2``.

For mini-batch-robust bookkeeping see :mod:`evotinyml.soo.pso_robust`.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct
from evosax.algorithms.population_based.base import (
    Params as BaseParams,
    metrics_fn,
)
from evosax.algorithms.population_based.pso import PSO as _PSO
from evosax.core.fitness_shaping import identity_fitness_shaping_fn
from evosax.types import Fitness, Population, Solution


@struct.dataclass
class Params(BaseParams):
    inertia_coeff: float  # w
    cognitive_coeff: float  # c1
    social_coeff: float  # c2
    v_max: float  # absolute velocity clamp


class FixedPSO(_PSO):
    """PSO with correct best seeding and velocity clamping."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        super().__init__(
            population_size,
            solution,
            fitness_shaping_fn=fitness_shaping_fn,
            metrics_fn=metrics_fn,
        )

    @property
    def _default_params(self) -> Params:
        return Params(
            inertia_coeff=0.75,
            cognitive_coeff=1.5,
            social_coeff=2.0,
            v_max=0.8,
        )

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        params: Params,
    ):
        """Initialize PSO, seeding archive and personal bests from init eval."""
        state = self._init(key, params)

        # Ravel population
        population = jax.vmap(self._ravel_solution)(population)

        # Seed archive from the evaluated initial population (raw fitness).
        # Mirrors PopulationBasedAlgorithm.init from evosax PR #109.
        best_idx = jnp.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        # Shape fitness
        fitness = self.fitness_shaping_fn(population, fitness, state, params)

        return state.replace(
            population=population,
            fitness=fitness,
            best_solution=best_solution,
            best_fitness=best_fitness,
            population_best=population,
            fitness_best=fitness,
        )

    def _ask(
        self,
        key: jax.Array,
        state,
        params: Params,
    ) -> tuple[Population, object]:
        # Global best from personal bests (seeded at init).
        best_global_idx = jnp.argmin(state.fitness_best)
        best_global = state.population_best[best_global_idx]
        v_max = params.v_max

        def _ask_member(key, velocity, member, member_best):
            r1, r2 = jax.random.uniform(key, (2,))
            inertia = params.inertia_coeff * velocity
            cognitive = params.cognitive_coeff * r1 * (member_best - member)
            social = params.social_coeff * r2 * (best_global - member)
            vp = inertia + cognitive + social
            vp = jnp.clip(vp, -v_max, v_max)
            xp = member + vp
            return xp, vp

        keys = jax.random.split(key, self.population_size)
        x, velocity = jax.vmap(_ask_member)(
            keys, state.velocity, state.population, state.population_best
        )
        return x, state.replace(velocity=velocity)
