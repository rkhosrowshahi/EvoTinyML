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

Optional ``ea_coeval``: ``ask`` returns ``[offspring; population_best]``
(``2 * popsize``) so offspring and the personal-best archive are scored on the
same minibatch; ``tell`` then replaces pbest/gbest using that co-batch fitness.
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
from evosax.types import Fitness, Metrics, Population, Solution


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
        *,
        ea_coeval: bool = False,
    ):
        super().__init__(
            population_size,
            solution,
            fitness_shaping_fn=fitness_shaping_fn,
            metrics_fn=metrics_fn,
        )
        self.ea_coeval = bool(ea_coeval)

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

    def ask(
        self,
        key: jax.Array,
        state,
        params: Params,
    ) -> tuple[Population, object]:
        """Ask for candidates; optionally append personal-best archive."""
        population, state = self._ask(key, state, params)
        population = jax.vmap(self._unravel_solution)(population)
        if self.ea_coeval:
            incumbents = jax.vmap(self._unravel_solution)(state.population_best)
            population = jnp.concatenate([population, incumbents], axis=0)
        return population, state

    def tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state,
        params: Params,
    ) -> tuple[object, Metrics]:
        """Tell fitness; with ea_coeval, replace pbest using same-batch scores."""
        if not self.ea_coeval:
            return super().tell(key, population, fitness, state, params)

        population = jax.vmap(self._ravel_solution)(population)
        n = self.population_size
        if population.shape[0] != 2 * n or fitness.shape[0] != 2 * n:
            raise ValueError(
                f"ea_coeval expects ask size 2*{n}="
                f"{2 * n}, got population={population.shape[0]}, "
                f"fitness={fitness.shape[0]}"
            )

        x = population[:n]
        fit_x = fitness[:n]
        fit_p = fitness[n:]

        # Same-batch pbest replacement (offspring vs re-evaluated incumbents).
        replace = fit_x <= fit_p
        population_best = jnp.where(replace[..., None], x, state.population_best)
        fitness_best = jnp.where(replace, fit_x, fit_p)

        best_idx = jnp.argmin(fitness_best)
        best_solution = population_best[best_idx]
        best_fitness = fitness_best[best_idx]
        state = state.replace(
            best_solution=best_solution,
            best_fitness=best_fitness,
        )

        metrics = self.metrics_fn(key, x, fit_x, state, params)
        shaped = self.fitness_shaping_fn(x, fit_x, state, params)
        state = state.replace(
            population=x,
            fitness=shaped,
            population_best=population_best,
            fitness_best=fitness_best,
            generation_counter=state.generation_counter + 1,
        )
        return state, metrics

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
            # Per-dimension random coefficients (classic Kennedy & Eberhart).
            # key_r1, key_r2 = jax.random.split(key)
            # r1 = jax.random.uniform(key_r1, (self.num_dims,))
            # r2 = jax.random.uniform(key_r2, (self.num_dims,))
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
