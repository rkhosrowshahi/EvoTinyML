"""Self-adaptive Differential Evolution (jDE; Brest et al., 2006).

Each population member carries its own ``F`` and ``CR``. Before generating a
trial, those control parameters are randomly reinitialized with probability
``τ`` (else kept). On successful replacement the new ``F``/``CR`` are retained,
so good parameter values proliferate.

Standard mutation/crossover is DE/rand/1/bin (or DE/best/1/bin with elitism).

[1] https://ieeexplore.ieee.org/document/1554902
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct
from evosax.algorithms.population_based.base import (
    Params as BaseParams,
    PopulationBasedAlgorithm,
    State as BaseState,
    metrics_fn,
)
from evosax.core.fitness_shaping import identity_fitness_shaping_fn
from evosax.types import Fitness, Population, Solution


@struct.dataclass
class State(BaseState):
    population: jax.Array
    fitness: jax.Array
    differential_weights: jax.Array  # (popsize,)
    crossover_rates: jax.Array  # (popsize,)
    # Proposed F/CR used for the current trial generation (written in ask).
    trial_differential_weights: jax.Array
    trial_crossover_rates: jax.Array


@struct.dataclass
class Params(BaseParams):
    elitism: bool  # base = best if True, else random (rand/1)
    f_l: float  # lower bound for F resampling
    f_u: float  # range width: F ← f_l + U(0,1)·f_u  ∈ [f_l, f_l+f_u]
    tau_f: float  # probability of resampling F
    tau_cr: float  # probability of resampling CR
    f_init: float  # initial F for all members
    cr_init: float  # initial CR for all members


class JDE(PopulationBasedAlgorithm):
    """Self-adaptive Differential Evolution (jDE)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        assert population_size >= 4, "jDE requires population_size >= 4."
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

    @property
    def _default_params(self) -> Params:
        # Brest et al. (2006) defaults.
        return Params(
            elitism=False,
            f_l=0.1,
            f_u=0.9,
            tau_f=0.1,
            tau_cr=0.1,
            f_init=0.5,
            cr_init=0.9,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        n = self.population_size
        return State(
            population=jnp.full((n, self.num_dims), jnp.nan),
            fitness=jnp.full(n, jnp.inf),
            differential_weights=jnp.full(n, params.f_init),
            crossover_rates=jnp.full(n, params.cr_init),
            trial_differential_weights=jnp.full(n, params.f_init),
            trial_crossover_rates=jnp.full(n, params.cr_init),
            best_solution=jnp.full((self.num_dims,), jnp.nan),
            best_fitness=jnp.inf,
            generation_counter=0,
        )

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        params: Params,
    ) -> State:
        """Initialize population and set archive best from the evaluated pop."""
        state = super().init(key, population, fitness, params)
        best_idx = jnp.argmin(state.fitness)
        return state.replace(
            best_solution=state.population[best_idx],
            best_fitness=state.fitness[best_idx],
            differential_weights=jnp.full(
                self.population_size, params.f_init, dtype=jnp.float32
            ),
            crossover_rates=jnp.full(
                self.population_size, params.cr_init, dtype=jnp.float32
            ),
            trial_differential_weights=jnp.full(
                self.population_size, params.f_init, dtype=jnp.float32
            ),
            trial_crossover_rates=jnp.full(
                self.population_size, params.cr_init, dtype=jnp.float32
            ),
        )

    def _ask(
        self,
        key: jax.Array,
        state: State,
        params: Params,
    ) -> tuple[Population, State]:
        keys = jax.random.split(key, self.population_size)
        member_ids = jnp.arange(self.population_size)
        best_index = jnp.argmin(state.fitness)

        def _ask_member(key, member_id, f_i, cr_i):
            x = state.population[member_id]
            (
                key_f_val,
                key_f_tau,
                key_cr_val,
                key_cr_tau,
                key_a,
                key_R,
                key_r,
                key_bc,
            ) = jax.random.split(key, 8)

            # Self-adaptive F / CR (Brest et al., 2006).
            f_new = params.f_l + jax.random.uniform(key_f_val) * params.f_u
            f = jnp.where(jax.random.uniform(key_f_tau) < params.tau_f, f_new, f_i)
            cr_new = jax.random.uniform(key_cr_val)
            cr = jnp.where(
                jax.random.uniform(key_cr_tau) < params.tau_cr, cr_new, cr_i
            )

            # Base vector a (rand or best).
            p = jnp.ones(self.population_size).at[member_id].set(0.0)
            a_index = jax.random.choice(key_a, self.population_size, p=p)
            a_index = jnp.where(params.elitism, best_index, a_index)
            a = state.population[a_index]

            # Distinct donors b, c ≠ member, a.
            p = p.at[a_index].set(0.0)
            b, c = jax.random.choice(key_bc, state.population, (2,), replace=False, p=p)
            mutant = a + f * (b - c)

            # Binomial crossover with at least one mutated dimension.
            R = jax.nn.one_hot(
                jax.random.choice(key_R, self.num_dims), self.num_dims
            )
            r = jax.random.uniform(key_r, (self.num_dims,))
            mask = jnp.logical_or(r < cr, R)
            trial = jnp.where(mask, mutant, x)
            return trial, f, cr

        trials, trial_f, trial_cr = jax.vmap(_ask_member)(
            keys,
            member_ids,
            state.differential_weights,
            state.crossover_rates,
        )
        state = state.replace(
            trial_differential_weights=trial_f,
            trial_crossover_rates=trial_cr,
        )
        return trials, state

    def _tell(
        self,
        key: jax.Array,
        population: Solution,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        # Greedy 1:1 replacement; keep adapted F/CR only when the trial wins.
        replace = fitness <= state.fitness
        population = jnp.where(replace[..., None], population, state.population)
        fitness = jnp.where(replace, fitness, state.fitness)
        differential_weights = jnp.where(
            replace, state.trial_differential_weights, state.differential_weights
        )
        crossover_rates = jnp.where(
            replace, state.trial_crossover_rates, state.crossover_rates
        )
        return state.replace(
            population=population,
            fitness=fitness,
            differential_weights=differential_weights,
            crossover_rates=crossover_rates,
        )
