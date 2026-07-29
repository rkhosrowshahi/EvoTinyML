"""Filtered-Attractor PSO (FA-PSO): PSO for noisy mini-batch objectives.

Canonical PSO commits information to state by *hard replacement*: ``pbest`` /
``gbest`` are overwritten iff a newly measured fitness beats a value archived in
an earlier generation. Under a mini-batch objective
``f̂(x; B) = mean_{i∈B} ℓ(x, z_i)`` that comparison is invalid — the two sides
are measured on different batches — and it is also monotone, so the archive can
only ratchet downward. The stored value therefore drifts below the true risk
(the optimistic bias of an extreme order statistic, i.e. the winner's curse),
the acceptance test becomes unsatisfiable, and the swarm stagnates around a
solution that was lucky rather than good. ``gbest`` is the worst offender: an
``argmin`` over ``λ`` noisy values is the most bias-prone statistic available.

FA-PSO removes the archives. Both attractors become *filtered estimators*:

1. Resample one eval pool per generation, shared by the whole swarm (CRN) —
   done by the caller, see ``WeightOptimizationProblem.sample_eval_pool``.
2. Evaluate every particle on that pool.
3. Rank *within* the generation. Fitness values are never stored across
   generations, so there is no stale value to compare against.
4. Social attractor: rank-weighted centroid of the top ``μ`` particles
   (CMA-ES / OpenES recombination weights), then an EMA across generations::

       ĝ = Σ_k w_k · x_(k)          w_k ∝ log(μ + 1/2) − log k
       g ← (1 − α)·g + α·ĝ

5. Personal attractor: Polyak average of the particle's own positions, gated on
   intra-generation rank rather than on an archived value::

       p_i ← (1 − β)·p_i + β·x_i    if rank(i) ≤ λ/2, else unchanged

6. Move with the usual velocity update, or (``use_adam=True``, **not
   recommended** — see below) treat the attractor pull as a gradient surrogate
   and apply Adam per particle, then add isotropic sampling noise ``σ_t · ε_i``,
   ``ε_i ~ N(0, I)``.

Selection bias is spread over ``μ`` particles instead of concentrating in one
lucky draw, and what survives is damped by ``α`` / ``β`` — the same role the
learning rate and moment estimates play in OpenES.

Why step 6 needs the sampling term
----------------------------------
Without it the swarm collapses — and it does so precisely in the noise-dominated
regime this algorithm was built for.

Every term of the PSO velocity update is proportional to a *difference between
existing points* (``p_i − x_i``, ``g − x_i``); there is no additive source of new
positions. Canonical PSO gets away with that because ``pbest`` / ``gbest`` are
frozen archived points that stay put while particles move, keeping those
differences nonzero. Filtering the attractors removes those anchors: the
rank-weighted centroid lies strictly inside the convex hull of the swarm, and
the personal EMA drags ``p_i`` toward the position the particle already occupies.

Whether that is fatal depends on the signal-to-noise ratio of the ranking. When
fitness differences exceed the evaluation noise, ``ĝ`` is displaced toward
genuinely better particles and the social term *transports* the swarm. When
batch noise dominates the ranking, the ordering is close to random, so
``E[ĝ] = mean(x)`` — the recombination degenerates to the plain swarm centroid,
the social term becomes a restoring force toward the mean carrying no
directional information, and the swarm contracts onto its own centroid.

Measured on a 64-d sphere over 300 generations, sweeping additive fitness noise
against a signal of O(100), final swarm diversity was::

    noise      0      10     1e3      1e5
    sigma=0    4.50   2.63   3.7e-5   3.9e-5     <- collapse once noise dominates
    sigma=0.1  4.60   2.56   1.06     1.06

and on MNIST (tinycnn_mnist_4k, d=4266, popsize 16, 32k FEs), where every
particle sat at CE ~= 2.29 so fitness gaps were far below batch noise, diversity
fell 2.98 -> 2e-6 by generation ~1000, mean step norm fell 6.0 -> 4e-6, and
cross-entropy froze at 2.285 against a chance level of ln(10) = 2.303.

The ``use_adam`` variant does not work (kept for ablation only)
--------------------------------------------------------------
Adam normalizes per coordinate, so the step magnitude is ~``lr`` regardless of
how far the particle is from the attractor. In SGD that is fine because a true
gradient grows as you leave the optimum. Here the "gradient" is a direction
toward ``g`` whose *magnitude* is exactly the restoring force that keeps a swarm
bounded — normalizing it away leaves nothing to counteract the per-generation
sampling noise, so the swarm random-walks outward. Measured on MNIST under the
same budget as above: diversity grew 8.1 -> 25.9 and cross-entropy stayed at
2.33 (chance), against 0.41 for the default velocity update. Use the velocity
update; if per-coordinate scaling is wanted, it must preserve ``||g − x||``.

CMA-ES and OpenES use the same recombination centroid without this failure
because they pair it with fresh Gaussian sampling at scale ``σ`` every
generation: the centroid contracts, the sampling regenerates. ``σ_t`` here comes
from the same ``--es-sigma-*`` schedule OpenES uses, so a constant schedule puts
a hard floor under the swarm spread. Setting it to 0 reproduces the collapse and
is only useful as an ablation.

Note on ``best_solution`` / ``best_fitness``: the evosax base ``tell`` maintains
these as a greedy cross-generation archive, i.e. exactly the ratchet described
above. They are kept for API compatibility and metrics only, and must **not** be
used as the reported solution — ``es._state_solution`` returns ``state.social``
for this algorithm instead.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
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
    population: Population  # x_i
    fitness: Fitness  # current generation only; never compared across generations
    personal: Population  # p_i, filtered personal attractors
    social: jax.Array  # g, filtered social attractor
    velocity: jax.Array
    std: float  # sigma_t, isotropic sampling scale (see module docstring)
    opt_m: jax.Array  # Adam first moment (zeros when use_adam=False)
    opt_v: jax.Array  # Adam second moment (zeros when use_adam=False)


@struct.dataclass
class Params(BaseParams):
    inertia_coeff: float  # w   (ignored when use_adam)
    cognitive_coeff: float  # c1
    social_coeff: float  # c2
    v_max: float  # velocity clamp (ignored when use_adam)
    social_decay: float  # alpha, social attractor EMA rate
    personal_decay: float  # beta, personal attractor EMA rate
    lr: float  # eta (Adam only)
    beta_1: float  # Adam only
    beta_2: float  # Adam only
    eps: float  # Adam only


def rank_weights(num_elites: int) -> jax.Array:
    """Positive, decreasing recombination weights summing to 1."""
    k = jnp.arange(1, num_elites + 1, dtype=jnp.float32)
    w = jnp.log(num_elites + 0.5) - jnp.log(k)
    return w / jnp.sum(w)


class FilteredPSO(PopulationBasedAlgorithm):
    """PSO whose attractors are filtered estimators instead of greedy archives."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        elite_ratio: float = 0.5,
        use_adam: bool = False,
        # Default matches this repo's --init-sigma; the runner always overrides
        # it with the --es-sigma-* schedule. A 0.0 schedule disables sampling
        # and collapses the swarm (see module docstring), so never default it.
        std_schedule: Callable = optax.constant_schedule(0.1),
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        super().__init__(
            population_size,
            solution,
            fitness_shaping_fn=fitness_shaping_fn,
            metrics_fn=metrics_fn,
        )
        if population_size < 2:
            raise ValueError(f"FA-PSO requires popsize >= 2, got {population_size}")
        if not 0.0 < elite_ratio <= 1.0:
            raise ValueError(f"elite_ratio must be in (0, 1], got {elite_ratio}")
        # mu = num_elites: truncation for the social centroid. The evosax base
        # derives num_elites from elite_ratio as a read-only property.
        self.elite_ratio = float(elite_ratio)
        # Gate for the personal EMA: better half of the current generation.
        self.num_gated = max(1, population_size // 2)
        self.use_adam = bool(use_adam)
        self.std_schedule = std_schedule
        self.weights = rank_weights(self.num_elites)

    @property
    def _default_params(self) -> Params:
        return Params(
            inertia_coeff=0.75,
            cognitive_coeff=1.5,
            social_coeff=2.0,
            v_max=0.2,
            social_decay=0.2,
            personal_decay=0.3,
            lr=0.01,
            beta_1=0.9,
            beta_2=0.999,
            eps=1e-8,
        )

    def _recombine(self, population: Population, fitness: Fitness) -> jax.Array:
        """Rank-weighted centroid of the top-mu particles of this generation."""
        elite_idx = jnp.argsort(fitness)[: self.num_elites]
        return jnp.sum(self.weights[:, None] * population[elite_idx], axis=0)

    def _init(self, key: jax.Array, params: Params) -> State:
        return State(
            population=jnp.full((self.population_size, self.num_dims), jnp.nan),
            fitness=jnp.full((self.population_size,), jnp.inf),
            personal=jnp.full((self.population_size, self.num_dims), jnp.nan),
            social=jnp.zeros((self.num_dims,)),
            velocity=jnp.zeros((self.population_size, self.num_dims)),
            std=self.std_schedule(0),
            opt_m=jnp.zeros((self.population_size, self.num_dims)),
            opt_v=jnp.zeros((self.population_size, self.num_dims)),
            best_solution=jnp.full((self.num_dims,), jnp.nan),
            best_fitness=jnp.inf,
            generation_counter=0,
        )

    def init(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        params: Params,
    ) -> State:
        """Seed both attractors from the evaluated initial population."""
        state = self._init(key, params)
        population = jax.vmap(self._ravel_solution)(population)

        best_idx = jnp.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        fitness = self.fitness_shaping_fn(population, fitness, state, params)

        return state.replace(
            population=population,
            fitness=fitness,
            # Each particle starts as its own personal attractor.
            personal=population,
            social=self._recombine(population, fitness),
            best_solution=best_solution,
            best_fitness=best_fitness,
        )

    def _ask(
        self,
        key: jax.Array,
        state: State,
        params: Params,
    ) -> tuple[Population, State]:
        social = state.social
        # generation_counter is a Python int before the first tell, a tracer after.
        t = jnp.asarray(state.generation_counter, dtype=jnp.float32) + 1.0

        def _move(key, x, personal, velocity, opt_m, opt_v):
            key_r1, key_r2, key_eps = jax.random.split(key, 3)
            r1 = jax.random.uniform(key_r1, (self.num_dims,))
            r2 = jax.random.uniform(key_r2, (self.num_dims,))
            direction = params.cognitive_coeff * r1 * (personal - x) + (
                params.social_coeff * r2 * (social - x)
            )
            # Regenerates the diversity that the attractor pulls consume; this
            # is what keeps the swarm from contracting to a point.
            noise = state.std * jax.random.normal(key_eps, (self.num_dims,))

            if self.use_adam:
                # Attractor pull as a descent direction; beta_1 replaces inertia
                # and per-coordinate scaling replaces the v_max clamp.
                grad = -direction
                opt_m = params.beta_1 * opt_m + (1.0 - params.beta_1) * grad
                opt_v = params.beta_2 * opt_v + (1.0 - params.beta_2) * jnp.square(grad)
                m_hat = opt_m / (1.0 - params.beta_1**t)
                v_hat = opt_v / (1.0 - params.beta_2**t)
                step = -params.lr * m_hat / (jnp.sqrt(v_hat) + params.eps)
                return x + step + noise, step, opt_m, opt_v

            velocity = params.inertia_coeff * velocity + direction
            velocity = jnp.clip(velocity, -params.v_max, params.v_max)
            # Noise is added to the position, not folded into the velocity, so
            # inertia does not accumulate it into a random walk.
            return x + velocity + noise, velocity, opt_m, opt_v

        keys = jax.random.split(key, self.population_size)
        x, velocity, opt_m, opt_v = jax.vmap(_move)(
            keys,
            state.population,
            state.personal,
            state.velocity,
            state.opt_m,
            state.opt_v,
        )
        return x, state.replace(velocity=velocity, opt_m=opt_m, opt_v=opt_v)

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        # Social: rank-weighted centroid of this generation, then EMA.
        alpha = params.social_decay
        social = (1.0 - alpha) * state.social + alpha * self._recombine(
            population, fitness
        )

        # Personal: Polyak average gated on intra-generation rank. No archived
        # fitness is consulted, so nothing can go stale.
        rank_of = jnp.argsort(jnp.argsort(fitness))
        gate = rank_of < self.num_gated
        beta = params.personal_decay
        personal = jnp.where(
            gate[:, None],
            (1.0 - beta) * state.personal + beta * population,
            state.personal,
        )

        return state.replace(
            population=population,
            fitness=fitness,
            personal=personal,
            social=social,
            std=self.std_schedule(state.generation_counter),
        )
