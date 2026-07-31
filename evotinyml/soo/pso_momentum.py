"""Soft-replacement PSO with learning-rate momentum on anchors.

Two algorithms (see ``contexts/pso_momentum.md``):

- ``FirstMomentumPSO`` (1MPSO): first-moment EMA of gated displacements.
  Step stays proportional to ``(x - p)``.
- ``SecondMomentumPSO`` (2MPSO): Adam-style first + second moments.
  Steps are roughly ``η · sign(x - p)``, so use much smaller ``η``.

``ask`` returns ``[population; personal anchors; global anchor]``
(``2 * popsize + 1``) for shared-batch co-evaluation. ``tell`` soft-updates
anchors, then applies standard PSO velocity/position dynamics.
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
from evosax.types import Fitness, Metrics, Population, Solution


# ---------------------------------------------------------------------------
# Shared PSO params / soft-anchor base
# ---------------------------------------------------------------------------


@struct.dataclass
class SoftPSOParams(BaseParams):
    inertia_coeff: float  # w
    cognitive_coeff: float  # c1
    social_coeff: float  # c2
    v_max: float  # absolute velocity clamp
    eta_personal: float  # η_p
    eta_global: float  # η_g
    beta1: float  # first-moment decay
    gate_temperature: float  # absolute gate temperature τ (loss units)
    gate_ema_decay: float  # EMA decay for the robust loss spread (tracked; unused by τ)
    global_topk_fraction: float  # rank-weighted fraction used for global pull
    eps: float  # numerical stability


class _SoftAnchorPSO(PopulationBasedAlgorithm):
    """Shared ask/tell / PSO dynamics for soft-replacement PSO variants."""

    _algo_name: str = "SoftAnchorPSO"

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
        *,
        soft_gate: bool = True,
        use_bias_correction: bool = True,
    ):
        if population_size < 2:
            raise ValueError(
                f"{self._algo_name} requires popsize >= 2, got {population_size}"
            )
        super().__init__(
            population_size,
            solution,
            fitness_shaping_fn=fitness_shaping_fn,
            metrics_fn=metrics_fn,
        )
        self.soft_gate = bool(soft_gate)
        self.use_bias_correction = bool(use_bias_correction)

    def ask(
        self,
        key: jax.Array,
        state,
        params: SoftPSOParams,
    ) -> tuple[Population, object]:
        """Return [x; p; g] for shared-batch co-evaluation (no move yet)."""
        del key, params
        x = jax.vmap(self._unravel_solution)(state.population)
        p = jax.vmap(self._unravel_solution)(state.population_best)
        g = self._unravel_solution(state.best_solution)
        g = jnp.expand_dims(g, axis=0)
        return jnp.concatenate([x, p, g], axis=0), state

    def tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state,
        params: SoftPSOParams,
    ) -> tuple[object, Metrics]:
        """Soft-update anchors from co-batch scores, then PSO dynamics."""
        population = jax.vmap(self._ravel_solution)(population)
        n = self.population_size
        expected = 2 * n + 1
        if population.shape[0] != expected or fitness.shape[0] != expected:
            raise ValueError(
                f"{self._algo_name} expects ask size 2*{n}+1={expected}, "
                f"got population={population.shape[0]}, fitness={fitness.shape[0]}"
            )

        x = population[:n]
        p = population[n : 2 * n]
        g = population[2 * n]
        a = fitness[:n]
        q = fitness[n : 2 * n]
        qg = fitness[2 * n]

        state = self._soft_update_anchors(x, p, g, a, q, qg, state, params)
        metrics = self.metrics_fn(key, x, a, state, params)
        shaped = self.fitness_shaping_fn(x, a, state, params)
        state = self._pso_step(key, state, params)
        state = state.replace(
            fitness=shaped,
            generation_counter=state.generation_counter + 1,
        )
        return state, metrics

    def _gated_steps(
        self,
        x: jax.Array,
        p: jax.Array,
        g: jax.Array,
        a: jax.Array,
        q: jax.Array,
        qg: jax.Array,
        previous_spread: jax.Array,
        params: SoftPSOParams,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Return gated steps, best fitness, and updated robust loss spread.

        Gate temperature is an absolute τ in loss units:
        ``γ = sigmoid((q - a) / τ)``. A robust MAD spread of paired
        candidate-anchor fitness differences is still tracked (EMA) for
        diagnostics / a future dynamic-τ mode, but does not scale τ.
        """
        paired_diff = a - q
        paired_median = jnp.median(paired_diff)
        raw_spread = 1.4826 * jnp.median(jnp.abs(paired_diff - paired_median))
        spread_ema = (
            params.gate_ema_decay * previous_spread
            + (1.0 - params.gate_ema_decay) * raw_spread
        )
        spread = jnp.where(previous_spread > params.eps, spread_ema, raw_spread)
        tau = params.gate_temperature
        d_p = x - p
        if self.soft_gate:
            gate_p = jax.nn.sigmoid((q - a) / tau)
        else:
            gate_p = (a <= q).astype(x.dtype)
        s_p = gate_p[:, None] * d_p

        # Global pull: combine a rank-weighted top-k rather than discarding all
        # information except the single batch winner. Log-rank utilities are
        # robust to loss scale and emphasize the strongest elites.
        order = jnp.argsort(a)
        ranked_a = a[order]
        ranked_x = x[order]
        n = a.shape[0]
        elite_count = jnp.maximum(
            1,
            jnp.ceil(params.global_topk_fraction * n).astype(jnp.int32),
        )
        ranks = jnp.arange(n, dtype=x.dtype) + 1.0
        utilities = jnp.maximum(
            0.0,
            jnp.log(elite_count.astype(x.dtype) + 0.5) - jnp.log(ranks),
        )
        rank_weights = utilities / (jnp.sum(utilities) + params.eps)

        a_star = ranked_a[0]
        if self.soft_gate:
            gate_g = jax.nn.sigmoid((qg - ranked_a) / tau)
        else:
            gate_g = (ranked_a <= qg).astype(x.dtype)
        weighted_gates = rank_weights * gate_g
        s_g = jnp.sum(weighted_gates[:, None] * (ranked_x - g), axis=0)
        return s_p, s_g, a_star, spread

    def _soft_update_anchors(
        self,
        x: jax.Array,
        p: jax.Array,
        g: jax.Array,
        a: jax.Array,
        q: jax.Array,
        qg: jax.Array,
        state,
        params: SoftPSOParams,
    ):
        raise NotImplementedError

    def _pso_step(self, key: jax.Array, state, params: SoftPSOParams):
        """Standard PSO velocity/position update toward soft anchors."""
        best_global = state.best_solution
        v_max = params.v_max

        def _ask_member(key_i, velocity, member, member_best):
            r1, r2 = jax.random.uniform(key_i, (2,))
            inertia = params.inertia_coeff * velocity
            cognitive = params.cognitive_coeff * r1 * (member_best - member)
            social = params.social_coeff * r2 * (best_global - member)
            vp = jnp.clip(inertia + cognitive + social, -v_max, v_max)
            return member + vp, vp

        keys = jax.random.split(key, self.population_size)
        x, velocity = jax.vmap(_ask_member)(
            keys, state.velocity, state.population, state.population_best
        )
        return state.replace(population=x, velocity=velocity)


# ---------------------------------------------------------------------------
# 1MPSO — first-moment only
# ---------------------------------------------------------------------------


@struct.dataclass
class FirstMomentumState(BaseState):
    population: Population
    fitness: Fitness
    population_best: Population
    fitness_best: Fitness
    velocity: jax.Array
    m_personal: jax.Array
    m_global: jax.Array
    gate_spread: jax.Array


@struct.dataclass
class FirstMomentumParams(SoftPSOParams):
    pass


class FirstMomentumPSO(_SoftAnchorPSO):
    """1MPSO: soft anchors with first-moment (momentum) EMA only."""

    _algo_name = "FirstMomentumPSO"

    @property
    def _default_params(self) -> FirstMomentumParams:
        return FirstMomentumParams(
            inertia_coeff=0.75,
            cognitive_coeff=1.5,
            social_coeff=2.0,
            v_max=0.8,
            eta_personal=0.3,
            eta_global=0.1,
            beta1=0.9,
            gate_temperature=0.75,
            gate_ema_decay=0.9,
            global_topk_fraction=0.2,
            eps=1e-8,
        )

    def _init(self, key: jax.Array, params: FirstMomentumParams) -> FirstMomentumState:
        n, d = self.population_size, self.num_dims
        return FirstMomentumState(
            population=jnp.full((n, d), jnp.nan),
            fitness=jnp.full((n,), jnp.inf),
            population_best=jnp.full((n, d), jnp.nan),
            fitness_best=jnp.full((n,), jnp.inf),
            velocity=jnp.zeros((n, d)),
            m_personal=jnp.zeros((n, d)),
            m_global=jnp.zeros((d,)),
            gate_spread=jnp.asarray(0.0),
            best_solution=jnp.full((d,), jnp.nan),
            best_fitness=jnp.inf,
            generation_counter=0,
        )

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        params: FirstMomentumParams,
    ) -> FirstMomentumState:
        state = self._init(key, params)
        population = jax.vmap(self._ravel_solution)(population)
        best_idx = jnp.argmin(fitness)
        shaped = self.fitness_shaping_fn(population, fitness, state, params)
        return state.replace(
            population=population,
            fitness=shaped,
            best_solution=population[best_idx],
            best_fitness=fitness[best_idx],
            population_best=population,
            fitness_best=fitness,
        )

    def _soft_update_anchors(
        self,
        x: jax.Array,
        p: jax.Array,
        g: jax.Array,
        a: jax.Array,
        q: jax.Array,
        qg: jax.Array,
        state: FirstMomentumState,
        params: FirstMomentumParams,
    ) -> FirstMomentumState:
        s_p, s_g, a_star, gate_spread = self._gated_steps(
            x, p, g, a, q, qg, state.gate_spread, params
        )
        t = state.generation_counter + 1
        beta1 = params.beta1

        m_p = beta1 * state.m_personal + (1.0 - beta1) * s_p
        m_g = beta1 * state.m_global + (1.0 - beta1) * s_g
        if self.use_bias_correction:
            mhat_p = m_p / (1.0 - beta1**t)
            mhat_g = m_g / (1.0 - beta1**t)
        else:
            mhat_p, mhat_g = m_p, m_g

        return state.replace(
            population=x,
            population_best=p + params.eta_personal * mhat_p,
            fitness_best=q,
            best_solution=g + params.eta_global * mhat_g,
            best_fitness=a_star,
            m_personal=m_p,
            m_global=m_g,
            gate_spread=gate_spread,
        )


# ---------------------------------------------------------------------------
# 2MPSO — Adam-style first + second moments
# ---------------------------------------------------------------------------


@struct.dataclass
class SecondMomentumState(BaseState):
    population: Population
    fitness: Fitness
    population_best: Population
    fitness_best: Fitness
    velocity: jax.Array
    m_personal: jax.Array
    u_personal: jax.Array
    m_global: jax.Array
    u_global: jax.Array
    gate_spread: jax.Array


@struct.dataclass
class SecondMomentumParams(SoftPSOParams):
    beta2: float  # second-moment decay


class SecondMomentumPSO(_SoftAnchorPSO):
    """2MPSO: soft anchors with Adam-style first + second moments.

    Because ``m/√u ≈ sign(s)``, ``η`` is an absolute per-coordinate step.
    Defaults use small learning rates suitable for NN weights.
    """

    _algo_name = "SecondMomentumPSO"

    @property
    def _default_params(self) -> SecondMomentumParams:
        return SecondMomentumParams(
            inertia_coeff=0.75,
            cognitive_coeff=1.5,
            social_coeff=2.0,
            v_max=0.8,
            # Small: Adam normalizes displacements to ~unit steps.
            eta_personal=1e-3,
            eta_global=1e-3,
            beta1=0.9,
            beta2=0.999,
            gate_temperature=0.75,
            gate_ema_decay=0.9,
            global_topk_fraction=0.2,
            eps=1e-8,
        )

    def _init(
        self, key: jax.Array, params: SecondMomentumParams
    ) -> SecondMomentumState:
        n, d = self.population_size, self.num_dims
        return SecondMomentumState(
            population=jnp.full((n, d), jnp.nan),
            fitness=jnp.full((n,), jnp.inf),
            population_best=jnp.full((n, d), jnp.nan),
            fitness_best=jnp.full((n,), jnp.inf),
            velocity=jnp.zeros((n, d)),
            m_personal=jnp.zeros((n, d)),
            u_personal=jnp.zeros((n, d)),
            m_global=jnp.zeros((d,)),
            u_global=jnp.zeros((d,)),
            gate_spread=jnp.asarray(0.0),
            best_solution=jnp.full((d,), jnp.nan),
            best_fitness=jnp.inf,
            generation_counter=0,
        )

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        params: SecondMomentumParams,
    ) -> SecondMomentumState:
        state = self._init(key, params)
        population = jax.vmap(self._ravel_solution)(population)
        best_idx = jnp.argmin(fitness)
        shaped = self.fitness_shaping_fn(population, fitness, state, params)
        return state.replace(
            population=population,
            fitness=shaped,
            best_solution=population[best_idx],
            best_fitness=fitness[best_idx],
            population_best=population,
            fitness_best=fitness,
        )

    def _soft_update_anchors(
        self,
        x: jax.Array,
        p: jax.Array,
        g: jax.Array,
        a: jax.Array,
        q: jax.Array,
        qg: jax.Array,
        state: SecondMomentumState,
        params: SecondMomentumParams,
    ) -> SecondMomentumState:
        s_p, s_g, a_star, gate_spread = self._gated_steps(
            x, p, g, a, q, qg, state.gate_spread, params
        )
        t = state.generation_counter + 1
        beta1, beta2, eps = params.beta1, params.beta2, params.eps

        m_p = beta1 * state.m_personal + (1.0 - beta1) * s_p
        m_g = beta1 * state.m_global + (1.0 - beta1) * s_g
        u_p = beta2 * state.u_personal + (1.0 - beta2) * jnp.square(s_p)
        u_g = beta2 * state.u_global + (1.0 - beta2) * jnp.square(s_g)

        if self.use_bias_correction:
            mhat_p = m_p / (1.0 - beta1**t)
            mhat_g = m_g / (1.0 - beta1**t)
            uhat_p = u_p / (1.0 - beta2**t)
            uhat_g = u_g / (1.0 - beta2**t)
        else:
            mhat_p, mhat_g, uhat_p, uhat_g = m_p, m_g, u_p, u_g

        step_p = mhat_p / (jnp.sqrt(uhat_p) + eps)
        step_g = mhat_g / (jnp.sqrt(uhat_g) + eps)

        return state.replace(
            population=x,
            population_best=p + params.eta_personal * step_p,
            fitness_best=q,
            best_solution=g + params.eta_global * step_g,
            best_fitness=a_star,
            m_personal=m_p,
            u_personal=u_p,
            m_global=m_g,
            u_global=u_g,
            gate_spread=gate_spread,
        )
