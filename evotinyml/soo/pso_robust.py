"""Robust PSO under mini-batch fitness noise.

Standalone algorithm (``--algo pso_robust``), separate from canonical
:class:`~evotinyml.soo.pso_fixed.FixedPSO`.

Bookkeeping (NumPy):
  A. Paired co-batch differences (common random numbers)
  B. Confident replacement (paired t-test / Wilcoxon)
  C. EMA-filtered incumbent values (steady-state Kalman)
  D. Soft annealed acceptance (optional)
  E. Uncertainty-aware gbest (LCB / Thompson)

JAX swarm (``RobustPSO``): init seeding, LCB/Thompson social attractor,
optional η-damped / Adam-normalized move, and ``robust_tell`` to apply
precomputed replacements without hard cross-batch comparisons.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from evosax.algorithms.population_based.base import (
    Params as BaseParams,
    metrics_fn,
)
from evosax.algorithms.population_based.pso import PSO as _PSO
from evosax.algorithms.population_based.pso import State as _PSOState
from evosax.core.fitness_shaping import identity_fitness_shaping_fn
from evosax.types import Fitness, Metrics, Population, Solution

ReevalMode = Literal["full", "g_only", "gated", "stochastic"]
GbestMode = Literal["lcb", "thompson", "argmin"]
TestMode = Literal["ttest", "wilcoxon"]

# gbest selection codes (static-friendly for JIT).
GBEST_ARGMIN = 0
GBEST_LCB = 1
GBEST_THOMPSON = 2


@dataclass(frozen=True)
class RobustPSOConfig:
    """Knobs for robust PSO selection / replacement."""

    enabled: bool = True
    ema_alpha: float = 0.3
    kappa: float = 1.25
    lcb_beta: float = 1.0
    gbest_mode: GbestMode = "lcb"
    soft_accept: bool = False
    T0: float = 1.0
    T_gamma: float = 0.99
    test: TestMode = "ttest"
    reeval_mode: ReevalMode = "full"
    gate_band_kappa: float = 1.0
    reeval_prob: float = 0.5
    # Optional ES-flavored damped move (applied in RobustPSO._ask).
    damp_eta0: float = 1.0
    damp_eta_gamma: float = 1.0
    damp_adam: bool = False
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8


@dataclass
class RobustBookkeepingResult:
    """Updated PSO memory after one generation of robust bookkeeping."""

    population_best: np.ndarray
    fitness_best: np.ndarray
    fitness_se: np.ndarray
    best_solution: np.ndarray
    best_fitness: float
    best_se: float
    n_pbest_replaced: int
    n_reeval: int
    gbest_changed: bool


def paired_diff_stats(
    cand_losses: np.ndarray, incumbent_losses: np.ndarray
) -> tuple[float, float, float]:
    """Paired gap stats: ``(d̄, SE, z)`` for ``d_j = ℓ(x) − ℓ(p)``."""
    d = np.asarray(cand_losses, dtype=np.float64).ravel() - np.asarray(
        incumbent_losses, dtype=np.float64
    ).ravel()
    m = int(d.size)
    if m < 2:
        d_bar = float(d.mean()) if m else 0.0
        return d_bar, 0.0, 0.0
    d_bar = float(d.mean())
    s = float(d.std(ddof=1))
    se = s / np.sqrt(m)
    z = d_bar / se if se > 0.0 else (0.0 if d_bar == 0.0 else np.sign(d_bar) * np.inf)
    return d_bar, float(se), float(z)


def sample_se(losses: np.ndarray) -> float:
    """Standard error of the mean of per-sample losses."""
    x = np.asarray(losses, dtype=np.float64).ravel()
    m = int(x.size)
    if m < 2:
        return 0.0
    return float(x.std(ddof=1) / np.sqrt(m))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks (1..n) with tie averaging; ``values`` already sorted order."""
    n = values.size
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and values[j] == values[i]:
            j += 1
        # ranks i..j-1 share the average of (i+1)..j
        avg = 0.5 * ((i + 1) + j)
        ranks[i:j] = avg
        i = j
    return ranks


def wilcoxon_signed_rank_z(d: np.ndarray) -> float:
    """Normal approx z for Wilcoxon signed-rank; negative ⇒ candidate better.

    Uses W− (sum of ranks where ``d < 0``). Under H0, E[W−]=n(n+1)/4.
    Returns ``−(W− − mean) / sd`` so the interface matches the paired t z-score
    (accept when ``z < −κ``).
    """
    d = np.asarray(d, dtype=np.float64).ravel()
    nonzero = d[d != 0.0]
    n = int(nonzero.size)
    if n < 2:
        return 0.0
    abs_d = np.abs(nonzero)
    order = np.argsort(abs_d, kind="mergesort")
    sorted_abs = abs_d[order]
    ranks_sorted = _average_ranks(sorted_abs)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    w_minus = float(ranks[nonzero < 0.0].sum())
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    # Tie correction for variance.
    _, counts = np.unique(sorted_abs, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (counts + 1))
    var_w -= tie_term / 48.0
    if var_w <= 0.0:
        return 0.0
    # Continuity correction toward the null.
    z_w = (w_minus - mean_w - 0.5 * np.sign(w_minus - mean_w)) / np.sqrt(var_w)
    return float(-z_w)


def confident_accept(
    cand_losses: np.ndarray,
    incumbent_losses: np.ndarray,
    *,
    kappa: float,
    test: TestMode = "ttest",
    soft_accept: bool = False,
    temperature: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[bool, float, float, float]:
    """Step B / D: evidence-based replacement decision.

    Returns ``(accept, d_bar, se, z)``.
    """
    d = np.asarray(cand_losses, dtype=np.float64).ravel() - np.asarray(
        incumbent_losses, dtype=np.float64
    ).ravel()
    d_bar, se, z_t = paired_diff_stats(cand_losses, incumbent_losses)
    if test == "wilcoxon":
        z = wilcoxon_signed_rank_z(d)
    else:
        z = z_t

    if soft_accept:
        t = max(float(temperature), 1e-8)
        # σ(−z / T): high T ⇒ generous; as T→0 recovers a hard threshold at 0.
        prob = 1.0 / (1.0 + np.exp(z / t))
        if rng is None:
            rng = np.random.default_rng()
        accept = bool(rng.random() < prob)
    else:
        accept = bool(d_bar < -float(kappa) * se) if se > 0.0 else bool(d_bar < 0.0)
    return accept, d_bar, se, z


def ema_update(phi: float, observation: float, alpha: float) -> float:
    """Step C: EMA / steady-state Kalman refresh of the stored incumbent value."""
    a = float(np.clip(alpha, 0.0, 1.0))
    if not np.isfinite(phi):
        return float(observation)
    return float((1.0 - a) * phi + a * observation)


def select_gbest_index(
    fitness_best: np.ndarray,
    fitness_se: np.ndarray,
    *,
    mode: GbestMode = "lcb",
    beta: float = 1.0,
    rng: np.random.Generator | None = None,
) -> int:
    """Step E: uncertainty-aware attractor selection."""
    phi = np.asarray(fitness_best, dtype=np.float64).ravel()
    se = np.asarray(fitness_se, dtype=np.float64).ravel()
    if phi.size == 0:
        raise ValueError("empty fitness_best")
    if mode == "thompson":
        if rng is None:
            rng = np.random.default_rng()
        se_safe = np.maximum(se, 0.0)
        samples = rng.normal(loc=phi, scale=se_safe)
        return int(np.argmin(samples))
    if mode == "lcb":
        return int(np.argmin(phi + float(beta) * se))
    return int(np.argmin(phi))


def temperature_at(gen: int, T0: float, gamma: float) -> float:
    return float(T0) * (float(gamma) ** max(int(gen), 0))


def eta_at(gen: int, eta0: float, gamma: float) -> float:
    return float(eta0) * (float(gamma) ** max(int(gen), 0))


def choose_reeval_mask(
    cand_fitness: np.ndarray,
    fitness_best: np.ndarray,
    cand_se: np.ndarray,
    *,
    mode: ReevalMode,
    gate_band_kappa: float,
    reeval_prob: float,
    gbest_idx: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Which personal bests need a co-batch re-evaluation this generation."""
    n = int(cand_fitness.size)
    mask = np.zeros(n, dtype=bool)
    if mode == "full":
        mask[:] = True
        return mask
    if mode == "g_only":
        # No pbest re-eval; g is handled separately.
        return mask
    if mode == "stochastic":
        mask[:] = rng.random(n) < float(reeval_prob)
        return mask
    # gated: re-eval near-misses relative to filtered φ.
    band = float(gate_band_kappa) * np.maximum(cand_se, 0.0)
    delta = cand_fitness - fitness_best
    mask[:] = np.abs(delta) <= band
    return mask


def apply_robust_bookkeeping(
    *,
    population: np.ndarray,
    cand_losses: np.ndarray,
    population_best: np.ndarray,
    fitness_best: np.ndarray,
    fitness_se: np.ndarray,
    best_solution: np.ndarray,
    best_fitness: float,
    best_se: float,
    config: RobustPSOConfig,
    gen: int,
    rng: np.random.Generator,
    pbest_losses: np.ndarray | None = None,
    g_losses: np.ndarray | None = None,
    reeval_mask: np.ndarray | None = None,
) -> RobustBookkeepingResult:
    """Update pbest / gbest with paired confidence tests and EMA filtering.

    Parameters
    ----------
    cand_losses
        Shape ``(n, m)`` per-sample losses for current positions on ``B_t``.
    pbest_losses
        Shape ``(n, m)`` when re-evaluated; rows may be unused if mask is False.
    g_losses
        Shape ``(m,)`` re-evaluation of the current archive / social gbest.
    """
    pop = np.asarray(population, dtype=np.float64)
    cand = np.asarray(cand_losses, dtype=np.float64)
    pbest = np.asarray(population_best, dtype=np.float64).copy()
    phi = np.asarray(fitness_best, dtype=np.float64).copy()
    se = np.asarray(fitness_se, dtype=np.float64).copy()
    n = pop.shape[0]
    if cand.ndim != 2 or cand.shape[0] != n:
        raise ValueError(
            f"cand_losses must have shape (n, m)=({n}, m), got {cand.shape}"
        )
    cand_mean = cand.mean(axis=1)
    cand_se = np.array([sample_se(cand[i]) for i in range(n)], dtype=np.float64)

    if reeval_mask is None:
        reeval_mask = choose_reeval_mask(
            cand_mean,
            phi,
            cand_se,
            mode=config.reeval_mode,
            gate_band_kappa=config.gate_band_kappa,
            reeval_prob=config.reeval_prob,
            gbest_idx=int(np.argmin(phi)),
            rng=rng,
        )
    else:
        reeval_mask = np.asarray(reeval_mask, dtype=bool).ravel()
        if reeval_mask.size != n:
            raise ValueError("reeval_mask length must equal population size")

    T = temperature_at(gen, config.T0, config.T_gamma)
    n_replaced = 0
    n_reeval = int(reeval_mask.sum())

    for i in range(n):
        f_x = float(cand_mean[i])
        if reeval_mask[i]:
            if pbest_losses is None:
                raise ValueError("pbest_losses required when reeval_mask is set")
            p_loss = np.asarray(pbest_losses[i], dtype=np.float64).ravel()
            f_p = float(p_loss.mean())
            # Step C: always refresh incumbent memory on co-batch re-eval.
            phi[i] = ema_update(float(phi[i]), f_p, config.ema_alpha)
            se[i] = sample_se(p_loss)
            accept, _, _, _ = confident_accept(
                cand[i],
                p_loss,
                kappa=config.kappa,
                test=config.test,
                soft_accept=config.soft_accept,
                temperature=T,
                rng=rng,
            )
            if accept:
                pbest[i] = pop[i]
                phi[i] = f_x
                se[i] = float(cand_se[i])
                n_replaced += 1
            continue

        # No incumbent re-eval: gated clear-win / clear-loss using candidate SE.
        band = float(config.gate_band_kappa) * float(cand_se[i])
        if config.reeval_mode in {"gated", "g_only", "stochastic"} and f_x < float(
            phi[i]
        ) - max(band, float(config.kappa) * float(cand_se[i])):
            # Clear winner without paired re-eval.
            pbest[i] = pop[i]
            phi[i] = f_x
            se[i] = float(cand_se[i])
            n_replaced += 1
        # Clear losers and ambiguous cases without re-eval: keep incumbent.

    # Step E: refresh archive g on B_t, then promote via LCB / Thompson + paired test.
    g_pos = np.asarray(best_solution, dtype=np.float64).copy()
    phi_g = float(best_fitness)
    se_g = float(best_se)
    if g_losses is not None:
        g_loss = np.asarray(g_losses, dtype=np.float64).ravel()
        phi_g = ema_update(phi_g, float(g_loss.mean()), config.ema_alpha)
        se_g = sample_se(g_loss)
        n_reeval += 1

    g_cand = select_gbest_index(
        phi, se, mode=config.gbest_mode, beta=config.lcb_beta, rng=rng
    )
    gbest_changed = False
    # Promote only with a co-batch confidence test against current g.
    if g_losses is not None and not np.allclose(pbest[g_cand], g_pos):
        # Need challenger losses on B_t. Prefer re-evaluated pbest row; else
        # if g_cand was just replaced this gen, use cand losses; else skip.
        if reeval_mask[g_cand] and pbest_losses is not None:
            chall_loss = np.asarray(pbest_losses[g_cand], dtype=np.float64).ravel()
            # If we just replaced, challenger is the candidate.
            if np.allclose(pbest[g_cand], pop[g_cand]):
                chall_loss = cand[g_cand]
        elif np.allclose(pbest[g_cand], pop[g_cand]):
            chall_loss = cand[g_cand]
        else:
            chall_loss = None

        if chall_loss is not None:
            accept_g, _, _, _ = confident_accept(
                chall_loss,
                g_losses,
                kappa=config.kappa,
                test=config.test,
                soft_accept=config.soft_accept,
                temperature=T,
                rng=rng,
            )
            if accept_g:
                g_pos = pbest[g_cand].copy()
                phi_g = float(phi[g_cand])
                se_g = float(se[g_cand])
                gbest_changed = True
    elif g_losses is None:
        # No g re-eval: fall back to LCB among filtered pbests.
        g_pos = pbest[g_cand].copy()
        phi_g = float(phi[g_cand])
        se_g = float(se[g_cand])
        gbest_changed = True

    return RobustBookkeepingResult(
        population_best=pbest,
        fitness_best=phi,
        fitness_se=se,
        best_solution=g_pos,
        best_fitness=phi_g,
        best_se=se_g,
        n_pbest_replaced=n_replaced,
        n_reeval=n_reeval,
        gbest_changed=gbest_changed,
    )


# ---------------------------------------------------------------------------
# JAX swarm: RobustPSO (standalone from FixedPSO)
# ---------------------------------------------------------------------------


@struct.dataclass
class Params(BaseParams):
    inertia_coeff: float  # w
    cognitive_coeff: float  # c1
    social_coeff: float  # c2
    v_max: float  # absolute velocity clamp
    gbest_mode: int = GBEST_LCB  # 0=argmin, 1=lcb, 2=thompson
    lcb_beta: float = 1.0
    damp_eta0: float = 1.0
    damp_eta_gamma: float = 1.0
    damp_adam: bool = False
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8


@struct.dataclass
class State(_PSOState):
    fitness_se: jax.Array  # SE of filtered personal-best values
    velocity_sq_ema: jax.Array  # Adam second moment of velocity (optional)


class RobustPSO(_PSO):
    """PSO with robust mini-batch bookkeeping (paired tests + filtered φ).

    Personal-best / archive replacement is computed in NumPy
    (:func:`apply_robust_bookkeeping`) and applied with :meth:`robust_tell`.
    ``_ask`` uses LCB / Thompson social selection and optional damped moves.
    """

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
            gbest_mode=GBEST_LCB,
            lcb_beta=1.0,
            damp_eta0=1.0,
            damp_eta_gamma=1.0,
            damp_adam=False,
            adam_beta2=0.999,
            adam_eps=1e-8,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        return State(
            population=jnp.full((self.population_size, self.num_dims), jnp.nan),
            fitness=jnp.full((self.population_size,), jnp.inf),
            population_best=jnp.full((self.population_size, self.num_dims), jnp.nan),
            fitness_best=jnp.full((self.population_size,), jnp.inf),
            fitness_se=jnp.zeros((self.population_size,)),
            velocity=jnp.zeros((self.population_size, self.num_dims)),
            velocity_sq_ema=jnp.zeros((self.population_size, self.num_dims)),
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
    ):
        """Initialize, seeding archive and personal bests from init eval."""
        state = self._init(key, params)
        population = jax.vmap(self._ravel_solution)(population)
        best_idx = jnp.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        shaped = self.fitness_shaping_fn(population, fitness, state, params)
        return state.replace(
            population=population,
            fitness=shaped,
            best_solution=best_solution,
            best_fitness=best_fitness,
            population_best=population,
            fitness_best=fitness,
            fitness_se=jnp.zeros_like(fitness),
        )

    def _social_best(self, key: jax.Array, state: State, params: Params):
        """Pick social attractor: argmin / LCB / Thompson over personal bests."""
        mode = params.gbest_mode
        phi = state.fitness_best
        se = state.fitness_se

        def _argmin(_key):
            return jnp.argmin(phi)

        def _lcb(_key):
            return jnp.argmin(phi + params.lcb_beta * se)

        def _thompson(key_t):
            noise = jax.random.normal(key_t, (self.population_size,))
            return jnp.argmin(phi + se * noise)

        idx = jax.lax.switch(mode, [_argmin, _lcb, _thompson], key)
        return state.population_best[idx]

    def _ask(
        self,
        key: jax.Array,
        state: State,
        params: Params,
    ) -> tuple[Population, State]:
        key_g, key_pop = jax.random.split(key)
        best_global = self._social_best(key_g, state, params)
        v_max = params.v_max
        eta = params.damp_eta0 * (params.damp_eta_gamma**state.generation_counter)

        def _ask_member(key_m, velocity, member, member_best, u_ema):
            r1, r2 = jax.random.uniform(key_m, (2,))
            inertia = params.inertia_coeff * velocity
            cognitive = params.cognitive_coeff * r1 * (member_best - member)
            social = params.social_coeff * r2 * (best_global - member)
            vp = inertia + eta * (cognitive + social)
            vp = jnp.clip(vp, -v_max, v_max)

            def _adam_step(vp_in, u_in):
                u_out = params.adam_beta2 * u_in + (1.0 - params.adam_beta2) * (
                    vp_in * vp_in
                )
                step = vp_in / (jnp.sqrt(u_out) + params.adam_eps)
                return step, u_out

            def _plain_step(vp_in, u_in):
                return vp_in, u_in

            step, u_out = jax.lax.cond(
                params.damp_adam,
                _adam_step,
                _plain_step,
                vp,
                u_ema,
            )
            return member + step, vp, u_out

        keys = jax.random.split(key_pop, self.population_size)
        x, velocity, u_ema = jax.vmap(_ask_member)(
            keys,
            state.velocity,
            state.population,
            state.population_best,
            state.velocity_sq_ema,
        )
        return x, state.replace(velocity=velocity, velocity_sq_ema=u_ema)

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        """Unused in the robust loop (see :meth:`robust_tell`); hard fallback."""
        replace = fitness <= state.fitness_best
        population_best = jnp.where(
            replace[..., None], population, state.population_best
        )
        fitness_best = jnp.where(replace, fitness, state.fitness_best)
        fitness_se = jnp.where(replace, jnp.zeros_like(fitness), state.fitness_se)
        return state.replace(
            population=population,
            fitness=fitness,
            population_best=population_best,
            fitness_best=fitness_best,
            fitness_se=fitness_se,
        )

    def robust_tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
        *,
        population_best: Population,
        fitness_best: Fitness,
        fitness_se: Fitness,
        best_solution: Solution,
        best_fitness: float,
    ) -> tuple[State, Metrics]:
        """Apply precomputed robust pbest / gbest (skips hard archive replace)."""
        population = jax.vmap(self._ravel_solution)(population)
        population_best = jax.vmap(self._ravel_solution)(population_best)
        best_solution = self._ravel_solution(best_solution)

        state = state.replace(
            population=population,
            fitness=fitness,
            population_best=population_best,
            fitness_best=fitness_best,
            fitness_se=fitness_se,
            best_solution=best_solution,
            best_fitness=jnp.asarray(best_fitness, dtype=fitness.dtype),
        )
        metrics = self.metrics_fn(key, population, fitness, state, params)
        shaped = self.fitness_shaping_fn(population, fitness, state, params)
        state = state.replace(
            fitness=shaped,
            generation_counter=state.generation_counter + 1,
        )
        return state, metrics
