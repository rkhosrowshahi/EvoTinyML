"""Uncertainty-Aware PSO (UA-PSO): confidence-gated archives for noisy fitness.

Where FA-PSO (``soo.pso_filtered``) removes the archives and relies on averaging,
UA-PSO keeps them and makes every comparison statistically explicit. Personal and
social bests stay *actual evaluated points* — which makes them anchors that do
not move when particles move, so ``p_i - x_i`` and ``g_i - x_i`` cannot vanish
the way FA-PSO's in-hull centroid can.

Per generation, on one shared minibatch ``B_t`` (CRN):

1. Evaluate the particles ``x_i``, their personal incumbents ``p_i``, and the
   ``K`` social archive members — all on ``B_t``. That is ``2*lambda + K``
   evaluations per generation, and it is the whole cost story of this algorithm.
2. Paired difference ``d_i = F(p_i) - F(x_i)``. Because both terms share a batch
   and the networks are similar, ``Cov`` is large and ``Var(d)`` is far below
   the variance of either level — the difference is estimable at batch sizes
   where the absolute loss is not.
3. Pooled noise scale ``sigma_hat`` from a robust (MAD) spread of ``{d_i}``
   across the swarm. One shared nuisance parameter, ``lambda`` signals: this is
   what makes ``q`` well defined from a single paired sample per particle.
4. Confidences, with ``Phi`` the normal CDF::

       q^p_i = P(F(p_i) + delta < F(x_i)) = Phi((-d_i - delta) / sigma_hat)
       q^g_i = P(F(g_i) + delta < F(x_i)) = Phi((F(x_i) - F(g_i) - delta) / sigma_hat)

5. Personal archive: promote ``x_i`` over ``p_i`` only on strong evidence
   (``q^p_i < q_promote``); otherwise retain. With one slot per particle,
   "uncertain" and "retain" coincide.
6. Social archive: rank by an *accumulated* centered score. This accumulation is
   legitimate precisely because archive members are fixed points — see the
   warning below.
7. Attraction weights ``w(q) = max(w_min, 2q - 1)``, so an attractor exerts no
   pull until there is evidence it is better.
8. Move, then add ``sigma_t * eps_i`` sampling noise.

What you may and may not accumulate evidence over
-------------------------------------------------
``d_i,t`` compares ``p_i`` against ``x_i``, and ``x_i`` moves every generation.
Averaging ``d_i,t`` over ``t`` therefore estimates ``F(p_i) - E_t[F(x_i,t)]``,
an average over the particle's trajectory, not the fixed-pair quantity ``q``
claims to measure. Worse, it is biased: the particle is being pulled toward
``p_i``, so ``d`` shrinks over the window and the statistic drifts toward
"uncertain" regardless of the truth. So ``q^p`` and ``q^g`` are computed
**single-shot** from the current batch, with the variance pooled across the
swarm rather than across time. Only the social archive — whose members are fixed
points — accumulates evidence across generations.

Known risk: the confidence gate can freeze the swarm
----------------------------------------------------
Late in training real improvements shrink, so ``d -> 0``, ``q -> 0.5`` and
``w(q) -> 0`` for both terms; the velocity update degenerates to ``v <- omega*v``
and decays geometrically. The swarm then stops for an impeccable reason — there
is genuinely no evidence any attractor is better — which is what makes it
dangerous. ``w_min`` and the ``sigma_t`` sampling term are the insurance; the
principled fix is to grow the batch when ``q`` sits at 0.5 across the swarm,
which ``diag_q_*`` in the logged metrics is there to detect.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from evosax.algorithms.population_based.base import (
    Params as BaseParams,
    PopulationBasedAlgorithm,
    State as BaseState,
    metrics_fn,
)
from evosax.core.fitness_shaping import identity_fitness_shaping_fn
from evosax.types import Fitness, Population, Solution
from flax import struct
from jax.scipy.stats import norm


@struct.dataclass
class State(BaseState):
    population: Population  # concatenated [particles; personal; archive]
    fitness: Fitness
    particles: jax.Array  # x_i
    velocity: jax.Array  # v_i
    personal: jax.Array  # p_i, one slot per particle
    archive: jax.Array  # K social candidates (fixed points)
    archive_mean: jax.Array  # running mean of centered score per member
    archive_count: jax.Array  # generations of evidence per member
    std: float  # sigma_t sampling scale
    # Diagnostics (see the freeze risk in the module docstring).
    diag_sigma: float  # pooled noise scale of the paired differences
    diag_q_p: float  # mean personal confidence
    diag_q_g: float  # mean leader confidence
    diag_w_p: float  # mean personal attraction weight
    diag_w_g: float  # mean leader attraction weight
    diag_uncertain: float  # fraction of personal decisions left undecided


@struct.dataclass
class Params(BaseParams):
    inertia_coeff: float  # omega
    cognitive_coeff: float  # c_p
    social_coeff: float  # c_g
    v_max: float
    delta: float  # practical-significance margin
    sigma_scale: float  # multiplier on the pooled noise estimate
    w_min: float  # floor on the attraction weights
    q_promote: float  # promote x_i over p_i below this confidence
    score_decay: float  # EMA rate for archive scores
    shrink_prior: float  # n0 in the n/(n+n0) shrinkage of archive scores
    leader_temp: float  # softmax temperature for leader sampling
    # Ablation: when False the attraction weights are pinned to w_min and the
    # confidence gate is bypassed, while q still drives archive promotion. This
    # isolates the gate; raising sigma_scale instead would ALSO freeze promotion
    # (q -> 0.5 never clears q_promote) and conflate the two effects.
    gate_weights: bool


def _robust_scale(values: jax.Array) -> jax.Array:
    """MAD-based scale estimate, robust to the heavy tails of batch losses."""
    med = jnp.median(values)
    mad = jnp.median(jnp.abs(values - med))
    return 1.4826 * mad


class UncertaintyAwarePSO(PopulationBasedAlgorithm):
    """PSO whose archives are evidence-weighted and whose pulls are confidence-gated."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        archive_size: int = 5,
        protect_generations: int = 3,
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
        archive_size = int(archive_size)
        if archive_size < 1:
            raise ValueError(f"archive_size must be >= 1, got {archive_size}")
        # population_size is the number of solutions evaluated per generation:
        # lambda particles + lambda personal incumbents + K archive members.
        num_particles = (population_size - archive_size) // 2
        if num_particles < 2:
            raise ValueError(
                f"UA-PSO needs popsize >= {2 * 2 + archive_size} for "
                f"archive_size={archive_size} (got {population_size}); "
                "popsize counts 2*particles + archive evaluations per generation."
            )
        self.archive_size = archive_size
        self.num_particles = num_particles
        self.protect_generations = int(protect_generations)
        self.std_schedule = std_schedule
        # Trailing slots when popsize - K is odd; padded with extra archive evals.
        self.num_used = 2 * num_particles + archive_size

    @property
    def _default_params(self) -> Params:
        return Params(
            inertia_coeff=0.75,
            cognitive_coeff=1.5,
            social_coeff=2.0,
            v_max=0.2,
            delta=0.0,
            sigma_scale=1.0,
            w_min=0.05,
            q_promote=0.4,
            score_decay=0.3,
            shrink_prior=3.0,
            leader_temp=1.0,
            gate_weights=True,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        lam, k, d = self.num_particles, self.archive_size, self.num_dims
        return State(
            population=jnp.full((self.population_size, d), jnp.nan),
            fitness=jnp.full((self.population_size,), jnp.inf),
            particles=jnp.zeros((lam, d)),
            velocity=jnp.zeros((lam, d)),
            personal=jnp.zeros((lam, d)),
            archive=jnp.zeros((k, d)),
            archive_mean=jnp.zeros((k,)),
            archive_count=jnp.zeros((k,)),
            std=self.std_schedule(0),
            diag_sigma=0.0,
            diag_q_p=0.5,
            diag_q_g=0.5,
            diag_w_p=0.0,
            diag_w_g=0.0,
            diag_uncertain=1.0,
            best_solution=jnp.full((d,), jnp.nan),
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
        """Seed particles from the initial draw and the archive from its best K."""
        state = self._init(key, params)
        population = jax.vmap(self._ravel_solution)(population)

        lam, k = self.num_particles, self.archive_size
        particles = population[:lam]
        order = jnp.argsort(fitness)
        archive = population[order[:k]]

        best_idx = order[0]
        shaped = self.fitness_shaping_fn(population, fitness, state, params)
        return state.replace(
            population=population,
            fitness=shaped,
            particles=particles,
            # Each particle starts as its own incumbent; nothing to compare yet.
            personal=particles,
            archive=archive,
            archive_count=jnp.ones((k,)),
            best_solution=population[best_idx],
            best_fitness=fitness[best_idx],
        )

    def _ask(
        self,
        key: jax.Array,
        state: State,
        params: Params,
    ) -> tuple[Population, State]:
        """Emit everything that must be measured on this generation's batch."""
        block = jnp.concatenate([state.particles, state.personal, state.archive])
        pad = self.population_size - self.num_used
        if pad > 0:
            # Odd popsize: re-measure the leading archive members in the spare
            # slots rather than silently wasting the budget.
            block = jnp.concatenate([block, state.archive[:pad]])
        return block, state

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        lam, k = self.num_particles, self.archive_size
        f_x = fitness[:lam]
        f_p = fitness[lam : 2 * lam]
        f_a = fitness[2 * lam : 2 * lam + k]
        x = population[:lam]
        p = population[lam : 2 * lam]
        archive = population[2 * lam : 2 * lam + k]

        # --- Evidence -------------------------------------------------------
        # Paired difference on a shared batch; d > 0 means x looks better.
        d = f_p - f_x
        sigma = params.sigma_scale * _robust_scale(d)
        sigma = jnp.maximum(sigma, 1e-8)

        q_p = norm.cdf((-d - params.delta) / sigma)
        # Reuses the paired-difference scale for the leader comparison: both are
        # same-batch differences between similar networks. Conservative, since
        # the spread of {d_i} also contains real between-particle signal.

        # --- Personal archive: promote only on strong evidence ---------------
        promote = q_p < params.q_promote
        personal = jnp.where(promote[:, None], x, p)

        # --- Social archive --------------------------------------------------
        # Centered, scale-normalised score for this generation. Centering removes
        # the batch-level shift, which is what makes accumulation across
        # generations valid for these (fixed) points.
        f_all = fitness[: 2 * lam + k]
        centre = jnp.median(f_all)
        spread = jnp.maximum(_robust_scale(f_all), 1e-8)
        z_a = (f_a - centre) / spread

        rho = params.score_decay
        archive_mean = (1.0 - rho) * state.archive_mean + rho * z_a
        archive_count = state.archive_count + 1.0

        # Insert this generation's best particle, evicting the worst member that
        # is out of its protection window. Newcomers are protected for
        # protect_generations so they can accumulate evidence before being
        # judged -- this is the "retain and re-evaluate later" branch.
        cand_idx = jnp.argmin(f_x)
        z_cand = (f_x[cand_idx] - centre) / spread
        n0 = params.shrink_prior
        eff = archive_mean * (archive_count / (archive_count + n0))
        protected = archive_count < self.protect_generations
        evictable = jnp.where(protected, -jnp.inf, eff)
        worst = jnp.argmax(evictable)
        # Only replace if some member is actually evictable.
        can_evict = jnp.any(~protected)

        onehot = (jnp.arange(k) == worst) & can_evict
        archive = jnp.where(onehot[:, None], x[cand_idx], archive)
        archive_mean = jnp.where(onehot, rho * z_cand, archive_mean)
        archive_count = jnp.where(onehot, 1.0, archive_count)

        # --- Leader selection: probability-weighted over the archive ---------
        eff = archive_mean * (archive_count / (archive_count + n0))
        key_leader, key_move = jax.random.split(key)
        leader_idx = jax.random.categorical(
            key_leader, -eff / params.leader_temp, shape=(lam,)
        )
        g = archive[leader_idx]
        f_g = jnp.where(onehot[leader_idx], f_x[cand_idx], f_a[leader_idx])
        q_g = norm.cdf((f_x - f_g - params.delta) / sigma)

        # --- Confidence-gated move -------------------------------------------
        w_p = jnp.where(
            params.gate_weights,
            jnp.maximum(params.w_min, 2.0 * q_p - 1.0),
            jnp.full_like(q_p, params.w_min),
        )
        w_g = jnp.where(
            params.gate_weights,
            jnp.maximum(params.w_min, 2.0 * q_g - 1.0),
            jnp.full_like(q_g, params.w_min),
        )

        def _move(key, x_i, p_i, g_i, v_i, wp_i, wg_i):
            key_rp, key_rg, key_eps = jax.random.split(key, 3)
            r_p = jax.random.uniform(key_rp, (self.num_dims,))
            r_g = jax.random.uniform(key_rg, (self.num_dims,))
            v_i = (
                params.inertia_coeff * v_i
                + params.cognitive_coeff * wp_i * r_p * (p_i - x_i)
                + params.social_coeff * wg_i * r_g * (g_i - x_i)
            )
            v_i = jnp.clip(v_i, -params.v_max, params.v_max)
            noise = state.std * jax.random.normal(key_eps, (self.num_dims,))
            return x_i + v_i + noise, v_i

        keys = jax.random.split(key_move, lam)
        particles, velocity = jax.vmap(_move)(
            keys, x, personal, g, state.velocity, w_p, w_g
        )

        undecided = jnp.mean(
            ((q_p >= params.q_promote) & (q_p <= 1.0 - params.q_promote)).astype(
                jnp.float32
            )
        )

        return state.replace(
            population=population,
            fitness=fitness,
            particles=particles,
            velocity=velocity,
            personal=personal,
            archive=archive,
            archive_mean=archive_mean,
            archive_count=archive_count,
            std=self.std_schedule(state.generation_counter),
            diag_sigma=sigma,
            diag_q_p=jnp.mean(q_p),
            diag_q_g=jnp.mean(q_g),
            diag_w_p=jnp.mean(w_p),
            diag_w_g=jnp.mean(w_g),
            diag_uncertain=undecided,
        )
