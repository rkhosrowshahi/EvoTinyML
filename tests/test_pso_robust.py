"""Tests for robust PSO bookkeeping under mini-batch fitness noise."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from evotinyml.soo.pso_fixed import FixedPSO
from evotinyml.soo.pso_robust import (
    GBEST_LCB,
    RobustPSO,
    RobustPSOConfig,
    apply_robust_bookkeeping,
    confident_accept,
    ema_update,
    paired_diff_stats,
    select_gbest_index,
    wilcoxon_signed_rank_z,
)


def test_paired_diff_cancels_shared_batch_shift():
    """Common batch offset cancels in the paired gap (CRN)."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=64)
    # Easy batch shifts both losses by −1; gap unchanged.
    cand = base + 0.5 - 1.0
    pbest = base - 1.0
    d_bar, se, z = paired_diff_stats(cand, pbest)
    assert abs(d_bar - 0.5) < 1e-9
    assert se >= 0.0
    assert z > 0.0  # candidate worse


def test_confident_accept_rejects_noise_flip():
    """Improvement smaller than κ·SE must be rejected."""
    # Three ties and one lucky win ⇒ d̄ = −0.25, SE large enough that
    # −κ·SE is more negative than d̄ for κ=1.5.
    cand = np.array([1.0, 1.0, 1.0, 0.0])
    inc = np.array([1.0, 1.0, 1.0, 1.0])
    accept, d_bar, se, _ = confident_accept(cand, inc, kappa=1.5, test="ttest")
    assert d_bar < 0.0
    assert d_bar >= -1.5 * se
    assert not accept


def test_confident_accept_takes_clear_improvement():
    p = np.ones(64)
    x = np.full(64, 0.5)
    accept, d_bar, se, _ = confident_accept(x, p, kappa=1.25, test="ttest")
    assert accept
    assert d_bar < -1.25 * se


def test_ema_pulls_optimistic_incumbent_up():
    phi = 0.1  # lucky snapshot
    for _ in range(20):
        phi = ema_update(phi, 1.0, alpha=0.3)
    assert phi > 0.9


def test_lcb_prefers_stable_over_noisy_lucky():
    phi = np.array([1.0, 0.5])  # particle 1 looks better
    se = np.array([0.05, 2.0])  # but is very noisy
    idx = select_gbest_index(phi, se, mode="lcb", beta=1.0)
    assert idx == 0  # LCB: 1.05 vs 2.5


def test_wilcoxon_negative_z_when_candidate_better():
    d = np.full(32, -0.5)
    z = wilcoxon_signed_rank_z(d)
    assert z < 0.0


def test_apply_robust_bookkeeping_replaces_and_filters():
    rng = np.random.default_rng(0)
    n, m, d = 4, 32, 3
    pop = rng.normal(size=(n, d))
    pbest = pop + 1.0
    # Candidates clearly better on every sample.
    cand = np.full((n, m), 0.5)
    p_loss = np.full((n, m), 1.5)
    g_loss = np.full(m, 1.5)
    cfg = RobustPSOConfig(kappa=1.0, ema_alpha=0.5, gbest_mode="argmin")
    out = apply_robust_bookkeeping(
        population=pop,
        cand_losses=cand,
        population_best=pbest,
        fitness_best=np.full(n, 1.5),
        fitness_se=np.zeros(n),
        best_solution=pbest[0],
        best_fitness=1.5,
        best_se=0.0,
        config=cfg,
        gen=1,
        rng=rng,
        pbest_losses=p_loss,
        g_losses=g_loss,
        reeval_mask=np.ones(n, dtype=bool),
    )
    assert out.n_pbest_replaced == n
    assert np.allclose(out.population_best, pop)
    assert np.allclose(out.fitness_best, 0.5)


def test_fixed_pso_robust_tell_and_lcb_ask():
    """robust_tell seeds SE; ask with LCB must stay finite and clamped."""
    n_particles, n_var = 4, 8
    algo = RobustPSO(population_size=n_particles, solution=jnp.zeros(n_var))
    params = algo.default_params.replace(
        gbest_mode=GBEST_LCB,
        lcb_beta=1.0,
        v_max=0.1,
    )
    rng = np.random.default_rng(0)
    pop = rng.normal(size=(n_particles, n_var)).astype(np.float32)
    fit = np.sum(pop * pop, axis=1).astype(np.float32)
    key = jax.random.key(0)
    key, k_init = jax.random.split(key)
    state = algo.init(k_init, jnp.asarray(pop), jnp.asarray(fit), params)

    key, k_tell = jax.random.split(key)
    se = np.linspace(0.0, 1.0, n_particles).astype(np.float32)
    state, _ = algo.robust_tell(
        k_tell,
        jnp.asarray(pop),
        jnp.asarray(fit),
        state,
        params,
        population_best=jnp.asarray(pop),
        fitness_best=jnp.asarray(fit),
        fitness_se=jnp.asarray(se),
        best_solution=jnp.asarray(pop[int(np.argmin(fit))]),
        best_fitness=float(np.min(fit)),
    )
    assert np.allclose(np.asarray(state.fitness_se), se)

    key, k_ask = jax.random.split(key)
    trial, state = algo.ask(k_ask, state, params)
    assert trial.shape == (n_particles, n_var)
    assert np.max(np.abs(np.asarray(state.velocity))) <= 0.1 + 1e-5


def test_pso_robust_registered_as_separate_algo():
    from evotinyml.soo.algorithms import SOO_ALGORITHMS
    from evotinyml.soo.es import EVOSAX_SOO_ALGOS, POPULATION_BASED_ALGOS, PSO_ALGOS
    from evotinyml.soo import pso_fixed, pso_robust

    assert "pso_robust" in SOO_ALGORITHMS
    assert "pso_robust" in EVOSAX_SOO_ALGOS
    assert "pso_robust" in POPULATION_BASED_ALGOS
    assert PSO_ALGOS == frozenset({"pso", "pso_robust"})
    assert hasattr(pso_robust, "RobustPSO")
    assert not hasattr(pso_fixed, "RobustPSO")
    assert RobustPSO(2, jnp.zeros(1)).default_params.gbest_mode == GBEST_LCB
