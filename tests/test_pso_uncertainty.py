"""Tests for UA-PSO (Uncertainty-Aware PSO), evotinyml.soo.pso_uncertainty."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from evotinyml.soo.pso_uncertainty import UncertaintyAwarePSO


def _make(popsize=21, d=4, k=5, sigma=0.0, **kw):
    return UncertaintyAwarePSO(
        population_size=popsize,
        solution=jnp.zeros(d),
        archive_size=k,
        std_schedule=optax.constant_schedule(sigma),
        **kw,
    )


def _init(algo, pop, fit, params, seed=0):
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    return key, algo.init(subkey, jnp.asarray(pop), jnp.asarray(fit), params)


def test_population_size_accounts_for_every_evaluation():
    """popsize must equal 2*particles + archive: FE accounting has to be honest."""
    algo = _make(popsize=21, k=5)
    assert algo.num_particles == 8
    assert 2 * algo.num_particles + algo.archive_size == 21

    key = jax.random.key(0)
    rng = np.random.default_rng(0)
    pop = rng.normal(size=(21, 4)).astype(np.float32)
    fit = np.sum(pop * pop, axis=1).astype(np.float32)
    _, state = _init(algo, pop, fit, algo.default_params)
    asked, _ = algo.ask(key, state, algo.default_params)
    assert asked.shape == (21, 4)

    # The asked block really is [particles; personal; archive].
    a = np.asarray(asked)
    assert np.allclose(a[:8], np.asarray(state.particles))
    assert np.allclose(a[8:16], np.asarray(state.personal))
    assert np.allclose(a[16:21], np.asarray(state.archive))


def test_rejects_popsize_too_small_for_archive():
    with pytest.raises(ValueError):
        _make(popsize=8, k=5)  # (8-5)//2 = 1 particle
    with pytest.raises(ValueError):
        UncertaintyAwarePSO(
            population_size=21, solution=jnp.zeros(4), archive_size=0
        )


def test_personal_incumbent_promoted_only_on_strong_evidence():
    """q^p low (x clearly better) promotes; ambiguous evidence retains p."""
    algo = _make(popsize=13, d=2, k=5)  # 4 particles
    params = algo.default_params.replace(q_promote=0.4, sigma_scale=1.0)
    lam = algo.num_particles

    rng = np.random.default_rng(1)
    pop = rng.normal(size=(13, 2)).astype(np.float32)
    key, state = _init(algo, pop, np.arange(13, dtype=np.float32), params)

    asked, state = algo.ask(key, state, params)
    p_before = np.asarray(state.personal).copy()

    # Particle 0's incumbent is far worse than the particle (d << 0 => q^p ~ 0);
    # the rest are ties, which must leave their incumbents untouched.
    f = np.zeros(13, dtype=np.float32)
    f[lam + 0] = 50.0  # personal incumbent of particle 0 is terrible
    f[1:lam] = 0.0
    key, key_tell = jax.random.split(key)
    state, _ = algo.tell(key_tell, asked, jnp.asarray(f), state, params)

    personal = np.asarray(state.personal)
    x_before = np.asarray(asked)[:lam]
    assert np.allclose(personal[0], x_before[0])  # promoted
    for i in range(1, lam):
        assert np.allclose(personal[i], p_before[i])  # retained


def test_confidence_weights_collapse_when_evidence_vanishes():
    """The design's central risk: no evidence => no pull.

    With every candidate identical, q -> 0.5 and w(q) must fall to w_min.
    """
    algo = _make(popsize=13, d=2, k=5)
    params = algo.default_params.replace(w_min=0.0)

    pop = np.zeros((13, 2), dtype=np.float32)
    key, state = _init(algo, pop, np.zeros(13, dtype=np.float32), params)
    asked, state = algo.ask(key, state, params)

    key, key_tell = jax.random.split(key)
    state, _ = algo.tell(
        key_tell, asked, jnp.zeros(13, dtype=jnp.float32), state, params
    )

    assert float(state.diag_q_p) == pytest.approx(0.5, abs=1e-5)
    assert float(state.diag_w_p) == pytest.approx(0.0, abs=1e-6)
    assert float(state.diag_w_g) == pytest.approx(0.0, abs=1e-6)
    assert float(state.diag_uncertain) == pytest.approx(1.0)
    # And with w_min = 0 and no sampling noise, the swarm stops moving.
    assert float(np.abs(np.asarray(state.velocity)).max()) == pytest.approx(0.0)


def test_w_min_floor_keeps_the_swarm_moving_without_evidence():
    algo = _make(popsize=13, d=2, k=5)
    params = algo.default_params.replace(w_min=0.2)

    rng = np.random.default_rng(2)
    pop = rng.normal(size=(13, 2)).astype(np.float32)
    key, state = _init(algo, pop, np.zeros(13, dtype=np.float32), params)
    asked, state = algo.ask(key, state, params)
    key, key_tell = jax.random.split(key)
    state, _ = algo.tell(
        key_tell, asked, jnp.zeros(13, dtype=jnp.float32), state, params
    )

    assert float(state.diag_w_p) == pytest.approx(0.2)
    assert float(np.abs(np.asarray(state.velocity)).max()) > 0.0


def test_archive_scores_shrink_toward_zero_for_new_members():
    """A single lucky batch must not install a candidate at the top."""
    algo = _make(popsize=13, d=2, k=5)
    params = algo.default_params.replace(score_decay=0.3, shrink_prior=3.0)

    rng = np.random.default_rng(3)
    pop = rng.normal(size=(13, 2)).astype(np.float32)
    key, state = _init(algo, pop, np.arange(13, dtype=np.float32), params)

    # All members enter at init with one observation, so nothing is evictable
    # until the protection window expires and turnover can begin.
    for _ in range(3):
        asked, state = algo.ask(key, state, params)
        key, key_tell = jax.random.split(key)
        state, _ = algo.tell(
            key_tell, asked, jnp.asarray(np.arange(13, dtype=np.float32)), state, params
        )

    # Now particle 0 draws a spectacular batch and enters the archive.
    asked, state = algo.ask(key, state, params)
    f = np.full(13, 10.0, dtype=np.float32)
    f[0] = -1000.0
    key, key_tell = jax.random.split(key)
    state, _ = algo.tell(key_tell, asked, jnp.asarray(f), state, params)

    counts = np.asarray(state.archive_count)
    means = np.asarray(state.archive_mean)
    entered = int(np.argmin(counts))
    assert counts[entered] == 1.0
    assert means[entered] < 0.0  # it did look good on that batch
    # But n/(n+n0) = 1/4 shrinkage keeps one lucky batch from dominating.
    eff = means * (counts / (counts + 3.0))
    assert eff[entered] > means[entered]
    assert eff[entered] == pytest.approx(means[entered] * 0.25)


def test_protected_members_are_not_evicted():
    algo = _make(popsize=13, d=2, k=5, protect_generations=10)
    params = algo.default_params

    rng = np.random.default_rng(4)
    pop = rng.normal(size=(13, 2)).astype(np.float32)
    key, state = _init(algo, pop, np.arange(13, dtype=np.float32), params)
    archive_before = np.asarray(state.archive).copy()

    for _ in range(3):
        asked, state = algo.ask(key, state, params)
        key, key_tell = jax.random.split(key)
        state, _ = algo.tell(
            key_tell, asked, jnp.asarray(np.arange(13, dtype=np.float32)), state, params
        )

    # All members are inside the protection window, so nothing can be replaced.
    assert np.allclose(np.asarray(state.archive), archive_before)


def test_full_loop_minimizes_noisy_sphere():
    d = 16
    algo = _make(popsize=37, d=d, k=5, sigma=0.05)  # 16 particles
    params = algo.default_params.replace(v_max=0.5)

    rng = np.random.default_rng(7)
    pop = (rng.normal(size=(37, d)) * 2.0).astype(np.float32)
    key, state = _init(
        algo, pop, np.sum(pop * pop, axis=1).astype(np.float32), params
    )
    f_start = float(np.mean(np.sum(np.asarray(state.particles) ** 2, axis=1)))

    for _ in range(200):
        key, key_ask, key_tell, key_noise = jax.random.split(key, 4)
        x, state = algo.ask(key_ask, state, params)
        f = jnp.sum(jnp.square(x), axis=1)
        f = f + 5.0 * jax.random.normal(key_noise, (37,))
        state, _ = algo.tell(key_tell, x, f, state, params)

    f_end = float(np.mean(np.sum(np.asarray(state.particles) ** 2, axis=1)))
    assert f_end < f_start
    assert np.isfinite(np.asarray(state.archive)).all()
