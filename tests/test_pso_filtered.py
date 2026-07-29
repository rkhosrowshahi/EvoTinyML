"""Tests for FA-PSO (Filtered-Attractor PSO), evotinyml.soo.pso_filtered."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from evotinyml.soo.pso_filtered import FilteredPSO, rank_weights


def _no_sampling(**kwargs):
    """FilteredPSO with sigma=0 (the pre-sampling-term behaviour)."""
    return FilteredPSO(std_schedule=optax.constant_schedule(0.0), **kwargs)


def _diversity(state) -> float:
    pop = np.asarray(state.population)
    return float(np.linalg.norm(pop - pop.mean(axis=0), axis=1).mean())


def _init(algo, pop, fit, params, seed=0):
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    return key, algo.init(subkey, jnp.asarray(pop), jnp.asarray(fit), params)


def test_rank_weights_positive_decreasing_normalized():
    for mu in (1, 2, 5, 16):
        w = np.asarray(rank_weights(mu))
        assert w.shape == (mu,)
        assert (w > 0).all()
        assert np.isclose(w.sum(), 1.0)
        assert (np.diff(w) <= 0).all()


def test_state_has_no_cross_generation_fitness_archive():
    """The point of FA-PSO: no pbest/gbest value survives a generation."""
    algo = FilteredPSO(population_size=4, solution=jnp.zeros(2))
    _, state = _init(
        algo,
        np.zeros((4, 2), dtype=np.float32),
        np.zeros(4, dtype=np.float32),
        algo.default_params,
    )
    assert not hasattr(state, "fitness_best")
    assert not hasattr(state, "population_best")


def test_social_attractor_is_rank_weighted_centroid_at_init():
    algo = FilteredPSO(population_size=4, solution=jnp.zeros(1), elite_ratio=0.5)
    params = algo.default_params
    pop = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    fit = np.array([3.0, 1.0, 0.0, 2.0], dtype=np.float32)  # ranking: 2, 1, 3, 0

    _, state = _init(algo, pop, fit, params)

    assert algo.num_elites == 2
    w = np.asarray(rank_weights(2))
    expected = w[0] * 2.0 + w[1] * 1.0
    assert float(state.social[0]) == pytest.approx(expected, rel=1e-6)
    # Personal attractors start at each particle's own position.
    assert np.allclose(np.asarray(state.personal), pop)


def test_social_attractor_is_ema_filtered_across_generations():
    """A single lucky generation may move g by at most alpha."""
    algo = FilteredPSO(population_size=4, solution=jnp.zeros(1), elite_ratio=0.25)
    alpha = 0.2
    params = algo.default_params.replace(social_decay=alpha)

    pop0 = np.zeros((4, 1), dtype=np.float32)
    fit0 = np.arange(4, dtype=np.float32)
    key, state = _init(algo, pop0, fit0, params)
    assert float(state.social[0]) == pytest.approx(0.0)

    # New generation whose best particle sits at 10.0 (an extreme outlier).
    pop1 = np.array([[10.0], [0.0], [0.0], [0.0]], dtype=np.float32)
    fit1 = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    key, key_tell = jax.random.split(key)
    state, _ = algo.tell(
        key_tell, jnp.asarray(pop1), jnp.asarray(fit1), state, params
    )

    # mu = 1, so ghat = 10.0; the EMA admits only alpha of it.
    assert algo.num_elites == 1
    assert float(state.social[0]) == pytest.approx(alpha * 10.0, rel=1e-6)


def test_personal_attractor_gated_on_intra_generation_rank():
    algo = FilteredPSO(population_size=4, solution=jnp.zeros(1))
    beta = 0.5
    params = algo.default_params.replace(personal_decay=beta)

    pop0 = np.zeros((4, 1), dtype=np.float32)
    fit0 = np.zeros(4, dtype=np.float32)
    key, state = _init(algo, pop0, fit0, params)

    pop1 = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    # Particles 3 and 1 are the better half of this generation.
    fit1 = np.array([5.0, 1.0, 9.0, 0.0], dtype=np.float32)
    key, key_tell = jax.random.split(key)
    state, _ = algo.tell(
        key_tell, jnp.asarray(pop1), jnp.asarray(fit1), state, params
    )

    personal = np.asarray(state.personal).ravel()
    assert algo.num_gated == 2
    # Gated in: EMA toward the new position.
    assert personal[3] == pytest.approx(beta * 4.0)
    assert personal[1] == pytest.approx(beta * 2.0)
    # Gated out: unchanged, and crucially not overwritten by a worse position.
    assert personal[0] == pytest.approx(0.0)
    assert personal[2] == pytest.approx(0.0)


def test_stale_good_fitness_cannot_freeze_the_swarm():
    """The regression that motivates FA-PSO.

    Generation 1 draws a wildly favourable batch (fitness -1000). Under hard
    replacement that value is archived and no later generation can beat it, so
    the attractors freeze. FA-PSO stores no value, so generation 2 still moves
    the attractors.
    """
    algo = FilteredPSO(population_size=4, solution=jnp.zeros(1), elite_ratio=0.25)
    params = algo.default_params.replace(social_decay=0.5, personal_decay=1.0)

    key, state = _init(
        algo,
        np.zeros((4, 1), dtype=np.float32),
        np.zeros(4, dtype=np.float32),
        params,
    )

    # Lucky generation: particle 0 at position 1.0 measured at -1000.
    lucky_pop = np.array([[1.0], [0.0], [0.0], [0.0]], dtype=np.float32)
    lucky_fit = np.array([-1000.0, 0.0, 0.0, 0.0], dtype=np.float32)
    key, k1 = jax.random.split(key)
    state, _ = algo.tell(
        key, jnp.asarray(lucky_pop), jnp.asarray(lucky_fit), state, params
    )
    social_after_lucky = float(state.social[0])

    # Honest generation: best particle at 5.0, measured at a much worse value
    # than the stale -1000. A greedy archive would reject it outright.
    honest_pop = np.array([[0.0], [5.0], [0.0], [0.0]], dtype=np.float32)
    honest_fit = np.array([2.0, 1.0, 3.0, 4.0], dtype=np.float32)
    state, _ = algo.tell(
        k1, jnp.asarray(honest_pop), jnp.asarray(honest_fit), state, params
    )

    assert float(state.social[0]) != pytest.approx(social_after_lucky)
    # And it moved toward the honest generation's best particle.
    assert float(state.social[0]) > social_after_lucky
    assert float(np.asarray(state.personal).ravel()[1]) == pytest.approx(5.0)


def test_ask_clamps_velocity_and_moves_particles():
    n, d, v_max = 8, 16, 0.05
    algo = FilteredPSO(population_size=n, solution=jnp.zeros(d))
    params = algo.default_params.replace(
        inertia_coeff=0.9, cognitive_coeff=2.0, social_coeff=2.0, v_max=v_max
    )

    rng = np.random.default_rng(0)
    pop = rng.normal(size=(n, d)).astype(np.float32)
    fit = np.sum(pop * pop, axis=1).astype(np.float32)
    key, state = _init(algo, pop, fit, params)

    state = state.replace(velocity=jnp.full((n, d), 10.0, dtype=jnp.float32))
    key, key_ask = jax.random.split(key)
    x, state = algo.ask(key_ask, state, params)

    assert np.max(np.abs(np.asarray(state.velocity))) <= v_max + 1e-6
    assert np.isfinite(np.asarray(x)).all()


def test_adam_variant_bounds_step_by_learning_rate():
    """Adam's per-coordinate scaling caps |step| near lr, without v_max."""
    n, d, lr = 6, 12, 0.01
    algo = _no_sampling(population_size=n, solution=jnp.zeros(d), use_adam=True)
    params = algo.default_params.replace(lr=lr)

    rng = np.random.default_rng(1)
    pop = (rng.normal(size=(n, d)) * 100.0).astype(np.float32)  # huge pulls
    fit = np.sum(pop * pop, axis=1).astype(np.float32)
    key, state = _init(algo, pop, fit, params)

    key, key_ask = jax.random.split(key)
    x, state = algo.ask(key_ask, state, params)

    step = np.asarray(state.velocity)  # holds the Adam step in this mode
    assert np.isfinite(step).all()
    # First bias-corrected Adam step is exactly +/- lr per coordinate.
    assert np.max(np.abs(step)) <= lr * 1.01
    assert np.allclose(np.asarray(x), pop + step, atol=1e-5)


def _run_noisy_sphere(algo, params, *, steps=300, d=64, seed=5, fitness_noise=0.0):
    """Sphere with additive fitness noise, mimicking minibatch estimation error."""
    n = algo.population_size
    rng = np.random.default_rng(seed)
    pop = (rng.normal(size=(n, d)) * 2.0).astype(np.float32)
    key, state = _init(
        algo, pop, np.sum(pop * pop, axis=1).astype(np.float32), params, seed=seed
    )
    for _ in range(steps):
        key, key_ask, key_tell, key_noise = jax.random.split(key, 4)
        x, state = algo.ask(key_ask, state, params)
        f = jnp.sum(jnp.square(x), axis=1)
        f = f + fitness_noise * jax.random.normal(key_noise, (n,))
        state, _ = algo.tell(key_tell, x, f, state, params)
    return state


def test_noise_dominated_ranking_collapses_the_swarm_without_sampling():
    """The failure the sampling term exists to prevent.

    When batch noise dominates the ranking the ordering is near-random, so the
    rank-weighted centroid degenerates to the plain swarm mean and the social
    term becomes a restoring force toward that mean with no directional
    information. With no additive noise source, the swarm contracts onto its own
    centroid. This is the regime the algorithm is built for, so it matters.
    """
    d = 64
    algo = _no_sampling(population_size=16, solution=jnp.zeros(d))
    params = algo.default_params.replace(v_max=0.2)

    # Signal is O(100) across the swarm; 1e3 noise swamps it.
    noisy = _run_noisy_sphere(algo, params, d=d, fitness_noise=1e3)
    assert _diversity(noisy) < 1e-3

    # Same algorithm, clean ranking: no collapse. The landscape is not the cause.
    clean = _run_noisy_sphere(algo, params, d=d, fitness_noise=0.0)
    assert _diversity(clean) > 1.0


def test_sampling_term_keeps_a_floor_under_diversity_under_noise():
    """The fix: with sigma > 0 the swarm cannot contract, even at high noise."""
    d, sigma = 64, 0.1
    algo = FilteredPSO(
        population_size=16,
        solution=jnp.zeros(d),
        std_schedule=optax.constant_schedule(sigma),
    )
    params = algo.default_params.replace(v_max=0.2)

    state = _run_noisy_sphere(algo, params, d=d, fitness_noise=1e3)
    # Independent N(0, sigma^2 I) per particle => expected spread ~ sigma*sqrt(d).
    assert _diversity(state) > 0.5 * sigma * np.sqrt(d)


def test_sampling_term_does_not_prevent_convergence():
    """Sigma > 0 must not turn the attractor into a random walk."""
    d, sigma = 8, 0.05
    algo = FilteredPSO(
        population_size=16,
        solution=jnp.zeros(d),
        std_schedule=optax.constant_schedule(sigma),
    )
    params = algo.default_params.replace(v_max=0.5)
    state = _run_noisy_sphere(algo, params, d=d, fitness_noise=0.0)
    assert float(np.sum(np.asarray(state.social) ** 2)) < 1.0


def test_std_follows_the_schedule():
    algo = FilteredPSO(
        population_size=4,
        solution=jnp.zeros(2),
        std_schedule=optax.exponential_decay(0.5, transition_steps=1, decay_rate=0.5),
    )
    params = algo.default_params
    pop = np.zeros((4, 2), dtype=np.float32)
    fit = np.arange(4, dtype=np.float32)
    key, state = _init(algo, pop, fit, params)
    assert float(state.std) == pytest.approx(0.5)

    for expected in (0.5, 0.25):
        key, key_tell = jax.random.split(key)
        state, _ = algo.tell(
            key_tell, jnp.asarray(pop), jnp.asarray(fit), state, params
        )
        assert float(state.std) == pytest.approx(expected)


def test_sampling_noise_is_not_accumulated_by_inertia():
    """Noise enters the position, not the velocity, so inertia cannot amplify it."""
    n, d = 8, 16
    algo = FilteredPSO(
        population_size=n,
        solution=jnp.zeros(d),
        std_schedule=optax.constant_schedule(1.0),
    )
    params = algo.default_params.replace(v_max=0.05)
    pop = np.zeros((n, d), dtype=np.float32)
    key, state = _init(algo, pop, np.zeros(n, dtype=np.float32), params)

    key, key_ask = jax.random.split(key)
    _, state = algo.ask(key_ask, state, params)
    # Large sigma, but velocity stays inside the clamp.
    assert np.max(np.abs(np.asarray(state.velocity))) <= 0.05 + 1e-6


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        FilteredPSO(population_size=1, solution=jnp.zeros(2))
    with pytest.raises(ValueError):
        FilteredPSO(population_size=4, solution=jnp.zeros(2), elite_ratio=0.0)
    with pytest.raises(ValueError):
        FilteredPSO(population_size=4, solution=jnp.zeros(2), elite_ratio=1.5)


def test_full_ask_tell_loop_minimizes_sphere():
    """Sanity check on a noiseless problem: it must still optimize."""
    n, d = 16, 8
    algo = FilteredPSO(population_size=n, solution=jnp.zeros(d))
    params = algo.default_params.replace(v_max=0.5)

    rng = np.random.default_rng(3)
    pop = (rng.normal(size=(n, d)) * 2.0).astype(np.float32)
    fit = np.sum(pop * pop, axis=1).astype(np.float32)
    key, state = _init(algo, pop, fit, params)
    f_start = float(np.sum(np.asarray(state.social) ** 2))

    for _ in range(60):
        key, key_ask, key_tell = jax.random.split(key, 3)
        x, state = algo.ask(key_ask, state, params)
        f = jnp.sum(jnp.square(x), axis=1)
        state, _ = algo.tell(key_tell, x, f, state, params)

    f_end = float(np.sum(np.asarray(state.social) ** 2))
    assert f_end < f_start
    assert np.isfinite(np.asarray(state.social)).all()
