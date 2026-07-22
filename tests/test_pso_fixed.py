"""Regression tests for FixedPSO (evosax PR #109 + velocity clamping)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from evotinyml.soo.pso_fixed import FixedPSO


def test_pso_seeds_personal_and_global_best_from_init():
    """PSO must keep the best initial particle after the first ask/tell."""
    population_size = 3
    num_dims = 1
    algo = FixedPSO(population_size=population_size, solution=jnp.zeros(num_dims))
    params = algo.default_params

    # True best at particle 2 (fitness 0); particle 0 is a distractor.
    population_init = jnp.array([[10.0], [20.0], [0.0]])
    fitness_init = jnp.array([100.0, 400.0, 0.0])

    key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    state = algo.init(subkey, population_init, fitness_init, params)

    assert jnp.allclose(state.population_best, population_init)
    assert jnp.allclose(state.fitness_best, fitness_init)
    assert jnp.allclose(state.best_solution, jnp.array([0.0]))
    assert float(state.best_fitness) == 0.0

    # First ask must use true gbest (particle 2), not particle 0.
    key, key_ask, key_tell = jax.random.split(key, 3)
    _, state = algo.ask(key_ask, state, params)

    # Worse post-move positions (as under the old bogus-gbest pull).
    population = jnp.array([[10.0], [10.57], [0.57]])
    fitness = jnp.array([100.0, 105.7, 0.57])
    state, _ = algo.tell(key_tell, population, fitness, state, params)

    # Initial optimum must remain personal best for particle 2 and archive best.
    assert jnp.allclose(state.population_best[2], jnp.array([0.0]))
    assert float(state.fitness_best[2]) == 0.0
    assert jnp.allclose(state.best_solution, jnp.array([0.0]))
    assert float(state.best_fitness) == 0.0


def test_pso_clamps_velocity_to_v_max():
    """Velocity components must stay within [-V_max, V_max]."""
    n_particles, n_var = 8, 16
    v_max = 0.05
    algo = FixedPSO(population_size=n_particles, solution=jnp.zeros(n_var))
    params = algo.default_params.replace(
        inertia_coeff=0.9,
        cognitive_coeff=2.0,
        social_coeff=2.0,
        v_max=v_max,
    )

    rng = np.random.default_rng(0)
    pop = rng.normal(size=(n_particles, n_var)).astype(np.float32)
    fit = np.sum(pop * pop, axis=1).astype(np.float32)

    key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    state = algo.init(subkey, jnp.asarray(pop), jnp.asarray(fit), params)
    # Large prior velocity that would exceed V_max without clamping.
    state = state.replace(
        velocity=jnp.full((n_particles, n_var), 10.0, dtype=jnp.float32)
    )

    key, key_ask = jax.random.split(key)
    _, state = algo.ask(key_ask, state, params)
    vel = np.asarray(state.velocity)
    assert np.max(np.abs(vel)) <= v_max + 1e-6


def test_upstream_pso_loses_init_optimum():
    """Document the upstream bug that FixedPSO repairs."""
    from evosax.algorithms import PSO

    algo = PSO(population_size=3, solution=jnp.zeros(1))
    params = algo.default_params
    population_init = jnp.array([[10.0], [20.0], [0.0]])
    fitness_init = jnp.array([100.0, 400.0, 0.0])

    key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    state = algo.init(subkey, population_init, fitness_init, params)
    # Stock init leaves personal bests unset.
    assert np.isinf(np.asarray(state.fitness_best)).all()

    key, key_ask, key_tell = jax.random.split(key, 3)
    trial, state = algo.ask(key_ask, state, params)
    fitness = jnp.sum(trial * trial, axis=1)
    state, _ = algo.tell(key_tell, trial, fitness, state, params)

    # Optimum at 0 is lost from personal / archive best.
    assert not jnp.allclose(state.population_best[2], jnp.array([0.0]))
    assert float(state.best_fitness) > 0.0
