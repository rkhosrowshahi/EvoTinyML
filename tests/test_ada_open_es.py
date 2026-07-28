"""Tests for AdaOpenES (OpenES with self-adapted per-parameter sigma)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from evotinyml.soo.algorithms import SOO_ALGORITHMS
from evotinyml.soo.es import (
    ADAPTIVE_SIGMA_ALGOS,
    EVEN_POPSIZE_ALGOS,
    EVOSAX_SOO_ALGOS,
    MEAN_OPTIMIZER_ALGOS,
    SIGMA_SCHEDULE_ALGOS,
)
from evotinyml.soo.params_opt_es import AdaOpenES


def test_ada_open_es_is_registered_consistently():
    """CLI/algo-set wiring: ada_open_es is an adaptive-sigma, even-popsize,
    mean-optimizer algo, and (unlike open_es) does *not* use a sigma schedule.
    """
    assert "ada_open_es" in SOO_ALGORITHMS
    assert "ada_open_es" in EVOSAX_SOO_ALGOS
    assert "ada_open_es" in ADAPTIVE_SIGMA_ALGOS
    assert "ada_open_es" in EVEN_POPSIZE_ALGOS
    assert "ada_open_es" in MEAN_OPTIMIZER_ALGOS
    assert "ada_open_es" not in SIGMA_SCHEDULE_ALGOS


def test_ada_open_es_requires_even_population():
    with pytest.raises(AssertionError):
        AdaOpenES(population_size=7, solution=jnp.zeros(4))


def test_ada_open_es_improves_and_shrinks_sigma_on_high_curvature_dims():
    """On an anisotropic sphere, the mean must still descend (like OpenES),
    and per-parameter sigma must separate: shrink where curvature is high
    (dim 0) and grow where curvature is low (last dim), relative to the
    isotropic middle dims.
    """
    np.random.seed(0)
    n_dims = 20
    popsize = 64

    coeffs = np.ones(n_dims, dtype=np.float32)
    coeffs[0] = 100.0
    coeffs[-1] = 0.01
    coeffs = jnp.asarray(coeffs)

    def fitness_fn(x):
        return jnp.sum(coeffs * x**2, axis=-1)

    algo = AdaOpenES(
        population_size=popsize,
        solution=jnp.zeros(n_dims),
        optimizer=optax.adam(learning_rate=0.05),
    )
    params = algo.default_params.replace(std_init=0.5, sigma_min=1e-4, sigma_max=1e3)

    key = jax.random.key(0)
    key, key_init = jax.random.split(key)
    mean0 = jnp.asarray(np.random.normal(0, 1.0, size=n_dims), dtype=jnp.float32)
    state = algo.init(key_init, mean0, params)
    f0 = float(fitness_fn(state.mean))

    for _ in range(300):
        key, key_ask, key_tell = jax.random.split(key, 3)
        population, state = algo.ask(key_ask, state, params)
        fitness = fitness_fn(population)
        state, _ = algo.tell(key_tell, population, fitness, state, params)
        std = np.asarray(state.std)
        assert np.isfinite(std).all()
        assert (std >= params.sigma_min - 1e-6).all()
        assert (std <= params.sigma_max + 1e-6).all()

    f_final = float(fitness_fn(state.mean))
    std_final = np.asarray(state.std)
    mid_mean = std_final[1:-1].mean()

    assert f_final < f0
    assert std_final[0] < mid_mean, "sigma should shrink on the high-curvature dim"
    assert std_final[-1] > mid_mean, "sigma should grow on the low-curvature dim"


def test_ada_open_es_sigma_clips_at_bounds():
    """An unbounded-reward fitness (favors ever-larger |x0|) must drive
    sigma[0] up against ``sigma_max`` rather than diverging.
    """
    n_dims = 4
    popsize = 32

    def fitness_fn(x):
        # evosax minimizes: reward large |x0| with very negative fitness.
        return -(x[..., 0] ** 2)

    algo = AdaOpenES(population_size=popsize, solution=jnp.zeros(n_dims))
    params = algo.default_params.replace(
        std_init=0.1, sigma_lr_init=1.0, sigma_min=1e-3, sigma_max=1.0
    )

    key = jax.random.key(1)
    key, key_init = jax.random.split(key)
    state = algo.init(key_init, jnp.zeros(n_dims), params)

    for _ in range(50):
        key, key_ask, key_tell = jax.random.split(key, 3)
        population, state = algo.ask(key_ask, state, params)
        fitness = fitness_fn(population)
        state, _ = algo.tell(key_tell, population, fitness, state, params)

    std = np.asarray(state.std)
    assert np.isfinite(std).all()
    assert std[0] == pytest.approx(params.sigma_max, rel=1e-3)
    assert (std <= params.sigma_max + 1e-6).all()
