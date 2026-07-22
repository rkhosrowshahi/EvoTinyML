"""Numerically stable ASEBO (evosax upstream is brittle).

Upstream issues (evosax ``ASEBO``):
- ``U_ort = Vt[pop/2:]`` is empty whenever ``subspace_dims < pop/2``, so
  ``alpha → 0`` and the covariance collapses to a rank-``k`` matrix; Cholesky
  then returns NaNs.
- ``alpha = ‖g U⊥‖ / ‖g U‖`` has no epsilon / clamp, so ``0/0`` or ``α>1``
  makes the covariance indefinite.
- Unit-normalizing samples after the Cholesky cancels the ``σ`` scale.

This subclass keeps the same Optax mean update / FIFO gradient archive, but
uses the orthogonal projector ``I − UUᵀ``, clamps ``α∈[0,1]``, and adds
covariance jitter.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from evosax.algorithms.distribution_based.asebo import ASEBO as _ASEBO
from evosax.core.fitness_shaping import identity_fitness_shaping_fn
from evosax.types import Fitness, Population, Solution

from evosax.algorithms.distribution_based.base import metrics_fn


class StableASEBO(_ASEBO):
    """ASEBO with stable covariance / α updates."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        subspace_dims: int = 1,
        optimizer: optax.GradientTransformation = optax.adam(learning_rate=1e-3),
        std_schedule: Callable = optax.constant_schedule(1.0),
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
        cov_jitter: float = 1e-6,
        alpha_eps: float = 1e-8,
    ):
        super().__init__(
            population_size,
            solution,
            subspace_dims=subspace_dims,
            optimizer=optimizer,
            std_schedule=std_schedule,
            fitness_shaping_fn=fitness_shaping_fn,
            metrics_fn=metrics_fn,
        )
        self.cov_jitter = float(cov_jitter)
        self.alpha_eps = float(alpha_eps)

    def _ask(
        self,
        key: jax.Array,
        state,
        params,
    ) -> tuple[Population, object]:
        X = state.grad_subspace
        X = X - jnp.mean(X, axis=0, keepdims=True)
        _U, _S, Vt = jnp.linalg.svd(X, full_matrices=False)

        # Right singular vectors span the gradient archive (rank ≤ subspace_dims).
        k = self.subspace_dims
        U = Vt[:k]
        UUT = U.T @ U
        UUT_ort = jnp.eye(self.num_dims, dtype=UUT.dtype) - UUT

        subspace_ready = state.generation_counter > self.subspace_dims
        UUT_use = jax.lax.select(
            subspace_ready, UUT, jnp.zeros((self.num_dims, self.num_dims), dtype=UUT.dtype)
        )

        alpha = jnp.clip(state.alpha, 0.0, 1.0)
        # When the subspace is not ready, force isotropic sampling.
        alpha = jax.lax.select(subspace_ready, alpha, jnp.asarray(1.0, dtype=alpha.dtype))

        cov = (
            state.std * (alpha / self.num_dims) * jnp.eye(self.num_dims)
            + ((1.0 - alpha) / max(k, 1)) * UUT_use
            + self.cov_jitter * jnp.eye(self.num_dims)
        )
        chol = jnp.linalg.cholesky(cov)

        z_plus = jax.random.normal(key, (self.population_size // 2, self.num_dims))
        z_plus = z_plus @ chol.T
        z = jnp.concatenate([z_plus, -z_plus], axis=0)
        population = state.mean + z
        return population, state.replace(UUT=UUT_use, UUT_ort=UUT_ort)

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state,
        params,
    ):
        half = self.population_size // 2
        fitness_plus = fitness[:half]
        fitness_minus = fitness[half:]
        grad = 0.5 * jnp.dot(
            fitness_plus - fitness_minus,
            (population[:half] - state.mean) / state.std,
        )

        num = jnp.linalg.norm(jnp.dot(grad, state.UUT_ort))
        den = jnp.linalg.norm(jnp.dot(grad, state.UUT))
        alpha = num / (den + self.alpha_eps)
        alpha = jnp.clip(alpha, 0.0, 1.0)
        subspace_ready = state.generation_counter > self.subspace_dims
        alpha = jax.lax.select(subspace_ready, alpha, jnp.asarray(1.0, dtype=alpha.dtype))

        grad_subspace = jnp.roll(state.grad_subspace, shift=-1, axis=0)
        grad_subspace = grad_subspace.at[-1, :].set(grad)

        grad = grad / (jnp.linalg.norm(grad) / self.num_dims + 1e-8)

        updates, opt_state = self.optimizer.update(
            grad, state.opt_state, state.mean
        )
        mean = optax.apply_updates(state.mean, updates)

        return state.replace(
            mean=mean,
            std=self.std_schedule(state.generation_counter),
            opt_state=opt_state,
            grad_subspace=grad_subspace,
            alpha=alpha,
        )
