"""Multi-objective OpenES variants over vector-valued problems (e.g. CWRM-CE).

Search math (antithetic sampling, fitness shaping, MGDA / UPGrad,
Tchebycheff / weighted-sum, Optax mean update) is JAX + evosax; fitness
evaluation stays PyTorch; the non-dominated archive stays NumPy/pymoo.

- ``mgda_open_es`` / ``upgrad_open_es`` (Design A): a single mean. Per-objective
  ES gradients are estimated from the *same* population sample, then combined
  into one descent direction by either MGDA (min-norm in the convex hull;
  Desideri 2012 / Sener & Koltun) or UPGrad (unconflicting projection onto the
  dual cone of the Jacobian rows, then average; Jacobian Descent,
  arXiv:2406.16232). Converges toward a Pareto-stationary compromise; all
  candidates feed a non-dominated archive.

- ``moead_open_es`` (Design B): K means, each paired with a weight vector on
  the objective simplex (MOEA/D decomposition; Zhang & Li 2007). Each mean
  runs OpenES on its augmented-Tchebycheff scalarization with a shared,
  online-updated ideal point. The K means plus the shared archive
  approximate the Pareto front.

The archive is pruned to ``archive_size`` with either NSGA-II rank+crowding
or NSGA-III reference-direction survival (``archive_selection``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pymoo.core.population import Population
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from torch.utils.data import DataLoader

from evotinyml.moo.algorithms import make_reference_directions
from evotinyml.moo.display import _hv_unit_front, _normalize_by_nadir
from evotinyml.moo.pareto_plot import pareto_front_images, val_class_pareto_images
from evotinyml.fitness import (
    L1_HV_REF,
    l1_problem_metric_log_dict,
    l1_val_front,
    mean_abs_weights,
)
from evotinyml.problem import (
    CE_PROBLEMS,
    L1_PROBLEMS,
    WeightOptimizationProblem,
    problem_obj_labels,
)
from evotinyml.soo.es import (
    DEFAULT_ES_OPTIM,
    DEFAULT_ES_OPTIM_LR,
    DEFAULT_ES_OPTIM_MOMENTUM,
    DEFAULT_ES_OPTIM_SCHEDULER,
    DEFAULT_ES_OPTIM_WD,
    DEFAULT_ES_SIGMA_SCHEDULER,
    _init_mean,
    build_es_sigma_schedule,
    build_open_es_optimizer,
    default_soo_popsize,
    sigma_at,
)
from evotinyml.validation import (
    class_metric_log_dict,
    evaluate_weights_on_loader,
    per_class_acc_ce_on_eval_pool,
    per_class_acc_ce_on_loader,
    validate_nd_set,
)
from evotinyml.wandb_logger import log_metrics

MO_ES_ALGORITHMS = ("mgda_open_es", "upgrad_open_es", "moead_open_es")
# Design-A aggregators: combine per-objective ES grads into one mean update.
MO_AGGREGATORS = ("mgda", "upgrad")
DEFAULT_MO_AGGREGATOR = "mgda"
ARCHIVE_SELECTIONS = ("nsga2", "nsga3")
# Per-objective ES-gradient rescaling before MGDA / UPGrad (Sener & Koltun style).
# ``l2``: g <- g/||g||. ``loss`` / ``loss+`` need current objective values and
# are unsafe when an objective is near 0 (e.g. L1 with --init zeros).
# ``range``: MGDA weights from l2-normalized grads are further multiplied by
# each objective's range-normalized distance to its ideal (F_c / running
# nadir_c, ideal = 0). Rank shaping makes ES gradients scale-invariant, so
# this re-injects the magnitude information ranks destroy: a converged
# objective (e.g. L1 = 0 at --init zeros) loses its pull instead of vetoing
# progress on the others.
MGDA_NORMALIZATIONS = ("none", "l2", "loss", "loss+", "range")
DEFAULT_MGDA_NORMALIZE = "none"
# Per-objective ES fitness shaping for the MGDA gradient estimate.
# ``centered_ranks``: OpenES default; robust to outliers but stretches any
# ordering (even pure batch noise) to full [-0.5, 0.5] spread and discards
# within-population magnitudes, so a noise-dominated objective yields a
# gradient with the same norm as an informative one.
# ``z_score``: (F - mean) / std per objective; keeps relative magnitudes
# within the population (a 3-sigma-better candidate contributes 3x) while
# staying scale-normalized across objectives.
# ``raw``: F - mean; the plain ES gradient estimate with true per-objective
# scale (a flat/noisy objective yields a small-norm gradient), but sensitive
# to outliers (e.g. one exploded CE candidate).
ES_FITNESS_SHAPINGS = ("centered_ranks", "z_score", "raw")
DEFAULT_ES_FITNESS_SHAPING = "centered_ranks"

DEFAULT_ARCHIVE_SIZE = 100
DEFAULT_MOEAD_K = 10
DEFAULT_MOEAD_RHO = 1e-4
# Shrink weight vectors toward the centroid: lam <- (1-a)*lam + a/n_obj.
# At the Tchebycheff optimum with ideal z=0 the objectives satisfy
# F_c ~ 1/lam_c, so a simplex corner asks for a class-specialist network
# that sacrifices every other class. Shrink 0.7 caps the implied loss ratio
# between classes at roughly 5:1 for n_obj=10 while keeping the spread.
DEFAULT_MOEAD_WEIGHT_SHRINK = 0.7
# Ideal-point mode for the Tchebycheff scalarization. All problem objectives
# (class-wise CE, 1-P, 1-R) are lower-bounded by 0, so "zero" pins z* = 0 and
# keeps the scalarization stationary. "adaptive" tracks the running minimum
# per objective (classic MOEA/D), but with class-wise CE a degenerate
# always-predict-class-c candidate drags z*_c to ~0 anyway while destabilizing
# the ranking in the meantime.
MOEAD_IDEAL_MODES = ("zero", "adaptive")
DEFAULT_MOEAD_IDEAL = "zero"
# Scalarization for the per-mean subproblems. Tchebycheff (max-based) is the
# classic MOEA/D choice and covers non-convex fronts, but with many
# objectives its rank signal flows through a single argmax objective per
# candidate, which is weak for ES gradient estimation. Weighted sum gives a
# dense signal on every objective (only reaches convex front regions).
MOEAD_SCALARIZATIONS = ("tchebycheff", "weighted_sum")
DEFAULT_MOEAD_SCALARIZATION = "tchebycheff"
# Objective values >= this are treated as sentinels (class absent from pool).
_SENTINEL_F = 1e5


# ---------------------------------------------------------------------------
# Shared ES / MO math (JAX + evosax); NumPy wrappers for archive / callers
# ---------------------------------------------------------------------------


def _jnp():
    import jax.numpy as jnp

    return jnp


def _centered_ranks(values):
    """Centered ranks in [-0.5, 0.5] via evosax (average ties). Returns JAX array."""
    from evosax.core.fitness_shaping import centered_rank_fitness_shaping_fn

    jnp = _jnp()
    x = jnp.asarray(values).ravel()
    n = int(x.shape[0])
    if n <= 1:
        return jnp.zeros((n,))
    return centered_rank_fitness_shaping_fn(None, x, None, None)


def _shape_objective(values, mode: str):
    """Map one objective / scalar fitness column to ES utilities (JAX)."""
    from evosax.core.fitness_shaping import (
        centered_rank_fitness_shaping_fn,
        standardize_fitness_shaping_fn,
    )

    jnp = _jnp()
    x = jnp.asarray(values).ravel()
    n = int(x.shape[0])
    if n == 0:
        return x
    if mode == "centered_ranks":
        if n <= 1:
            return jnp.zeros((n,))
        return centered_rank_fitness_shaping_fn(None, x, None, None)
    if mode == "z_score":
        if n <= 1:
            return jnp.zeros((n,))
        return standardize_fitness_shaping_fn(None, x, None, None)
    # raw: centered fitness (keeps per-objective scale)
    return x - jnp.mean(x)


def _shape_objectives(F, mode: str):
    """Shape each column of ``F`` (n_pop, n_obj) -> utilities (n_pop, n_obj)."""
    jnp = _jnp()
    F = jnp.asarray(F)
    cols = [_shape_objective(F[:, c], mode) for c in range(int(F.shape[1]))]
    return jnp.column_stack(cols)


def _antithetic_noise(key, half: int, n_var: int):
    """Split ``key``; return ``(new_key, eps)`` with antithetic Gaussian rows."""
    import jax

    jnp = _jnp()
    key, sub = jax.random.split(key)
    eps = jax.random.normal(sub, (half, n_var))
    return key, jnp.concatenate([eps, -eps], axis=0)


def _es_gradient(utilities, eps_eff, sigma: float):
    """OpenES gradient ``(u @ eps_eff) / (n σ)``; ``utilities`` shape (n,)."""
    jnp = _jnp()
    u = jnp.asarray(utilities).ravel()
    n = u.shape[0]
    return (u @ eps_eff) / (n * sigma)


def _es_gradients_multi(U, eps_eff, sigma: float):
    """Per-objective OpenES grads; ``U`` is (n_pop, n_obj) -> ``G`` (n_obj, n_var)."""
    jnp = _jnp()
    U = jnp.asarray(U)
    n = U.shape[0]
    return (U.T @ eps_eff) / (n * sigma)


def _evaluate_F(problem: WeightOptimizationProblem, X: np.ndarray) -> np.ndarray:
    """Vector objectives for each row of ``X`` via the problem's evaluator."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    F = np.empty((X.shape[0], problem.n_obj), dtype=np.float64)
    for i in range(X.shape[0]):
        F[i] = problem._evaluate_individual(X[i])
    if np.any(F >= _SENTINEL_F):
        raise RuntimeError(
            "Sentinel objective value detected (a class is absent from the eval "
            "pool). Rank shaping / Tchebycheff would be corrupted; use a "
            "class-balanced sampler or a larger --batch-size."
        )
    return F


class _MeanOptimizer:
    """Optax-backed OpenES mean update with box-bound clipping."""

    def __init__(
        self,
        mean0: np.ndarray,
        *,
        xl: float,
        xu: float,
        es_optim: str,
        es_optim_lr: float,
        es_optim_scheduler: str,
        es_optim_momentum: float,
        es_optim_wd: float,
        steps: int,
    ) -> None:
        import jax.numpy as jnp

        self._jnp = jnp
        self.xl = float(xl)
        self.xu = float(xu)
        self._optimizer = build_open_es_optimizer(
            es_optim,
            es_optim_lr,
            es_optim_scheduler,
            steps=steps,
            momentum=es_optim_momentum,
            weight_decay=es_optim_wd,
        )
        self.mean = np.clip(np.asarray(mean0, dtype=np.float64), self.xl, self.xu)
        self._opt_state = self._optimizer.init(jnp.asarray(self.mean, dtype=jnp.float32))

    def step(self, grad) -> np.ndarray:
        """Apply one Optax update along ``grad`` (ascent on grad -> we pass +grad
        of the loss so Optax's negative step descends). Returns the new mean."""
        import optax

        jnp = self._jnp
        params = jnp.asarray(self.mean, dtype=jnp.float32)
        updates, self._opt_state = self._optimizer.update(
            jnp.asarray(grad, dtype=jnp.float32), self._opt_state, params
        )
        new_mean = optax.apply_updates(params, updates)
        self.mean = np.clip(np.asarray(new_mean, dtype=np.float64), self.xl, self.xu)
        return self.mean

    def reset_at(self, mean: np.ndarray) -> None:
        """Move the mean and reset optimizer state (used by MOEA/D migration)."""
        jnp = self._jnp
        self.mean = np.clip(np.asarray(mean, dtype=np.float64), self.xl, self.xu)
        self._opt_state = self._optimizer.init(jnp.asarray(self.mean, dtype=jnp.float32))


# ---------------------------------------------------------------------------
# MGDA (min-norm point in the convex hull of gradients) — JAX
# ---------------------------------------------------------------------------


def _normalize_mgda_gradients(
    G,
    *,
    mode: str = DEFAULT_MGDA_NORMALIZE,
    losses=None,
    eps: float = 1e-8,
):
    """Rescale per-objective gradient rows before MGDA (JAX arrays)."""
    jnp = _jnp()
    mode = mode.lower()
    if mode not in MGDA_NORMALIZATIONS:
        raise ValueError(
            f"Unknown MGDA normalize mode: {mode!r}. Use one of {MGDA_NORMALIZATIONS}."
        )
    G = jnp.asarray(G)
    if mode == "none":
        return G

    norms = jnp.linalg.norm(G, axis=1)
    if mode in ("loss", "loss+"):
        if losses is None:
            raise ValueError(f"MGDA normalize mode {mode!r} requires losses.")
        L = jnp.maximum(jnp.asarray(losses).ravel(), eps)
        if int(L.shape[0]) != int(G.shape[0]):
            raise ValueError(
                f"losses length {int(L.shape[0])} != n_obj={int(G.shape[0])} "
                "for MGDA normalize."
            )
    else:
        L = None

    scales = jnp.ones(G.shape[0])
    if mode in ("l2", "loss+"):
        scales = scales * jnp.maximum(norms, eps)
    if mode in ("loss", "loss+"):
        scales = scales * L
    return G / scales[:, None]


def _mgda_weights(G, max_iter: int = 250, tol: float = 1e-12):
    """Frank-Wolfe solve of ``min_w ||w^T G||^2`` over the simplex (JAX)."""
    jnp = _jnp()
    G = jnp.asarray(G)
    M = G @ G.T
    n_obj = int(M.shape[0])
    w = jnp.full((n_obj,), 1.0 / n_obj)
    for _ in range(max_iter):
        t = int(jnp.argmin(M @ w))
        e_t = jnp.zeros((n_obj,)).at[t].set(1.0)
        diff = e_t - w
        denom = diff @ M @ diff
        if float(denom) <= tol:
            break
        gamma = jnp.clip(-(w @ M @ diff) / denom, 0.0, 1.0)
        if float(gamma) <= tol:
            break
        w = w + gamma * diff
    return w


def normalize_mgda_gradients(
    G: np.ndarray,
    *,
    mode: str = DEFAULT_MGDA_NORMALIZE,
    losses: np.ndarray | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Rescale per-objective gradient rows before MGDA (Sener & Koltun).

    ``G`` is ``(n_obj, n_var)``. Modes match their ``gradient_normalizers``:
    ``none``, ``l2`` (``g/||g||``), ``loss`` (``g/L``), ``loss+`` (``g/(L·||g||)``).
    """
    return np.asarray(
        _normalize_mgda_gradients(G, mode=mode, losses=losses, eps=eps),
        dtype=np.float64,
    )


def mgda_weights(G: np.ndarray, max_iter: int = 250, tol: float = 1e-12) -> np.ndarray:
    """Frank-Wolfe solve of ``min_w ||w^T G||^2`` over the simplex.

    ``G`` is (n_obj, n_var). Returns convex weights ``w`` (n_obj,).
    """
    return np.asarray(_mgda_weights(G, max_iter=max_iter, tol=tol), dtype=np.float64)


# ---------------------------------------------------------------------------
# UPGrad (unconflicting projection onto the dual cone) — NumPy / SciPy
# ---------------------------------------------------------------------------


def _dual_cone_qp_weights(M: np.ndarray, u: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Solve ``min_v v^T M v`` s.t. ``v >= u`` (UPGrad dual, Prop. 1)."""
    from scipy.optimize import minimize

    m = int(M.shape[0])
    M = 0.5 * (M + M.T) + eps * np.eye(m)
    u = np.asarray(u, dtype=np.float64).ravel()

    def fun(v: np.ndarray) -> float:
        return float(v @ M @ v)

    def jac(v: np.ndarray) -> np.ndarray:
        return 2.0 * (M @ v)

    bounds = [(float(u[i]), None) for i in range(m)]
    v0 = np.maximum(u, 0.0)
    # Feasible start slightly above the bound helps L-BFGS-B.
    v0 = v0 + 0.1 * (np.abs(v0) + 1.0)
    res = minimize(
        fun,
        v0,
        jac=jac,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 500},
    )
    return np.asarray(res.x, dtype=np.float64)


def _upgrad_aggregate(G, eps: float = 1e-10):
    """UPGrad aggregator (Jacobian Descent): project each row onto the dual cone, average.

    ``G`` is ``(n_obj, n_var)``. Returns ``(direction, w_bar)`` where
    ``direction = w_bar @ G`` lies in the dual cone of the rows of ``G``
    (``G @ direction >= 0`` up to numerics), and ``w_bar`` is the average of
    the dual QP solutions for ``u = e_i``.
    """
    G_np = np.asarray(G, dtype=np.float64)
    m = int(G_np.shape[0])
    if m == 0:
        raise ValueError("UPGrad needs at least one objective gradient row.")
    M = G_np @ G_np.T
    w_sum = np.zeros(m, dtype=np.float64)
    for i in range(m):
        u = np.zeros(m, dtype=np.float64)
        u[i] = 1.0
        w_sum += _dual_cone_qp_weights(M, u, eps=eps)
    w_bar = w_sum / m
    direction = w_bar @ G_np
    return direction, w_bar


def upgrad_aggregate(G: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """UPGrad direction for per-objective gradients ``G`` (n_obj, n_var)."""
    direction, _ = _upgrad_aggregate(G, eps=eps)
    return np.asarray(direction, dtype=np.float64)


def aggregate_mo_gradients(
    G,
    *,
    aggregator: str = DEFAULT_MO_AGGREGATOR,
    normalize: str = DEFAULT_MGDA_NORMALIZE,
    losses=None,
):
    """Combine per-objective ES grads into one descent direction.

    Returns ``(direction, weights_or_dual, info)``.
    For ``mgda``, ``weights`` are simplex weights and ``direction = w @ G_raw``
    (weights from optionally normalized grads). For ``upgrad``, ``direction`` is
    the UPGrad aggregate of the (optionally normalized) Jacobian and
    ``weights`` are the averaged dual QP coefficients.
    """
    jnp = _jnp()
    aggregator = aggregator.lower()
    if aggregator not in MO_AGGREGATORS:
        raise ValueError(
            f"Unknown mo aggregator: {aggregator!r}. Use one of {MO_AGGREGATORS}."
        )
    normalize = normalize.lower()
    G_raw = jnp.asarray(G)
    info: dict[str, float] = {}

    if aggregator == "mgda":
        if normalize == "range":
            w = _mgda_weights(_normalize_mgda_gradients(G_raw, mode="l2"))
            # range modulation applied by caller (needs running nadir / mean_F)
            direction = w @ G_raw
            return direction, w, info
        G_mgda = _normalize_mgda_gradients(G_raw, mode=normalize, losses=losses)
        w = _mgda_weights(G_mgda)
        direction = w @ G_raw
        return direction, w, info

    # UPGrad: ``range`` falls back to l2 geometry (no simplex weights to rescale).
    norm_mode = "l2" if normalize == "range" else normalize
    if normalize == "range":
        info["upgrad_range_fallback"] = 1.0
    G_use = _normalize_mgda_gradients(G_raw, mode=norm_mode, losses=losses)
    direction_np, w_bar = _upgrad_aggregate(np.asarray(G_use))
    return jnp.asarray(direction_np), jnp.asarray(w_bar), info


# ---------------------------------------------------------------------------
# Tchebycheff scalarization (MOEA/D) — JAX core, NumPy wrappers
# ---------------------------------------------------------------------------


def moead_weight_vectors(
    n_obj: int, k: int, method: str = "energy", seed: int = 1
) -> np.ndarray:
    """``k`` well-spread weight vectors on the objective simplex.

    pymoo's "energy" generator fails for ``n_points < n_obj``; in that case we
    over-generate and greedily subselect ``k`` maximally spread vectors,
    seeding with the one closest to the uniform weighting.
    """
    n_points = max(int(k), n_obj + 2)
    W = make_reference_directions(n_obj=n_obj, pop_size=n_points, method=method, seed=seed)
    if len(W) <= k:
        return np.asarray(W, dtype=np.float64)

    W = np.asarray(W, dtype=np.float64)
    center = np.full(n_obj, 1.0 / n_obj)
    chosen = [int(np.argmin(np.linalg.norm(W - center, axis=1)))]
    for _ in range(k - 1):
        dists = np.min(
            np.linalg.norm(W[:, None, :] - W[chosen][None, :, :], axis=2), axis=1
        )
        dists[chosen] = -np.inf
        chosen.append(int(np.argmax(dists)))
    return W[chosen]


def _tchebycheff(F, lam, ideal, rho: float = DEFAULT_MOEAD_RHO):
    """Augmented Tchebycheff (JAX)."""
    jnp = _jnp()
    diff = jnp.asarray(F) - jnp.asarray(ideal)
    if diff.ndim == 1:
        diff = diff[None, :]
    weighted = jnp.asarray(lam) * diff
    return weighted.max(axis=1) + rho * weighted.sum(axis=1)


def _weighted_sum(F, lam, ideal):
    """Weighted-sum scalarization (JAX)."""
    jnp = _jnp()
    diff = jnp.asarray(F) - jnp.asarray(ideal)
    if diff.ndim == 1:
        diff = diff[None, :]
    return (jnp.asarray(lam) * diff).sum(axis=1)


def _scalarize(
    F,
    lam,
    ideal,
    *,
    method: str = DEFAULT_MOEAD_SCALARIZATION,
    rho: float = DEFAULT_MOEAD_RHO,
):
    if method == "weighted_sum":
        return _weighted_sum(F, lam, ideal)
    return _tchebycheff(F, lam, ideal, rho)


def tchebycheff(
    F: np.ndarray, lam: np.ndarray, ideal: np.ndarray, rho: float = DEFAULT_MOEAD_RHO
) -> np.ndarray:
    """Augmented Tchebycheff ``max_c lam_c (F_c - z_c) + rho * sum_c lam_c (F_c - z_c)``."""
    return np.asarray(_tchebycheff(F, lam, ideal, rho), dtype=np.float64)


def weighted_sum(F: np.ndarray, lam: np.ndarray, ideal: np.ndarray) -> np.ndarray:
    """Weighted-sum scalarization ``sum_c lam_c (F_c - z_c)``."""
    return np.asarray(_weighted_sum(F, lam, ideal), dtype=np.float64)


def scalarize(
    F: np.ndarray,
    lam: np.ndarray,
    ideal: np.ndarray,
    *,
    method: str = DEFAULT_MOEAD_SCALARIZATION,
    rho: float = DEFAULT_MOEAD_RHO,
) -> np.ndarray:
    return np.asarray(
        _scalarize(F, lam, ideal, method=method, rho=rho), dtype=np.float64
    )


# ---------------------------------------------------------------------------
# Non-dominated archive with NSGA2 / NSGA3 survival pruning
# ---------------------------------------------------------------------------


class NDArchive:
    """Keep the non-dominated set of all evaluated candidates, capped in size."""

    def __init__(
        self,
        problem: WeightOptimizationProblem,
        *,
        max_size: int = DEFAULT_ARCHIVE_SIZE,
        selection: str = "nsga2",
        ref_dirs_method: str = "energy",
        seed: int = 1,
    ) -> None:
        selection = selection.lower()
        if selection not in ARCHIVE_SELECTIONS:
            raise ValueError(
                f"Unknown archive selection: {selection!r}. Use one of {ARCHIVE_SELECTIONS}."
            )
        self.problem = problem
        self.max_size = int(max_size)
        self.selection = selection
        if selection == "nsga3":
            from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival

            ref_dirs = make_reference_directions(
                n_obj=problem.n_obj,
                pop_size=self.max_size,
                method=ref_dirs_method,
                seed=seed,
            )
            self._survival = ReferenceDirectionSurvival(ref_dirs)
        else:
            from pymoo.operators.survival.rank_and_crowding import RankAndCrowding

            self._survival = RankAndCrowding()

        self.X = np.zeros((0, problem.n_var), dtype=np.float64)
        self.F = np.zeros((0, problem.n_obj), dtype=np.float64)

    def __len__(self) -> int:
        return len(self.X)

    def update(self, X: np.ndarray, F: np.ndarray) -> None:
        X_all = np.vstack([self.X, np.asarray(X, dtype=np.float64)])
        F_all = np.vstack([self.F, np.asarray(F, dtype=np.float64)])
        nd_idx = NonDominatedSorting().do(F_all, only_non_dominated_front=True)
        X_nd, F_nd = X_all[nd_idx], F_all[nd_idx]
        if len(X_nd) > self.max_size:
            pop = Population.new(X=X_nd, F=F_nd)
            kept = self._survival.do(self.problem, pop, n_survive=self.max_size)
            X_nd = np.asarray(kept.get("X"), dtype=np.float64)
            F_nd = np.asarray(kept.get("F"), dtype=np.float64)
        self.X, self.F = X_nd, F_nd

    def best_scalarized(
        self,
        lam: np.ndarray,
        ideal: np.ndarray,
        rho: float,
        method: str = DEFAULT_MOEAD_SCALARIZATION,
    ) -> tuple[int, float]:
        s = scalarize(self.F, lam, ideal, method=method, rho=rho)
        i = int(np.argmin(s))
        return i, float(s[i])


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class MOESResult:
    """Final state of a multi-objective OpenES run."""

    X: np.ndarray  # archive weights (n_nd, n_var)
    F: np.ndarray  # archive objectives (n_nd, n_obj)
    means: np.ndarray  # distribution centers (K, n_var); K=1 for MGDA
    means_F: np.ndarray  # objectives at the centers (K, n_obj)
    steps: int
    popsize: int  # evaluations per generation (total across means)
    n_eval: int
    algo: str
    weights: np.ndarray | None = None  # MOEA/D weight vectors (K, n_obj)
    details: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared logging / validation
# ---------------------------------------------------------------------------


class _RunLogger:
    """Console + W&B logging and periodic test validation of the ND archive."""

    def __init__(
        self,
        problem: WeightOptimizationProblem,
        archive: NDArchive,
        *,
        test_loader: DataLoader | None,
        val_every: int,
        n_classes: int,
        use_wandb: bool,
        verbose: bool,
        pareto_every: int = 100,
        hv_samples: int = 10_000,
        hv_seed: int = 1,
    ) -> None:
        self.problem = problem
        self.archive = archive
        self.test_loader = test_loader
        self.val_every = max(1, int(val_every))
        self.pareto_every = max(1, int(pareto_every))
        self.n_classes = n_classes
        self.use_wandb = use_wandb
        self.verbose = verbose
        self.hv_samples = hv_samples
        self.hv_seed = hv_seed
        self.nadir: np.ndarray | None = None
        self.last_val: dict[str, float] = {}

    def ensure_hv_anchors(self, F: np.ndarray) -> None:
        if self.nadir is not None:
            return
        problem_name = getattr(self.problem, "problem_name", "")
        if problem_name in L1_PROBLEMS:
            self.nadir = L1_HV_REF.copy()
            print("HV: L1 problems use fixed ref = [1.0, 1.0] (no nadir scaling)")
            return
        self.problem.set_hv_anchors_from_F(F)
        self.nadir = np.asarray(self.problem.hv_nadir, dtype=np.float64)
        print(
            "HV: F /= initial-population nadir, ref = ones "
            f"(n_obj={self.problem.n_obj}); nadir="
            + np.array2string(self.nadir, precision=4, separator=", ")
        )

    def archive_hv(self) -> float:
        if self.nadir is None or len(self.archive) == 0:
            return 0.0
        F_use = np.asarray(self.archive.F, dtype=float)
        problem_name = getattr(self.problem, "problem_name", "")
        if problem_name not in L1_PROBLEMS:
            F_use = _normalize_by_nadir(F_use, self.nadir)
        return _hv_unit_front(
            F_use,
            n_obj=self.problem.n_obj,
            hv_exact=None,
            hv_samples=self.hv_samples,
            hv_seed=self.hv_seed,
        )

    def log_train(
        self,
        *,
        step: int,
        n_eval: int,
        extra: dict[str, float],
        label: str,
        report_x: np.ndarray | None = None,
        report_F: np.ndarray | None = None,
    ) -> None:
        hv = self.archive_hv()
        per_ind = self.archive.F.mean(axis=1) if len(self.archive) else np.zeros(1)
        metrics: dict[str, Any] = {
            "train/step": int(step),
            "train/n_nds": len(self.archive),
            "train/hv": float(hv),
            "train/mean_f": float(per_ind.mean()),
            "train/min_f": float(per_ind.min()),
            "train/f": float(per_ind.min()),
            **{f"train/{k}": float(v) for k, v in extra.items()},
        }
        if getattr(self.problem, "problem_name", "") in CE_PROBLEMS:
            x = report_x
            if x is None and len(self.archive):
                # Fallback when no center is passed (should be rare).
                knee_i = int(np.argmin(np.sum(np.square(self.archive.F), axis=1)))
                x = self.archive.X[knee_i]
            if x is not None:
                acc_c, ce_c = per_class_acc_ce_on_eval_pool(self.problem, x)
                metrics.update(class_metric_log_dict("train", acc_c, ce_c))
                # Prefer center scalar fitness when reporting the mean.
                if report_x is not None and "center_f" in extra:
                    metrics["train/f"] = float(extra["center_f"])
                    metrics["train/mean_f"] = float(extra["center_f"])
        problem_name = getattr(self.problem, "problem_name", "")
        if problem_name in L1_PROBLEMS:
            F_rep = report_F
            if F_rep is None and len(self.archive):
                knee_i = int(np.argmin(np.sum(np.square(self.archive.F), axis=1)))
                F_rep = self.archive.F[knee_i]
            if F_rep is not None:
                metrics.update(l1_problem_metric_log_dict("train", problem_name, F_rep))
                if report_x is not None and "center_f" in extra:
                    metrics["train/f"] = float(extra["center_f"])
                    metrics["train/mean_f"] = float(extra["center_f"])
        if self.verbose:
            extra_str = "  ".join(
                f"{k}={v:.6f}" for k, v in extra.items() if not str(k).startswith("class_")
            )
            print(
                f"[{label}] n_eval={n_eval}  n_nds={len(self.archive)}  "
                f"hv={hv:.4f}  min_f={metrics['train/min_f']:.6f}  {extra_str}"
            )
        if self.use_wandb:
            if len(self.archive) > 0 and (
                step == 0 or step == 1 or step % self.pareto_every == 0
            ):
                labels = problem_obj_labels(
                    getattr(self.problem, "problem_name", ""),
                    self.archive.F.shape[1],
                )
                highlight = None
                if report_F is not None:
                    highlight = np.asarray(report_F, dtype=float).ravel()
                metrics.update(
                    pareto_front_images(
                        self.archive.F,
                        problem_name=getattr(self.problem, "problem_name", ""),
                        step=step,
                        key_prefix="train",
                        highlight=highlight,
                        highlight_label="center",
                        obj_labels=labels,
                    )
                )
            log_metrics(metrics, n_eval=n_eval)

    def maybe_validate(
        self,
        *,
        step: int,
        n_eval: int,
        center: np.ndarray | None = None,
        force: bool = False,
    ) -> None:
        if self.test_loader is None:
            return
        if not force and step != 1 and step % self.val_every != 0:
            return
        # Need either an archive or a center to report.
        if len(self.archive) == 0 and center is None:
            return

        metrics: dict[str, Any] = {"train/step": int(step)}
        msg_parts: list[str] = [f"[step {step}]"]

        problem_name = getattr(self.problem, "problem_name", "")
        is_l1 = problem_name in L1_PROBLEMS
        result = None
        knee_i = 0

        if len(self.archive) > 0:
            result = validate_nd_set(
                self.problem, self.archive.X, self.test_loader, self.n_classes
            )
            metrics.update(
                {
                    "val/n_nd": len(self.archive),
                    "val/acc_best": float(result.best_acc),
                    "val/f1_best": float(result.best_f1),
                    "val/mean_acc": float(result.mean_acc),
                    "val/mean_f1": float(result.mean_f1),
                }
            )
            if is_l1:
                F_val = l1_val_front(
                    problem_name,
                    self.archive.X,
                    mean_ce=result.mean_ce,
                    macro_f1=result.macro_f1,
                )
                hv_l1 = _hv_unit_front(
                    F_val,
                    n_obj=2,
                    hv_exact=None,
                    hv_samples=self.hv_samples,
                    hv_seed=self.hv_seed,
                )
                knee_i = int(np.argmin(np.sum(np.square(F_val), axis=1)))
                metrics["val/hv"] = float(hv_l1)
                metrics["val/knee_acc"] = float(result.overall_acc[knee_i])
                metrics["val/knee_f1"] = float(result.macro_f1[knee_i])
                metrics["val/knee_l1"] = float(F_val[knee_i, 1])
                metrics.update(
                    l1_problem_metric_log_dict("val", problem_name, F_val[knee_i])
                )
                msg_parts.append(
                    f"test ND (n={len(self.archive)}): "
                    f"acc_best={result.best_acc:.4f}  f1_best={result.best_f1:.4f}  "
                    f"knee_acc={metrics['val/knee_acc']:.4f}  "
                    f"knee_l1={metrics['val/knee_l1']:.6f}  "
                    f"mean_acc={result.mean_acc:.4f}  HV={hv_l1:.6f}"
                )
                if self.use_wandb and (
                    force or step == 1 or step % self.pareto_every == 0
                ):
                    labels = problem_obj_labels(problem_name, 2)
                    metrics.update(
                        pareto_front_images(
                            F_val,
                            problem_name=problem_name,
                            step=step,
                            key_prefix="val",
                            highlight=np.mean(F_val, axis=0),
                            highlight_label="center",
                            obj_labels=labels,
                        )
                    )
            else:
                hv_acc = _hv_unit_front(
                    result.error_acc,
                    n_obj=self.n_classes,
                    hv_exact=None,
                    hv_samples=self.hv_samples,
                    hv_seed=self.hv_seed,
                )
                hv_f1 = _hv_unit_front(
                    result.error_f1,
                    n_obj=self.n_classes,
                    hv_exact=None,
                    hv_samples=self.hv_samples,
                    hv_seed=self.hv_seed,
                )
                metrics["val/hv_acc"] = float(hv_acc)
                metrics["val/hv_f1"] = float(hv_f1)
                # Archive knee (diagnostic); primary report prefers the center below.
                knee_i = int(np.argmin(np.sum(np.square(result.per_class_ce), axis=1)))
                metrics["val/knee_acc"] = float(result.overall_acc[knee_i])
                metrics["val/knee_f1"] = float(result.macro_f1[knee_i])
                msg_parts.append(
                    f"test ND (n={len(self.archive)}): "
                    f"acc_best={result.best_acc:.4f}  f1_best={result.best_f1:.4f}  "
                    f"knee_acc={metrics['val/knee_acc']:.4f}  "
                    f"mean_acc={result.mean_acc:.4f}  HV_acc={hv_acc:.6f}"
                )
                if self.use_wandb and (
                    force or step == 1 or step % self.pareto_every == 0
                ):
                    metrics.update(
                        val_class_pareto_images(
                            result.per_class_ce,
                            result.per_class_acc,
                            step=step,
                        )
                    )

        if center is not None:
            acc, f1, prec, rec, _, _ = evaluate_weights_on_loader(
                self.problem, center, self.test_loader, self.n_classes
            )
            # Primary reported solution = distribution center (OpenES mean).
            metrics["val/acc"] = float(acc)
            metrics["val/f1"] = float(f1)
            metrics["val/center_acc"] = float(acc)
            metrics["val/center_f1"] = float(f1)
            if problem_name in CE_PROBLEMS:
                acc_c, ce_c = per_class_acc_ce_on_loader(
                    self.problem, center, self.test_loader, self.n_classes
                )
                metrics.update(class_metric_log_dict("val", acc_c, ce_c))
            if is_l1:
                # L1 is a weight-norm objective; identical on train/val.
                center_l1 = mean_abs_weights(center)
                metrics["val/l1"] = float(center_l1)
                metrics["val/center_l1"] = float(center_l1)
                msg_parts.append(
                    f"center_acc={acc:.4f}  center_f1={f1:.4f}  "
                    f"center_l1={center_l1:.6f}"
                )
            else:
                msg_parts.append(f"center_acc={acc:.4f}  center_f1={f1:.4f}")
        elif result is not None:
            # No center: fall back to archive knee for primary val/* keys.
            metrics["val/acc"] = float(metrics["val/knee_acc"])
            metrics["val/f1"] = float(metrics["val/knee_f1"])
            if problem_name in CE_PROBLEMS:
                metrics.update(
                    class_metric_log_dict(
                        "val",
                        result.per_class_acc[knee_i],
                        result.per_class_ce[knee_i],
                    )
                )
            if is_l1:
                msg_parts.append(f"knee_l1={metrics['val/knee_l1']:.6f}")

        msg = "  ".join(msg_parts)
        print(msg)
        self.last_val = {
            k: float(v)
            for k, v in metrics.items()
            if k.startswith("val/") and isinstance(v, (int, float, np.floating))
        }
        if self.use_wandb:
            log_metrics(metrics, n_eval=n_eval)



# ---------------------------------------------------------------------------
# Design A: MGDA-OpenES
# ---------------------------------------------------------------------------


def run_mgda_open_es(
    problem: WeightOptimizationProblem,
    *,
    steps: int,
    popsize: int | None = None,
    init: str = "gaussian",
    init_sigma: float = 0.1,
    seed: int = 1,
    val_every: int = 50,
    pareto_every: int = 100,
    test_loader: DataLoader | None = None,
    n_classes: int = 10,
    verbose: bool = False,
    use_wandb: bool = True,
    es_optim: str = DEFAULT_ES_OPTIM,
    es_optim_lr: float = DEFAULT_ES_OPTIM_LR,
    es_optim_scheduler: str = DEFAULT_ES_OPTIM_SCHEDULER,
    es_optim_momentum: float = DEFAULT_ES_OPTIM_MOMENTUM,
    es_optim_wd: float = DEFAULT_ES_OPTIM_WD,
    es_sigma_scheduler: str = DEFAULT_ES_SIGMA_SCHEDULER,
    es_sigma_end: float | None = None,
    archive_size: int = DEFAULT_ARCHIVE_SIZE,
    archive_selection: str = "nsga2",
    ref_dirs_method: str = "energy",
    mgda_normalize: str = DEFAULT_MGDA_NORMALIZE,
    es_fitness_shaping: str = DEFAULT_ES_FITNESS_SHAPING,
    aggregator: str = DEFAULT_MO_AGGREGATOR,
) -> MOESResult:
    """OpenES with per-objective gradients combined by MGDA or UPGrad (Design A).

    Each generation samples one antithetic population around a single mean and
    evaluates the full objective vector per candidate. Per-objective ES
    gradients are built with ``es_fitness_shaping`` of each objective column,
    optionally rescaled (``mgda_normalize``), then combined by ``aggregator``
    (``mgda`` or ``upgrad``) into a common descent direction for the Optax
    OpenES mean update. All candidates feed a non-dominated archive pruned by
    NSGA-II or NSGA-III survival.
    """
    n_var = int(problem.n_var)
    n_obj = int(problem.n_obj)
    steps = int(steps)
    aggregator = aggregator.lower()
    if aggregator not in MO_AGGREGATORS:
        raise ValueError(
            f"Unknown aggregator: {aggregator!r}. Use one of {MO_AGGREGATORS}."
        )
    mgda_normalize = mgda_normalize.lower()
    if mgda_normalize not in MGDA_NORMALIZATIONS:
        raise ValueError(
            f"Unknown mgda_normalize: {mgda_normalize!r}. Use one of {MGDA_NORMALIZATIONS}."
        )
    if aggregator == "upgrad" and mgda_normalize == "range":
        print(
            "Note: --mgda-normalize range is MGDA-specific; "
            "UPGrad uses l2-normalized grads instead."
        )
    es_fitness_shaping = es_fitness_shaping.lower()
    if es_fitness_shaping not in ES_FITNESS_SHAPINGS:
        raise ValueError(
            f"Unknown es_fitness_shaping: {es_fitness_shaping!r}. "
            f"Use one of {ES_FITNESS_SHAPINGS}."
        )
    if popsize is None:
        popsize = default_soo_popsize(n_var, "open_es")
    popsize = int(popsize)
    if popsize % 2 == 1:
        print(f"Note: antithetic sampling needs even popsize; bumping {popsize} -> {popsize + 1}.")
        popsize += 1

    sigma0 = float(init_sigma)
    sigma_schedule = build_es_sigma_schedule(
        es_sigma_scheduler, sigma0, steps=steps, end=es_sigma_end
    )
    import jax
    import jax.numpy as jnp

    xl = float(np.asarray(problem.xl).reshape(-1)[0]) if problem.xl is not None else -10.0
    xu = float(np.asarray(problem.xu).reshape(-1)[0]) if problem.xu is not None else 10.0

    rng = np.random.default_rng(seed)
    key = jax.random.key(int(seed))
    mean_opt = _MeanOptimizer(
        _init_mean(
            n_var, init, sigma0, rng, theta0=getattr(problem, "theta0", None)
        ),
        xl=xl,
        xu=xu,
        es_optim=es_optim,
        es_optim_lr=es_optim_lr,
        es_optim_scheduler=es_optim_scheduler,
        es_optim_momentum=es_optim_momentum,
        es_optim_wd=es_optim_wd,
        steps=steps,
    )
    archive = NDArchive(
        problem,
        max_size=archive_size,
        selection=archive_selection,
        ref_dirs_method=ref_dirs_method,
        seed=seed,
    )
    logger = _RunLogger(
        problem,
        archive,
        test_loader=test_loader,
        val_every=val_every,
        pareto_every=pareto_every,
        n_classes=n_classes,
        use_wandb=use_wandb,
        verbose=verbose,
    )

    # Initial center evaluation (not counted, matching run_soo_es).
    # HV anchors are set from the first sampled generation, like NSGA's
    # initial-population nadir.
    mean_F = _evaluate_F(problem, mean_opt.mean)[0]
    archive.update(mean_opt.mean[None, :], mean_F[None, :])
    # Running per-objective max used as the nadir for "range" weighting.
    running_nadir = np.clip(mean_F, 0.0, None).copy()
    n_eval = 0
    init_extra: dict[str, float] = {
        "center_f": float(mean_F.mean()),
        "es_sigma": sigma0,
    }
    if getattr(problem, "problem_name", "") in L1_PROBLEMS and mean_F.size >= 2:
        init_extra["l1"] = float(mean_F[1])
        if getattr(problem, "problem_name", "") == "cross_entropy_l1":
            init_extra["ce"] = float(mean_F[0])
        else:
            init_extra["f1"] = float(1.0 - mean_F[0])
    logger.log_train(
        step=0,
        n_eval=n_eval,
        extra=init_extra,
        label="init",
        report_x=mean_opt.mean,
        report_F=mean_F,
    )
    logger.maybe_validate(step=0, n_eval=n_eval, center=mean_opt.mean, force=True)

    for gen in range(1, steps + 1):
        sigma = sigma_at(sigma_schedule, gen - 1)
        mean = jnp.asarray(mean_opt.mean)
        key, eps = _antithetic_noise(key, popsize // 2, n_var)
        X = jnp.clip(mean[None, :] + sigma * eps, xl, xu)
        problem.sample_eval_pool()
        F_np = _evaluate_F(problem, np.asarray(X))
        F = jnp.asarray(F_np)
        n_eval = gen * popsize
        logger.ensure_hv_anchors(F_np)

        # Per-objective ES gradients from shared samples (JAX + evosax shaping).
        eps_eff = (X - mean[None, :]) / sigma
        U = _shape_objectives(F, es_fitness_shaping)
        G = _es_gradients_multi(U, eps_eff, sigma)
        running_nadir = np.maximum(running_nadir, F_np.max(axis=0))
        if aggregator == "mgda" and mgda_normalize == "range":
            # Solve MGDA on unit grads (pure geometry), then modulate the
            # weights by range-normalized distance-to-ideal so an objective at
            # its ideal (scale 0) stops pulling and a far-from-ideal one
            # (scale ~1) dominates.
            w = _mgda_weights(_normalize_mgda_gradients(G, mode="l2"))
            range_scale = jnp.clip(jnp.asarray(mean_F), 0.0, None) / jnp.maximum(
                jnp.asarray(running_nadir), 1e-12
            )
            w = w * range_scale
            w_sum = float(jnp.sum(w))
            w = (
                jnp.full((n_obj,), 1.0 / n_obj)
                if w_sum <= 1e-12
                else w / w_sum
            )
            direction = w @ G
        else:
            direction, w, _ = aggregate_mo_gradients(
                G,
                aggregator=aggregator,
                normalize=mgda_normalize,
                losses=mean_F,
            )
        mean_opt.step(direction)

        mean_F = _evaluate_F(problem, mean_opt.mean)[0]
        running_nadir = np.maximum(running_nadir, mean_F)
        archive.update(
            np.vstack([np.asarray(X), mean_opt.mean[None, :]]),
            np.vstack([F_np, mean_F[None, :]]),
        )

        center_objs: dict[str, float] = {
            f"center_f{c}": float(mean_F[c]) for c in range(n_obj) if n_obj <= 4
        }
        problem_name = getattr(problem, "problem_name", "")
        if problem_name in L1_PROBLEMS and n_obj >= 2:
            # Named keys for terminal + wandb (center_f0/f1 stay as aliases).
            center_objs["l1"] = float(mean_F[1])
            if problem_name == "cross_entropy_l1":
                center_objs["ce"] = float(mean_F[0])
            else:
                center_objs["f1"] = float(1.0 - mean_F[0])
        # Alignment of combined direction with each obj grad (common-descent check).
        G_np = np.asarray(G)
        d_np = np.asarray(direction)
        d_norm = float(np.linalg.norm(d_np) + 1e-12)
        min_align = min(
            float(np.dot(G_np[c], d_np) / (np.linalg.norm(G_np[c]) * d_norm + 1e-12))
            for c in range(n_obj)
        )
        logger.log_train(
            step=gen,
            n_eval=n_eval,
            extra={
                "center_f": float(mean_F.mean()),
                **center_objs,
                "mo_dir_norm": float(d_norm),
                "mo_w_max": float(np.max(np.asarray(w))),
                "mo_w_min": float(np.min(np.asarray(w))),
                "mo_min_align": float(min_align),
                # Keep legacy keys for existing W&B panels.
                "mgda_dir_norm": float(d_norm),
                "mgda_w_max": float(np.max(np.asarray(w))),
                "mgda_w_min": float(np.min(np.asarray(w))),
                "es_sigma": float(sigma),
            },
            label=f"step {gen}",
            report_x=mean_opt.mean,
            report_F=mean_F,
        )
        logger.maybe_validate(step=gen, n_eval=n_eval, center=mean_opt.mean)

    algo_name = "upgrad_open_es" if aggregator == "upgrad" else "mgda_open_es"
    return MOESResult(
        X=archive.X,
        F=archive.F,
        means=mean_opt.mean[None, :],
        means_F=mean_F[None, :],
        steps=steps,
        popsize=popsize,
        n_eval=n_eval,
        algo=algo_name,
        details={
            "center_f": float(mean_F.mean()),
            "aggregator": aggregator,
            **logger.last_val,
        },
    )


def run_upgrad_open_es(*args, **kwargs) -> MOESResult:
    """Design A with UPGrad dual-cone aggregation (see ``run_mgda_open_es``)."""
    kwargs["aggregator"] = "upgrad"
    return run_mgda_open_es(*args, **kwargs)


# ---------------------------------------------------------------------------
# Design B: MOEA/D-OpenES
# ---------------------------------------------------------------------------


def run_moead_open_es(
    problem: WeightOptimizationProblem,
    *,
    steps: int,
    popsize: int | None = None,
    k: int = DEFAULT_MOEAD_K,
    init: str = "gaussian",
    init_sigma: float = 0.1,
    seed: int = 1,
    val_every: int = 50,
    pareto_every: int = 100,
    test_loader: DataLoader | None = None,
    n_classes: int = 10,
    verbose: bool = False,
    use_wandb: bool = True,
    es_optim: str = DEFAULT_ES_OPTIM,
    es_optim_lr: float = DEFAULT_ES_OPTIM_LR,
    es_optim_scheduler: str = DEFAULT_ES_OPTIM_SCHEDULER,
    es_optim_momentum: float = DEFAULT_ES_OPTIM_MOMENTUM,
    es_optim_wd: float = DEFAULT_ES_OPTIM_WD,
    es_sigma_scheduler: str = DEFAULT_ES_SIGMA_SCHEDULER,
    es_sigma_end: float | None = None,
    archive_size: int = DEFAULT_ARCHIVE_SIZE,
    archive_selection: str = "nsga2",
    ref_dirs_method: str = "energy",
    rho: float = DEFAULT_MOEAD_RHO,
    weight_shrink: float = DEFAULT_MOEAD_WEIGHT_SHRINK,
    ideal_mode: str = DEFAULT_MOEAD_IDEAL,
    scalarization: str = DEFAULT_MOEAD_SCALARIZATION,
    migrate_every: int = 0,
) -> MOESResult:
    """Decomposed multi-mean OpenES (Design B, MOEA/D-style).

    ``k`` means, each tied to a weight vector on the objective simplex and
    optimized by OpenES on its augmented-Tchebycheff scalarization. The ideal
    point is shared and updated online from every evaluation. ``popsize`` is
    the *total* evaluations per generation (split evenly across means, forced
    even per mean for antithetic sampling). Optional migration restarts a mean
    at the archive point that best serves its weight vector.
    """
    n_var = int(problem.n_var)
    n_obj = int(problem.n_obj)
    steps = int(steps)
    k = int(k)
    if k < 2:
        raise ValueError(f"moead_open_es needs k >= 2 means, got {k}")

    if popsize is None:
        popsize = k * 8
    per_mean = max(2, int(popsize) // k)
    if per_mean % 2 == 1:
        per_mean += 1
    total_popsize = per_mean * k
    if total_popsize != int(popsize):
        print(
            f"Note: --popsize {popsize} split over k={k} means -> "
            f"{per_mean}/mean ({total_popsize} evals per generation)."
        )

    # Weight vectors on the simplex, shrunk toward the centroid so no
    # subproblem degenerates into single-class optimization.
    weights = moead_weight_vectors(n_obj, k, method=ref_dirs_method, seed=seed)
    if len(weights) != k:
        print(f"Note: ref-dirs method {ref_dirs_method!r} produced {len(weights)} weight vectors (k={k}).")
        k = len(weights)
        total_popsize = per_mean * k
    shrink = float(np.clip(weight_shrink, 0.0, 1.0))
    lam = (1.0 - shrink) * np.asarray(weights, dtype=np.float64) + shrink / n_obj
    lam = np.maximum(lam, 1e-6)
    lam /= lam.sum(axis=1, keepdims=True)

    import jax
    import jax.numpy as jnp

    sigma0 = float(init_sigma)
    sigma_schedule = build_es_sigma_schedule(
        es_sigma_scheduler, sigma0, steps=steps, end=es_sigma_end
    )
    xl = float(np.asarray(problem.xl).reshape(-1)[0]) if problem.xl is not None else -10.0
    xu = float(np.asarray(problem.xu).reshape(-1)[0]) if problem.xu is not None else 10.0

    rng = np.random.default_rng(seed)
    key = jax.random.key(int(seed))
    mean_opts = [
        _MeanOptimizer(
            _init_mean(
                n_var, init, sigma0, rng, theta0=getattr(problem, "theta0", None)
            ),
            xl=xl,
            xu=xu,
            es_optim=es_optim,
            es_optim_lr=es_optim_lr,
            es_optim_scheduler=es_optim_scheduler,
            es_optim_momentum=es_optim_momentum,
            es_optim_wd=es_optim_wd,
            steps=steps,
        )
        for _ in range(k)
    ]
    archive = NDArchive(
        problem,
        max_size=archive_size,
        selection=archive_selection,
        ref_dirs_method=ref_dirs_method,
        seed=seed,
    )
    logger = _RunLogger(
        problem,
        archive,
        test_loader=test_loader,
        val_every=val_every,
        pareto_every=pareto_every,
        n_classes=n_classes,
        use_wandb=use_wandb,
        verbose=verbose,
    )

    ideal_mode = ideal_mode.lower()
    if ideal_mode not in MOEAD_IDEAL_MODES:
        raise ValueError(
            f"Unknown ideal_mode: {ideal_mode!r}. Use one of {MOEAD_IDEAL_MODES}."
        )
    scalarization = scalarization.lower()
    if scalarization not in MOEAD_SCALARIZATIONS:
        raise ValueError(
            f"Unknown scalarization: {scalarization!r}. Use one of {MOEAD_SCALARIZATIONS}."
        )

    # Initial means evaluation (not counted, matching run_soo_es conventions).
    means_X = np.stack([m.mean for m in mean_opts], axis=0)
    means_F = _evaluate_F(problem, means_X)
    ideal = (
        np.zeros(n_obj) if ideal_mode == "zero" else means_F.min(axis=0).copy()
    )
    archive.update(means_X, means_F)
    logger.ensure_hv_anchors(means_F)
    n_eval = 0
    best_center_i = int(np.argmin(means_F.mean(axis=1)))
    lam_j = jnp.asarray(lam)
    ideal_j = jnp.asarray(ideal)
    logger.log_train(
        step=0,
        n_eval=n_eval,
        extra={
            "tcheby_mean": float(
                jnp.mean(
                    jnp.stack(
                        [
                            _scalarize(
                                means_F[j : j + 1],
                                lam_j[j],
                                ideal_j,
                                method=scalarization,
                                rho=rho,
                            )[0]
                            for j in range(k)
                        ]
                    )
                )
            ),
            "es_sigma": sigma0,
            "center_f": float(means_F[best_center_i].mean()),
        },
        label="init",
        report_x=means_X[best_center_i],
        report_F=means_F[best_center_i],
    )
    logger.maybe_validate(
        step=0, n_eval=n_eval, center=means_X[best_center_i], force=True
    )

    tcheby_bests = np.full(k, np.nan, dtype=np.float64)
    for gen in range(1, steps + 1):
        # Sample and evaluate all subpopulations first so the ideal point
        # update is shared within the generation.
        sigma = sigma_at(sigma_schedule, gen - 1)
        # One shared fitness batch for all means this generation (CRN).
        problem.sample_eval_pool()
        X_all: list[np.ndarray] = []
        F_all: list[np.ndarray] = []
        eps_eff_all = []
        for j in range(k):
            mean_j = jnp.asarray(mean_opts[j].mean)
            key, eps = _antithetic_noise(key, per_mean // 2, n_var)
            X_j = jnp.clip(mean_j[None, :] + sigma * eps, xl, xu)
            F_j = _evaluate_F(problem, np.asarray(X_j))
            eps_eff_all.append((X_j - mean_j[None, :]) / sigma)
            X_all.append(np.asarray(X_j))
            F_all.append(F_j)
        n_eval = gen * total_popsize

        F_gen = np.vstack(F_all)
        if ideal_mode == "adaptive":
            ideal = np.minimum(ideal, F_gen.min(axis=0))
            ideal_j = jnp.asarray(ideal)
        logger.ensure_hv_anchors(F_gen)

        tcheby_bests = np.empty(k, dtype=np.float64)
        for j in range(k):
            s = _scalarize(
                F_all[j], lam_j[j], ideal_j, method=scalarization, rho=rho
            )
            tcheby_bests[j] = float(jnp.min(s))
            u = _centered_ranks(s)
            g = _es_gradient(u, eps_eff_all[j], sigma)
            mean_opts[j].step(g)

        archive.update(np.vstack(X_all), F_gen)

        if migrate_every > 0 and gen % migrate_every == 0 and len(archive) > 0:
            means_X = np.stack([m.mean for m in mean_opts], axis=0)
            means_F = _evaluate_F(problem, means_X)  # not counted (diagnostic)
            n_migrated = 0
            for j in range(k):
                mean_s = float(
                    _scalarize(
                        means_F[j : j + 1],
                        lam_j[j],
                        ideal_j,
                        method=scalarization,
                        rho=rho,
                    )[0]
                )
                idx, best_s = archive.best_scalarized(
                    lam[j], ideal, rho, method=scalarization
                )
                if best_s < mean_s:
                    mean_opts[j].reset_at(archive.X[idx])
                    n_migrated += 1
            if verbose and n_migrated:
                print(f"[step {gen}] migrated {n_migrated}/{k} means to archive points")

        # Report the best distribution center (lowest mean class-wise CE).
        means_X = np.stack([m.mean for m in mean_opts], axis=0)
        means_F = _evaluate_F(problem, means_X)  # not counted (reporting)
        best_center_i = int(np.argmin(means_F.mean(axis=1)))
        logger.log_train(
            step=gen,
            n_eval=n_eval,
            extra={
                "tcheby_mean": float(tcheby_bests.mean()),
                "tcheby_max": float(tcheby_bests.max()),
                "es_sigma": float(sigma),
                "center_f": float(means_F[best_center_i].mean()),
            },
            label=f"step {gen}",
            report_x=means_X[best_center_i],
            report_F=means_F[best_center_i],
        )
        logger.maybe_validate(
            step=gen, n_eval=n_eval, center=means_X[best_center_i]
        )

    # Final means evaluation for reporting and archive inclusion.
    means_X = np.stack([m.mean for m in mean_opts], axis=0)
    means_F = _evaluate_F(problem, means_X)
    archive.update(means_X, means_F)

    return MOESResult(
        X=archive.X,
        F=archive.F,
        means=means_X,
        means_F=means_F,
        steps=steps,
        popsize=total_popsize,
        n_eval=n_eval,
        algo="moead_open_es",
        weights=np.asarray(weights, dtype=np.float64),
        details={
            "tcheby_mean": float(tcheby_bests.mean()) if steps > 0 else float("nan"),
            **logger.last_val,
        },
    )


# ---------------------------------------------------------------------------
# W&B config
# ---------------------------------------------------------------------------


def build_mo_es_wandb_config(
    args: Any,
    *,
    n_var: int,
    n_obj: int,
    popsize: int,
) -> dict[str, Any]:
    """W&B config for multi-objective OpenES runs (mirrors the SOO builder)."""
    algo = getattr(args, "algo", "mgda_open_es")
    config = {
        "dataset": args.dataset,
        "problem": args.problem,
        "activation": args.activation,
        "algo": algo,
        "fitness": "vector",
        "init": args.init,
        "init_sigma": getattr(args, "init_sigma", 0.1),
        "xl": getattr(args, "xl", -10.0),
        "xu": getattr(args, "xu", 10.0),
        "es_optim": getattr(args, "es_optim", DEFAULT_ES_OPTIM),
        "es_optim_lr": getattr(args, "es_optim_lr", DEFAULT_ES_OPTIM_LR),
        "es_optim_scheduler": getattr(args, "es_optim_scheduler", DEFAULT_ES_OPTIM_SCHEDULER),
        "es_optim_momentum": getattr(args, "es_optim_momentum", DEFAULT_ES_OPTIM_MOMENTUM),
        "es_optim_wd": getattr(args, "es_optim_wd", DEFAULT_ES_OPTIM_WD),
        "es_sigma_scheduler": getattr(args, "es_sigma_scheduler", DEFAULT_ES_SIGMA_SCHEDULER),
        "es_sigma_end": getattr(args, "es_sigma_end", None),
        "steps": args.steps,
        "evals": getattr(args, "evals", None),
        "popsize": popsize,
        "batch_size": args.batch_size,
        "eval_mode": getattr(args, "eval_mode", "multi"),
        "eval_batches": getattr(args, "eval_batches", 50),
        "sampler": getattr(args, "sampler", "auto"),
        "val_every": args.val_every,
        "pareto_every": getattr(args, "pareto_every", 100),
        "val_batch_size": args.val_batch_size,
        "seed": args.seed,
        "device": args.device,
        "n_var": n_var,
        "n_obj": n_obj,
        "archive_size": getattr(args, "archive_size", DEFAULT_ARCHIVE_SIZE),
        "archive_selection": getattr(args, "archive_selection", "nsga2"),
        "mgda_normalize": getattr(args, "mgda_normalize", DEFAULT_MGDA_NORMALIZE),
        "es_fitness_shaping": getattr(
            args, "es_fitness_shaping", DEFAULT_ES_FITNESS_SHAPING
        ),
        "wandb_x_axis": "Function Evaluations",
    }
    algo_config: dict[str, Any] = {
        "name": algo,
        "popsize": popsize,
        "std_init": getattr(args, "init_sigma", 0.1),
        "es_optim": config["es_optim"],
        "es_optim_lr": config["es_optim_lr"],
        "es_optim_scheduler": config["es_optim_scheduler"],
        "es_optim_momentum": config["es_optim_momentum"],
        "es_optim_wd": config["es_optim_wd"],
        "es_sigma_scheduler": config["es_sigma_scheduler"],
        "es_sigma_end": config["es_sigma_end"],
        "archive_size": config["archive_size"],
        "archive_selection": config["archive_selection"],
    }
    if algo in {"mgda_open_es", "upgrad_open_es"}:
        algo_config["aggregator"] = "upgrad" if algo == "upgrad_open_es" else "mgda"
        algo_config["mgda_normalize"] = config["mgda_normalize"]
        algo_config["es_fitness_shaping"] = config["es_fitness_shaping"]
    if algo == "moead_open_es":
        moead = {
            "k": getattr(args, "moead_k", DEFAULT_MOEAD_K),
            "rho": getattr(args, "moead_rho", DEFAULT_MOEAD_RHO),
            "weight_shrink": getattr(
                args, "moead_weight_shrink", DEFAULT_MOEAD_WEIGHT_SHRINK
            ),
            "ideal": getattr(args, "moead_ideal", DEFAULT_MOEAD_IDEAL),
            "scalarization": getattr(
                args, "moead_scalarization", DEFAULT_MOEAD_SCALARIZATION
            ),
            "migrate_every": getattr(args, "moead_migrate_every", 0),
            "ref_dirs": getattr(args, "ref_dirs", "energy"),
        }
        config.update({f"moead_{key}": value for key, value in moead.items()})
        algo_config.update(moead)
    config["algo_config"] = algo_config
    return config
