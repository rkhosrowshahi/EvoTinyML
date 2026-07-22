"""SOO/MOO fitness adapters over shared torch eval metrics.

MOO soft-P/R returns a vector ``F = (1-P, 1-R)``.
SOO on any multi-objective problem uses weighted sum ``f = w · F``
(problem defaults; override via CLI ``--scalar-weights``).
SOO cross-entropy returns mean CE on the eval pool.
SOO F1 / soft F1 returns ``f = 1 − macro-F1`` on the eval pool.
MOO CE + soft-P/R returns ``F = (CE, 1-P, 1-R)``.
MOO task+L1 returns ``F = (task, mean|θ|)`` (sparsity as a second objective).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def default_scalar_weights(problem_name: str, n_obj: int) -> np.ndarray:
    """Problem-specific default SOO scalarization weights (length ``n_obj``).

    Defaults are equal ones so ``w · F`` matches an unweighted sum (legacy
    soft-P/R SOO behavior). Override per problem here when a different balance
    is preferred.
    """
    n = int(n_obj)
    if n < 1:
        raise ValueError(f"n_obj must be >= 1, got {n_obj}")
    # Equal ones for all problems today (soft-P/R SOO ≡ unweighted sum).
    # Branch on ``problem_name`` here for non-uniform defaults later.
    return np.ones(n, dtype=np.float64)


def parse_scalar_weights(
    spec: str | Sequence[float] | None,
    n_obj: int,
    *,
    problem_name: str = "",
) -> np.ndarray:
    """Parse CLI / list weights, or fall back to :func:`default_scalar_weights`."""
    n = int(n_obj)
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        return default_scalar_weights(problem_name, n)
    if isinstance(spec, str):
        parts = [p.strip() for p in spec.replace(" ", "").split(",") if p.strip()]
        try:
            values = [float(p) for p in parts]
        except ValueError as exc:
            raise ValueError(
                f"Invalid --scalar-weights {spec!r}: expected comma-separated floats."
            ) from exc
    else:
        values = [float(v) for v in spec]
    w = np.asarray(values, dtype=np.float64).ravel()
    if w.size != n:
        raise ValueError(
            f"--scalar-weights length {w.size} != n_obj={n} "
            f"(problem={problem_name!r})."
        )
    if not np.all(np.isfinite(w)):
        raise ValueError(f"--scalar-weights must be finite, got {w.tolist()}")
    if np.any(w < 0):
        raise ValueError(f"--scalar-weights must be non-negative, got {w.tolist()}")
    if float(w.sum()) <= 0.0:
        raise ValueError("--scalar-weights must not be all zeros.")
    return w


def weighted_sum(F: np.ndarray, weights: np.ndarray) -> float:
    """Scalarize ``f = w · F`` (minimization)."""
    f = np.asarray(F, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    if f.size != w.size:
        raise ValueError(f"F length {f.size} != weights length {w.size}")
    return float(np.dot(w, f))


def mean_abs_weights(flat: np.ndarray) -> float:
    """Mean absolute weight ``‖θ‖₁ / n`` (scale-friendly L1 sparsity objective)."""
    x = np.asarray(flat, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    return float(np.mean(np.abs(x)))


def hard_macro_f1(
    pred: torch.Tensor, targets: torch.Tensor, n_classes: int
) -> float:
    """Hard macro-F1 from argmax predictions (per-class P/R harmonic mean)."""
    f1s: list[float] = []
    for c in range(n_classes):
        true_c = targets == c
        pred_c = pred == c
        tp = float((true_c & pred_c).sum().item())
        fp = float(((~true_c) & pred_c).sum().item())
        fn = float((true_c & (~pred_c)).sum().item())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append((2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s))


def soft_macro_f1(
    probs: torch.Tensor, targets: torch.Tensor, n_classes: int
) -> float:
    """Soft macro-F1 from class probabilities (soft TP/FP/FN per class)."""
    y = F.one_hot(targets.long(), num_classes=n_classes).to(dtype=probs.dtype)
    soft_tp = (probs * y).sum(dim=0)
    soft_fp = (probs * (1.0 - y)).sum(dim=0)
    soft_fn = ((1.0 - probs) * y).sum(dim=0)
    prec_den = soft_tp + soft_fp
    rec_den = soft_tp + soft_fn
    prec = torch.where(prec_den > 0, soft_tp / prec_den, torch.zeros_like(soft_tp))
    rec = torch.where(rec_den > 0, soft_tp / rec_den, torch.zeros_like(soft_tp))
    f1_den = prec + rec
    f1 = torch.where(f1_den > 0, 2.0 * prec * rec / f1_den, torch.zeros_like(prec))
    return float(f1.mean().item())


def l1_problem_metric_log_dict(
    prefix: str,
    problem_name: str,
    F: np.ndarray,
) -> dict[str, float]:
    """Named train/val keys for task+L1 objective vectors ``F = (task, L1)``.

    ``cross_entropy_l1`` -> ``{prefix}/ce``, ``{prefix}/l1``.
    ``f1_l1`` / ``soft_f1_l1`` -> ``{prefix}/f1`` (as ``1 − F[0]``), ``{prefix}/l1``.
    """
    f = np.asarray(F, dtype=float).ravel()
    if f.size < 2:
        return {}
    name = str(problem_name).lower()
    out: dict[str, float] = {f"{prefix}/l1": float(f[1])}
    if name == "cross_entropy_l1":
        out[f"{prefix}/ce"] = float(f[0])
    else:
        # F[0] is 1−F1 (or 1−soft F1); report the quality score.
        out[f"{prefix}/f1"] = float(1.0 - f[0])
    return out


# Fixed HV reference for all task+L1 problems (CE/1−F1, L1) in [0, 1]².
L1_HV_REF = np.asarray([1.0, 1.0], dtype=np.float64)


def l1_val_front(
    problem_name: str,
    X: np.ndarray,
    *,
    mean_ce: np.ndarray | None = None,
    macro_f1: np.ndarray | None = None,
) -> np.ndarray:
    """Build val minimization front ``F = (task, L1)`` matching train objectives.

    ``cross_entropy_l1``: task = mean test CE.
    ``f1_l1`` / ``soft_f1_l1``: task = 1 − hard macro-F1 on the test set.
    L1 is ``mean|θ|`` (weight-only; identical on train/val).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    n = len(X)
    l1 = np.asarray([mean_abs_weights(X[i]) for i in range(n)], dtype=np.float64)
    name = str(problem_name).lower()
    if name == "cross_entropy_l1":
        if mean_ce is None:
            raise ValueError("l1_val_front requires mean_ce for cross_entropy_l1")
        task = np.asarray(mean_ce, dtype=np.float64).ravel()
    else:
        if macro_f1 is None:
            raise ValueError(f"l1_val_front requires macro_f1 for {name}")
        task = 1.0 - np.asarray(macro_f1, dtype=np.float64).ravel()
    if task.shape[0] != n:
        raise ValueError(f"task length {task.shape[0]} != n_nd={n}")
    return np.column_stack([task, l1])


class HasEvalPool(Protocol):
    """Minimal surface shared with ``WeightOptimizationProblem``."""

    model: nn.Module
    device: torch.device
    batch_sampler: object
    eval_batch_pool: list[tuple[torch.Tensor, torch.Tensor]]
    _param_shapes: list[tuple[int, ...]]

    def set_weights(self, flat: np.ndarray) -> None: ...


class SoftPrecisionRecallMetrics:
    """Shared soft macro-P / soft macro-R computation on an eval batch pool."""

    def __init__(self, problem: HasEvalPool, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.problem = problem
        self.temperature = float(temperature)
        self.n_obj = 2
        self.fitness_name = "1-P,1-R"

    @property
    def num_classes(self) -> int:
        return int(self.problem.batch_sampler.num_classes)

    def proba_pooled(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward current weights on the eval pool; return (probs [N,C], targets [N])."""
        probs: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        model = self.problem.model
        with torch.no_grad():
            for inputs, batch_targets in self.problem.eval_batch_pool:
                logits = model(inputs) / self.temperature
                probs.append(torch.softmax(logits, dim=1))
                targets.append(batch_targets)
        return torch.cat(probs, dim=0), torch.cat(targets, dim=0)

    def soft_macro_precision_recall(
        self, probs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[float, float]:
        n_classes = self.num_classes
        y = F.one_hot(targets.long(), num_classes=n_classes).to(dtype=probs.dtype)
        soft_tp = (probs * y).sum(dim=0)
        soft_fp = (probs * (1.0 - y)).sum(dim=0)
        soft_fn = ((1.0 - probs) * y).sum(dim=0)

        prec_den = soft_tp + soft_fp
        rec_den = soft_tp + soft_fn
        prec = torch.where(prec_den > 0, soft_tp / prec_den, torch.zeros_like(soft_tp))
        rec = torch.where(rec_den > 0, soft_tp / rec_den, torch.zeros_like(soft_tp))
        return float(prec.mean().item()), float(rec.mean().item())

    def vector_for_weights(self, flat: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Return ``F=(1-P, 1-R)`` and the soft (P, R) for one weight vector."""
        self.problem.set_weights(flat)
        probs, targets = self.proba_pooled()
        precision, recall = self.soft_macro_precision_recall(probs, targets)
        f_vec = np.asarray([1.0 - precision, 1.0 - recall], dtype=np.float64)
        return f_vec, precision, recall

    def scalar_for_weights(self, flat: np.ndarray) -> float:
        """Equal-weight sum ``f=(1-P)+(1-R)`` (SOO uses :class:`SOOFitness` weights)."""
        f_vec, _, _ = self.vector_for_weights(flat)
        return float(f_vec.sum())

    def scalar_details_for_weights(self, flat: np.ndarray) -> dict[str, float]:
        f_vec, precision, recall = self.vector_for_weights(flat)
        return {
            "f": float(f_vec.sum()),
            "precision": float(precision),
            "recall": float(recall),
        }


class CrossEntropyMetrics:
    """Mean cross-entropy on the eval batch pool (ERM–CE, single objective)."""

    def __init__(self, problem: HasEvalPool) -> None:
        self.problem = problem
        self.n_obj = 1
        self.fitness_name = "erm_ce"

    def mean_ce_for_weights(self, flat: np.ndarray) -> float:
        self.problem.set_weights(flat)
        losses: list[torch.Tensor] = []
        with torch.no_grad():
            for inputs, targets in self.problem.eval_batch_pool:
                logits = self.problem.model(inputs)
                losses.append(F.cross_entropy(logits, targets, reduction="none"))
        all_losses = torch.cat(losses, dim=0)
        return float(all_losses.mean().item())

    def scalar_for_weights(self, flat: np.ndarray) -> float:
        return self.mean_ce_for_weights(flat)

    def scalar_details_for_weights(self, flat: np.ndarray) -> dict[str, float]:
        ce = self.mean_ce_for_weights(flat)
        return {"f": ce, "ce": ce}


class F1Metrics:
    """Hard macro-F1 on the eval batch pool (ERM–F1, single objective).

    Minimization scalar ``f = 1 − macro-F1``.
    """

    def __init__(self, problem: HasEvalPool) -> None:
        self.problem = problem
        self.n_obj = 1
        self.fitness_name = "erm_f1"

    @property
    def num_classes(self) -> int:
        return int(self.problem.batch_sampler.num_classes)

    def macro_f1_for_weights(self, flat: np.ndarray) -> float:
        self.problem.set_weights(flat)
        preds: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        with torch.no_grad():
            for inputs, batch_targets in self.problem.eval_batch_pool:
                logits = self.problem.model(inputs)
                preds.append(logits.argmax(dim=1))
                targets.append(batch_targets)
        return hard_macro_f1(
            torch.cat(preds, dim=0),
            torch.cat(targets, dim=0),
            self.num_classes,
        )

    def scalar_for_weights(self, flat: np.ndarray) -> float:
        return 1.0 - self.macro_f1_for_weights(flat)

    def scalar_details_for_weights(self, flat: np.ndarray) -> dict[str, float]:
        f1 = self.macro_f1_for_weights(flat)
        return {"f": 1.0 - f1, "f1": f1}


class SoftF1Metrics:
    """Soft macro-F1 on the eval batch pool (ERM–soft-F1, single objective).

    Minimization scalar ``f = 1 − soft macro-F1``.
    """

    def __init__(self, problem: HasEvalPool, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.problem = problem
        self.temperature = float(temperature)
        self.n_obj = 1
        self.fitness_name = "erm_soft_f1"

    @property
    def num_classes(self) -> int:
        return int(self.problem.batch_sampler.num_classes)

    def soft_macro_f1_for_weights(self, flat: np.ndarray) -> float:
        self.problem.set_weights(flat)
        probs: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        with torch.no_grad():
            for inputs, batch_targets in self.problem.eval_batch_pool:
                logits = self.problem.model(inputs) / self.temperature
                probs.append(torch.softmax(logits, dim=1))
                targets.append(batch_targets)
        return soft_macro_f1(
            torch.cat(probs, dim=0),
            torch.cat(targets, dim=0),
            self.num_classes,
        )

    def scalar_for_weights(self, flat: np.ndarray) -> float:
        return 1.0 - self.soft_macro_f1_for_weights(flat)

    def scalar_details_for_weights(self, flat: np.ndarray) -> dict[str, float]:
        f1 = self.soft_macro_f1_for_weights(flat)
        return {"f": 1.0 - f1, "f1": f1}


class CESoftPrecisionRecallMetrics:
    """Joint mean CE + soft macro-P/R on one forward (3 MOO objectives).

    Minimization vector ``F = (CE, 1−soft_P, 1−soft_R)``.
    """

    OBJ_LABELS = ("CE", "1−soft P", "1−soft R")

    def __init__(self, problem: HasEvalPool, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.problem = problem
        self.temperature = float(temperature)
        self.n_obj = 3
        self.fitness_name = "ce_soft_pr"

    @property
    def num_classes(self) -> int:
        return int(self.problem.batch_sampler.num_classes)

    def _logits_pooled(self) -> tuple[torch.Tensor, torch.Tensor]:
        logits_all: list[torch.Tensor] = []
        targets_all: list[torch.Tensor] = []
        model = self.problem.model
        with torch.no_grad():
            for inputs, batch_targets in self.problem.eval_batch_pool:
                logits_all.append(model(inputs))
                targets_all.append(batch_targets)
        return torch.cat(logits_all, dim=0), torch.cat(targets_all, dim=0)

    def soft_macro_precision_recall(
        self, probs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[float, float]:
        n_classes = self.num_classes
        y = F.one_hot(targets.long(), num_classes=n_classes).to(dtype=probs.dtype)
        soft_tp = (probs * y).sum(dim=0)
        soft_fp = (probs * (1.0 - y)).sum(dim=0)
        soft_fn = ((1.0 - probs) * y).sum(dim=0)
        prec_den = soft_tp + soft_fp
        rec_den = soft_tp + soft_fn
        prec = torch.where(prec_den > 0, soft_tp / prec_den, torch.zeros_like(soft_tp))
        rec = torch.where(rec_den > 0, soft_tp / rec_den, torch.zeros_like(soft_tp))
        return float(prec.mean().item()), float(rec.mean().item())

    def vector_for_weights(
        self, flat: np.ndarray
    ) -> tuple[np.ndarray, float, float, float]:
        """Return ``F=(CE, 1-P, 1-R)`` and ``(CE, soft_P, soft_R)``."""
        self.problem.set_weights(flat)
        logits, targets = self._logits_pooled()
        ce = float(F.cross_entropy(logits, targets, reduction="mean").item())
        probs = torch.softmax(logits / self.temperature, dim=1)
        precision, recall = self.soft_macro_precision_recall(probs, targets)
        f_vec = np.asarray([ce, 1.0 - precision, 1.0 - recall], dtype=np.float64)
        return f_vec, ce, precision, recall

    def scalar_for_weights(self, flat: np.ndarray) -> float:
        f_vec, _, _, _ = self.vector_for_weights(flat)
        return float(f_vec.sum())

    def scalar_details_for_weights(self, flat: np.ndarray) -> dict[str, float]:
        f_vec, ce, precision, recall = self.vector_for_weights(flat)
        return {
            "f": float(f_vec.sum()),
            "ce": float(ce),
            "precision": float(precision),
            "recall": float(recall),
        }


class CrossEntropyL1Metrics:
    """Bi-objective: mean CE + mean |θ|. Minimization ``F = (CE, L1)``."""

    OBJ_LABELS = ("CE", "L1")

    def __init__(self, problem: HasEvalPool) -> None:
        self.problem = problem
        self.n_obj = 2
        self.fitness_name = "ce_l1"

    def vector_for_weights(self, flat: np.ndarray) -> tuple[np.ndarray, float, float]:
        self.problem.set_weights(flat)
        losses: list[torch.Tensor] = []
        with torch.no_grad():
            for inputs, targets in self.problem.eval_batch_pool:
                logits = self.problem.model(inputs)
                losses.append(F.cross_entropy(logits, targets, reduction="none"))
        ce = float(torch.cat(losses, dim=0).mean().item())
        l1 = mean_abs_weights(flat)
        return np.asarray([ce, l1], dtype=np.float64), ce, l1

    def scalar_for_weights(self, flat: np.ndarray) -> float:
        return float(self.vector_for_weights(flat)[0].sum())

    def scalar_details_for_weights(self, flat: np.ndarray) -> dict[str, float]:
        f_vec, ce, l1 = self.vector_for_weights(flat)
        return {"f": float(f_vec.sum()), "ce": float(ce), "l1": float(l1)}


class F1L1Metrics:
    """Bi-objective: hard macro-F1 + mean |θ|. Minimization ``F = (1−F1, L1)``."""

    OBJ_LABELS = ("1−F1", "L1")

    def __init__(self, problem: HasEvalPool) -> None:
        self.problem = problem
        self.n_obj = 2
        self.fitness_name = "f1_l1"

    @property
    def num_classes(self) -> int:
        return int(self.problem.batch_sampler.num_classes)

    def vector_for_weights(self, flat: np.ndarray) -> tuple[np.ndarray, float, float]:
        self.problem.set_weights(flat)
        preds: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        with torch.no_grad():
            for inputs, batch_targets in self.problem.eval_batch_pool:
                logits = self.problem.model(inputs)
                preds.append(logits.argmax(dim=1))
                targets.append(batch_targets)
        f1 = hard_macro_f1(
            torch.cat(preds, dim=0),
            torch.cat(targets, dim=0),
            self.num_classes,
        )
        l1 = mean_abs_weights(flat)
        return np.asarray([1.0 - f1, l1], dtype=np.float64), f1, l1

    def scalar_for_weights(self, flat: np.ndarray) -> float:
        return float(self.vector_for_weights(flat)[0].sum())

    def scalar_details_for_weights(self, flat: np.ndarray) -> dict[str, float]:
        f_vec, f1, l1 = self.vector_for_weights(flat)
        return {"f": float(f_vec.sum()), "f1": float(f1), "l1": float(l1)}


class SoftF1L1Metrics:
    """Bi-objective: soft macro-F1 + mean |θ|. Minimization ``F = (1−soft F1, L1)``."""

    OBJ_LABELS = ("1−soft F1", "L1")

    def __init__(self, problem: HasEvalPool, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.problem = problem
        self.temperature = float(temperature)
        self.n_obj = 2
        self.fitness_name = "soft_f1_l1"

    @property
    def num_classes(self) -> int:
        return int(self.problem.batch_sampler.num_classes)

    def vector_for_weights(self, flat: np.ndarray) -> tuple[np.ndarray, float, float]:
        self.problem.set_weights(flat)
        probs: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        with torch.no_grad():
            for inputs, batch_targets in self.problem.eval_batch_pool:
                logits = self.problem.model(inputs) / self.temperature
                probs.append(torch.softmax(logits, dim=1))
                targets.append(batch_targets)
        f1 = soft_macro_f1(
            torch.cat(probs, dim=0),
            torch.cat(targets, dim=0),
            self.num_classes,
        )
        l1 = mean_abs_weights(flat)
        return np.asarray([1.0 - f1, l1], dtype=np.float64), f1, l1

    def scalar_for_weights(self, flat: np.ndarray) -> float:
        return float(self.vector_for_weights(flat)[0].sum())

    def scalar_details_for_weights(self, flat: np.ndarray) -> dict[str, float]:
        f_vec, f1, l1 = self.vector_for_weights(flat)
        return {"f": float(f_vec.sum()), "f1": float(f1), "l1": float(l1)}


class ProblemObjectiveMetrics:
    """Expose ``problem._evaluate_individual`` as a vector for SOO weighted sum."""

    def __init__(self, problem: HasEvalPool, fitness_name: str = "F") -> None:
        self.problem = problem
        self.n_obj = int(getattr(problem, "n_obj", 1))
        self.fitness_name = fitness_name

    def vector_for_weights(self, flat: np.ndarray) -> tuple[np.ndarray]:
        F = np.asarray(
            self.problem._evaluate_individual(flat), dtype=np.float64  # type: ignore[attr-defined]
        ).ravel()
        return (F,)

    def scalar_for_weights(self, flat: np.ndarray) -> float:
        return float(self.vector_for_weights(flat)[0].sum())

    def scalar_details_for_weights(self, flat: np.ndarray) -> dict[str, float]:
        F = self.vector_for_weights(flat)[0]
        out: dict[str, float] = {"f": float(F.sum())}
        for i, v in enumerate(F):
            out[f"f{i}"] = float(v)
        return out


_MOO_METRICS = (
    SoftPrecisionRecallMetrics
    | CESoftPrecisionRecallMetrics
    | CrossEntropyL1Metrics
    | F1L1Metrics
    | SoftF1L1Metrics
    | ProblemObjectiveMetrics
)

_SOO_METRICS = (
    SoftPrecisionRecallMetrics
    | CrossEntropyMetrics
    | F1Metrics
    | SoftF1Metrics
    | CESoftPrecisionRecallMetrics
    | CrossEntropyL1Metrics
    | F1L1Metrics
    | SoftF1L1Metrics
    | ProblemObjectiveMetrics
)


class MOOFitness:
    """Multi-objective adapter: ``evaluate(X) -> (n, n_obj)``."""

    def __init__(self, metrics: _MOO_METRICS) -> None:
        self.metrics = metrics
        self.n_obj = metrics.n_obj

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        out = np.empty((X.shape[0], self.n_obj), dtype=np.float64)
        for i in range(X.shape[0]):
            out[i] = self.evaluate_one(X[i])
        return out

    def evaluate_one(self, flat: np.ndarray) -> np.ndarray:
        result = self.metrics.vector_for_weights(flat)
        return np.asarray(result[0], dtype=np.float64)


class SOOFitness:
    """Single-objective adapter: ``evaluate(X) -> (n,)`` scalar fitness.

    Multi-objective metrics are scalarized as ``f = w · F`` with
    :attr:`scalar_weights` (problem defaults; set via CLI).
    """

    def __init__(
        self,
        metrics: _SOO_METRICS,
        scalar_weights: np.ndarray | Sequence[float] | None = None,
    ) -> None:
        self.metrics = metrics
        self.n_obj = int(getattr(metrics, "n_obj", 1))
        base = getattr(metrics, "fitness_name", "f")
        self._base_fitness_name = str(base)
        self._scalar_weights = parse_scalar_weights(
            scalar_weights, self.n_obj, problem_name=self._base_fitness_name
        )
        self._refresh_fitness_name()

    def _refresh_fitness_name(self) -> None:
        if self.n_obj > 1:
            self.fitness_name = f"w·({self._base_fitness_name})"
        else:
            self.fitness_name = self._base_fitness_name

    @property
    def scalar_weights(self) -> np.ndarray:
        return np.asarray(self._scalar_weights, dtype=np.float64).copy()

    def set_scalar_weights(self, weights: np.ndarray | Sequence[float]) -> None:
        self._scalar_weights = parse_scalar_weights(
            weights, self.n_obj, problem_name=self._base_fitness_name
        )
        self._refresh_fitness_name()

    def _scalarize_vector(
        self, result: tuple
    ) -> tuple[float, np.ndarray, dict[str, float]]:
        F = np.asarray(result[0], dtype=np.float64).ravel()
        f = weighted_sum(F, self._scalar_weights)
        details: dict[str, float] = {"f": f}
        for i, v in enumerate(F):
            details[f"f{i}"] = float(v)
            details[f"w{i}"] = float(self._scalar_weights[i])
        name = self._base_fitness_name
        if name in {"1-P,1-R"} and len(result) >= 3:
            details["precision"] = float(result[1])
            details["recall"] = float(result[2])
        elif name == "ce_soft_pr" and len(result) >= 4:
            details["ce"] = float(result[1])
            details["precision"] = float(result[2])
            details["recall"] = float(result[3])
        elif name in {"ce_l1", "f1_l1", "soft_f1_l1"} and len(result) >= 3:
            if "f1" in name:
                details["f1"] = float(result[1])
            else:
                details["ce"] = float(result[1])
            details["l1"] = float(result[2])
        return f, F, details

    def _scalar_one(self, flat: np.ndarray) -> float:
        if hasattr(self.metrics, "vector_for_weights"):
            result = self.metrics.vector_for_weights(flat)
            f, _, _ = self._scalarize_vector(result)
            return f
        # True single-objective metrics (ERM).
        f = float(self.metrics.scalar_for_weights(flat))
        return float(self._scalar_weights[0] * f)

    def evaluate(
        self, X: np.ndarray, *, details: bool = False
    ) -> np.ndarray | tuple[np.ndarray, list[dict[str, float]]]:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        n = X.shape[0]
        f = np.empty(n, dtype=np.float64)
        if not details:
            for i in range(n):
                f[i] = self._scalar_one(X[i])
            return f

        detail_rows: list[dict[str, float]] = []
        for i in range(n):
            row = self.evaluate_one(X[i])
            f[i] = float(row["f"])
            detail_rows.append(row)
        return f, detail_rows

    def evaluate_one(self, flat: np.ndarray) -> dict[str, float]:
        if hasattr(self.metrics, "vector_for_weights"):
            result = self.metrics.vector_for_weights(flat)
            _, _, details = self._scalarize_vector(result)
            return details
        row = dict(self.metrics.scalar_details_for_weights(flat))
        row["f"] = float(self._scalar_weights[0] * float(row["f"]))
        row["w0"] = float(self._scalar_weights[0])
        return row
