"""SOO/MOO fitness adapters over shared torch eval metrics.

MOO soft-P/R returns a vector ``F = (1-P, 1-R)``.
SOO soft-P/R returns the scalar ``f = (1-P) + (1-R)``.
SOO cross-entropy returns mean CE on the eval pool.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


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
        self.fitness_name = "sum(1-P,1-R)"

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
        """Return ``f=(1-P)+(1-R)``."""
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


class MOOFitness:
    """Multi-objective adapter: ``evaluate(X) -> (n, n_obj)``."""

    def __init__(self, metrics: SoftPrecisionRecallMetrics) -> None:
        self.metrics = metrics
        self.n_obj = metrics.n_obj

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        out = np.empty((X.shape[0], self.n_obj), dtype=np.float64)
        for i in range(X.shape[0]):
            f_vec, _, _ = self.metrics.vector_for_weights(X[i])
            out[i] = f_vec
        return out

    def evaluate_one(self, flat: np.ndarray) -> np.ndarray:
        f_vec, _, _ = self.metrics.vector_for_weights(flat)
        return f_vec


class SOOFitness:
    """Single-objective adapter: ``evaluate(X) -> (n,)`` scalar fitness."""

    def __init__(self, metrics: SoftPrecisionRecallMetrics | CrossEntropyMetrics) -> None:
        self.metrics = metrics
        self.fitness_name = getattr(metrics, "fitness_name", "f")

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
                f[i] = self.metrics.scalar_for_weights(X[i])
            return f

        detail_rows: list[dict[str, float]] = []
        for i in range(n):
            row = self.metrics.scalar_details_for_weights(X[i])
            f[i] = float(row["f"])
            detail_rows.append(row)
        return f, detail_rows

    def evaluate_one(self, flat: np.ndarray) -> dict[str, float]:
        return self.metrics.scalar_details_for_weights(flat)
