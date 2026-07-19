"""Test-set validation metrics for a population of weight vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from evotinyml.problem import WeightOptimizationProblem


@dataclass
class NDValidationResult:
    """Per-individual and front-level test metrics for the ND set."""

    overall_acc: np.ndarray  # (n_nd,)
    macro_f1: np.ndarray  # (n_nd,)
    macro_precision: np.ndarray  # (n_nd,)
    macro_recall: np.ndarray  # (n_nd,)
    per_class_acc: np.ndarray  # (n_nd, n_classes)
    per_class_f1: np.ndarray  # (n_nd, n_classes)
    per_class_ce: np.ndarray  # (n_nd, n_classes)
    error_acc: np.ndarray  # (n_nd, n_classes) = 1 - per_class_acc
    error_f1: np.ndarray  # (n_nd, n_classes) = 1 - per_class_f1
    error_pr: np.ndarray  # (n_nd, 2) = [1 - macro_precision, 1 - macro_recall]

    @property
    def mean_acc(self) -> float:
        return float(self.overall_acc.mean()) if len(self.overall_acc) else float("nan")

    @property
    def mean_f1(self) -> float:
        return float(self.macro_f1.mean()) if len(self.macro_f1) else float("nan")

    @property
    def mean_precision(self) -> float:
        return float(self.macro_precision.mean()) if len(self.macro_precision) else float("nan")

    @property
    def mean_recall(self) -> float:
        return float(self.macro_recall.mean()) if len(self.macro_recall) else float("nan")

    @property
    def best_acc(self) -> float:
        return float(self.overall_acc.max()) if len(self.overall_acc) else float("nan")

    @property
    def best_f1(self) -> float:
        return float(self.macro_f1.max()) if len(self.macro_f1) else float("nan")

    @property
    def best_acc_index(self) -> int:
        return int(np.argmax(self.overall_acc)) if len(self.overall_acc) else 0

    @property
    def best_precision(self) -> float:
        return float(self.macro_precision.max()) if len(self.macro_precision) else float("nan")

    @property
    def best_recall(self) -> float:
        return float(self.macro_recall.max()) if len(self.macro_recall) else float("nan")

    @property
    def knee_index(self) -> int:
        """Index of the ND point closest to utopia (P=1, R=1) in L2."""
        if len(self.error_pr) == 0:
            return 0
        return int(np.argmin(np.sum(np.square(self.error_pr), axis=1)))

    def knee_metrics(self) -> dict[str, float]:
        """Test metrics for the knee individual on the P–R front."""
        i = self.knee_index
        p = float(self.macro_precision[i])
        r = float(self.macro_recall[i])
        return {
            "precision": p,
            "recall": r,
            "pr_mean": 0.5 * (p + r),
            "f1": float(self.macro_f1[i]),
            "acc": float(self.overall_acc[i]),
        }


def class_metric_fields(
    per_class_acc: np.ndarray, per_class_ce: np.ndarray
) -> dict[str, float]:
    """Build ``class_{c}_acc`` / ``class_{c}_ce`` entries (no prefix)."""
    acc = np.asarray(per_class_acc, dtype=np.float64).ravel()
    ce = np.asarray(per_class_ce, dtype=np.float64).ravel()
    out: dict[str, float] = {}
    for c in range(min(len(acc), len(ce))):
        out[f"class_{c}_acc"] = float(acc[c])
        out[f"class_{c}_ce"] = float(ce[c])
    return out


def class_metric_log_dict(
    prefix: str, per_class_acc: np.ndarray, per_class_ce: np.ndarray
) -> dict[str, float]:
    """Build ``{prefix}/class_{c}_acc`` and ``{prefix}/class_{c}_ce`` entries."""
    return {
        f"{prefix}/{k}": v
        for k, v in class_metric_fields(per_class_acc, per_class_ce).items()
    }


def _per_class_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """Return overall acc, macro-F1/P/R, per-class acc, per-class F1."""
    acc_c = np.zeros(n_classes, dtype=np.float64)
    f1_c = np.zeros(n_classes, dtype=np.float64)
    prec_c = np.zeros(n_classes, dtype=np.float64)
    rec_c = np.zeros(n_classes, dtype=np.float64)

    correct = 0
    for c in range(n_classes):
        true_c = y_true == c
        pred_c = y_pred == c
        support = int(true_c.sum())
        tp = int((true_c & pred_c).sum())
        fp = int((~true_c & pred_c).sum())
        fn = support - tp

        acc_c[c] = (tp / support) if support > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec_c[c] = prec
        rec_c[c] = rec
        f1_c[c] = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        correct += tp

    overall_acc = correct / max(len(y_true), 1)
    return (
        overall_acc,
        float(f1_c.mean()),
        float(prec_c.mean()),
        float(rec_c.mean()),
        acc_c,
        f1_c,
    )


def _per_class_ce_from_losses(
    losses: np.ndarray, y_true: np.ndarray, n_classes: int
) -> np.ndarray:
    ce_c = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        mask = y_true == c
        if int(mask.sum()) == 0:
            ce_c[c] = float("nan")
        else:
            ce_c[c] = float(losses[mask].mean())
    return ce_c


@torch.no_grad()
def _forward_collect(
    model: torch.nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, y_pred, per_sample_ce) for the given batches."""
    model.eval()
    losses: list[torch.Tensor] = []
    preds: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    for inputs, targets in batches:
        if device is not None:
            inputs = inputs.to(device)
            targets = targets.to(device)
        logits = model(inputs)
        losses.append(F.cross_entropy(logits, targets, reduction="none"))
        preds.append(logits.argmax(dim=1))
        targets_all.append(targets)
    loss_np = torch.cat(losses, dim=0).detach().cpu().numpy()
    y_pred = torch.cat(preds, dim=0).detach().cpu().numpy()
    y_true = torch.cat(targets_all, dim=0).detach().cpu().numpy()
    return y_true, y_pred, loss_np


@torch.no_grad()
def per_class_acc_ce_on_batches(
    model: torch.nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    n_classes: int,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class accuracy and mean CE over a sequence of (inputs, targets) batches."""
    y_true, y_pred, loss_np = _forward_collect(model, batches, device=device)
    _, _, _, _, acc_c, _ = _per_class_scores(y_true, y_pred, n_classes)
    ce_c = _per_class_ce_from_losses(loss_np, y_true, n_classes)
    return acc_c, ce_c


@torch.no_grad()
def per_class_acc_ce_on_eval_pool(
    problem: WeightOptimizationProblem, flat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class acc / CE for ``flat`` on the problem's train eval batch pool."""
    problem.set_weights(flat)
    n_classes = int(problem.batch_sampler.num_classes)
    return per_class_acc_ce_on_batches(
        problem.model, problem.eval_batch_pool, n_classes, device=None
    )


@torch.no_grad()
def per_class_acc_ce_on_loader(
    problem: WeightOptimizationProblem,
    flat: np.ndarray,
    loader: DataLoader,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class acc / CE for ``flat`` on a DataLoader (e.g. test set)."""
    problem.set_weights(flat)
    batches = (
        (inputs.to(problem.device), targets.to(problem.device))
        for inputs, targets in loader
    )
    return per_class_acc_ce_on_batches(problem.model, batches, n_classes, device=None)


@torch.no_grad()
def evaluate_weights_on_loader(
    problem: WeightOptimizationProblem,
    flat: np.ndarray,
    loader: DataLoader,
    n_classes: int,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """Run one weight vector on ``loader``; return acc / F1 / P / R metrics."""
    problem.set_weights(flat)
    batches = (
        (inputs.to(problem.device), targets.to(problem.device))
        for inputs, targets in loader
    )
    y_true, y_pred, _ = _forward_collect(problem.model, batches, device=None)
    return _per_class_scores(y_true, y_pred, n_classes)


def validate_nd_set(
    problem: WeightOptimizationProblem,
    X: np.ndarray,
    test_loader: DataLoader,
    n_classes: int,
) -> NDValidationResult:
    """Evaluate each ND weight vector on the test set."""
    n = len(X)
    overall_acc = np.empty(n, dtype=np.float64)
    macro_f1 = np.empty(n, dtype=np.float64)
    macro_precision = np.empty(n, dtype=np.float64)
    macro_recall = np.empty(n, dtype=np.float64)
    per_class_acc = np.empty((n, n_classes), dtype=np.float64)
    per_class_f1 = np.empty((n, n_classes), dtype=np.float64)
    per_class_ce = np.empty((n, n_classes), dtype=np.float64)

    for i in range(n):
        problem.set_weights(X[i])
        batches = (
            (inputs.to(problem.device), targets.to(problem.device))
            for inputs, targets in test_loader
        )
        y_true, y_pred, loss_np = _forward_collect(problem.model, batches, device=None)
        acc, f1, prec, rec, acc_c, f1_c = _per_class_scores(y_true, y_pred, n_classes)
        ce_c = _per_class_ce_from_losses(loss_np, y_true, n_classes)
        overall_acc[i] = acc
        macro_f1[i] = f1
        macro_precision[i] = prec
        macro_recall[i] = rec
        per_class_acc[i] = acc_c
        per_class_f1[i] = f1_c
        per_class_ce[i] = ce_c

    error_pr = np.column_stack(
        [
            np.clip(1.0 - macro_precision, 0.0, 1.0),
            np.clip(1.0 - macro_recall, 0.0, 1.0),
        ]
    )

    return NDValidationResult(
        overall_acc=overall_acc,
        macro_f1=macro_f1,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        per_class_acc=per_class_acc,
        per_class_f1=per_class_f1,
        per_class_ce=per_class_ce,
        error_acc=np.clip(1.0 - per_class_acc, 0.0, 1.0),
        error_f1=np.clip(1.0 - per_class_f1, 0.0, 1.0),
        error_pr=error_pr,
    )


def make_test_loader(
    dataset: Dataset,
    batch_size: int = 512,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
