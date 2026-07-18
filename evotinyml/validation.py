"""Test-set validation metrics for a population of weight vectors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
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


@torch.no_grad()
def evaluate_weights_on_loader(
    problem: WeightOptimizationProblem,
    flat: np.ndarray,
    loader: DataLoader,
    n_classes: int,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """Run one weight vector on ``loader``; return acc / F1 / P / R metrics."""
    problem.set_weights(flat)
    model = problem.model
    model.eval()

    ys = []
    preds = []
    for inputs, targets in loader:
        inputs = inputs.to(problem.device)
        targets = targets.to(problem.device)
        logits = model(inputs)
        pred = logits.argmax(dim=1)
        ys.append(targets.cpu().numpy())
        preds.append(pred.cpu().numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
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

    for i in range(n):
        acc, f1, prec, rec, acc_c, f1_c = evaluate_weights_on_loader(
            problem, X[i], test_loader, n_classes
        )
        overall_acc[i] = acc
        macro_f1[i] = f1
        macro_precision[i] = prec
        macro_recall[i] = rec
        per_class_acc[i] = acc_c
        per_class_f1[i] = f1_c

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
