"""pymoo multi-objective problems over CNN weight vectors."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from pymoo.core.problem import Problem
from torch import nn

from evotinyml.data import EVAL_MODES, ClassGuaranteedBatchSampler, RandomBatchSampler

PROBLEMS = ("per_class_ce", "precision_recall", "soft_precision_recall")
# Problems that expose a 2-obj (1-P, 1-R) front in train/display/W&B.
PR_PROBLEMS = frozenset({"precision_recall", "soft_precision_recall"})


class BatchSampler(Protocol):
    num_classes: int

    def sample_batch(self) -> tuple[torch.Tensor, torch.Tensor]: ...

    def sample_batches(self, n_batches: int) -> list[tuple[torch.Tensor, torch.Tensor]]: ...


class WeightOptimizationProblem(Problem):
    """Shared machinery for evolving TinyCNN weights on an eval batch pool."""

    def __init__(
        self,
        model: nn.Module,
        batch_sampler: BatchSampler,
        n_obj: int,
        xl: float = -1.0,
        xu: float = 1.0,
        device: torch.device | None = None,
        resample_every: int = 50,
        eval_mode: str = "single",
        eval_batches: int = 8,
    ) -> None:
        self.model = model
        self.batch_sampler = batch_sampler
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

        eval_mode = eval_mode.lower()
        if eval_mode not in EVAL_MODES:
            raise ValueError(f"Unknown eval_mode: {eval_mode!r}. Use one of {EVAL_MODES}.")
        self.eval_mode = eval_mode
        self.eval_batches = 1 if eval_mode == "single" else max(1, int(eval_batches))

        self.resample_every = max(1, int(resample_every))
        self.batch_version = 0
        self.hv_ideal: np.ndarray | None = None
        self.hv_nadir: np.ndarray | None = None
        self.problem_name = "weight"

        self._param_shapes = [tuple(p.shape) for p in self.model.parameters()]
        self._n_var = sum(p.numel() for p in self.model.parameters())

        # Cached pool of eval batches shared by the whole population.
        self.eval_batch_pool: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.resample_batch()

        super().__init__(n_var=self._n_var, n_obj=n_obj, xl=xl, xu=xu)

    def resample_batch(self) -> None:
        """Draw a new eval batch pool and bump ``batch_version``."""
        raw = self.batch_sampler.sample_batches(self.eval_batches)
        self.eval_batch_pool = [
            (inputs.to(self.device), targets.to(self.device)) for inputs, targets in raw
        ]
        self.batch_version += 1

    def set_hv_anchors_from_F(self, F: np.ndarray) -> None:
        """Set HV ideal/nadir once from the initial population (no-op if already set)."""
        if self.hv_nadir is not None:
            return
        F = np.asarray(F, dtype=float)
        self.hv_ideal = F.min(axis=0).copy()
        self.hv_nadir = F.max(axis=0).copy()

    def set_weights(self, flat: np.ndarray) -> None:
        """Load a flat numpy vector into the model parameters."""
        offset = 0
        with torch.no_grad():
            for param, shape in zip(self.model.parameters(), self._param_shapes):
                n = int(np.prod(shape))
                chunk = flat[offset : offset + n].reshape(shape)
                param.copy_(torch.from_numpy(chunk.astype(np.float32)).to(self.device))
                offset += n

    def _predict_pooled(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the current weights on the full eval pool; return pooled (pred, target)."""
        preds: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        with torch.no_grad():
            for inputs, batch_targets in self.eval_batch_pool:
                logits = self.model(inputs)
                preds.append(logits.argmax(dim=1))
                targets.append(batch_targets)
        return torch.cat(preds, dim=0), torch.cat(targets, dim=0)

    def _evaluate_individual(self, flat: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _evaluate(self, X, out, *args, **kwargs):
        F_mat = np.empty((X.shape[0], self.n_obj), dtype=np.float64)
        for i in range(X.shape[0]):
            F_mat[i] = self._evaluate_individual(X[i])
        out["F"] = F_mat


class PerClassCEProblem(WeightOptimizationProblem):
    """One minimization objective per class: mean CE on pooled eval samples."""

    def __init__(
        self,
        model: nn.Module,
        batch_sampler: BatchSampler,
        xl: float = -1.0,
        xu: float = 1.0,
        device: torch.device | None = None,
        resample_every: int = 50,
        eval_mode: str = "single",
        eval_batches: int = 8,
    ) -> None:
        super().__init__(
            model=model,
            batch_sampler=batch_sampler,
            n_obj=batch_sampler.num_classes,
            xl=xl,
            xu=xu,
            device=device,
            resample_every=resample_every,
            eval_mode=eval_mode,
            eval_batches=eval_batches,
        )
        self.problem_name = "per_class_ce"

    def _evaluate_individual(self, flat: np.ndarray) -> np.ndarray:
        self.set_weights(flat)
        losses_all: list[torch.Tensor] = []
        targets_all: list[torch.Tensor] = []
        with torch.no_grad():
            for inputs, targets in self.eval_batch_pool:
                logits = self.model(inputs)
                losses_all.append(F.cross_entropy(logits, targets, reduction="none"))
                targets_all.append(targets)
        losses = torch.cat(losses_all, dim=0)
        targets = torch.cat(targets_all, dim=0)

        n_obj = self.batch_sampler.num_classes
        objectives = np.empty(n_obj, dtype=np.float64)
        for c in range(n_obj):
            mask = targets == c
            if int(mask.sum().item()) == 0:
                # Class absent from the pooled samples.
                objectives[c] = 1e6
            else:
                objectives[c] = float(losses[mask].mean().item())
        return objectives


class PrecisionRecallProblem(WeightOptimizationProblem):
    """Two-objective problem: minimize (1 - macro precision) and (1 - macro recall).

    Predictions are collected over the entire eval batch pool first; precision and
    recall are computed once on the pooled labels/predictions (not per mini-batch).
    """

    def __init__(
        self,
        model: nn.Module,
        batch_sampler: BatchSampler,
        xl: float = -1.0,
        xu: float = 1.0,
        device: torch.device | None = None,
        resample_every: int = 50,
        eval_mode: str = "single",
        eval_batches: int = 8,
    ) -> None:
        super().__init__(
            model=model,
            batch_sampler=batch_sampler,
            n_obj=2,
            xl=xl,
            xu=xu,
            device=device,
            resample_every=resample_every,
            eval_mode=eval_mode,
            eval_batches=eval_batches,
        )
        self.problem_name = "precision_recall"

    def _macro_precision_recall(
        self, pred: torch.Tensor, targets: torch.Tensor
    ) -> tuple[float, float]:
        n_classes = self.batch_sampler.num_classes
        precisions = []
        recalls = []
        for c in range(n_classes):
            true_c = targets == c
            pred_c = pred == c
            tp = float((true_c & pred_c).sum().item())
            fp = float(((~true_c) & pred_c).sum().item())
            fn = float((true_c & (~pred_c)).sum().item())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precisions.append(prec)
            recalls.append(rec)
        return float(np.mean(precisions)), float(np.mean(recalls))

    def _evaluate_individual(self, flat: np.ndarray) -> np.ndarray:
        self.set_weights(flat)
        pred, targets = self._predict_pooled()
        precision, recall = self._macro_precision_recall(pred, targets)
        return np.asarray([1.0 - precision, 1.0 - recall], dtype=np.float64)


class SoftPrecisionRecallProblem(WeightOptimizationProblem):
    """Two-objective problem with soft (probabilistic) macro precision / recall.

    Uses softmax probabilities instead of hard argmax counts:

        soft_tp_c = Σ_i p_i(c) · 1[y_i = c]
        soft_fp_c = Σ_i p_i(c) · 1[y_i ≠ c]
        soft_fn_c = Σ_i (1 - p_i(c)) · 1[y_i = c]

    then per-class soft P/R and macro-average. Objectives are still
    ``(1 - macro_P, 1 - macro_R)``. The soft counts make the landscape
    continuous in the weights (unlike hard precision/recall).
    """

    def __init__(
        self,
        model: nn.Module,
        batch_sampler: BatchSampler,
        xl: float = -1.0,
        xu: float = 1.0,
        device: torch.device | None = None,
        resample_every: int = 50,
        eval_mode: str = "single",
        eval_batches: int = 8,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(
            model=model,
            batch_sampler=batch_sampler,
            n_obj=2,
            xl=xl,
            xu=xu,
            device=device,
            resample_every=resample_every,
            eval_mode=eval_mode,
            eval_batches=eval_batches,
        )
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = float(temperature)
        self.problem_name = "soft_precision_recall"

    def _proba_pooled(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Run current weights on the eval pool; return (probs [N,C], targets [N])."""
        probs: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        with torch.no_grad():
            for inputs, batch_targets in self.eval_batch_pool:
                logits = self.model(inputs) / self.temperature
                probs.append(torch.softmax(logits, dim=1))
                targets.append(batch_targets)
        return torch.cat(probs, dim=0), torch.cat(targets, dim=0)

    def _soft_macro_precision_recall(
        self, probs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[float, float]:
        n_classes = self.batch_sampler.num_classes
        # one-hot targets: [N, C]
        y = F.one_hot(targets.long(), num_classes=n_classes).to(dtype=probs.dtype)
        # soft_tp[c] = Σ_i p_i(c) y_i(c)
        soft_tp = (probs * y).sum(dim=0)
        # soft_fp[c] = Σ_i p_i(c) (1 - y_i(c))
        soft_fp = (probs * (1.0 - y)).sum(dim=0)
        # soft_fn[c] = Σ_i (1 - p_i(c)) y_i(c)
        soft_fn = ((1.0 - probs) * y).sum(dim=0)

        prec_den = soft_tp + soft_fp
        rec_den = soft_tp + soft_fn
        prec = torch.where(prec_den > 0, soft_tp / prec_den, torch.zeros_like(soft_tp))
        rec = torch.where(rec_den > 0, soft_tp / rec_den, torch.zeros_like(soft_tp))
        return float(prec.mean().item()), float(rec.mean().item())

    def _evaluate_individual(self, flat: np.ndarray) -> np.ndarray:
        self.set_weights(flat)
        probs, targets = self._proba_pooled()
        precision, recall = self._soft_macro_precision_recall(probs, targets)
        return np.asarray([1.0 - precision, 1.0 - recall], dtype=np.float64)


# Backward-compatible alias.
CNNWeightProblem = PerClassCEProblem


def build_problem(
    name: str,
    model: nn.Module,
    batch_sampler: BatchSampler,
    device: torch.device | None = None,
    resample_every: int = 50,
    eval_mode: str = "single",
    eval_batches: int = 8,
) -> WeightOptimizationProblem:
    name = name.lower()
    kwargs = dict(
        model=model,
        batch_sampler=batch_sampler,
        device=device,
        resample_every=resample_every,
        eval_mode=eval_mode,
        eval_batches=eval_batches,
    )
    if name == "per_class_ce":
        return PerClassCEProblem(**kwargs)
    if name == "precision_recall":
        return PrecisionRecallProblem(**kwargs)
    if name == "soft_precision_recall":
        return SoftPrecisionRecallProblem(**kwargs)
    raise ValueError(f"Unknown problem: {name!r}. Use one of {PROBLEMS}.")


def build_eval_sampler(
    problem_name: str,
    dataset,
    batch_size: int,
    num_classes: int,
    seed: int | None = None,
) -> BatchSampler:
    """P/R problems use random batches; per-class CE keeps class-guaranteed."""
    if problem_name in PR_PROBLEMS:
        return RandomBatchSampler(
            dataset, batch_size=batch_size, num_classes=num_classes, seed=seed
        )
    return ClassGuaranteedBatchSampler(
        dataset, batch_size=batch_size, num_classes=num_classes, seed=seed
    )
