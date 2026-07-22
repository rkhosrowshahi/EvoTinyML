"""pymoo multi-objective problems over CNN weight vectors."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from pymoo.core.problem import Problem
from torch import nn

from evotinyml.data import EVAL_MODES, ClassBalancedSampler, RandomBatchSampler
from evotinyml.fitness import (
    CESoftPrecisionRecallMetrics,
    CrossEntropyL1Metrics,
    CrossEntropyMetrics,
    F1L1Metrics,
    F1Metrics,
    MOOFitness,
    ProblemObjectiveMetrics,
    SOOFitness,
    SoftF1L1Metrics,
    SoftF1Metrics,
    SoftPrecisionRecallMetrics,
    default_scalar_weights,
    parse_scalar_weights,
)

PROBLEMS = (
    "cwrm_cross_entropy",
    "precision_recall",
    "soft_precision_recall",
    "ce_soft_precision_recall",
    "erm_cross_entropy",
    "erm_f1",
    "erm_soft_f1",
    "cross_entropy_l1",
    "f1_l1",
    "soft_f1_l1",
)
# Problems that expose a 2-obj (1-P, 1-R) front in train/display/W&B.
PR_PROBLEMS = frozenset({"precision_recall", "soft_precision_recall"})
# 3-obj CE + soft P/R (minimized as CE, 1-P, 1-R).
CE_SOFT_PR_PROBLEMS = frozenset({"ce_soft_precision_recall"})
# Bi-objective task + mean |θ| sparsity (no new genotype vars).
L1_PROBLEMS = frozenset({"cross_entropy_l1", "f1_l1", "soft_f1_l1"})
# Cross-entropy problems that log per-class acc / CE.
CE_PROBLEMS = frozenset({"cwrm_cross_entropy", "erm_cross_entropy"})
# All problems accept SOO ES (multi-obj via weighted-sum scalarization).
SOO_PROBLEMS = frozenset(PROBLEMS)
# SOO-only (reject NSGA / MO-ES).
SOO_ONLY_PROBLEMS = frozenset({"erm_cross_entropy", "erm_f1", "erm_soft_f1"})
# Multi-objective problems (vector F; SOO uses w·F).
MOO_PROBLEMS = frozenset(p for p in PROBLEMS if p not in SOO_ONLY_PROBLEMS)
# Backward-compatible CLI aliases → canonical problem name.
PROBLEM_ALIASES = {
    "per_class_ce": "cwrm_cross_entropy",
    "cwce": "cwrm_cross_entropy",
    "cross_entropy": "erm_cross_entropy",
    "f1": "erm_f1",
    "soft_f1": "erm_soft_f1",
    "ce_soft_pr": "ce_soft_precision_recall",
    "ce_l1": "cross_entropy_l1",
}


def problem_obj_labels(problem_name: str, n_obj: int) -> list[str] | None:
    """Axis labels for Pareto plots, or ``None`` to use plot defaults."""
    name = PROBLEM_ALIASES.get(problem_name.lower(), problem_name.lower())
    if name in CE_SOFT_PR_PROBLEMS:
        return list(CESoftPrecisionRecallMetrics.OBJ_LABELS)
    if name == "cross_entropy_l1":
        return list(CrossEntropyL1Metrics.OBJ_LABELS)
    if name == "f1_l1":
        return list(F1L1Metrics.OBJ_LABELS)
    if name == "soft_f1_l1":
        return list(SoftF1L1Metrics.OBJ_LABELS)
    if name in CE_PROBLEMS and name != "erm_cross_entropy":
        return [f"Class {i}" for i in range(int(n_obj))]
    return None


class BatchSampler(Protocol):
    num_classes: int

    def sample_batch(self) -> tuple[torch.Tensor, torch.Tensor]: ...

    def sample_batches(self, n_batches: int) -> list[tuple[torch.Tensor, torch.Tensor]]: ...


class WeightOptimizationProblem(Problem):
    """Shared machinery for evolving TinyCNN weights on an eval batch pool.

    The pool is drawn by the batch sampler and shared by all individuals in one
    evaluation call (CRN). Call ``sample_eval_pool`` once per generation.
    ``eval_mode`` only controls pool *size* (one batch vs many).
    """

    def __init__(
        self,
        model: nn.Module,
        batch_sampler: BatchSampler,
        n_obj: int,
        xl: float = -10.0,
        xu: float = 10.0,
        device: torch.device | None = None,
        eval_mode: str = "multi",
        eval_batches: int = 50,
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

        self.hv_ideal: np.ndarray | None = None
        self.hv_nadir: np.ndarray | None = None
        self.problem_name = "weight"

        self._param_shapes = [tuple(p.shape) for p in self.model.parameters()]
        self._n_var = sum(p.numel() for p in self.model.parameters())
        # Snapshot of PyTorch default weights at construction (before any set_weights).
        self.theta0 = np.concatenate(
            [p.detach().cpu().numpy().ravel() for p in self.model.parameters()]
        ).astype(np.float64)

        # Eval batch pool shared by the whole population (CRN within a generation).
        # Drawn via the sampler each generation (see sample_eval_pool).
        self.eval_batch_pool: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.sample_eval_pool()

        super().__init__(n_var=self._n_var, n_obj=n_obj, xl=xl, xu=xu)

    def sample_eval_pool(self) -> None:
        """Sample a fresh eval minibatch pool with ``batch_sampler``.

        ``eval_mode=single`` → one batch of ``batch_size``; ``multi`` →
        ``eval_batches`` batches pooled for metrics. Call once per generation
        so the population shares the same data (CRN); do not call between
        population fitness and the reported mean/best on that generation.
        """
        raw = self.batch_sampler.sample_batches(self.eval_batches)
        self.eval_batch_pool = [
            (inputs.to(self.device), targets.to(self.device)) for inputs, targets in raw
        ]

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
        # Fresh minibatch(es) via the sampler once per pymoo evaluation call.
        self.sample_eval_pool()
        F_mat = np.empty((X.shape[0], self.n_obj), dtype=np.float64)
        for i in range(X.shape[0]):
            F_mat[i] = self._evaluate_individual(X[i])
        out["F"] = F_mat


class CWRMCrossEntropyProblem(WeightOptimizationProblem):
    """Class-wise risk minimization with cross-entropy (CWRM–CE).

    One minimization objective per class: mean CE on pooled eval samples of
    that class.
    """

    def __init__(
        self,
        model: nn.Module,
        batch_sampler: BatchSampler,
        xl: float = -10.0,
        xu: float = 10.0,
        device: torch.device | None = None,
        eval_mode: str = "multi",
        eval_batches: int = 50,
    ) -> None:
        super().__init__(
            model=model,
            batch_sampler=batch_sampler,
            n_obj=batch_sampler.num_classes,
            xl=xl,
            xu=xu,
            device=device,
            eval_mode=eval_mode,
            eval_batches=eval_batches,
        )
        self.problem_name = "cwrm_cross_entropy"
        self.metrics = ProblemObjectiveMetrics(self, fitness_name="cwrm_ce")
        self.soo_fitness = SOOFitness(
            self.metrics,
            scalar_weights=default_scalar_weights(self.problem_name, self.n_obj),
        )

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


# Backward-compatible aliases.
CWCEProblem = CWRMCrossEntropyProblem
PerClassCEProblem = CWRMCrossEntropyProblem
CNNWeightProblem = CWRMCrossEntropyProblem


class PrecisionRecallProblem(WeightOptimizationProblem):
    """Two-objective problem: minimize (1 - macro precision) and (1 - macro recall).

    Predictions are collected over the entire eval batch pool first; precision and
    recall are computed once on the pooled labels/predictions (not per mini-batch).
    """

    def __init__(
        self,
        model: nn.Module,
        batch_sampler: BatchSampler,
        xl: float = -10.0,
        xu: float = 10.0,
        device: torch.device | None = None,
        eval_mode: str = "multi",
        eval_batches: int = 50,
    ) -> None:
        super().__init__(
            model=model,
            batch_sampler=batch_sampler,
            n_obj=2,
            xl=xl,
            xu=xu,
            device=device,
            eval_mode=eval_mode,
            eval_batches=eval_batches,
        )
        self.problem_name = "precision_recall"
        self.metrics = ProblemObjectiveMetrics(self, fitness_name="1-P,1-R")
        self.soo_fitness = SOOFitness(
            self.metrics,
            scalar_weights=default_scalar_weights(self.problem_name, self.n_obj),
        )

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

    Uses softmax probabilities instead of hard argmax counts via
    :class:`~evotinyml.fitness.SoftPrecisionRecallMetrics`. Objectives are
    ``(1 - macro_P, 1 - macro_R)``.
    """

    def __init__(
        self,
        model: nn.Module,
        batch_sampler: BatchSampler,
        xl: float = -10.0,
        xu: float = 10.0,
        device: torch.device | None = None,
        eval_mode: str = "multi",
        eval_batches: int = 50,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(
            model=model,
            batch_sampler=batch_sampler,
            n_obj=2,
            xl=xl,
            xu=xu,
            device=device,
            eval_mode=eval_mode,
            eval_batches=eval_batches,
        )
        self.temperature = float(temperature)
        self.problem_name = "soft_precision_recall"
        self.metrics = SoftPrecisionRecallMetrics(self, temperature=temperature)
        self.moo_fitness = MOOFitness(self.metrics)
        self.soo_fitness = SOOFitness(
            self.metrics,
            scalar_weights=default_scalar_weights(self.problem_name, self.n_obj),
        )

    def _evaluate_individual(self, flat: np.ndarray) -> np.ndarray:
        return self.moo_fitness.evaluate_one(flat)


class CESoftPrecisionRecallProblem(WeightOptimizationProblem):
    """Three-objective problem: mean CE, soft precision, soft recall.

    Minimization vector ``F = (CE, 1−soft_P, 1−soft_R)`` computed from one
    forward on the eval pool
    (:class:`~evotinyml.fitness.CESoftPrecisionRecallMetrics`).
    """

    def __init__(
        self,
        model: nn.Module,
        batch_sampler: BatchSampler,
        xl: float = -10.0,
        xu: float = 10.0,
        device: torch.device | None = None,
        eval_mode: str = "multi",
        eval_batches: int = 50,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(
            model=model,
            batch_sampler=batch_sampler,
            n_obj=3,
            xl=xl,
            xu=xu,
            device=device,
            eval_mode=eval_mode,
            eval_batches=eval_batches,
        )
        self.temperature = float(temperature)
        self.problem_name = "ce_soft_precision_recall"
        self.metrics = CESoftPrecisionRecallMetrics(self, temperature=temperature)
        self.moo_fitness = MOOFitness(self.metrics)
        self.soo_fitness = SOOFitness(
            self.metrics,
            scalar_weights=default_scalar_weights(self.problem_name, self.n_obj),
        )

    def _evaluate_individual(self, flat: np.ndarray) -> np.ndarray:
        return self.moo_fitness.evaluate_one(flat)


class ERMCrossEntropyProblem(WeightOptimizationProblem):
    """Empirical risk minimization with mean cross-entropy (ERM–CE).

    Single-objective mean CE on the eval batch pool. Intended for SOO ES
    (``--algo cmaes`` / ``snes`` / ``xnes`` / ``open_es`` / ``cr_fm_nes`` /
    ``asebo`` / ``lm_ma_es`` / ``de`` / ``jde`` / ``pso``). Not for NSGA.
    """

    def __init__(
        self,
        model: nn.Module,
        batch_sampler: BatchSampler,
        xl: float = -10.0,
        xu: float = 10.0,
        device: torch.device | None = None,
        eval_mode: str = "multi",
        eval_batches: int = 50,
    ) -> None:
        super().__init__(
            model=model,
            batch_sampler=batch_sampler,
            n_obj=1,
            xl=xl,
            xu=xu,
            device=device,
            eval_mode=eval_mode,
            eval_batches=eval_batches,
        )
        self.problem_name = "erm_cross_entropy"
        self.metrics = CrossEntropyMetrics(self)
        self.soo_fitness = SOOFitness(self.metrics)

    def _evaluate_individual(self, flat: np.ndarray) -> np.ndarray:
        return np.asarray([self.soo_fitness.evaluate_one(flat)["f"]], dtype=np.float64)


# Backward-compatible alias.
CrossEntropyProblem = ERMCrossEntropyProblem


class ERMF1Problem(WeightOptimizationProblem):
    """Empirical risk minimization with hard macro-F1 (ERM–F1).

    Single-objective ``f = 1 − macro-F1`` on the eval batch pool. Intended for
    SOO ES; not for NSGA.
    """

    def __init__(
        self,
        model: nn.Module,
        batch_sampler: BatchSampler,
        xl: float = -10.0,
        xu: float = 10.0,
        device: torch.device | None = None,
        eval_mode: str = "multi",
        eval_batches: int = 50,
    ) -> None:
        super().__init__(
            model=model,
            batch_sampler=batch_sampler,
            n_obj=1,
            xl=xl,
            xu=xu,
            device=device,
            eval_mode=eval_mode,
            eval_batches=eval_batches,
        )
        self.problem_name = "erm_f1"
        self.metrics = F1Metrics(self)
        self.soo_fitness = SOOFitness(self.metrics)

    def _evaluate_individual(self, flat: np.ndarray) -> np.ndarray:
        return np.asarray([self.soo_fitness.evaluate_one(flat)["f"]], dtype=np.float64)


class ERMSoftF1Problem(WeightOptimizationProblem):
    """Empirical risk minimization with soft macro-F1 (ERM–soft-F1).

    Single-objective ``f = 1 − soft macro-F1`` on the eval batch pool. Intended
    for SOO ES; not for NSGA.
    """

    def __init__(
        self,
        model: nn.Module,
        batch_sampler: BatchSampler,
        xl: float = -10.0,
        xu: float = 10.0,
        device: torch.device | None = None,
        eval_mode: str = "multi",
        eval_batches: int = 50,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(
            model=model,
            batch_sampler=batch_sampler,
            n_obj=1,
            xl=xl,
            xu=xu,
            device=device,
            eval_mode=eval_mode,
            eval_batches=eval_batches,
        )
        self.temperature = float(temperature)
        self.problem_name = "erm_soft_f1"
        self.metrics = SoftF1Metrics(self, temperature=temperature)
        self.soo_fitness = SOOFitness(self.metrics)

    def _evaluate_individual(self, flat: np.ndarray) -> np.ndarray:
        return np.asarray([self.soo_fitness.evaluate_one(flat)["f"]], dtype=np.float64)


class CrossEntropyL1Problem(WeightOptimizationProblem):
    """Two-objective: mean CE and mean |θ| (``F = (CE, L1)``)."""

    def __init__(
        self,
        model: nn.Module,
        batch_sampler: BatchSampler,
        xl: float = -10.0,
        xu: float = 10.0,
        device: torch.device | None = None,
        eval_mode: str = "multi",
        eval_batches: int = 50,
    ) -> None:
        super().__init__(
            model=model,
            batch_sampler=batch_sampler,
            n_obj=2,
            xl=xl,
            xu=xu,
            device=device,
            eval_mode=eval_mode,
            eval_batches=eval_batches,
        )
        self.problem_name = "cross_entropy_l1"
        self.metrics = CrossEntropyL1Metrics(self)
        self.moo_fitness = MOOFitness(self.metrics)
        self.soo_fitness = SOOFitness(
            self.metrics,
            scalar_weights=default_scalar_weights(self.problem_name, self.n_obj),
        )

    def _evaluate_individual(self, flat: np.ndarray) -> np.ndarray:
        return self.moo_fitness.evaluate_one(flat)


class F1L1Problem(WeightOptimizationProblem):
    """Two-objective: hard macro-F1 and mean |θ| (``F = (1−F1, L1)``)."""

    def __init__(
        self,
        model: nn.Module,
        batch_sampler: BatchSampler,
        xl: float = -10.0,
        xu: float = 10.0,
        device: torch.device | None = None,
        eval_mode: str = "multi",
        eval_batches: int = 50,
    ) -> None:
        super().__init__(
            model=model,
            batch_sampler=batch_sampler,
            n_obj=2,
            xl=xl,
            xu=xu,
            device=device,
            eval_mode=eval_mode,
            eval_batches=eval_batches,
        )
        self.problem_name = "f1_l1"
        self.metrics = F1L1Metrics(self)
        self.moo_fitness = MOOFitness(self.metrics)
        self.soo_fitness = SOOFitness(
            self.metrics,
            scalar_weights=default_scalar_weights(self.problem_name, self.n_obj),
        )

    def _evaluate_individual(self, flat: np.ndarray) -> np.ndarray:
        return self.moo_fitness.evaluate_one(flat)


class SoftF1L1Problem(WeightOptimizationProblem):
    """Two-objective: soft macro-F1 and mean |θ| (``F = (1−soft F1, L1)``)."""

    def __init__(
        self,
        model: nn.Module,
        batch_sampler: BatchSampler,
        xl: float = -10.0,
        xu: float = 10.0,
        device: torch.device | None = None,
        eval_mode: str = "multi",
        eval_batches: int = 50,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(
            model=model,
            batch_sampler=batch_sampler,
            n_obj=2,
            xl=xl,
            xu=xu,
            device=device,
            eval_mode=eval_mode,
            eval_batches=eval_batches,
        )
        self.temperature = float(temperature)
        self.problem_name = "soft_f1_l1"
        self.metrics = SoftF1L1Metrics(self, temperature=temperature)
        self.moo_fitness = MOOFitness(self.metrics)
        self.soo_fitness = SOOFitness(
            self.metrics,
            scalar_weights=default_scalar_weights(self.problem_name, self.n_obj),
        )

    def _evaluate_individual(self, flat: np.ndarray) -> np.ndarray:
        return self.moo_fitness.evaluate_one(flat)


DEFAULT_XL = -10.0
DEFAULT_XU = 10.0


def apply_scalar_weights(
    problem: WeightOptimizationProblem,
    weights_spec: str | list[float] | tuple[float, ...] | None,
) -> np.ndarray:
    """Set SOO weighted-sum weights on ``problem.soo_fitness``; return the weights used."""
    soo = getattr(problem, "soo_fitness", None)
    if soo is None:
        raise ValueError(
            f"Problem {type(problem).__name__} has no soo_fitness for scalarization."
        )
    name = getattr(problem, "problem_name", "")
    w = parse_scalar_weights(weights_spec, int(problem.n_obj), problem_name=name)
    soo.set_scalar_weights(w)
    return w


def build_problem(
    name: str,
    model: nn.Module,
    batch_sampler: BatchSampler,
    device: torch.device | None = None,
    eval_mode: str = "multi",
    eval_batches: int = 50,
    xl: float = DEFAULT_XL,
    xu: float = DEFAULT_XU,
) -> WeightOptimizationProblem:
    name = PROBLEM_ALIASES.get(name.lower(), name.lower())
    xl = float(xl)
    xu = float(xu)
    if not (xu > xl):
        raise ValueError(f"Need xu > xl, got xl={xl}, xu={xu}")
    kwargs = dict(
        model=model,
        batch_sampler=batch_sampler,
        device=device,
        eval_mode=eval_mode,
        eval_batches=eval_batches,
        xl=xl,
        xu=xu,
    )
    if name == "cwrm_cross_entropy":
        return CWRMCrossEntropyProblem(**kwargs)
    if name == "precision_recall":
        return PrecisionRecallProblem(**kwargs)
    if name == "soft_precision_recall":
        return SoftPrecisionRecallProblem(**kwargs)
    if name == "ce_soft_precision_recall":
        return CESoftPrecisionRecallProblem(**kwargs)
    if name == "erm_cross_entropy":
        return ERMCrossEntropyProblem(**kwargs)
    if name == "erm_f1":
        return ERMF1Problem(**kwargs)
    if name == "erm_soft_f1":
        return ERMSoftF1Problem(**kwargs)
    if name == "cross_entropy_l1":
        return CrossEntropyL1Problem(**kwargs)
    if name == "f1_l1":
        return F1L1Problem(**kwargs)
    if name == "soft_f1_l1":
        return SoftF1L1Problem(**kwargs)
    raise ValueError(f"Unknown problem: {name!r}. Use one of {PROBLEMS}.")


EVAL_SAMPLER_NAMES = ("auto", "random", "balanced")


def build_eval_sampler(
    problem_name: str,
    dataset,
    batch_size: int,
    num_classes: int,
    seed: int | None = None,
    sampler: str = "auto",
) -> BatchSampler:
    """Build a ``RandomBatchSampler`` or ``ClassBalancedSampler``.

    ``sampler="auto"``: CWRM–CE uses class-balanced batches; P/R, CE+soft-P/R,
    task+L1, and ERM use uniform random batches.
    """
    name = str(sampler).lower()
    if name not in EVAL_SAMPLER_NAMES:
        raise ValueError(
            f"Unknown sampler: {sampler!r}. Use one of {EVAL_SAMPLER_NAMES}."
        )
    problem_name = PROBLEM_ALIASES.get(problem_name.lower(), problem_name.lower())
    if name == "auto":
        use_balanced = not (
            problem_name in PR_PROBLEMS
            or problem_name in CE_SOFT_PR_PROBLEMS
            or problem_name in L1_PROBLEMS
            or problem_name in SOO_ONLY_PROBLEMS
        )
    else:
        use_balanced = name == "balanced"
    cls = ClassBalancedSampler if use_balanced else RandomBatchSampler
    return cls(dataset, batch_size=batch_size, num_classes=num_classes, seed=seed)