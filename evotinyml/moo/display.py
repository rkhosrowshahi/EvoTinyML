"""Console display that reports progress in steps, not generations."""

from __future__ import annotations

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from torch.utils.data import DataLoader

from evotinyml.moo.pareto_history import FrontHistory
from evotinyml.moo.pareto_plot import pareto_front_images, val_class_pareto_images
from evotinyml.fitness import (
    L1_HV_REF,
    l1_problem_metric_log_dict,
    l1_val_front,
)
from evotinyml.problem import (
    CE_PROBLEMS,
    CE_SOFT_PR_PROBLEMS,
    L1_PROBLEMS,
    PR_PROBLEMS,
    WeightOptimizationProblem,
    problem_obj_labels,
)
from evotinyml.validation import (
    class_metric_log_dict,
    per_class_acc_ce_on_eval_pool,
    validate_nd_set,
)
from evotinyml.wandb_logger import log_metrics, to_wandb_step


def _knee_index_utopia(F: np.ndarray) -> int:
    """Index closest to the origin in L2 (utopia for minimization objectives)."""
    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        return 0
    if len(F) == 0:
        return 0
    return int(np.argmin(np.sum(np.square(F), axis=1)))


def _normalize_by_nadir(F: np.ndarray, nadir: np.ndarray) -> np.ndarray:
    """Scale objectives by the nadir so the unit vector is the HV reference."""
    scale = np.maximum(np.asarray(nadir, dtype=float), 1e-12)
    return np.asarray(F, dtype=float) / scale


def _monte_carlo_hv_unit(
    F_norm: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    """Hit-or-miss HV for minimization with ref = ones (unit hypercube)."""
    mask = np.all(F_norm < 1.0, axis=1)
    F = np.maximum(F_norm[mask], 0.0)
    if len(F) == 0:
        return 0.0

    n_obj = F.shape[1]
    samples = rng.uniform(0.0, 1.0, size=(n_samples, n_obj))
    covered = np.any(np.all(F[:, None, :] <= samples[None, :, :], axis=2), axis=0)
    return float(covered.mean())


def _update_nd_archive(archive: np.ndarray | None, F: np.ndarray) -> np.ndarray:
    """Merge ``F`` into the archive and keep only non-dominated points."""
    if archive is None or len(archive) == 0:
        merged = np.asarray(F, dtype=float)
    else:
        merged = np.vstack([archive, np.asarray(F, dtype=float)])

    nd_idx = NonDominatedSorting().do(merged, only_non_dominated_front=True)
    return merged[nd_idx]


def _hv_unit_front(
    F: np.ndarray,
    *,
    n_obj: int,
    hv_exact: HV | None,
    hv_samples: int,
    hv_seed: int,
) -> float:
    """HV in [0, 1] for a minimization front already scaled to the unit cube."""
    if F is None or len(F) == 0:
        return 0.0
    if hv_exact is not None or n_obj <= 3:
        ind = hv_exact or HV(ref_point=np.ones(n_obj), norm_ref_point=False)
        mask = np.all(F < 1.0, axis=1)
        F_dom = F[mask]
        return 0.0 if len(F_dom) == 0 else float(ind(F_dom))
    rng = np.random.default_rng(hv_seed)
    return _monte_carlo_hv_unit(F, n_samples=hv_samples, rng=rng)


class StepOutput(Output):
    """Progress table with train HV and periodic test validation."""

    def __init__(
        self,
        hv_samples: int = 10_000,
        hv_seed: int = 1,
        test_loader: DataLoader | None = None,
        val_every: int = 50,
        n_classes: int = 10,
        problem_name: str = "cwrm_cross_entropy",
        use_wandb: bool = False,
        max_steps: int | None = None,
        pareto_every: int = 100,
        train_history_path: str = "train_history.npz",
        val_history_path: str = "val_history.npz",
    ) -> None:
        super().__init__()
        self.problem_name = problem_name
        self.is_precision_recall = problem_name in PR_PROBLEMS
        self.is_ce_problem = problem_name in CE_PROBLEMS
        self.is_ce_soft_pr = problem_name in CE_SOFT_PR_PROBLEMS
        self.is_l1_problem = problem_name in L1_PROBLEMS
        self.use_wandb = use_wandb
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.pareto_every = max(1, int(pareto_every))
        # Always persist train/val front histories for P–R problems.
        self.train_history = (
            FrontHistory(train_history_path) if self.is_precision_recall else None
        )
        self.val_history = (
            FrontHistory(val_history_path) if self.is_precision_recall else None
        )
        # Alias used by older call sites / prints.
        self.pareto_history = self.train_history

        self.step = Column("step", width=8)
        self.n_eval = Column("n_eval", width=10)
        self.n_nds = Column("n_nds", width=8)
        self.hv = Column("hv", width=10)

        if self.is_precision_recall:
            self.pf_p_min = Column("pf_P_min", width=10)
            self.pf_p_max = Column("pf_P_max", width=10)
            self.pf_r_min = Column("pf_R_min", width=10)
            self.pf_r_max = Column("pf_R_max", width=10)
            self.pf_prm_min = Column("pf_PRm_min", width=10)
            self.pf_prm_max = Column("pf_PRm_max", width=10)
            self.knee_p = Column("knee_P", width=10)
            self.knee_r = Column("knee_R", width=10)
            self.knee_pr_mean = Column("knee_PR_mean", width=12)
            self.columns = [
                self.step,
                self.n_eval,
                self.n_nds,
                self.pf_p_min,
                self.pf_p_max,
                self.pf_r_min,
                self.pf_r_max,
                self.pf_prm_min,
                self.pf_prm_max,
                self.knee_p,
                self.knee_r,
                self.knee_pr_mean,
                self.hv,
            ]
        else:
            self.mean_f = Column("mean_f", width=12)
            self.min_f = Column("min_f", width=12)
            self.columns = [
                self.step,
                self.n_eval,
                self.n_nds,
                self.mean_f,
                self.min_f,
                self.hv,
            ]
            if self.is_l1_problem:
                if self.problem_name == "cross_entropy_l1":
                    knee_task_name = "knee_CE"
                elif self.problem_name == "soft_f1_l1":
                    knee_task_name = "knee_softF1"
                else:
                    knee_task_name = "knee_F1"
                self.knee_task = Column(knee_task_name, width=max(12, len(knee_task_name) + 2))
                self.knee_l1 = Column("knee_L1", width=10)
                self.columns[4:4] = [self.knee_task, self.knee_l1]

        self.hv_samples = hv_samples
        self.hv_seed = hv_seed
        self.test_loader = test_loader
        self.val_every = max(1, int(val_every))
        self.n_classes = n_classes

        self.nadir: np.ndarray | None = None
        self.ref_point: np.ndarray | None = None
        self._archive: np.ndarray | None = None
        self._hv_exact: HV | None = None
        self._last_hv: float = 0.0
        self._last_val_acc: float | None = None
        self._last_val_f1: float | None = None
        self._last_best_acc: float | None = None
        self._last_best_f1: float | None = None
        self._last_hv_acc: float | None = None
        self._last_hv_f1: float | None = None
        self._last_hv_l1: float | None = None
        self._last_knee_prec: float | None = None
        self._last_knee_rec: float | None = None
        self._last_knee_pr_mean: float | None = None
        self._last_knee_f1: float | None = None
        self._last_knee_acc: float | None = None
        self._last_knee_l1: float | None = None
        self._last_hv_pr: float | None = None
        self._last_val_n_nd: int | None = None
        self._last_wandb_n_eval: int | None = None
        self._validated_this_step: bool = False
        self._last_class_train: dict[str, float] = {}
        self._last_class_val: dict[str, float] = {}
        self._last_l1_train: dict[str, float] = {}
        self._last_l1_val: dict[str, float] = {}
        self._pending_pareto_images: dict = {}

    def collect_train_metrics(self, algorithm) -> dict[str, float | int]:
        n_eval = getattr(getattr(algorithm, "evaluator", None), "n_eval", None)
        F_opt = algorithm.opt.get("F")
        if F_opt is None or len(F_opt) == 0:
            return {}
        F_opt = np.asarray(F_opt, dtype=float)
        opt_step = to_wandb_step(algorithm.n_gen, self.max_steps)
        metrics: dict[str, float | int] = {
            "train/n_eval": int(n_eval) if n_eval is not None else 0,
            "train/step": int(opt_step),
            "train/n_nds": len(F_opt),
            "train/hv": float(self._last_hv),
        }
        if self.is_precision_recall and F_opt.ndim == 2 and F_opt.shape[1] >= 2:
            precision = 1.0 - F_opt[:, 0]
            recall = 1.0 - F_opt[:, 1]
            pr_mean = 0.5 * (precision + recall)
            metrics["train/pf_precision_min"] = float(np.min(precision))
            metrics["train/pf_precision_max"] = float(np.max(precision))
            metrics["train/pf_recall_min"] = float(np.min(recall))
            metrics["train/pf_recall_max"] = float(np.max(recall))
            metrics["train/pf_pr_mean_min"] = float(np.min(pr_mean))
            metrics["train/pf_pr_mean_max"] = float(np.max(pr_mean))
            # Shared with SOO: train knee (closest to P=R=1) as scalar report.
            knee_i = int(np.argmin(np.sum(np.square(F_opt[:, :2]), axis=1)))
            metrics["train/precision"] = float(precision[knee_i])
            metrics["train/recall"] = float(recall[knee_i])
            metrics["train/f"] = float(F_opt[knee_i, 0] + F_opt[knee_i, 1])
            metrics["train/mean_f"] = float(metrics["train/f"])
        elif self.is_ce_soft_pr and F_opt.ndim == 2 and F_opt.shape[1] >= 3:
            knee_i = _knee_index_utopia(F_opt)
            metrics["train/ce"] = float(F_opt[knee_i, 0])
            metrics["train/precision"] = float(1.0 - F_opt[knee_i, 1])
            metrics["train/recall"] = float(1.0 - F_opt[knee_i, 2])
            metrics["train/f"] = float(np.mean(F_opt[knee_i]))
            metrics["train/mean_f"] = float(np.mean(F_opt, axis=0).mean())
            metrics["train/min_f"] = float(np.min(np.mean(F_opt, axis=1)))
            metrics["train/knee_f"] = float(metrics["train/f"])
        elif self.is_l1_problem and F_opt.ndim == 2 and F_opt.shape[1] >= 2:
            knee_i = _knee_index_utopia(F_opt)
            self._last_l1_train = l1_problem_metric_log_dict(
                "train", self.problem_name, F_opt[knee_i]
            )
            metrics.update(self._last_l1_train)
            metrics["train/f"] = float(np.mean(F_opt[knee_i]))
            metrics["train/mean_f"] = float(np.mean(F_opt, axis=0).mean())
            metrics["train/min_f"] = float(np.min(np.mean(F_opt, axis=1)))
            metrics["train/knee_f"] = float(metrics["train/f"])
            metrics["train/knee_l1"] = float(F_opt[knee_i, 1])
        else:
            per_ind = np.mean(F_opt, axis=1)
            metrics["train/mean_f"] = float(np.mean(per_ind))
            metrics["train/min_f"] = float(np.min(per_ind))
            # CWRM: knee = closest to utopia (all CE=0) in L2.
            knee_i = _knee_index_utopia(F_opt)
            metrics["train/f"] = float(per_ind[knee_i])
            metrics["train/knee_f"] = float(per_ind[knee_i])
            if self.is_ce_problem:
                X_opt = algorithm.opt.get("X")
                problem = algorithm.problem
                if (
                    X_opt is not None
                    and len(X_opt) > knee_i
                    and isinstance(problem, WeightOptimizationProblem)
                ):
                    acc_c, ce_c = per_class_acc_ce_on_eval_pool(problem, X_opt[knee_i])
                    self._last_class_train = class_metric_log_dict("train", acc_c, ce_c)
                    metrics.update(self._last_class_train)
        return metrics

    def collect_val_metrics(self) -> dict[str, float | int]:
        if self._last_best_acc is None:
            return {}
        metrics: dict[str, float | int] = {
            "val/acc_best": float(self._last_best_acc),
            "val/f1_best": float(self._last_best_f1 or 0.0),
        }
        if self._last_val_n_nd is not None:
            metrics["val/n_nd"] = int(self._last_val_n_nd)
        if self.is_precision_recall:
            if self._last_knee_prec is not None:
                metrics["val/knee_precision"] = float(self._last_knee_prec)
                metrics["val/precision"] = float(self._last_knee_prec)
            if self._last_knee_rec is not None:
                metrics["val/knee_recall"] = float(self._last_knee_rec)
                metrics["val/recall"] = float(self._last_knee_rec)
            if self._last_knee_pr_mean is not None:
                metrics["val/knee_pr_mean"] = float(self._last_knee_pr_mean)
            if self._last_knee_f1 is not None:
                metrics["val/knee_f1"] = float(self._last_knee_f1)
                metrics["val/f1"] = float(self._last_knee_f1)
            if self._last_knee_acc is not None:
                metrics["val/knee_acc"] = float(self._last_knee_acc)
                metrics["val/acc"] = float(self._last_knee_acc)
            if self._last_hv_pr is not None:
                metrics["val/hv_pr"] = float(self._last_hv_pr)
        else:
            # CWRM / other: shared val/acc is the knee individual (not best-acc).
            if self._last_knee_acc is not None:
                metrics["val/acc"] = float(self._last_knee_acc)
                metrics["val/knee_acc"] = float(self._last_knee_acc)
            else:
                metrics["val/acc"] = float(self._last_best_acc)
            if self._last_knee_f1 is not None:
                metrics["val/f1"] = float(self._last_knee_f1)
                metrics["val/knee_f1"] = float(self._last_knee_f1)
            elif self._last_best_f1 is not None:
                metrics["val/f1"] = float(self._last_best_f1)
            if self._last_val_acc is not None:
                metrics["val/mean_acc"] = float(self._last_val_acc)
            if self._last_val_f1 is not None:
                metrics["val/mean_f1"] = float(self._last_val_f1)
            if self.is_l1_problem:
                if self._last_hv_l1 is not None:
                    metrics["val/hv"] = float(self._last_hv_l1)
            else:
                if self._last_hv_acc is not None:
                    metrics["val/hv_acc"] = float(self._last_hv_acc)
                if self._last_hv_f1 is not None:
                    metrics["val/hv_f1"] = float(self._last_hv_f1)
            metrics.update(self._last_class_val)
            metrics.update(self._last_l1_val)
            if self._last_knee_l1 is not None:
                metrics["val/knee_l1"] = float(self._last_knee_l1)
                metrics["val/l1"] = float(self._last_knee_l1)
        return metrics

    def final_summary_metrics(self) -> dict[str, float | int]:
        """Scalar summary fields for ``finish_wandb`` (knee + best test)."""
        out: dict[str, float | int] = {}
        if self._last_best_acc is not None:
            out["final/acc_best"] = float(self._last_best_acc)
        if self._last_best_f1 is not None:
            out["final/f1_best"] = float(self._last_best_f1)
        if self.is_precision_recall:
            if self._last_knee_prec is not None:
                out["final/knee_precision"] = float(self._last_knee_prec)
            if self._last_knee_rec is not None:
                out["final/knee_recall"] = float(self._last_knee_rec)
            if self._last_knee_pr_mean is not None:
                out["final/knee_pr_mean"] = float(self._last_knee_pr_mean)
            if self._last_knee_f1 is not None:
                out["final/knee_f1"] = float(self._last_knee_f1)
            if self._last_knee_acc is not None:
                out["final/knee_acc"] = float(self._last_knee_acc)
            if self._last_knee_l1 is not None:
                out["final/knee_l1"] = float(self._last_knee_l1)
        elif self._last_knee_acc is not None:
            out["final/knee_acc"] = float(self._last_knee_acc)
            if self._last_knee_f1 is not None:
                out["final/knee_f1"] = float(self._last_knee_f1)
            if self._last_knee_l1 is not None:
                out["final/knee_l1"] = float(self._last_knee_l1)
        return out

    def log_wandb(
        self,
        algorithm,
        *,
        wandb_step: int | None = None,
        force: bool = False,
        include_val: bool | None = None,
    ) -> None:
        if not self.use_wandb:
            return
        n_eval = getattr(getattr(algorithm, "evaluator", None), "n_eval", None)
        if n_eval is None:
            # Fallback if evaluator missing: approximate from opt step * popsize.
            opt_step = (
                to_wandb_step(algorithm.n_gen, self.max_steps)
                if wandb_step is None
                else int(wandb_step)
            )
            pop_size = len(algorithm.pop) if getattr(algorithm, "pop", None) is not None else 0
            n_eval = max(0, opt_step) * max(pop_size, 1)
        n_eval = int(n_eval)
        if not force and self._last_wandb_n_eval == n_eval:
            return

        metrics = self.collect_train_metrics(algorithm)
        if not metrics and not self._pending_pareto_images:
            return
        # Only emit val/* on steps where validation actually ran.
        if include_val if include_val is not None else self._validated_this_step:
            metrics.update(self.collect_val_metrics())
        if self._pending_pareto_images:
            metrics.update(self._pending_pareto_images)
            self._pending_pareto_images = {}
        log_metrics(metrics, n_eval=n_eval)
        self._last_wandb_n_eval = n_eval
        self._validated_this_step = False

    def _maybe_save_train_pareto(self, algorithm, *, step: int) -> None:
        """Persist P–R front history and queue a W&B Pareto plot."""
        # Snapshot at step 1, then every pareto_every.
        if step <= 0 or (step != 1 and step % self.pareto_every != 0):
            return
        F = algorithm.opt.get("F")
        if F is None or len(F) == 0:
            return
        F = np.asarray(F, dtype=float)
        if F.ndim != 2 or F.shape[1] < 2:
            return

        history = None
        if self.is_precision_recall and self.train_history is not None:
            precision = 1.0 - F[:, 0]
            recall = 1.0 - F[:, 1]
            self.train_history.append(step, precision, recall)
            history = self.train_history.iter_fronts()

        if self.use_wandb:
            labels = problem_obj_labels(self.problem_name, F.shape[1])
            # NSGA has no ES mean: center = average of the current ND front.
            center_F = np.mean(F, axis=0)
            self._pending_pareto_images.update(
                pareto_front_images(
                    F,
                    problem_name=self.problem_name,
                    step=step,
                    key_prefix="train",
                    history=history,
                    highlight=center_F,
                    highlight_label="center",
                    obj_labels=labels,
                )
            )


    def _ensure_hv_anchors(self, algorithm) -> None:
        if self.ref_point is not None:
            return

        # Task+L1: fixed ref [1, 1] on raw (task, L1); no nadir scaling.
        if self.is_l1_problem:
            self.nadir = L1_HV_REF.copy()
            self.ref_point = L1_HV_REF.copy()
            self._hv_exact = HV(ref_point=self.ref_point, norm_ref_point=False)
            print("HV: L1 problems use fixed ref = [1.0, 1.0] (no nadir scaling)")
            return

        problem = algorithm.problem
        if isinstance(problem, WeightOptimizationProblem) and problem.hv_nadir is not None:
            self.nadir = problem.hv_nadir.copy()
        else:
            F_init = algorithm.pop.get("F")
            if F_init is None or len(F_init) == 0:
                return
            F_init = np.asarray(F_init, dtype=float)
            self.nadir = F_init.max(axis=0).copy()
            if isinstance(problem, WeightOptimizationProblem):
                problem.set_hv_anchors_from_F(F_init)

        n_obj = int(self.nadir.shape[0])
        self.ref_point = np.ones(n_obj, dtype=float)
        print(
            "HV: F /= initial-population nadir, ref = ones "
            f"(n_obj={n_obj}); nadir="
            + np.array2string(self.nadir, precision=4, separator=", ")
        )
        if n_obj <= 3:
            self._hv_exact = HV(ref_point=self.ref_point, norm_ref_point=False)

    def _compute_train_hv(self, F: np.ndarray) -> float | None:
        if self.ref_point is None or F is None or len(F) == 0:
            return None
        F_use = np.asarray(F, dtype=float)
        if not self.is_l1_problem:
            if self.nadir is None:
                return None
            F_use = _normalize_by_nadir(F_use, self.nadir)
        return _hv_unit_front(
            F_use,
            n_obj=F_use.shape[1],
            hv_exact=self._hv_exact,
            hv_samples=self.hv_samples,
            hv_seed=self.hv_seed,
        )

    def _maybe_validate(self, algorithm, *, force: bool = False) -> None:
        if self.test_loader is None:
            return
        step = int(algorithm.n_gen)
        # Validate at step 1 (init / first reported gen), then every val_every.
        if not force and step > 0 and step != 1 and step % self.val_every != 0:
            return
        if not force and step <= 0:
            return

        problem = algorithm.problem
        if not isinstance(problem, WeightOptimizationProblem):
            return

        X = algorithm.opt.get("X")
        if X is None or len(X) == 0:
            return

        result = validate_nd_set(problem, X, self.test_loader, self.n_classes)
        self._last_best_acc = result.best_acc
        self._last_best_f1 = result.best_f1
        self._last_val_n_nd = len(X)
        self._validated_this_step = True

        if self.is_precision_recall:
            hv_pr = _hv_unit_front(
                result.error_pr,
                n_obj=2,
                hv_exact=HV(ref_point=np.ones(2), norm_ref_point=False),
                hv_samples=self.hv_samples,
                hv_seed=self.hv_seed,
            )
            knee = result.knee_metrics()
            self._last_knee_prec = knee["precision"]
            self._last_knee_rec = knee["recall"]
            self._last_knee_pr_mean = knee["pr_mean"]
            self._last_knee_f1 = knee["f1"]
            self._last_knee_acc = knee["acc"]
            self._last_hv_pr = hv_pr
            if self.val_history is not None:
                self.val_history.append(
                    step,
                    result.macro_precision,
                    result.macro_recall,
                    scalars={
                        "acc_best": float(result.best_acc),
                        "f1_best": float(result.best_f1),
                        "knee_precision": float(knee["precision"]),
                        "knee_recall": float(knee["recall"]),
                        "knee_pr_mean": float(knee["pr_mean"]),
                        "knee_f1": float(knee["f1"]),
                        "knee_acc": float(knee["acc"]),
                        "hv_pr": float(hv_pr),
                    },
                )
            if self.use_wandb:
                self._pending_pareto_images.update(
                    pareto_front_images(
                        result.error_pr,
                        problem_name=self.problem_name,
                        step=step,
                        key_prefix="val",
                        history=(
                            self.val_history.iter_fronts()
                            if self.val_history is not None
                            else None
                        ),
                        highlight=np.mean(result.error_pr, axis=0),
                        highlight_label="center",
                    )
                )
            print(
                f"[step {step}] test ND (n={len(X)}): "
                f"acc_best={result.best_acc:.4f}  f1_best={result.best_f1:.4f}  "
                f"knee_P={knee['precision']:.4f}  knee_R={knee['recall']:.4f}  "
                f"knee_pr_mean={knee['pr_mean']:.4f}  knee_f1={knee['f1']:.4f}  "
                f"knee_acc={knee['acc']:.4f}  HV_PR={hv_pr:.6f}"
            )
        elif self.is_l1_problem:
            F_val = l1_val_front(
                self.problem_name,
                X,
                mean_ce=result.mean_ce,
                macro_f1=result.macro_f1,
            )
            hv_l1 = _hv_unit_front(
                F_val,
                n_obj=2,
                hv_exact=HV(ref_point=L1_HV_REF, norm_ref_point=False),
                hv_samples=self.hv_samples,
                hv_seed=self.hv_seed,
            )
            knee_i = _knee_index_utopia(F_val)
            self._last_val_acc = result.mean_acc
            self._last_val_f1 = result.mean_f1
            self._last_hv_l1 = hv_l1
            self._last_knee_acc = float(result.overall_acc[knee_i])
            self._last_knee_f1 = float(result.macro_f1[knee_i])
            self._last_knee_l1 = float(F_val[knee_i, 1])
            self._last_l1_val = l1_problem_metric_log_dict(
                "val", self.problem_name, F_val[knee_i]
            )
            if self.use_wandb and (step == 1 or step % self.pareto_every == 0):
                labels = problem_obj_labels(self.problem_name, 2)
                self._pending_pareto_images.update(
                    pareto_front_images(
                        F_val,
                        problem_name=self.problem_name,
                        step=step,
                        key_prefix="val",
                        highlight=np.mean(F_val, axis=0),
                        highlight_label="center",
                        obj_labels=labels,
                    )
                )
            print(
                f"[step {step}] test ND (n={len(X)}): "
                f"acc_best={result.best_acc:.4f}  f1_best={result.best_f1:.4f}  "
                f"knee_acc={self._last_knee_acc:.4f}  knee_f1={self._last_knee_f1:.4f}  "
                f"knee_l1={self._last_knee_l1:.6f}  "
                f"mean_acc={result.mean_acc:.4f}  mean_f1={result.mean_f1:.4f}  "
                f"HV={hv_l1:.6f}"
            )
        else:
            hv_exact_cls = (
                self._hv_exact
                if (
                    self._hv_exact is not None
                    and self.ref_point is not None
                    and len(self.ref_point) == self.n_classes
                )
                else None
            )
            hv_acc = _hv_unit_front(
                result.error_acc,
                n_obj=self.n_classes,
                hv_exact=hv_exact_cls,
                hv_samples=self.hv_samples,
                hv_seed=self.hv_seed,
            )
            hv_f1 = _hv_unit_front(
                result.error_f1,
                n_obj=self.n_classes,
                hv_exact=hv_exact_cls,
                hv_samples=self.hv_samples,
                hv_seed=self.hv_seed,
            )
            self._last_val_acc = result.mean_acc
            self._last_val_f1 = result.mean_f1
            self._last_hv_acc = hv_acc
            self._last_hv_f1 = hv_f1
            # Knee on test class-wise CE (closest to utopia CE=0).
            knee_i = _knee_index_utopia(result.per_class_ce)
            self._last_knee_acc = float(result.overall_acc[knee_i])
            self._last_knee_f1 = float(result.macro_f1[knee_i])
            if self.is_ce_problem:
                self._last_class_val = class_metric_log_dict(
                    "val",
                    result.per_class_acc[knee_i],
                    result.per_class_ce[knee_i],
                )
            if self.use_wandb and self.is_ce_problem:
                # Log val CE + Acc fronts on the same cadence as train Pareto plots.
                if step == 1 or step % self.pareto_every == 0:
                    self._pending_pareto_images.update(
                        val_class_pareto_images(
                            result.per_class_ce,
                            result.per_class_acc,
                            step=step,
                        )
                    )
            print(
                f"[step {step}] test ND (n={len(X)}): "
                f"acc_best={result.best_acc:.4f}  f1_best={result.best_f1:.4f}  "
                f"knee_acc={self._last_knee_acc:.4f}  knee_f1={self._last_knee_f1:.4f}  "
                f"mean_acc={result.mean_acc:.4f}  mean_f1={result.mean_f1:.4f}  "
                f"HV_acc={hv_acc:.6f}  HV_F1={hv_f1:.6f}"
            )

    def _fmt_last(self, value: float | None) -> str:
        return "-" if value is None else f"{value:.4f}"

    def update(self, algorithm):
        super().update(algorithm)
        self.step.set(algorithm.n_gen)
        n_eval = getattr(getattr(algorithm, "evaluator", None), "n_eval", None)
        self.n_eval.set("-" if n_eval is None else int(n_eval))
        self._ensure_hv_anchors(algorithm)

        F_pop = algorithm.pop.get("F")
        F_opt = algorithm.opt.get("F")

        if F_opt is None or len(F_opt) == 0:
            self.n_nds.set("-")
            self.hv.set("-")
            if self.is_precision_recall:
                self.pf_p_min.set("-")
                self.pf_p_max.set("-")
                self.pf_r_min.set("-")
                self.pf_r_max.set("-")
                self.pf_prm_min.set("-")
                self.pf_prm_max.set("-")
                self.knee_p.set("-")
                self.knee_r.set("-")
                self.knee_pr_mean.set("-")
            else:
                self.mean_f.set("-")
                self.min_f.set("-")
                if self.is_l1_problem:
                    self.knee_task.set("-")
                    self.knee_l1.set("-")
            return

        F_opt = np.asarray(F_opt, dtype=float)
        self.n_nds.set(len(F_opt))
        if self.is_precision_recall and F_opt.ndim == 2 and F_opt.shape[1] >= 2:
            precision = 1.0 - F_opt[:, 0]
            recall = 1.0 - F_opt[:, 1]
            pr_mean = 0.5 * (precision + recall)
            self.pf_p_min.set(f"{float(np.min(precision)):.4f}")
            self.pf_p_max.set(f"{float(np.max(precision)):.4f}")
            self.pf_r_min.set(f"{float(np.min(recall)):.4f}")
            self.pf_r_max.set(f"{float(np.max(recall)):.4f}")
            self.pf_prm_min.set(f"{float(np.min(pr_mean)):.4f}")
            self.pf_prm_max.set(f"{float(np.max(pr_mean)):.4f}")
            # Train knee: closest to utopia (P=1, R=1) on the current ND front.
            knee_idx = int(np.argmin(np.sum(np.square(F_opt[:, :2]), axis=1)))
            knee_p = float(precision[knee_idx])
            knee_r = float(recall[knee_idx])
            self.knee_p.set(f"{knee_p:.4f}")
            self.knee_r.set(f"{knee_r:.4f}")
            self.knee_pr_mean.set(f"{0.5 * (knee_p + knee_r):.4f}")
        else:
            per_ind = np.mean(F_opt, axis=1)
            self.mean_f.set(f"{float(np.mean(per_ind)):.4e}")
            self.min_f.set(f"{float(np.min(per_ind)):.4e}")
            if self.is_l1_problem and F_opt.ndim == 2 and F_opt.shape[1] >= 2:
                knee_i = _knee_index_utopia(F_opt)
                if self.problem_name == "cross_entropy_l1":
                    self.knee_task.set(f"{float(F_opt[knee_i, 0]):.4f}")
                else:
                    self.knee_task.set(f"{float(1.0 - F_opt[knee_i, 0]):.4f}")
                self.knee_l1.set(f"{float(F_opt[knee_i, 1]):.4f}")

        source = F_pop if F_pop is not None and len(F_pop) else F_opt
        self._archive = _update_nd_archive(self._archive, source)

        hv_val = self._compute_train_hv(self._archive)
        if hv_val is None:
            self.hv.set("-")
        else:
            hv_val = max(hv_val, self._last_hv)
            self._last_hv = hv_val
            self.hv.set(f"{hv_val:.4f}")

        self._maybe_validate(algorithm)
        self._maybe_save_train_pareto(algorithm, step=int(algorithm.n_gen))

        if self.use_wandb:
            self.log_wandb(algorithm, wandb_step=algorithm.n_gen)