"""Train TinyCNN weights with NSGA-II / NSGA-III or evosax SOO (CMA-ES, ASEBO, ...)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import wandb
from pymoo.optimize import minimize

from evotinyml.moo.algorithms import (
    CROSSOVERS,
    MOO_ALGORITHMS,
    MUTATIONS,
    OperatorConfig,
    build_algorithm,
)
from evotinyml.moo.display import StepOutput
from evotinyml.moo.mo_es import (
    ARCHIVE_SELECTIONS,
    DEFAULT_ARCHIVE_SIZE,
    DEFAULT_MGDA_NORMALIZE,
    DEFAULT_ES_FITNESS_SHAPING,
    DEFAULT_MOEAD_IDEAL,
    DEFAULT_MOEAD_K,
    DEFAULT_MOEAD_RHO,
    DEFAULT_MOEAD_SCALARIZATION,
    DEFAULT_MOEAD_WEIGHT_SHRINK,
    MGDA_NORMALIZATIONS,
    ES_FITNESS_SHAPINGS,
    MO_ES_ALGORITHMS,
    MOEAD_IDEAL_MODES,
    MOEAD_SCALARIZATIONS,
    build_mo_es_wandb_config,
    run_mgda_open_es,
    run_moead_open_es,
    run_upgrad_open_es,
)
from evotinyml.moo.termination import MaximumStepTermination
from evotinyml.soo.algorithms import SOO_ALGORITHMS
from evotinyml.soo.es import (
    DEFAULT_ES_OPTIM,
    DEFAULT_ASEBO_SUBSPACE_DIMS,
    DEFAULT_DE_CR,
    DEFAULT_DE_ELITISM,
    DEFAULT_DE_F,
    DEFAULT_JDE_CR_INIT,
    DEFAULT_JDE_ELITISM,
    DEFAULT_JDE_F_INIT,
    DEFAULT_JDE_F_L,
    DEFAULT_JDE_F_U,
    DEFAULT_JDE_TAU_CR,
    DEFAULT_JDE_TAU_F,
    DEFAULT_PSO_COGNITIVE,
    DEFAULT_PSO_INERTIA,
    DEFAULT_PSO_MAX_VELOCITY,
    DEFAULT_PSO_SOCIAL,
    DEFAULT_ES_OPTIM_LR,
    DEFAULT_ES_OPTIM_MOMENTUM,
    DEFAULT_ES_OPTIM_SCHEDULER,
    DEFAULT_ES_OPTIM_WD,
    DEFAULT_ES_SIGMA_SCHEDULER,
    ES_OPTIMS,
    ES_OPTIM_SCHEDULERS,
    ES_SIGMA_SCHEDULERS,
    EVOSAX_SOO_ALGOS,
    MEAN_OPTIMIZER_ALGOS,
    POPULATION_BASED_ALGOS,
    SIGMA_SCHEDULE_ALGOS,
    build_soo_wandb_config,
    default_soo_popsize,
    run_soo_es,
)
from evotinyml.data import EVAL_MODES, load_dataset
from evotinyml.model import ACTIVATIONS, build_model
from evotinyml.problem import (
    CE_SOFT_PR_PROBLEMS,
    DEFAULT_XL,
    DEFAULT_XU,
    EVAL_SAMPLER_NAMES,
    L1_PROBLEMS,
    PROBLEM_ALIASES,
    PROBLEMS,
    PR_PROBLEMS,
    SOO_ONLY_PROBLEMS,
    SOO_PROBLEMS,
    apply_scalar_weights,
    build_eval_sampler,
    build_problem,
    problem_obj_labels,
)
from evotinyml.device import resolve_device
from evotinyml.sampling import get_population_init
from evotinyml.validation import make_test_loader
from evotinyml.wandb_logger import (
    DEFAULT_ENTITY,
    DEFAULT_PROJECT,
    define_wandb_n_eval_metric,
    finish_wandb,
    init_wandb,
    make_run_name,
    save_wandb_checkpoint,
)


def _primary_front_index(problem_name: str, F: np.ndarray) -> int:
    """Index of the representative final model on a MOO front (knee / best mean)."""
    F = np.asarray(F, dtype=float)
    if F.ndim == 1 or len(F) == 0:
        return 0
    if (
        problem_name in PR_PROBLEMS
        or problem_name in CE_SOFT_PR_PROBLEMS
        or problem_name in L1_PROBLEMS
    ):
        return int(np.argmin(np.sum(np.square(F), axis=1)))
    return int(np.argmin(F.mean(axis=1)))

ALGORITHMS = MOO_ALGORITHMS + SOO_ALGORITHMS + MO_ES_ALGORITHMS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evolutionary TinyCNN training (NSGA-II/III or evosax SOO)."
    )
    parser.add_argument(
        "--dataset",
        choices=("mnist", "mnist_2cls", "cifar10"),
        required=True,
        help="Dataset: mnist (10-class), mnist_2cls (digits 0/1), or cifar10.",
    )
    parser.add_argument(
        "--problem",
        choices=(*PROBLEMS, *PROBLEM_ALIASES),
        default="cwrm_cross_entropy",
        help=(
            "Objective: cwrm_cross_entropy (class-wise RM + CE, multi-obj), "
            "precision_recall / soft_precision_recall (2-obj 1-P/1-R), "
            "ce_soft_precision_recall (3-obj CE, 1-soft P, 1-soft R), "
            "cross_entropy_l1 / f1_l1 / soft_f1_l1 (2-obj task + mean |θ|), "
            "erm_cross_entropy (ERM + mean CE for SOO), "
            "or erm_f1 / erm_soft_f1 (ERM + 1-macro-F1 for SOO). "
            "MOO problems use a Pareto / MO-ES solver, or weighted-sum scalarization "
            "with SOO algos (--scalar-weights). "
            "Aliases: cwce/per_class_ce -> cwrm; ce_soft_pr -> ce_soft_precision_recall; "
            "ce_l1 -> cross_entropy_l1; cross_entropy -> erm; f1 -> erm_f1; soft_f1 -> erm_soft_f1."
        ),
    )
    parser.add_argument(
        "--scalar-weights",
        type=str,
        default=None,
        help=(
            "SOO weighted-sum weights for multi-objective problems: comma-separated "
            "non-negative floats, length = n_obj (e.g. 1,1 or 1,0.1). "
            "Default: problem-specific equal ones (unweighted sum). Ignored for MOO/MO-ES."
        ),
    )
    parser.add_argument(
        "--activation",
        choices=ACTIVATIONS,
        default="relu",
        help="Hidden activation: relu or tanh.",
    )
    parser.add_argument(
        "--algo",
        choices=ALGORITHMS,
        default="nsga2",
        help=(
            "Optimizer: nsga2 / nsga3 (MOO), cmaes / snes / xnes / open_es / "
            "cr_fm_nes / asebo / lm_ma_es / de / jde / pso "
            "(SOO: ERM or weighted-sum scalarization of any MOO problem), "
            "or mgda_open_es / upgrad_open_es / moead_open_es "
            "(multi-objective OpenES on vector problems, e.g. cwrm_cross_entropy)."
        ),
    )
    parser.add_argument(
        "--ref-dirs",
        choices=("energy", "das-dennis"),
        default="energy",
        help=(
            "Reference direction method: NSGA-III, moead_open_es weight vectors, "
            "and nsga3 archive selection (ignored for NSGA-II / SOO ES)."
        ),
    )
    parser.add_argument(
        "--n-partitions",
        type=int,
        default=None,
        help="Das-Dennis partitions for NSGA-III (default: largest with #dirs <= popsize).",
    )
    parser.add_argument(
        "--crossover",
        choices=CROSSOVERS,
        default="sbx",
        help="Crossover operator: sbx or none (ignored for SOO ES).",
    )
    parser.add_argument(
        "--crossover-prob",
        type=float,
        default=0.9,
        help="SBX crossover probability (ignored when --crossover none or SOO ES).",
    )
    parser.add_argument(
        "--crossover-eta",
        type=float,
        default=15.0,
        help="SBX distribution index eta (ignored when --crossover none or SOO ES).",
    )
    parser.add_argument(
        "--crossover-prob-var",
        type=float,
        default=0.5,
        help="SBX per-variable crossover probability (ignored when --crossover none or SOO ES).",
    )
    parser.add_argument(
        "--mutation",
        choices=MUTATIONS,
        default="pm",
        help=(
            "Mutation operator: pm (polynomial), gaussian (absolute N(0, sigma)), "
            "or layerwise (He fan-in scaled Gaussian). Ignored for SOO ES."
        ),
    )
    parser.add_argument(
        "--mutation-prob",
        type=float,
        default=0.9,
        help="Per-individual mutation probability (ignored for SOO ES).",
    )
    parser.add_argument(
        "--mutation-eta",
        type=float,
        default=20.0,
        help="Polynomial mutation distribution index eta (ignored for SOO ES).",
    )
    parser.add_argument(
        "--mutation-sigma",
        type=float,
        default=0.1,
        help=(
            "Gaussian mutation std: absolute for --mutation gaussian; "
            "mean per-variable std for --mutation layerwise. Ignored for SOO ES."
        ),
    )
    parser.add_argument(
        "--mutation-prob-var",
        type=float,
        default=None,
        help=(
            "Per-variable mutation probability. Default: pymoo's min(0.5, 1/n_var). "
            "Ignored for SOO ES."
        ),
    )
    parser.add_argument(
        "--init",
        choices=("uniform", "gaussian", "both", "zeros", "kaiming"),
        default="kaiming",
        help=(
            "Population / ES mean initialization: uniform[-init_sigma, init_sigma], "
            "gaussian N(0, init_sigma), both (half/half mix; SOO ES uses gaussian mean), "
            "zeros (all-zero), or kaiming (ES mean = PyTorch default / Kaiming-uniform "
            "weights; NSGA: one individual at that theta0, rest theta0+N(0,init_sigma)). "
            "Default: kaiming. --init-sigma still sets ES sampling std."
        ),
    )
    parser.add_argument(
        "--init-sigma",
        type=float,
        default=0.1,
        help="Init scale (also ES std_init / OpenES noise std): uniform [-sigma,sigma], gaussian N(0,sigma).",
    )
    parser.add_argument(
        "--xl",
        type=float,
        default=DEFAULT_XL,
        help=(
            f"Lower box bound for decision variables (weights). "
            f"Default: {DEFAULT_XL}."
        ),
    )
    parser.add_argument(
        "--xu",
        type=float,
        default=DEFAULT_XU,
        help=(
            f"Upper box bound for decision variables (weights). "
            f"Default: {DEFAULT_XU}."
        ),
    )
    parser.add_argument(
        "--es-optim",
        choices=ES_OPTIMS,
        default=DEFAULT_ES_OPTIM,
        help=(
            f"Mean-update optimizer (optax) for open_es / snes / xnes / asebo. "
            f"Default: {DEFAULT_ES_OPTIM}. "
            "Ignored for cmaes / cr_fm_nes / lm_ma_es / de / jde / pso."
        ),
    )
    parser.add_argument(
        "--es-optim-lr",
        type=float,
        default=DEFAULT_ES_OPTIM_LR,
        help=(
            f"Optimizer learning rate for open_es / snes / xnes / asebo "
            f"(initial value if scheduled). Default: {DEFAULT_ES_OPTIM_LR}. "
            "Ignored for cmaes / cr_fm_nes / lm_ma_es / de / jde / pso."
        ),
    )
    parser.add_argument(
        "--es-optim-scheduler",
        choices=ES_OPTIM_SCHEDULERS,
        default=DEFAULT_ES_OPTIM_SCHEDULER,
        help=(
            "LR schedule over steps for open_es / snes / xnes / asebo: constant, "
            "cosine (decay to 0 over steps), or exponential. "
            "Ignored for cmaes / cr_fm_nes / lm_ma_es / de / jde / pso."
        ),
    )
    parser.add_argument(
        "--es-optim-momentum",
        type=float,
        default=DEFAULT_ES_OPTIM_MOMENTUM,
        help=(
            f"Momentum for sgd / rmsprop (0 = off) on open_es / snes / xnes / asebo. "
            f"Default: {DEFAULT_ES_OPTIM_MOMENTUM}. "
            "Ignored for adam/adamw and cmaes / cr_fm_nes / lm_ma_es / de / jde / pso."
        ),
    )
    parser.add_argument(
        "--es-optim-wd",
        type=float,
        default=DEFAULT_ES_OPTIM_WD,
        help=(
            f"Weight decay for mean Optax update on open_es / snes / xnes / asebo "
            f"(0 = off). adamw: decoupled WD; sgd / adam / rmsprop: add_decayed_weights. "
            f"Default: {DEFAULT_ES_OPTIM_WD}. "
            "Ignored for cmaes / cr_fm_nes / lm_ma_es / de / jde / pso."
        ),
    )
    parser.add_argument(
        "--es-sigma-scheduler",
        choices=ES_SIGMA_SCHEDULERS,
        default=DEFAULT_ES_SIGMA_SCHEDULER,
        help=(
            "OpenES / ASEBO / MO-OpenES sampling-noise (σ) schedule over steps: "
            "constant, cosine, or exponential. Start value is --init-sigma. "
            "Ignored for cmaes / snes / xnes / cr_fm_nes / lm_ma_es / de / jde / pso."
        ),
    )
    parser.add_argument(
        "--es-sigma-end",
        type=float,
        default=None,
        help=(
            "Final σ for cosine / exponential --es-sigma-scheduler "
            "(default: max(0.01 * init_sigma, 1e-6)). "
            "Ignored for constant / non-scheduled algos."
        ),
    )
    parser.add_argument(
        "--asebo-subspace-dims",
        type=int,
        default=DEFAULT_ASEBO_SUBSPACE_DIMS,
        help=(
            f"ASEBO active-subspace rank (FIFO gradient history). "
            f"Default: {DEFAULT_ASEBO_SUBSPACE_DIMS}. Ignored for other algos."
        ),
    )
    parser.add_argument(
        "--de-f",
        type=float,
        default=DEFAULT_DE_F,
        help=(
            f"Differential Evolution differential weight F. "
            f"Default: {DEFAULT_DE_F}. Ignored for non-DE algos."
        ),
    )
    parser.add_argument(
        "--de-cr",
        type=float,
        default=DEFAULT_DE_CR,
        help=(
            f"Differential Evolution crossover rate CR. "
            f"Default: {DEFAULT_DE_CR}. Ignored for non-DE algos."
        ),
    )
    parser.add_argument(
        "--de-elitism",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DE_ELITISM,
        help=(
            "DE base-vector elitism (best/1/bin when on, rand/1/bin when off). "
            f"Default: {DEFAULT_DE_ELITISM}. Ignored for non-DE algos."
        ),
    )
    parser.add_argument(
        "--jde-f-init",
        type=float,
        default=DEFAULT_JDE_F_INIT,
        help=(
            f"jDE initial differential weight F for all members. "
            f"Default: {DEFAULT_JDE_F_INIT}. Ignored for non-jDE algos."
        ),
    )
    parser.add_argument(
        "--jde-cr-init",
        type=float,
        default=DEFAULT_JDE_CR_INIT,
        help=(
            f"jDE initial crossover rate CR for all members. "
            f"Default: {DEFAULT_JDE_CR_INIT}. Ignored for non-jDE algos."
        ),
    )
    parser.add_argument(
        "--jde-f-l",
        type=float,
        default=DEFAULT_JDE_F_L,
        help=(
            f"jDE lower bound Fl when resampling F (F <- Fl + U*Fu). "
            f"Default: {DEFAULT_JDE_F_L}."
        ),
    )
    parser.add_argument(
        "--jde-f-u",
        type=float,
        default=DEFAULT_JDE_F_U,
        help=(
            f"jDE range width Fu when resampling F (F <- Fl + U*Fu). "
            f"Default: {DEFAULT_JDE_F_U}."
        ),
    )
    parser.add_argument(
        "--jde-tau-f",
        type=float,
        default=DEFAULT_JDE_TAU_F,
        help=(
            f"jDE probability τ_F of resampling each member's F. "
            f"Default: {DEFAULT_JDE_TAU_F}."
        ),
    )
    parser.add_argument(
        "--jde-tau-cr",
        type=float,
        default=DEFAULT_JDE_TAU_CR,
        help=(
            f"jDE probability τ_CR of resampling each member's CR. "
            f"Default: {DEFAULT_JDE_TAU_CR}."
        ),
    )
    parser.add_argument(
        "--jde-elitism",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_JDE_ELITISM,
        help=(
            "jDE base-vector elitism (best/1/bin when on; paper default "
            f"rand/1/bin when off). Default: {DEFAULT_JDE_ELITISM}."
        ),
    )
    parser.add_argument(
        "--pso-inertia",
        type=float,
        default=DEFAULT_PSO_INERTIA,
        help=(
            f"PSO inertia coefficient w. Default: {DEFAULT_PSO_INERTIA}. "
            "Ignored for non-PSO algos."
        ),
    )
    parser.add_argument(
        "--pso-cognitive",
        type=float,
        default=DEFAULT_PSO_COGNITIVE,
        help=(
            f"PSO cognitive coefficient c1 (pull to personal best). "
            f"Default: {DEFAULT_PSO_COGNITIVE}."
        ),
    )
    parser.add_argument(
        "--pso-social",
        type=float,
        default=DEFAULT_PSO_SOCIAL,
        help=(
            f"PSO social coefficient c2 (pull to global best). "
            f"Default: {DEFAULT_PSO_SOCIAL}."
        ),
    )
    parser.add_argument(
        "--pso-max-velocity",
        type=float,
        default=DEFAULT_PSO_MAX_VELOCITY,
        help=(
            "PSO velocity clamp v_max: clip each velocity component to "
            f"[-v_max, v_max]. Default: {DEFAULT_PSO_MAX_VELOCITY}."
        ),
    )
    parser.add_argument(
        "--mgda-normalize",
        choices=MGDA_NORMALIZATIONS,
        default=DEFAULT_MGDA_NORMALIZE,
        help=(
            "Per-objective ES-gradient rescaling before MGDA / UPGrad "
            "(mgda_open_es / upgrad_open_es only): "
            "none, l2 (g/||g||), loss (g/L), loss+ (g/(L*||g||)), or range "
            "(MGDA weights scaled by each objective's range-normalized "
            "distance-to-ideal, F_c / running nadir_c; recommended for "
            "mixed-scale objectives like cross_entropy_l1); "
            f"default: {DEFAULT_MGDA_NORMALIZE}. "
            "Avoid loss/loss+ when an objective can be ~0 (e.g. L1 with "
            "--init zeros)."
        ),
    )
    parser.add_argument(
        "--es-fitness-shaping",
        choices=ES_FITNESS_SHAPINGS,
        default=DEFAULT_ES_FITNESS_SHAPING,
        help=(
            "Per-objective ES fitness shaping for MGDA gradient estimates "
            "(mgda_open_es / upgrad_open_es only): centered_ranks (OpenES default; robust but "
            "scale-blind and noise-amplifying), z_score ((F-mean)/std; keeps "
            "within-population magnitudes), raw (F-mean; true gradient scale, "
            f"outlier-sensitive); default: {DEFAULT_ES_FITNESS_SHAPING}."
        ),
    )
    parser.add_argument(
        "--archive-size",
        type=int,
        default=DEFAULT_ARCHIVE_SIZE,
        help=(
            f"Max non-dominated archive size for mgda_open_es / upgrad_open_es / "
            f"moead_open_es "
            f"(default: {DEFAULT_ARCHIVE_SIZE}). Ignored for other algos."
        ),
    )
    parser.add_argument(
        "--archive-selection",
        choices=ARCHIVE_SELECTIONS,
        default="nsga2",
        help=(
            "Archive pruning for mgda_open_es / upgrad_open_es / moead_open_es: nsga2 "
            "(rank + crowding) or nsga3 (reference-direction niching)."
        ),
    )
    parser.add_argument(
        "--moead-k",
        type=int,
        default=DEFAULT_MOEAD_K,
        help=(
            f"Number of means / weight vectors for moead_open_es "
            f"(default: {DEFAULT_MOEAD_K}). Ignored for other algos."
        ),
    )
    parser.add_argument(
        "--moead-rho",
        type=float,
        default=DEFAULT_MOEAD_RHO,
        help=(
            f"Augmented-Tchebycheff rho for moead_open_es "
            f"(default: {DEFAULT_MOEAD_RHO}). Ignored for other algos."
        ),
    )
    parser.add_argument(
        "--moead-weight-shrink",
        type=float,
        default=DEFAULT_MOEAD_WEIGHT_SHRINK,
        help=(
            "moead_open_es: shrink weight vectors toward uniform, "
            "lam <- (1-a)*lam + a/n_obj (default: "
            f"{DEFAULT_MOEAD_WEIGHT_SHRINK}; 0 = raw simplex corners allowed)."
        ),
    )
    parser.add_argument(
        "--moead-ideal",
        choices=MOEAD_IDEAL_MODES,
        default=DEFAULT_MOEAD_IDEAL,
        help=(
            "moead_open_es ideal point z*: zero (fixed at 0; objectives are "
            "lower-bounded by 0) or adaptive (running per-objective minimum)."
        ),
    )
    parser.add_argument(
        "--moead-scalarization",
        choices=MOEAD_SCALARIZATIONS,
        default=DEFAULT_MOEAD_SCALARIZATION,
        help=(
            "moead_open_es subproblem scalarization: tchebycheff (covers "
            "non-convex fronts; sparse rank signal with many objectives) or "
            "weighted_sum (dense signal; convex front regions only)."
        ),
    )
    parser.add_argument(
        "--moead-migrate-every",
        type=int,
        default=0,
        help=(
            "moead_open_es: every N steps, restart a mean at the archive point "
            "with the best Tchebycheff value for its weight vector (0 = off)."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Optimization steps after init. Default: 100000 if --evals omitted. "
            "Ignored when --evals is set (steps = ceil(evals / popsize))."
        ),
    )
    parser.add_argument(
        "--evals",
        type=int,
        default=None,
        help=(
            "Function-evaluation budget (train fitness calls). "
            "Sets steps = ceil(evals / popsize) (ES/NSGA: one generation costs popsize evals)."
        ),
    )
    parser.add_argument(
        "--popsize",
        type=int,
        default=None,
        help=(
            "Population size. Default: 100 for NSGA; "
            "4+3*ln(n_var) for SOO ES (~25 for TinyCNN; even for open_es / "
            "cr_fm_nes / asebo)."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="Eval batch size.")
    parser.add_argument(
        "--sampler",
        choices=EVAL_SAMPLER_NAMES,
        default="auto",
        help=(
            "Eval-batch sampler: random (uniform, RandomSampler-style), "
            "balanced (ClassBalancedSampler: >=1 sample per class), or auto "
            "(CWRM-CE -> balanced; P/R / ERM / L1 -> random). Default: auto."
        ),
    )
    parser.add_argument(
        "--eval-mode",
        choices=EVAL_MODES,
        default="single",
        help=(
            "How many fitness batches per generation (sampler draws each gen): "
            "single = one minibatch of --batch-size shared by the whole population "
            "(CRN); multi = pool --eval-batches minibatches then compute metrics once."
        ),
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=1,
        help=(
            "Number of minibatches to pool each generation when --eval-mode multi "
            "(ignored for single)."
        ),
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=50,
        help="Evaluate on the test set every N steps (ND set for MOO; ES mean for SOO).",
    )
    parser.add_argument(
        "--pareto-every",
        type=int,
        default=100,
        help=(
            "Save train Pareto front / log W&B Pareto plots at step 1 "
            "and every N steps thereafter (MOO only; ignored for SOO ES). "
            "P-R also writes train_history.npz; all MOO problems log "
            "train/pareto_front (and val/pareto_front on validation)."
        ),
    )
    parser.add_argument(
        "--train-history",
        type=str,
        default="train_history.npz",
        help="Path for train Pareto front history (MOO P-R problems only).",
    )
    parser.add_argument(
        "--val-history",
        type=str,
        default="val_history.npz",
        help="Path for val Pareto front history (MOO P-R problems only).",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=512,
        help="Batch size for test-set validation.",
    )
    parser.add_argument("--seed", type=int, default=1, help="RNG seed.")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset download/cache root.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "Torch device: cpu, cuda, mps, or gpu "
            "(gpu -> CUDA if available, else MPS)."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to save final weights (.npz).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print step progress.")
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log metrics to Weights & Biases (default: on).",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=DEFAULT_ENTITY,
        help=f"W&B entity (default: {DEFAULT_ENTITY}).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=DEFAULT_PROJECT,
        help=f"W&B project (default: {DEFAULT_PROJECT}).",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="W&B run name (default: {dataset}-{algo}-seed{seed}).",
    )
    return parser.parse_args()


def _resolve_popsize(args: argparse.Namespace, n_var: int) -> int:
    if args.popsize is not None:
        return int(args.popsize)
    if args.algo in SOO_ALGORITHMS:
        return default_soo_popsize(n_var, args.algo)
    if args.algo in {"mgda_open_es", "upgrad_open_es"}:
        return default_soo_popsize(n_var, "open_es")
    if args.algo == "moead_open_es":
        # Total per-generation evaluations: 8 antithetic samples per mean.
        return int(args.moead_k) * 8
    return 100


def _resolve_steps_and_evals(args: argparse.Namespace, popsize: int) -> tuple[int, int]:
    """Resolve ``(steps, evals)`` from ``--steps`` and/or ``--evals``.

    ``--evals`` wins when set: ``steps = ceil(evals / popsize)`` (may exceed budget).
    """
    popsize = int(popsize)
    if popsize < 1:
        raise SystemExit(f"popsize must be >= 1, got {popsize}")

    if args.evals is not None:
        evals = int(args.evals)
        if evals < 1:
            raise SystemExit(f"--evals must be >= 1, got {evals}")
        steps = int(np.ceil(evals / popsize))
        used = steps * popsize
        if used > evals:
            print(
                f"Note: --evals={evals} -> steps={steps} "
                f"({used} Function Evaluations)."
            )
        if args.steps is not None and int(args.steps) != steps:
            print(
                f"Note: --evals sets steps={steps}; ignoring --steps={args.steps}."
            )
        args.steps = steps
        args.evals = used
        return steps, used

    if args.steps is not None:
        steps = int(args.steps)
    else:
        steps = 100_000
    if steps < 0:
        raise SystemExit(f"--steps must be >= 0, got {steps}")
    evals = steps * popsize
    args.steps = steps
    args.evals = evals
    return steps, evals


def run_moo(args: argparse.Namespace, problem, test_loader, num_classes: int, batch_sampler):
    operators = OperatorConfig(
        crossover=args.crossover,
        crossover_prob=args.crossover_prob,
        crossover_eta=args.crossover_eta,
        crossover_prob_var=args.crossover_prob_var,
        mutation=args.mutation,
        mutation_prob=args.mutation_prob,
        mutation_eta=args.mutation_eta,
        mutation_sigma=args.mutation_sigma,
        mutation_prob_var=args.mutation_prob_var,
    )
    popsize = _resolve_popsize(args, problem.n_var)
    steps, evals = _resolve_steps_and_evals(args, popsize)
    # Keep resolved values on args so W&B / banners see them.
    args.popsize = popsize
    args.steps = steps
    args.evals = evals

    algorithm = build_algorithm(
        args.algo,
        pop_size=popsize,
        n_obj=problem.n_obj,
        sampling=get_population_init(
            args.init,
            init_sigma=args.init_sigma,
            theta0=getattr(problem, "theta0", None),
        ),
        output=StepOutput(
            test_loader=test_loader,
            val_every=args.val_every,
            n_classes=num_classes,
            problem_name=args.problem,
            use_wandb=args.wandb,
            max_steps=args.steps,
            pareto_every=args.pareto_every,
            train_history_path=args.train_history,
            val_history_path=args.val_history,
        ),
        seed=args.seed,
        ref_dirs_method=args.ref_dirs,
        n_partitions=args.n_partitions,
        operators=operators,
    )

    run_name = make_run_name(args.dataset, args.algo, args.seed, args.wandb_name)
    init_wandb(
        args,
        n_var=problem.n_var,
        n_obj=problem.n_obj,
        algo_config=getattr(algorithm, "algo_config", None),
        operators=operators,
        enabled=args.wandb,
    )

    n_ref = len(algorithm.ref_dirs) if args.algo == "nsga3" else None
    print(
        f"dataset={args.dataset}  problem={args.problem}  "
        f"activation={args.activation}  algo={args.algo}  "
        f"fitness=vector  "
        f"init={args.init}(sigma={args.init_sigma})  "
        f"n_var={problem.n_var}  n_obj={problem.n_obj}  "
        f"xl={args.xl}  xu={args.xu}  "
        f"popsize={popsize}  steps={args.steps}  evals={args.evals}  "
        + (
            "crossover=none  "
            if args.crossover == "none"
            else f"sbx(prob={args.crossover_prob}, eta={args.crossover_eta})  "
        )
        + (
            f"gauss_mut(prob={args.mutation_prob}, sigma={args.mutation_sigma}"
            f", prob_var={args.mutation_prob_var})  "
            if args.mutation == "gaussian"
            else (
                f"layerwise_mut(prob={args.mutation_prob}, sigma={args.mutation_sigma}"
                f", prob_var={args.mutation_prob_var})  "
                if args.mutation == "layerwise"
                else (
                    f"pm(prob={args.mutation_prob}, eta={args.mutation_eta}"
                    f", prob_var={args.mutation_prob_var})  "
                )
            )
        )
        +         f"batch_size={args.batch_size}  eval_mode={args.eval_mode}  "
        f"eval_batches={problem.eval_batches}  "
        f"sampler={type(batch_sampler).__name__}  "
        f"val_every={args.val_every}  pareto_every={args.pareto_every}  "
        f"train_history={args.train_history}  val_history={args.val_history}  "
        f"device={problem.device}"
        + (f"  ref_dirs={args.ref_dirs}({n_ref})" if n_ref is not None else "")
        + (f"  wandb={args.wandb_entity}/{args.wandb_project}/{run_name}" if args.wandb else "  wandb=off")
    )

    try:
        result = minimize(
            problem,
            algorithm,
            termination=MaximumStepTermination(args.steps),
            seed=args.seed,
            verbose=args.verbose,
            save_history=False,
        )
    except Exception:
        finish_wandb()
        raise

    F = result.F
    X = result.X
    mean_obj = F.mean(axis=1)
    best_idx = int(np.argmin(mean_obj))

    completed_steps = min(max(0, int(result.algorithm.n_gen) - 1), args.steps)
    print(f"Finished after {completed_steps} steps.")
    print(f"Pareto set size: {len(F)}")

    summary: dict = {
        "final/n_nds": len(F),
        "final/steps": completed_steps,
    }

    if args.problem in PR_PROBLEMS:
        err = np.asarray(F, dtype=float)
        knee_idx = int(np.argmin(np.sum(np.square(err), axis=1)))
        knee = err[knee_idx]
        knee_p = float(1.0 - knee[0])
        knee_r = float(1.0 - knee[1])
        print(
            f"Train knee individual: "
            f"P={knee_p:.6f}, R={knee_r:.6f}, "
            f"pr_mean={0.5 * (knee_p + knee_r):.6f}"
        )
        print(f"ND front objectives [1-P, 1-R]:\n{np.array2string(F, precision=4)}")
    elif args.problem in CE_SOFT_PR_PROBLEMS:
        err = np.asarray(F, dtype=float)
        knee_idx = int(np.argmin(np.sum(np.square(err), axis=1)))
        knee = err[knee_idx]
        print(
            f"Train knee individual: "
            f"CE={float(knee[0]):.6f}, "
            f"soft_P={float(1.0 - knee[1]):.6f}, "
            f"soft_R={float(1.0 - knee[2]):.6f}"
        )
        print(
            f"ND front objectives [CE, 1-soft P, 1-soft R]:\n"
            f"{np.array2string(F, precision=4)}"
        )
        summary["final/knee_ce"] = float(knee[0])
        summary["final/knee_precision"] = float(1.0 - knee[1])
        summary["final/knee_recall"] = float(1.0 - knee[2])
    elif args.problem in L1_PROBLEMS:
        err = np.asarray(F, dtype=float)
        knee_idx = int(np.argmin(np.sum(np.square(err), axis=1)))
        knee = err[knee_idx]
        labels = problem_obj_labels(args.problem, 2) or ["task", "L1"]
        if args.problem == "cross_entropy_l1":
            print(
                f"Train knee individual: "
                f"CE={float(knee[0]):.6f}, L1={float(knee[1]):.6f}"
            )
            summary["final/knee_ce"] = float(knee[0])
        else:
            task = float(1.0 - knee[0])
            tag = "soft_F1" if args.problem == "soft_f1_l1" else "F1"
            print(
                f"Train knee individual: "
                f"{tag}={task:.6f}, L1={float(knee[1]):.6f}"
            )
            summary["final/knee_f1"] = task
        summary["final/knee_l1"] = float(knee[1])
        print(
            f"ND front objectives [{labels[0]}, {labels[1]}]:\n"
            f"{np.array2string(F, precision=4)}"
        )
    else:
        print(f"Best mean CWRM-CE: {mean_obj[best_idx]:.6f}")
        print(f"Class-wise CE (best mean): {np.array2string(F[best_idx], precision=4)}")
        summary["final/mean_obj_best"] = float(mean_obj[best_idx])

    output = getattr(algorithm, "output", None)
    if output is not None and hasattr(output, "final_summary_metrics"):
        summary.update(output.final_summary_metrics())

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            X=X,
            F=F,
            dataset=args.dataset,
            problem=args.problem,
            activation=args.activation,
            algo=args.algo,
            init=args.init,
            init_sigma=args.init_sigma,
            steps=args.steps,
            popsize=popsize,
        )
        print(f"Saved Pareto set to {out_path}")

    output = getattr(result.algorithm, "output", None)
    if args.wandb and output is not None:
        if args.steps % args.val_every != 0:
            output._maybe_validate(result.algorithm, force=True)
        output.log_wandb(result.algorithm, wandb_step=args.steps, force=True)

    ckpt_idx = _primary_front_index(args.problem, F)
    ckpt_kind = (
        "knee"
        if args.problem in PR_PROBLEMS
        or args.problem in CE_SOFT_PR_PROBLEMS
        or args.problem in L1_PROBLEMS
        else "best_mean_obj"
    )
    ckpt_path = save_wandb_checkpoint(
        problem,
        X[ckpt_idx],
        meta={
            "dataset": args.dataset,
            "problem": args.problem,
            "activation": args.activation,
            "algo": args.algo,
            "num_classes": num_classes,
            "val_solution": ckpt_kind,
            "solution_index": ckpt_idx,
            "F": np.asarray(F, dtype=np.float64),
            "X": np.asarray(X, dtype=np.float64),
        },
    )
    if ckpt_path is not None:
        print(f"Saved W&B checkpoint to {ckpt_path}")

    finish_wandb(summary)
    return result


def run_soo(args: argparse.Namespace, problem, test_loader, num_classes: int, batch_sampler):
    algo = args.algo
    display = EVOSAX_SOO_ALGOS.get(algo, (algo, algo))[1]
    if args.problem not in SOO_PROBLEMS:
        raise SystemExit(
            f"--algo {algo} requires --problem in {sorted(SOO_PROBLEMS)} "
            f"(got {args.problem!r})."
        )
    soo = getattr(problem, "soo_fitness", None)
    if soo is None:
        raise SystemExit(
            f"Internal error: problem {type(problem).__name__} has no soo_fitness"
        )
    try:
        weights = apply_scalar_weights(problem, getattr(args, "scalar_weights", None))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.scalar_weights_resolved = weights.tolist()

    # Operator args are MOO-only; note when user passed non-defaults.
    if args.crossover != "sbx" or args.mutation != "pm":
        print(f"Note: --crossover / --mutation are ignored for --algo {algo}.")

    uses_mean_optimizer = algo in MEAN_OPTIMIZER_ALGOS
    uses_sigma_schedule = algo in SIGMA_SCHEDULE_ALGOS
    es_optim_opts_nondefault = (
        args.es_optim != DEFAULT_ES_OPTIM
        or float(args.es_optim_lr) != DEFAULT_ES_OPTIM_LR
        or args.es_optim_scheduler != DEFAULT_ES_OPTIM_SCHEDULER
        or float(args.es_optim_momentum) != DEFAULT_ES_OPTIM_MOMENTUM
        or float(args.es_optim_wd) != DEFAULT_ES_OPTIM_WD
    )
    es_sigma_opts_nondefault = (
        args.es_sigma_scheduler != DEFAULT_ES_SIGMA_SCHEDULER
        or args.es_sigma_end is not None
    )
    if not uses_mean_optimizer and es_optim_opts_nondefault:
        print(
            f"Note: --es-optim / --es-optim-lr / --es-optim-scheduler / "
            f"--es-optim-momentum / --es-optim-wd are ignored for --algo {algo} "
            f"(open_es / snes / xnes / asebo only)."
        )
    if not uses_sigma_schedule and es_sigma_opts_nondefault:
        print(
            f"Note: --es-sigma-scheduler / --es-sigma-end are ignored for "
            f"--algo {algo} (open_es / asebo only)."
        )
    if (
        algo != "asebo"
        and int(args.asebo_subspace_dims) != DEFAULT_ASEBO_SUBSPACE_DIMS
    ):
        print(
            f"Note: --asebo-subspace-dims is ignored for --algo {algo} "
            f"(asebo only)."
        )
    de_opts_nondefault = (
        float(args.de_f) != DEFAULT_DE_F
        or float(args.de_cr) != DEFAULT_DE_CR
        or bool(args.de_elitism) != DEFAULT_DE_ELITISM
    )
    if algo != "de" and de_opts_nondefault:
        print(
            f"Note: --de-f / --de-cr / --de-elitism are ignored for "
            f"--algo {algo} (de only)."
        )
    jde_opts_nondefault = (
        float(args.jde_f_init) != DEFAULT_JDE_F_INIT
        or float(args.jde_cr_init) != DEFAULT_JDE_CR_INIT
        or float(args.jde_f_l) != DEFAULT_JDE_F_L
        or float(args.jde_f_u) != DEFAULT_JDE_F_U
        or float(args.jde_tau_f) != DEFAULT_JDE_TAU_F
        or float(args.jde_tau_cr) != DEFAULT_JDE_TAU_CR
        or bool(args.jde_elitism) != DEFAULT_JDE_ELITISM
    )
    if algo != "jde" and jde_opts_nondefault:
        print(
            f"Note: --jde-* flags are ignored for --algo {algo} (jde only)."
        )
    pso_opts_nondefault = (
        float(args.pso_inertia) != DEFAULT_PSO_INERTIA
        or float(args.pso_cognitive) != DEFAULT_PSO_COGNITIVE
        or float(args.pso_social) != DEFAULT_PSO_SOCIAL
        or float(args.pso_max_velocity) != DEFAULT_PSO_MAX_VELOCITY
    )
    if algo != "pso" and pso_opts_nondefault:
        print(
            f"Note: --pso-* flags are ignored for --algo {algo} (pso only)."
        )

    popsize = _resolve_popsize(args, problem.n_var)
    steps, evals = _resolve_steps_and_evals(args, popsize)
    args.popsize = popsize
    args.steps = steps
    args.evals = evals
    run_name = make_run_name(args.dataset, args.algo, args.seed, args.wandb_name)
    fitness_name = soo.fitness_name

    if args.wandb:
        config = build_soo_wandb_config(
            args, n_var=problem.n_var, popsize=popsize, fitness_name=fitness_name
        )
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            config=config,
            reinit="finish_previous",
        )
        define_wandb_n_eval_metric()

    es_banner = ""
    if uses_mean_optimizer:
        es_banner = (
            f"  es_optim={args.es_optim}  es_optim_lr={args.es_optim_lr}  "
            f"es_optim_scheduler={args.es_optim_scheduler}  "
            f"es_optim_momentum={args.es_optim_momentum}  "
            f"es_optim_wd={args.es_optim_wd}"
        )
    if uses_sigma_schedule:
        end_str = (
            f"{args.es_sigma_end}" if args.es_sigma_end is not None else "default"
        )
        es_banner += (
            f"  es_sigma_scheduler={args.es_sigma_scheduler}  "
            f"es_sigma_end={end_str}"
        )
    if algo == "asebo":
        es_banner += f"  asebo_subspace_dims={args.asebo_subspace_dims}"
    if algo == "de":
        es_banner += (
            f"  de_f={args.de_f}  de_cr={args.de_cr}  de_elitism={args.de_elitism}"
        )
    if algo == "jde":
        es_banner += (
            f"  jde_f_init={args.jde_f_init}  jde_cr_init={args.jde_cr_init}  "
            f"jde_f_l={args.jde_f_l}  jde_f_u={args.jde_f_u}  "
            f"jde_tau_f={args.jde_tau_f}  jde_tau_cr={args.jde_tau_cr}  "
            f"jde_elitism={args.jde_elitism}"
        )
    if algo == "pso":
        es_banner += (
            f"  pso_inertia={args.pso_inertia}  "
            f"pso_cognitive={args.pso_cognitive}  "
            f"pso_social={args.pso_social}  "
            f"pso_max_velocity={args.pso_max_velocity}"
        )

    sol_tag = "best" if algo in POPULATION_BASED_ALGOS else "mean"
    w_str = ",".join(f"{x:g}" for x in args.scalar_weights_resolved)
    print(
        f"dataset={args.dataset}  problem={args.problem}  "
        f"activation={args.activation}  algo={algo} ({display})  "
        f"fitness={fitness_name}  scalar_weights=[{w_str}]  "
        f"init={args.init}(sigma={args.init_sigma})  "
        f"n_var={problem.n_var}  n_obj={problem.n_obj}->1  "
        f"xl={args.xl}  xu={args.xu}  "
        f"popsize={popsize}  steps={args.steps}  evals={args.evals}  "
        f"es_std_init={args.init_sigma}"
        f"{es_banner}  "
        f"batch_size={args.batch_size}  eval_mode={args.eval_mode}  "
        f"eval_batches={problem.eval_batches}  "
        f"sampler={type(batch_sampler).__name__}  "
        f"val_every={args.val_every}  "
        f"device={problem.device}"
        + (f"  wandb={args.wandb_entity}/{args.wandb_project}/{run_name}" if args.wandb else "  wandb=off")
    )

    try:
        result = run_soo_es(
            problem,
            algo=algo,
            steps=args.steps,
            popsize=popsize,
            init=args.init,
            init_sigma=args.init_sigma,
            seed=args.seed,
                val_every=args.val_every,
            test_loader=test_loader,
            n_classes=num_classes,
            verbose=args.verbose,
            use_wandb=args.wandb,
            es_optim=args.es_optim,
            es_optim_lr=args.es_optim_lr,
            es_optim_scheduler=args.es_optim_scheduler,
            es_optim_momentum=args.es_optim_momentum,
            es_optim_wd=args.es_optim_wd,
            es_sigma_scheduler=args.es_sigma_scheduler,
            es_sigma_end=args.es_sigma_end,
            asebo_subspace_dims=args.asebo_subspace_dims,
            de_f=args.de_f,
            de_cr=args.de_cr,
            de_elitism=args.de_elitism,
            jde_f_init=args.jde_f_init,
            jde_cr_init=args.jde_cr_init,
            jde_f_l=args.jde_f_l,
            jde_f_u=args.jde_f_u,
            jde_tau_f=args.jde_tau_f,
            jde_tau_cr=args.jde_tau_cr,
            jde_elitism=args.jde_elitism,
            pso_inertia=args.pso_inertia,
            pso_cognitive=args.pso_cognitive,
            pso_social=args.pso_social,
            pso_max_velocity=args.pso_max_velocity,
        )
    except Exception:
        finish_wandb()
        raise

    print(f"Finished after {result.steps} steps.")
    detail_str = "  ".join(
        f"{k}={v:.6f}" for k, v in result.details.items() if k != "f"
    )
    print(
        f"ES {sol_tag}: f={result.f:.6f}"
        + (f"  {detail_str}" if detail_str else "")
    )

    summary = {
        "final/steps": result.steps,
        "final/f": result.f,
        "final/popsize": result.popsize,
        "final/fitness": result.fitness_name,
        "final/algo": algo,
        "final/val_solution": (
            "de_best" if algo in POPULATION_BASED_ALGOS else "es_mean"
        ),
        **{f"final/{k}": v for k, v in result.details.items() if k != "f"},
    }

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            X=result.X,
            f=result.f,
            mean_f_history=result.mean_f_history,
            dataset=args.dataset,
            problem=args.problem,
            activation=args.activation,
            algo=args.algo,
            init=args.init,
            init_sigma=args.init_sigma,
            es_optim=args.es_optim,
            es_optim_lr=args.es_optim_lr,
            es_optim_scheduler=args.es_optim_scheduler,
            es_optim_momentum=args.es_optim_momentum,
            es_optim_wd=args.es_optim_wd,
            es_sigma_scheduler=args.es_sigma_scheduler,
            es_sigma_end=args.es_sigma_end if args.es_sigma_end is not None else -1.0,
            steps=args.steps,
            popsize=result.popsize,
            fitness=result.fitness_name,
            val_solution="es_mean",
            **{k: v for k, v in result.details.items() if k != "f"},
        )
        print(f"Saved ES mean weights to {out_path}")

    ckpt_path = save_wandb_checkpoint(
        problem,
        result.X,
        meta={
            "dataset": args.dataset,
            "problem": args.problem,
            "activation": args.activation,
            "algo": args.algo,
            "num_classes": num_classes,
            "val_solution": "es_mean",
            "f": float(result.f),
            "fitness": result.fitness_name,
            **{k: v for k, v in result.details.items() if k != "f"},
        },
    )
    if ckpt_path is not None:
        print(f"Saved W&B checkpoint to {ckpt_path}")

    finish_wandb(summary)
    return result


def run_mo_es(args: argparse.Namespace, problem, test_loader, num_classes: int, batch_sampler):
    """Multi-objective OpenES: Design A (MGDA/UPGrad) / Design B (MOEA/D)."""
    algo = args.algo
    if args.problem in SOO_ONLY_PROBLEMS or problem.n_obj < 2:
        raise SystemExit(
            f"--algo {algo} needs a multi-objective problem "
            f"(e.g. cwrm_cross_entropy); got {args.problem!r} (n_obj={problem.n_obj})."
        )
    if args.crossover != "sbx" or args.mutation != "pm":
        print(f"Note: --crossover / --mutation are ignored for --algo {algo}.")

    popsize = _resolve_popsize(args, problem.n_var)
    steps, evals = _resolve_steps_and_evals(args, popsize)
    args.popsize = popsize
    args.steps = steps
    args.evals = evals
    run_name = make_run_name(args.dataset, args.algo, args.seed, args.wandb_name)

    if args.wandb:
        config = build_mo_es_wandb_config(
            args, n_var=problem.n_var, n_obj=problem.n_obj, popsize=popsize
        )
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            config=config,
            reinit="finish_previous",
        )
        define_wandb_n_eval_metric()

    moead_banner = ""
    mgda_banner = ""
    if algo in {"mgda_open_es", "upgrad_open_es"}:
        mgda_banner = (
            f"  aggregator={'upgrad' if algo == 'upgrad_open_es' else 'mgda'}"
            f"  mgda_normalize={args.mgda_normalize}"
            f"  es_fitness_shaping={args.es_fitness_shaping}"
        )
    if algo == "moead_open_es":
        moead_banner = (
            f"  k={args.moead_k}  scalarization={args.moead_scalarization}  "
            f"rho={args.moead_rho}  "
            f"weight_shrink={args.moead_weight_shrink}  "
            f"ideal={args.moead_ideal}  "
            f"migrate_every={args.moead_migrate_every}  ref_dirs={args.ref_dirs}"
        )
    print(
        f"dataset={args.dataset}  problem={args.problem}  "
        f"activation={args.activation}  algo={algo}  "
        f"fitness=vector  "
        f"init={args.init}(sigma={args.init_sigma})  "
        f"n_var={problem.n_var}  n_obj={problem.n_obj}  "
        f"xl={args.xl}  xu={args.xu}  "
        f"popsize={popsize}  steps={args.steps}  evals={args.evals}  "
        f"es_optim={args.es_optim}  es_optim_lr={args.es_optim_lr}  "
        f"es_optim_scheduler={args.es_optim_scheduler}  "
        f"es_optim_momentum={args.es_optim_momentum}  "
        f"es_optim_wd={args.es_optim_wd}  "
        f"es_sigma_scheduler={args.es_sigma_scheduler}  "
        f"es_sigma_end={args.es_sigma_end if args.es_sigma_end is not None else 'default'}  "
        f"archive={args.archive_selection}({args.archive_size})"
        f"{mgda_banner}{moead_banner}  "
        f"batch_size={args.batch_size}  eval_mode={args.eval_mode}  "
        f"eval_batches={problem.eval_batches}  "
        f"sampler={type(batch_sampler).__name__}  "
        f"val_every={args.val_every}  "
        f"device={problem.device}"
        + (f"  wandb={args.wandb_entity}/{args.wandb_project}/{run_name}" if args.wandb else "  wandb=off")
    )

    shared_kwargs = dict(
        steps=args.steps,
        popsize=popsize,
        init=args.init,
        init_sigma=args.init_sigma,
        seed=args.seed,
        val_every=args.val_every,
        pareto_every=args.pareto_every,
        test_loader=test_loader,
        n_classes=num_classes,
        verbose=args.verbose,
        use_wandb=args.wandb,
        es_optim=args.es_optim,
        es_optim_lr=args.es_optim_lr,
        es_optim_scheduler=args.es_optim_scheduler,
        es_optim_momentum=args.es_optim_momentum,
        es_optim_wd=args.es_optim_wd,
        es_sigma_scheduler=args.es_sigma_scheduler,
        es_sigma_end=args.es_sigma_end,
        archive_size=args.archive_size,
        archive_selection=args.archive_selection,
        ref_dirs_method=args.ref_dirs,
    )
    try:
        if algo == "mgda_open_es":
            result = run_mgda_open_es(
                problem,
                mgda_normalize=args.mgda_normalize,
                es_fitness_shaping=args.es_fitness_shaping,
                aggregator="mgda",
                **shared_kwargs,
            )
        elif algo == "upgrad_open_es":
            result = run_upgrad_open_es(
                problem,
                mgda_normalize=args.mgda_normalize,
                es_fitness_shaping=args.es_fitness_shaping,
                **shared_kwargs,
            )
        else:
            result = run_moead_open_es(
                problem,
                k=args.moead_k,
                rho=args.moead_rho,
                weight_shrink=args.moead_weight_shrink,
                ideal_mode=args.moead_ideal,
                scalarization=args.moead_scalarization,
                migrate_every=args.moead_migrate_every,
                **shared_kwargs,
            )
    except Exception:
        finish_wandb()
        raise

    print(f"Finished after {result.steps} steps.")
    print(f"Archive ND set size: {len(result.F)}")
    mean_obj = result.F.mean(axis=1)
    best_idx = int(np.argmin(mean_obj))
    print(f"Best mean objective (archive): {mean_obj[best_idx]:.6f}")
    print(f"Objectives (best mean): {np.array2string(result.F[best_idx], precision=4)}")
    center_obj = result.means_F.mean(axis=1)
    print(
        f"Center(s) mean objective: "
        f"{np.array2string(center_obj, precision=4)}"
    )

    summary = {
        "final/steps": result.steps,
        "final/n_nds": len(result.F),
        "final/popsize": result.popsize,
        "final/algo": algo,
        "final/mean_obj_best": float(mean_obj[best_idx]),
        **{f"final/{k.removeprefix('val/')}": v for k, v in result.details.items() if k.startswith("val/")},
    }

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            X=result.X,
            F=result.F,
            means=result.means,
            means_F=result.means_F,
            dataset=args.dataset,
            problem=args.problem,
            activation=args.activation,
            algo=args.algo,
            init=args.init,
            init_sigma=args.init_sigma,
            es_optim=args.es_optim,
            es_optim_lr=args.es_optim_lr,
            es_optim_scheduler=args.es_optim_scheduler,
            es_optim_momentum=args.es_optim_momentum,
            es_optim_wd=args.es_optim_wd,
            es_sigma_scheduler=args.es_sigma_scheduler,
            es_sigma_end=args.es_sigma_end if args.es_sigma_end is not None else -1.0,
            steps=args.steps,
            popsize=result.popsize,
            archive_size=args.archive_size,
            archive_selection=args.archive_selection,
        )
        if result.weights is not None:
            payload["moead_weights"] = result.weights
        np.savez(out_path, **payload)
        print(f"Saved archive + centers to {out_path}")

    ckpt_path = save_wandb_checkpoint(
        problem,
        result.X[best_idx],
        meta={
            "dataset": args.dataset,
            "problem": args.problem,
            "activation": args.activation,
            "algo": args.algo,
            "num_classes": num_classes,
            "val_solution": "best_mean_obj",
            "solution_index": best_idx,
            "F": np.asarray(result.F, dtype=np.float64),
            "X": np.asarray(result.X, dtype=np.float64),
            "means": np.asarray(result.means, dtype=np.float64),
            "means_F": np.asarray(result.means_F, dtype=np.float64),
        },
    )
    if ckpt_path is not None:
        print(f"Saved W&B checkpoint to {ckpt_path}")

    finish_wandb(summary)
    return result


def run(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    args.device = str(device)
    dataset, num_classes = load_dataset(args.dataset, root=args.data_root, train=True)
    test_dataset, _ = load_dataset(args.dataset, root=args.data_root, train=False)
    model = build_model(args.dataset, num_classes, activation=args.activation)
    test_loader = make_test_loader(test_dataset, batch_size=args.val_batch_size)

    # Normalize aliases (e.g. cwce -> cwrm_cross_entropy).
    args.problem = PROBLEM_ALIASES.get(args.problem, args.problem)

    if args.algo in MOO_ALGORITHMS and args.problem in SOO_ONLY_PROBLEMS:
        raise SystemExit(
            f"--problem {args.problem} is single-objective; use --algo "
            f"{'/'.join(SOO_ALGORITHMS)} (not {args.algo})."
        )
    if (
        args.algo not in SOO_ALGORITHMS
        and getattr(args, "scalar_weights", None) is not None
    ):
        print(
            f"Note: --scalar-weights is only used for SOO algos; "
            f"ignored for --algo {args.algo}."
        )

    if not (float(args.xu) > float(args.xl)):
        raise SystemExit(
            f"--xu must be > --xl, got xl={args.xl}, xu={args.xu}"
        )

    batch_sampler = build_eval_sampler(
        args.problem,
        dataset,
        batch_size=args.batch_size,
        num_classes=num_classes,
        seed=args.seed,
        sampler=args.sampler,
    )
    problem = build_problem(
        args.problem,
        model=model,
        batch_sampler=batch_sampler,
        device=device,
        eval_mode=args.eval_mode,
        eval_batches=args.eval_batches,
        xl=args.xl,
        xu=args.xu,
    )

    if args.algo in MOO_ALGORITHMS:
        return run_moo(args, problem, test_loader, num_classes, batch_sampler)
    if args.algo in SOO_ALGORITHMS:
        return run_soo(args, problem, test_loader, num_classes, batch_sampler)
    if args.algo in MO_ES_ALGORITHMS:
        return run_mo_es(args, problem, test_loader, num_classes, batch_sampler)
    raise SystemExit(f"Unknown algo: {args.algo!r}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
