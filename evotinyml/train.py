"""Train TinyCNN weights with NSGA-II / NSGA-III or evosax SOO (CMA-ES, SNES, …)."""

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
from evotinyml.moo.callback import ResampleBatchCallback
from evotinyml.moo.display import StepOutput
from evotinyml.moo.mo_es import (
    ARCHIVE_SELECTIONS,
    DEFAULT_ARCHIVE_SIZE,
    DEFAULT_MOEAD_IDEAL,
    DEFAULT_MOEAD_K,
    DEFAULT_MOEAD_RHO,
    DEFAULT_MOEAD_SCALARIZATION,
    DEFAULT_MOEAD_WEIGHT_SHRINK,
    MO_ES_ALGORITHMS,
    MOEAD_IDEAL_MODES,
    MOEAD_SCALARIZATIONS,
    build_mo_es_wandb_config,
    run_mgda_open_es,
    run_moead_open_es,
)
from evotinyml.moo.termination import MaximumStepTermination
from evotinyml.soo.algorithms import SOO_ALGORITHMS
from evotinyml.soo.es import (
    DEFAULT_ES_OPTIM,
    DEFAULT_ES_OPTIM_LR,
    DEFAULT_ES_OPTIM_MOMENTUM,
    DEFAULT_ES_OPTIM_SCHEDULER,
    DEFAULT_ES_SIGMA_SCHEDULER,
    ES_OPTIMS,
    ES_OPTIM_SCHEDULERS,
    ES_SIGMA_SCHEDULERS,
    EVOSAX_SOO_ALGOS,
    build_soo_wandb_config,
    default_soo_popsize,
    run_soo_es,
)
from evotinyml.data import EVAL_MODES, load_dataset
from evotinyml.model import ACTIVATIONS, build_model
from evotinyml.problem import (
    PROBLEM_ALIASES,
    PROBLEMS,
    PR_PROBLEMS,
    SOO_ONLY_PROBLEMS,
    SOO_PROBLEMS,
    build_eval_sampler,
    build_problem,
)
from evotinyml.sampling import get_population_init
from evotinyml.validation import make_test_loader
from evotinyml.wandb_logger import (
    DEFAULT_ENTITY,
    DEFAULT_PROJECT,
    define_wandb_n_eval_metric,
    finish_wandb,
    init_wandb,
    make_run_name,
)

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
            "precision_recall / soft_precision_recall (2-obj 1-P/1-R; soft sum for SOO), "
            "or erm_cross_entropy (ERM + mean CE for SOO). "
            "Aliases: cwce/per_class_ce → cwrm; cross_entropy → erm."
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
            "Optimizer: nsga2 / nsga3 (MOO), cmaes / snes / xnes / open_es "
            "(SOO: soft P/R sum or mean CE), or mgda_open_es / moead_open_es "
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
        required=True,
        help=(
            "Population / ES mean initialization: uniform[-init_sigma, init_sigma], "
            "gaussian N(0, init_sigma), both (half/half mix; SOO ES uses gaussian mean), "
            "zeros (all-zero), or kaiming (ES mean = PyTorch default / Kaiming-uniform "
            "weights; NSGA: one individual at that theta0, rest theta0+N(0,init_sigma)). "
            "--init-sigma still sets ES sampling std."
        ),
    )
    parser.add_argument(
        "--init-sigma",
        type=float,
        default=0.1,
        help="Init scale (also ES std_init / OpenES noise std): uniform [-sigma,sigma], gaussian N(0,sigma).",
    )
    parser.add_argument(
        "--es-optim",
        choices=ES_OPTIMS,
        default=DEFAULT_ES_OPTIM,
        help=(
            f"OpenES mean-update optimizer (optax). Default: {DEFAULT_ES_OPTIM}. "
            "Ignored for cmaes / snes / xnes."
        ),
    )
    parser.add_argument(
        "--es-optim-lr",
        type=float,
        default=DEFAULT_ES_OPTIM_LR,
        help=(
            f"OpenES optimizer learning rate (initial value if scheduled). "
            f"Default: {DEFAULT_ES_OPTIM_LR}. Ignored for cmaes / snes / xnes."
        ),
    )
    parser.add_argument(
        "--es-optim-scheduler",
        choices=ES_OPTIM_SCHEDULERS,
        default=DEFAULT_ES_OPTIM_SCHEDULER,
        help=(
            "OpenES LR schedule over steps: constant, cosine (decay to 0 over steps), "
            "or exponential. Ignored for cmaes / snes / xnes."
        ),
    )
    parser.add_argument(
        "--es-optim-momentum",
        type=float,
        default=DEFAULT_ES_OPTIM_MOMENTUM,
        help=(
            f"OpenES SGD momentum (0 = off). Default: {DEFAULT_ES_OPTIM_MOMENTUM}. "
            "Only used with --es-optim sgd; ignored for adam/adamw and non-OpenES algos."
        ),
    )
    parser.add_argument(
        "--es-sigma-scheduler",
        choices=ES_SIGMA_SCHEDULERS,
        default=DEFAULT_ES_SIGMA_SCHEDULER,
        help=(
            "OpenES / MO-OpenES sampling-noise (σ) schedule over steps: constant, "
            "cosine, or exponential. Start value is --init-sigma. "
            "Ignored for cmaes / snes / xnes."
        ),
    )
    parser.add_argument(
        "--es-sigma-end",
        type=float,
        default=None,
        help=(
            "Final σ for cosine / exponential --es-sigma-scheduler "
            "(default: max(0.01 * init_sigma, 1e-6)). Ignored for constant / non-OpenES."
        ),
    )
    parser.add_argument(
        "--archive-size",
        type=int,
        default=DEFAULT_ARCHIVE_SIZE,
        help=(
            f"Max non-dominated archive size for mgda_open_es / moead_open_es "
            f"(default: {DEFAULT_ARCHIVE_SIZE}). Ignored for other algos."
        ),
    )
    parser.add_argument(
        "--archive-selection",
        choices=ARCHIVE_SELECTIONS,
        default="nsga2",
        help=(
            "Archive pruning for mgda_open_es / moead_open_es: nsga2 "
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
            "Ignored when --evals is set (steps = evals // popsize)."
        ),
    )
    parser.add_argument(
        "--evals",
        type=int,
        default=None,
        help=(
            "Function-evaluation budget (train fitness calls). "
            "Sets steps = evals // popsize (ES/NSGA: one generation costs popsize evals)."
        ),
    )
    parser.add_argument(
        "--popsize",
        type=int,
        default=None,
        help=(
            "Population size. Default: 100 for NSGA; "
            "4+3*ln(n_var) for SOO ES (~25 for TinyCNN; even for open_es)."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="Eval batch size.")
    parser.add_argument(
        "--eval-mode",
        choices=EVAL_MODES,
        default="single",
        help=(
            "Fitness evaluation mode: single (one random/class-balanced batch) or "
            "multi (pool several batches, then compute metrics once on all predictions)."
        ),
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=8,
        help="Number of random batches to pool when --eval-mode multi (ignored for single).",
    )
    parser.add_argument(
        "--resample-every",
        type=int,
        default=50,
        help="Redraw the eval batch pool every N steps (0 = never resample).",
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
            "P–R also writes train_history.npz; all MOO problems log "
            "train/pareto_front (and val/pareto_front on validation)."
        ),
    )
    parser.add_argument(
        "--train-history",
        type=str,
        default="train_history.npz",
        help="Path for train Pareto front history (MOO P–R problems only).",
    )
    parser.add_argument(
        "--val-history",
        type=str,
        default="val_history.npz",
        help="Path for val Pareto front history (MOO P–R problems only).",
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
        help="Torch device, e.g. cpu or cuda.",
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
    if args.algo == "mgda_open_es":
        return default_soo_popsize(n_var, "open_es")
    if args.algo == "moead_open_es":
        # Total per-generation evaluations: 8 antithetic samples per mean.
        return int(args.moead_k) * 8
    return 100


def _resolve_steps_and_evals(args: argparse.Namespace, popsize: int) -> tuple[int, int]:
    """Resolve ``(steps, evals)`` from ``--steps`` and/or ``--evals``.

    ``--evals`` wins when set: ``steps = evals // popsize``.
    """
    popsize = int(popsize)
    if popsize < 1:
        raise SystemExit(f"popsize must be >= 1, got {popsize}")

    if args.evals is not None:
        evals = int(args.evals)
        if evals < popsize:
            raise SystemExit(
                f"--evals ({evals}) must be >= popsize ({popsize})"
            )
        steps = evals // popsize
        used = steps * popsize
        if used != evals:
            print(
                f"Note: --evals={evals} not divisible by popsize={popsize}; "
                f"using steps={steps} ({used} Function Evaluations)."
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
        + f"batch_size={args.batch_size}  eval_mode={args.eval_mode}  "
        f"eval_batches={problem.eval_batches}  "
        f"sampler={type(batch_sampler).__name__}  "
        f"resample_every={args.resample_every if args.resample_every > 0 else 'never'}  "
        f"val_every={args.val_every}  pareto_every={args.pareto_every}  "
        f"train_history={args.train_history}  val_history={args.val_history}  "
        f"device={problem.device}"
        + (f"  ref_dirs={args.ref_dirs}({n_ref})" if n_ref is not None else "")
        + (f"  wandb={args.wandb_entity}/{args.wandb_project}/{run_name}" if args.wandb else "  wandb=off")
    )

    try:
        minimize_kwargs = dict(
            seed=args.seed,
            verbose=args.verbose,
            save_history=False,
        )
        if args.resample_every > 0:
            minimize_kwargs["callback"] = ResampleBatchCallback(
                every=args.resample_every
            )
        result = minimize(
            problem,
            algorithm,
            termination=MaximumStepTermination(args.steps),
            **minimize_kwargs,
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
    else:
        print(f"Best mean CWRM–CE: {mean_obj[best_idx]:.6f}")
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

    # Operator args are MOO-only; note when user passed non-defaults.
    if args.crossover != "sbx" or args.mutation != "pm":
        print(f"Note: --crossover / --mutation are ignored for --algo {algo}.")

    open_es_opts_nondefault = (
        args.es_optim != DEFAULT_ES_OPTIM
        or float(args.es_optim_lr) != DEFAULT_ES_OPTIM_LR
        or args.es_optim_scheduler != DEFAULT_ES_OPTIM_SCHEDULER
        or float(args.es_optim_momentum) != DEFAULT_ES_OPTIM_MOMENTUM
        or args.es_sigma_scheduler != DEFAULT_ES_SIGMA_SCHEDULER
        or args.es_sigma_end is not None
    )
    if algo != "open_es" and open_es_opts_nondefault:
        print(
            f"Note: --es-optim / --es-optim-lr / --es-optim-scheduler / "
            f"--es-optim-momentum / --es-sigma-scheduler / --es-sigma-end "
            f"are ignored for --algo {algo} (OpenES only)."
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
            reinit=True,
        )
        define_wandb_n_eval_metric()

    open_es_banner = ""
    if algo == "open_es":
        end_str = (
            f"{args.es_sigma_end}" if args.es_sigma_end is not None else "default"
        )
        open_es_banner = (
            f"  es_optim={args.es_optim}  es_optim_lr={args.es_optim_lr}  "
            f"es_optim_scheduler={args.es_optim_scheduler}  "
            f"es_optim_momentum={args.es_optim_momentum}  "
            f"es_sigma_scheduler={args.es_sigma_scheduler}  "
            f"es_sigma_end={end_str}"
        )

    print(
        f"dataset={args.dataset}  problem={args.problem}  "
        f"activation={args.activation}  algo={algo} ({display})  "
        f"fitness={fitness_name}  "
        f"init={args.init}(sigma={args.init_sigma})  "
        f"n_var={problem.n_var}  n_obj=1  "
        f"popsize={popsize}  steps={args.steps}  evals={args.evals}  "
        f"es_std_init={args.init_sigma}"
        f"{open_es_banner}  "
        f"batch_size={args.batch_size}  eval_mode={args.eval_mode}  "
        f"eval_batches={problem.eval_batches}  "
        f"sampler={type(batch_sampler).__name__}  "
        f"resample_every={args.resample_every if args.resample_every > 0 else 'never'}  "
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
            resample_every=args.resample_every,
            val_every=args.val_every,
            test_loader=test_loader,
            n_classes=num_classes,
            verbose=args.verbose,
            use_wandb=args.wandb,
            es_optim=args.es_optim,
            es_optim_lr=args.es_optim_lr,
            es_optim_scheduler=args.es_optim_scheduler,
            es_optim_momentum=args.es_optim_momentum,
            es_sigma_scheduler=args.es_sigma_scheduler,
            es_sigma_end=args.es_sigma_end,
        )
    except Exception:
        finish_wandb()
        raise

    print(f"Finished after {result.steps} steps.")
    detail_str = "  ".join(
        f"{k}={v:.6f}" for k, v in result.details.items() if k != "f"
    )
    print(f"ES mean: f={result.f:.6f}" + (f"  {detail_str}" if detail_str else ""))

    summary = {
        "final/steps": result.steps,
        "final/f": result.f,
        "final/popsize": result.popsize,
        "final/fitness": result.fitness_name,
        "final/algo": algo,
        "final/val_solution": "es_mean",
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
            es_sigma_scheduler=args.es_sigma_scheduler,
            es_sigma_end=args.es_sigma_end if args.es_sigma_end is not None else -1.0,
            steps=args.steps,
            popsize=result.popsize,
            fitness=result.fitness_name,
            val_solution="es_mean",
            **{k: v for k, v in result.details.items() if k != "f"},
        )
        print(f"Saved ES mean weights to {out_path}")

    finish_wandb(summary)
    return result


def run_mo_es(args: argparse.Namespace, problem, test_loader, num_classes: int, batch_sampler):
    """Multi-objective OpenES: mgda_open_es (Design A) / moead_open_es (Design B)."""
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
            reinit=True,
        )
        define_wandb_n_eval_metric()

    moead_banner = ""
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
        f"popsize={popsize}  steps={args.steps}  evals={args.evals}  "
        f"es_optim={args.es_optim}  es_optim_lr={args.es_optim_lr}  "
        f"es_optim_scheduler={args.es_optim_scheduler}  "
        f"es_optim_momentum={args.es_optim_momentum}  "
        f"es_sigma_scheduler={args.es_sigma_scheduler}  "
        f"es_sigma_end={args.es_sigma_end if args.es_sigma_end is not None else 'default'}  "
        f"archive={args.archive_selection}({args.archive_size})"
        f"{moead_banner}  "
        f"batch_size={args.batch_size}  eval_mode={args.eval_mode}  "
        f"eval_batches={problem.eval_batches}  "
        f"sampler={type(batch_sampler).__name__}  "
        f"resample_every={args.resample_every if args.resample_every > 0 else 'never'}  "
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
        resample_every=args.resample_every,
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
        es_sigma_scheduler=args.es_sigma_scheduler,
        es_sigma_end=args.es_sigma_end,
        archive_size=args.archive_size,
        archive_selection=args.archive_selection,
        ref_dirs_method=args.ref_dirs,
    )
    try:
        if algo == "mgda_open_es":
            result = run_mgda_open_es(problem, **shared_kwargs)
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

    finish_wandb(summary)
    return result


def run(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    dataset, num_classes = load_dataset(args.dataset, root=args.data_root, train=True)
    test_dataset, _ = load_dataset(args.dataset, root=args.data_root, train=False)
    model = build_model(args.dataset, num_classes, activation=args.activation)
    test_loader = make_test_loader(test_dataset, batch_size=args.val_batch_size)

    # Normalize aliases (e.g. cwce → cwrm_cross_entropy).
    args.problem = PROBLEM_ALIASES.get(args.problem, args.problem)

    # SOO ES defaults to soft_precision_recall when user left the MOO default.
    if args.algo in SOO_ALGORITHMS and args.problem == "cwrm_cross_entropy":
        print(
            f"Note: --algo {args.algo} with default cwrm_cross_entropy → soft_precision_recall. "
            "Use --problem erm_cross_entropy for scalar ERM–CE."
        )
        args.problem = "soft_precision_recall"

    if args.algo in MOO_ALGORITHMS and args.problem in SOO_ONLY_PROBLEMS:
        raise SystemExit(
            f"--problem {args.problem} is single-objective; use --algo "
            f"{'/'.join(SOO_ALGORITHMS)} (not {args.algo})."
        )

    batch_sampler = build_eval_sampler(
        args.problem,
        dataset,
        batch_size=args.batch_size,
        num_classes=num_classes,
        seed=args.seed,
    )
    problem = build_problem(
        args.problem,
        model=model,
        batch_sampler=batch_sampler,
        device=device,
        resample_every=args.resample_every,
        eval_mode=args.eval_mode,
        eval_batches=args.eval_batches,
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
