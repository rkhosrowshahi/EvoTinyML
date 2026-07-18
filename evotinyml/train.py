"""Train TinyCNN weights with NSGA-II / NSGA-III / CMA-ES."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import wandb
from pymoo.optimize import minimize

from evotinyml.algorithms import (
    ALGORITHMS,
    CROSSOVERS,
    MOO_ALGORITHMS,
    MUTATIONS,
    OperatorConfig,
    build_algorithm,
)
from evotinyml.callback import ResampleBatchCallback
from evotinyml.cmaes import build_cma_wandb_config, default_cma_popsize, run_soo_cma
from evotinyml.data import EVAL_MODES, load_dataset
from evotinyml.display import StepOutput
from evotinyml.model import ACTIVATIONS, build_model
from evotinyml.problem import (
    CMA_PROBLEMS,
    PROBLEM_ALIASES,
    PROBLEMS,
    PR_PROBLEMS,
    SOO_ONLY_PROBLEMS,
    build_eval_sampler,
    build_problem,
)
from evotinyml.sampling import get_population_init
from evotinyml.termination import MaximumStepTermination
from evotinyml.validation import make_test_loader
from evotinyml.wandb_logger import (
    DEFAULT_ENTITY,
    DEFAULT_PROJECT,
    define_wandb_n_eval_metric,
    finish_wandb,
    init_wandb,
    make_run_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evolutionary TinyCNN training (NSGA-II/III or CMA-ES)."
    )
    parser.add_argument(
        "--dataset",
        choices=("mnist", "cifar10"),
        required=True,
        help="Dataset to train on.",
    )
    parser.add_argument(
        "--problem",
        choices=(*PROBLEMS, *PROBLEM_ALIASES),
        default="cwrm_cross_entropy",
        help=(
            "Objective: cwrm_cross_entropy (class-wise RM + CE, multi-obj), "
            "precision_recall / soft_precision_recall (2-obj 1-P/1-R; soft sum for CMA), "
            "or erm_cross_entropy (ERM + mean CE for CMA). "
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
        help="Optimizer: nsga2 / nsga3 (MOO) or cmaes (SOO: soft P/R sum or mean CE).",
    )
    parser.add_argument(
        "--ref-dirs",
        choices=("energy", "das-dennis"),
        default="energy",
        help="NSGA-III reference direction method (ignored for NSGA-II / CMA-ES).",
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
        help="Crossover operator: sbx or none (ignored for CMA-ES).",
    )
    parser.add_argument(
        "--crossover-prob",
        type=float,
        default=0.9,
        help="SBX crossover probability (ignored when --crossover none or CMA-ES).",
    )
    parser.add_argument(
        "--crossover-eta",
        type=float,
        default=15.0,
        help="SBX distribution index eta (ignored when --crossover none or CMA-ES).",
    )
    parser.add_argument(
        "--crossover-prob-var",
        type=float,
        default=0.5,
        help="SBX per-variable crossover probability (ignored when --crossover none or CMA-ES).",
    )
    parser.add_argument(
        "--mutation",
        choices=MUTATIONS,
        default="pm",
        help=(
            "Mutation operator: pm (polynomial), gaussian (absolute N(0, sigma)), "
            "or layerwise (He fan-in scaled Gaussian). Ignored for CMA-ES."
        ),
    )
    parser.add_argument(
        "--mutation-prob",
        type=float,
        default=0.9,
        help="Per-individual mutation probability (ignored for CMA-ES).",
    )
    parser.add_argument(
        "--mutation-eta",
        type=float,
        default=20.0,
        help="Polynomial mutation distribution index eta (ignored for CMA-ES).",
    )
    parser.add_argument(
        "--mutation-sigma",
        type=float,
        default=0.1,
        help=(
            "Gaussian mutation std: absolute for --mutation gaussian; "
            "mean per-variable std for --mutation layerwise. Ignored for CMA-ES."
        ),
    )
    parser.add_argument(
        "--mutation-prob-var",
        type=float,
        default=None,
        help=(
            "Per-variable mutation probability. Default: pymoo's min(0.5, 1/n_var). "
            "Ignored for CMA-ES."
        ),
    )
    parser.add_argument(
        "--init",
        choices=("uniform", "gaussian", "both", "zeros"),
        required=True,
        help=(
            "Population / CMA mean initialization: uniform[-init_sigma, init_sigma], "
            "gaussian N(0, init_sigma), both (half/half mix; CMA uses gaussian mean), "
            "or zeros (all-zero vector / CMA mean at 0; --init-sigma still sets CMA std_init)."
        ),
    )
    parser.add_argument(
        "--init-sigma",
        type=float,
        default=0.1,
        help="Init scale (also CMA std_init): uniform [-sigma,sigma], gaussian N(0,sigma).",
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
            "Sets steps = evals // popsize (CMA/NSGA: one generation costs popsize evals)."
        ),
    )
    parser.add_argument(
        "--popsize",
        type=int,
        default=None,
        help=(
            "Population size. Default: 100 for NSGA; "
            "4+3*ln(n_var) for CMA-ES (~25 for TinyCNN)."
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
        help="Evaluate on the test set every N steps (ND set for MOO; best for CMA-ES).",
    )
    parser.add_argument(
        "--pareto-every",
        type=int,
        default=100,
        help=(
            "Save train Pareto front to train_history.npz at step 1 "
            "and every N steps thereafter (MOO P–R problems only; ignored for CMA-ES)."
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
    if args.algo == "cmaes":
        return default_cma_popsize(n_var)
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
        sampling=get_population_init(args.init, init_sigma=args.init_sigma),
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
    if args.problem not in CMA_PROBLEMS:
        raise SystemExit(
            f"--algo cmaes requires --problem in {sorted(CMA_PROBLEMS)} "
            f"(got {args.problem!r})."
        )
    soo = getattr(problem, "soo_fitness", None)
    if soo is None:
        raise SystemExit(
            f"Internal error: problem {type(problem).__name__} has no soo_fitness"
        )

    # Operator args are MOO-only; note when user passed non-defaults.
    if args.crossover != "sbx" or args.mutation != "pm":
        print("Note: --crossover / --mutation are ignored for --algo cmaes.")

    popsize = _resolve_popsize(args, problem.n_var)
    steps, evals = _resolve_steps_and_evals(args, popsize)
    args.popsize = popsize
    args.steps = steps
    args.evals = evals
    run_name = make_run_name(args.dataset, args.algo, args.seed, args.wandb_name)
    fitness_name = soo.fitness_name

    if args.wandb:
        config = build_cma_wandb_config(
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

    print(
        f"dataset={args.dataset}  problem={args.problem}  "
        f"activation={args.activation}  algo=cmaes  "
        f"fitness={fitness_name}  "
        f"init={args.init}(sigma={args.init_sigma})  "
        f"n_var={problem.n_var}  n_obj=1  "
        f"popsize={popsize}  steps={args.steps}  evals={args.evals}  "
        f"cma_std_init={args.init_sigma}  "
        f"batch_size={args.batch_size}  eval_mode={args.eval_mode}  "
        f"eval_batches={problem.eval_batches}  "
        f"sampler={type(batch_sampler).__name__}  "
        f"resample_every={args.resample_every if args.resample_every > 0 else 'never'}  "
        f"val_every={args.val_every}  "
        f"device={problem.device}"
        + (f"  wandb={args.wandb_entity}/{args.wandb_project}/{run_name}" if args.wandb else "  wandb=off")
    )

    try:
        result = run_soo_cma(
            problem,
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
        )
    except Exception:
        finish_wandb()
        raise

    print(f"Finished after {result.steps} steps.")
    detail_str = "  ".join(
        f"{k}={v:.6f}" for k, v in result.details.items() if k != "f"
    )
    print(f"CMA mean: f={result.f:.6f}" + (f"  {detail_str}" if detail_str else ""))

    summary = {
        "final/steps": result.steps,
        "final/f": result.f,
        "final/popsize": result.popsize,
        "final/fitness": result.fitness_name,
        "final/val_solution": "cma_mean",
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
            steps=args.steps,
            popsize=result.popsize,
            fitness=result.fitness_name,
            val_solution="cma_mean",
            **{k: v for k, v in result.details.items() if k != "f"},
        )
        print(f"Saved CMA mean weights to {out_path}")

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

    # CMA-ES defaults to soft_precision_recall when user left the MOO default.
    if args.algo == "cmaes" and args.problem == "cwrm_cross_entropy":
        print(
            "Note: --algo cmaes with default cwrm_cross_entropy → soft_precision_recall. "
            "Use --problem erm_cross_entropy for scalar ERM–CE."
        )
        args.problem = "soft_precision_recall"

    if args.algo in MOO_ALGORITHMS and args.problem in SOO_ONLY_PROBLEMS:
        raise SystemExit(
            f"--problem {args.problem} is single-objective; use --algo cmaes "
            f"(not {args.algo})."
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
    if args.algo == "cmaes":
        return run_soo(args, problem, test_loader, num_classes, batch_sampler)
    raise SystemExit(f"Unknown algo: {args.algo!r}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
