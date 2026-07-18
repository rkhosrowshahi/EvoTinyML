"""Train TinyCNN weights with NSGA-II / NSGA-III."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from pymoo.optimize import minimize

from evotinyml.algorithms import (
    ALGORITHMS,
    CROSSOVERS,
    MUTATIONS,
    OperatorConfig,
    build_algorithm,
)
from evotinyml.callback import ResampleBatchCallback
from evotinyml.data import EVAL_MODES, load_dataset
from evotinyml.display import StepOutput
from evotinyml.model import ACTIVATIONS, build_model
from evotinyml.problem import PROBLEMS, PR_PROBLEMS, build_eval_sampler, build_problem
from evotinyml.sampling import get_population_init
from evotinyml.termination import MaximumStepTermination
from evotinyml.validation import make_test_loader
from evotinyml.wandb_logger import (
    DEFAULT_ENTITY,
    DEFAULT_PROJECT,
    finish_wandb,
    init_wandb,
    make_run_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-objective TinyCNN training with NSGA-II/III."
    )
    parser.add_argument(
        "--dataset",
        choices=("mnist", "cifar10"),
        required=True,
        help="Dataset to train on.",
    )
    parser.add_argument(
        "--problem",
        choices=PROBLEMS,
        default="per_class_ce",
        help=(
            "Objective formulation: per_class_ce (10-obj class CE), "
            "precision_recall (2-obj hard 1-macroP/1-macroR), or "
            "soft_precision_recall (2-obj soft/softmax 1-macroP/1-macroR)."
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
        help="MOEA: nsga2 or nsga3.",
    )
    parser.add_argument(
        "--ref-dirs",
        choices=("energy", "das-dennis"),
        default="energy",
        help="NSGA-III reference direction method (ignored for NSGA-II).",
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
        help="Crossover operator: sbx or none (mutation-only / ES-style).",
    )
    parser.add_argument(
        "--crossover-prob",
        type=float,
        default=0.9,
        help="SBX crossover probability (ignored when --crossover none).",
    )
    parser.add_argument(
        "--crossover-eta",
        type=float,
        default=15.0,
        help="SBX distribution index eta (ignored when --crossover none).",
    )
    parser.add_argument(
        "--crossover-prob-var",
        type=float,
        default=0.5,
        help="SBX per-variable crossover probability (ignored when --crossover none).",
    )
    parser.add_argument(
        "--mutation",
        choices=MUTATIONS,
        default="pm",
        help=(
            "Mutation operator: pm (polynomial), gaussian (absolute N(0, sigma)), "
            "or layerwise (He fan-in scaled Gaussian, mean-normalized to sigma)."
        ),
    )
    parser.add_argument(
        "--mutation-prob",
        type=float,
        default=0.9,
        help="Per-individual mutation probability.",
    )
    parser.add_argument(
        "--mutation-eta",
        type=float,
        default=20.0,
        help="Polynomial mutation distribution index eta (used when --mutation pm).",
    )
    parser.add_argument(
        "--mutation-sigma",
        type=float,
        default=0.1,
        help=(
            "Gaussian mutation std: absolute for --mutation gaussian; "
            "mean per-variable std for --mutation layerwise."
        ),
    )
    parser.add_argument(
        "--mutation-prob-var",
        type=float,
        default=None,
        help=(
            "Per-variable mutation probability. Default: pymoo's min(0.5, 1/n_var). "
            "Set explicitly (e.g. 0.05–0.2) for weight-space evolution."
        ),
    )
    parser.add_argument(
        "--init",
        choices=("uniform", "gaussian", "both"),
        required=True,
        help=(
            "Population initialization: uniform[-init_sigma, init_sigma], "
            "gaussian N(0, init_sigma), or both (half/half mix)."
        ),
    )
    parser.add_argument(
        "--init-sigma",
        type=float,
        default=0.1,
        help="Init scale: uniform uses [-sigma, sigma], gaussian uses N(0, sigma).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100_000,
        help="Number of optimization steps after init (W&B logs 0=init, 1..steps=opt).",
    )
    parser.add_argument("--popsize", type=int, default=100, help="Population size.")
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
        help="Evaluate ND set on the test set every N steps.",
    )
    parser.add_argument(
        "--pareto-every",
        type=int,
        default=100,
        help=(
            "Save train precision–recall Pareto front to --pareto-history every N steps "
            "(precision_recall / soft_precision_recall only)."
        ),
    )
    parser.add_argument(
        "--pareto-history",
        type=str,
        default="history.npz",
        help=(
            "Path to local NPZ accumulating train Pareto fronts "
            "(default: history.npz). Empty string disables."
        ),
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
        help="Optional path to save final Pareto weight vectors (.npz).",
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
    return parser.parse_args()


def run(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    dataset, num_classes = load_dataset(args.dataset, root=args.data_root, train=True)
    test_dataset, _ = load_dataset(args.dataset, root=args.data_root, train=False)
    model = build_model(args.dataset, num_classes, activation=args.activation)
    test_loader = make_test_loader(test_dataset, batch_size=args.val_batch_size)

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

    algorithm = build_algorithm(
        args.algo,
        pop_size=args.popsize,
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
            pareto_history_path=args.pareto_history or None,
        ),
        seed=args.seed,
        ref_dirs_method=args.ref_dirs,
        n_partitions=args.n_partitions,
        operators=operators,
    )

    run_name = make_run_name(args.dataset, args.algo, args.seed)
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
        f"init={args.init}(sigma={args.init_sigma})  "
        f"n_var={problem.n_var}  n_obj={problem.n_obj}  "
        f"popsize={args.popsize}  steps={args.steps}  "
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
        f"pareto_history={args.pareto_history or 'off'}  "
        f"device={device}"
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
        # Train knee (console only): closest to (P=1, R=1) on the final ND front.
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
        print(f"Best mean per-class CE: {mean_obj[best_idx]:.6f}")
        print(f"Per-class CE (best mean): {np.array2string(F[best_idx], precision=4)}")
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
            popsize=args.popsize,
        )
        print(f"Saved Pareto set to {out_path}")

    output = getattr(result.algorithm, "output", None)
    if args.wandb and output is not None:
        if args.steps % args.val_every != 0:
            output._maybe_validate(result.algorithm, force=True)
        output.log_wandb(result.algorithm, wandb_step=args.steps, force=True)

    finish_wandb(summary)
    return result


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
