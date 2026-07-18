"""Weights & Biases helpers for EvoTinyML runs."""

from __future__ import annotations

from typing import Any

import wandb

from evotinyml.algorithms import OperatorConfig


DEFAULT_ENTITY = "rasa_research"
DEFAULT_PROJECT = "EvoTinyML-Precision-Recall"


def make_run_name(dataset: str, algo: str, seed: int) -> str:
    """Format: mnist-nsga2-seed0"""
    return f"{dataset}-{algo}-seed{seed}"


def build_wandb_config(
    args,
    *,
    n_var: int,
    n_obj: int,
    algo_config: dict[str, Any] | None = None,
    operators: OperatorConfig | None = None,
) -> dict[str, Any]:
    """Full run config, including NSGA-II/III operator hyperparameters."""
    ops = operators or OperatorConfig(
        crossover=getattr(args, "crossover", "sbx"),
        crossover_prob=getattr(args, "crossover_prob", 0.9),
        crossover_eta=getattr(args, "crossover_eta", 15.0),
        crossover_prob_var=getattr(args, "crossover_prob_var", 0.5),
        mutation=getattr(args, "mutation", "pm"),
        mutation_prob=getattr(args, "mutation_prob", 0.9),
        mutation_eta=getattr(args, "mutation_eta", 20.0),
        mutation_sigma=getattr(args, "mutation_sigma", 0.1),
        mutation_prob_var=getattr(args, "mutation_prob_var", None),
    )

    nsga_shared = {
        "popsize": args.popsize,
        **ops.to_dict(),
    }
    nsga2_config = {
        "name": "nsga2",
        **nsga_shared,
    }
    nsga3_config = {
        "name": "nsga3",
        **nsga_shared,
        "ref_dirs_method": getattr(args, "ref_dirs", "energy"),
        "n_partitions": getattr(args, "n_partitions", None),
    }
    if algo_config and args.algo == "nsga3":
        nsga3_config["n_ref_dirs"] = algo_config.get("n_ref_dirs")

    config = {
        "dataset": args.dataset,
        "problem": args.problem,
        "activation": args.activation,
        "algo": args.algo,
        "init": args.init,
        "init_sigma": getattr(args, "init_sigma", 0.1),
        "steps": args.steps,
        "popsize": args.popsize,
        "batch_size": args.batch_size,
        "eval_mode": getattr(args, "eval_mode", "single"),
        "eval_batches": getattr(args, "eval_batches", 8),
        "resample_every": args.resample_every,
        "val_every": args.val_every,
        "pareto_every": getattr(args, "pareto_every", 100),
        "pareto_history": getattr(args, "pareto_history", "history.npz"),
        "val_batch_size": args.val_batch_size,
        "seed": args.seed,
        "device": args.device,
        "n_var": n_var,
        "n_obj": n_obj,
        # Flat operator fields for easy filtering in the W&B UI.
        "crossover": ops.crossover,
        "crossover_prob": ops.crossover_prob,
        "crossover_eta": ops.crossover_eta,
        "crossover_prob_var": ops.crossover_prob_var,
        "mutation": ops.mutation,
        "mutation_prob": ops.mutation_prob,
        "mutation_eta": ops.mutation_eta,
        "mutation_sigma": ops.mutation_sigma,
        "mutation_prob_var": ops.mutation_prob_var,
        # Nested algorithm configs (both always present for comparison).
        "nsga2": nsga2_config,
        "nsga3": nsga3_config,
        "algo_config": algo_config or (nsga2_config if args.algo == "nsga2" else nsga3_config),
    }
    return config


def init_wandb(
    args,
    *,
    n_var: int,
    n_obj: int,
    algo_config: dict[str, Any] | None = None,
    operators: OperatorConfig | None = None,
    enabled: bool = True,
) -> Any | None:
    """Initialize a wandb run; return the run object or None if disabled."""
    if not enabled:
        return None

    run_name = make_run_name(args.dataset, args.algo, args.seed)
    config = build_wandb_config(
        args,
        n_var=n_var,
        n_obj=n_obj,
        algo_config=algo_config,
        operators=operators,
    )
    run = wandb.init(
        entity=getattr(args, "wandb_entity", DEFAULT_ENTITY),
        project=getattr(args, "wandb_project", DEFAULT_PROJECT),
        name=run_name,
        config=config,
        reinit=True,
    )
    define_wandb_step_metric(int(args.steps))
    return run


def to_wandb_step(n_gen: int, max_steps: int | None = None) -> int:
    """Map pymoo ``n_gen`` to a 0-indexed W&B step (0 = init, 1 = first opt)."""
    step = max(0, int(n_gen) - 1)
    if max_steps is not None:
        step = min(step, int(max_steps))
    return step


def define_wandb_step_metric(max_steps: int) -> None:
    """Register global step axis: 0 = init pop, 1..max_steps = optimization."""
    if wandb.run is None:
        return
    wandb.define_metric("step")
    wandb.define_metric("*", step_metric="step")
    wandb.run.config.update({"wandb_max_step": int(max_steps)}, allow_val_change=True)


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    if wandb.run is None:
        return
    payload = dict(metrics)
    if step is not None:
        payload["step"] = int(step)
    wandb.log(payload, step=step)


def finish_wandb(summary: dict[str, Any] | None = None) -> None:
    if wandb.run is None:
        return
    if summary:
        for key, value in summary.items():
            wandb.run.summary[key] = value
    wandb.finish()
