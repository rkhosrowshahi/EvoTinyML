"""Weights & Biases helpers for EvoTinyML runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb

from evotinyml.moo.algorithms import OperatorConfig
DEFAULT_ENTITY = "rasa_research"
DEFAULT_PROJECT = "EvoTinyML-Precision-Recall"
# W&B x-axis key (function evaluations).
FE_METRIC = "Function Evaluations"
# Final model file written into ``wandb.run.dir`` (auto-uploaded on finish).
CHECKPOINT_NAME = "checkpoint.pt.tar"


def make_run_name(dataset: str, algo: str, seed: int, name: str | None = None) -> str:
    """Format: mnist-nsga2-seed0 (or an explicit override)."""
    if name:
        return str(name)
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
        "xl": getattr(args, "xl", -10.0),
        "xu": getattr(args, "xu", 10.0),
        "steps": args.steps,
        "evals": getattr(args, "evals", None),
        "popsize": args.popsize,
        "batch_size": args.batch_size,
        "eval_mode": getattr(args, "eval_mode", "multi"),
        "eval_batches": getattr(args, "eval_batches", 50),
        "sampler": getattr(args, "sampler", "auto"),
        "val_every": args.val_every,
        "pareto_every": getattr(args, "pareto_every", 100),
        "train_history": getattr(args, "train_history", "train_history.npz"),
        "val_history": getattr(args, "val_history", "val_history.npz"),
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
        "wandb_x_axis": FE_METRIC,
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

    run_name = make_run_name(
        args.dataset, args.algo, args.seed, getattr(args, "wandb_name", None)
    )
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
    define_wandb_n_eval_metric()
    return run


def define_wandb_n_eval_metric(max_n_eval: int | None = None) -> None:
    """Register function evaluations as the global W&B x-axis."""
    if wandb.run is None:
        return
    wandb.define_metric(FE_METRIC)
    wandb.define_metric("*", step_metric=FE_METRIC)
    if max_n_eval is not None:
        wandb.run.config.update(
            {"wandb_max_function_evaluations": int(max_n_eval)}, allow_val_change=True
        )


def to_wandb_step(n_gen: int, max_steps: int | None = None) -> int:
    """Map pymoo ``n_gen`` to a 0-indexed opt step (0 = init, 1 = first opt).

    Kept for history / scheduling; W&B x-axis uses ``Function Evaluations``.
    """
    step = max(0, int(n_gen) - 1)
    if max_steps is not None:
        step = min(step, int(max_steps))
    return step


# Backward-compatible alias.
define_wandb_step_metric = define_wandb_n_eval_metric


def log_metrics(
    metrics: dict[str, Any],
    *,
    n_eval: int | None = None,
    step: int | None = None,
) -> None:
    """Log metrics against ``Function Evaluations``.

    ``step`` is accepted as a deprecated alias for ``n_eval``.
    """
    if wandb.run is None:
        return
    payload = dict(metrics)
    fe = n_eval if n_eval is not None else step
    if fe is not None:
        fe = int(fe)
        payload[FE_METRIC] = fe
        payload.setdefault("train/n_eval", fe)
        wandb.log(payload, step=fe)
    else:
        wandb.log(payload)


def save_wandb_checkpoint(
    problem,
    flat: np.ndarray,
    *,
    meta: dict[str, Any] | None = None,
) -> Path | None:
    """Save final weights into the active W&B run dir as ``checkpoint.pt.tar``.

    Files under ``wandb.run.dir`` are synced to the cloud when the run finishes.
    No-op (returns ``None``) when wandb is disabled / not initialized.
    """
    if wandb.run is None:
        return None

    flat = np.asarray(flat, dtype=np.float64).ravel()
    problem.set_weights(flat)
    payload: dict[str, Any] = {
        "model_state_dict": {
            k: v.detach().cpu().clone() for k, v in problem.model.state_dict().items()
        },
        "weights": flat,
    }
    if meta:
        payload.update(meta)

    path = Path(wandb.run.dir) / CHECKPOINT_NAME
    torch.save(payload, path)
    # Explicitly mark for upload (also covered by living under run.dir).
    wandb.save(str(path), base_path=str(path.parent), policy="now")
    return path


def finish_wandb(summary: dict[str, Any] | None = None) -> None:
    if wandb.run is None:
        return
    if summary:
        for key, value in summary.items():
            wandb.run.summary[key] = value
    wandb.finish()
