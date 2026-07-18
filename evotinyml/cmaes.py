"""CMA-ES (evosax) single-objective runner."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from torch.utils.data import DataLoader

from evotinyml.fitness import SOOFitness
from evotinyml.problem import WeightOptimizationProblem
from evotinyml.validation import evaluate_weights_on_loader
from evotinyml.wandb_logger import log_metrics


def default_cma_popsize(n_var: int) -> int:
    """CMA rule of thumb: ``4 + 3 * ln(n)``."""
    return max(4, int(4 + 3 * math.log(max(n_var, 1))))


def _init_mean(n_var: int, init: str, init_sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Draw CMA mean with the same scheme as MOO population init."""
    sigma = float(init_sigma)
    name = init.lower()
    if name in {"zeros", "zero"}:
        return np.zeros(n_var, dtype=np.float64)
    if name == "uniform":
        return rng.uniform(-sigma, sigma, size=n_var).astype(np.float64)
    if name in {"both", "mixed"}:
        # Single mean: prefer Gaussian half of the mixed scheme.
        return rng.normal(0.0, sigma, size=n_var).astype(np.float64)
    # gaussian (default for CMA)
    return rng.normal(0.0, sigma, size=n_var).astype(np.float64)


def _state_mean(state, xl: float, xu: float) -> np.ndarray:
    """CMA distribution center (mean), clipped to box bounds."""
    return np.clip(np.asarray(state.mean, dtype=np.float64), xl, xu)


@dataclass
class CMAESResult:
    """Final CMA-ES state for CLI / saving.

    ``X`` is the final CMA mean (center), not the best population member.
    """

    X: np.ndarray  # mean weights (n_var,)
    f: float
    mean_f_history: np.ndarray
    steps: int
    popsize: int
    fitness_name: str = "f"
    details: dict[str, float] = field(default_factory=dict)


def run_soo_cma(
    problem: WeightOptimizationProblem,
    *,
    steps: int,
    popsize: int | None = None,
    init: str = "gaussian",
    init_sigma: float = 0.1,
    seed: int = 1,
    resample_every: int = 0,
    val_every: int = 50,
    test_loader: DataLoader | None = None,
    n_classes: int = 10,
    verbose: bool = False,
    use_wandb: bool = True,
) -> CMAESResult:
    """Ask / eval / tell CMA-ES loop minimizing a scalar SOO fitness.

    Fitness is evaluated with torch (no JAX NN). evosax CMA-ES **minimizes**
    the fitness values passed to ``tell``, so the scalar objective is passed
    through directly (no negation).

    Train logging and test validation use the CMA **mean** (distribution
    center), not the best sampled individual.
    """
    from evosax.algorithms import CMA_ES

    soo = getattr(problem, "soo_fitness", None)
    if not isinstance(soo, SOOFitness):
        raise TypeError(
            f"CMA-ES requires a problem with SOOFitness, got {type(problem).__name__}"
        )
    fitness_name = soo.fitness_name

    n_var = int(problem.n_var)
    if popsize is None:
        popsize = default_cma_popsize(n_var)
    popsize = int(popsize)
    steps = int(steps)
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")

    rng = np.random.default_rng(seed)
    mean0 = _init_mean(n_var, init, init_sigma, rng)

    es = CMA_ES(population_size=popsize, solution=jnp.zeros(n_var))
    params = es.default_params.replace(std_init=float(init_sigma))
    key = jax.random.key(int(seed))
    key, key_init = jax.random.split(key)
    state = es.init(key_init, jnp.asarray(mean0, dtype=jnp.float32), params)

    xl = float(np.asarray(problem.xl).reshape(-1)[0]) if problem.xl is not None else -1.0
    xu = float(np.asarray(problem.xu).reshape(-1)[0]) if problem.xu is not None else 1.0

    mean_f_history = np.empty(steps + 1, dtype=np.float64)
    # FE budget: each CMA generation costs ``popsize`` train fitness calls
    # (mean logging / test val do not count toward the search budget).
    n_eval = 0

    # Step 0: evaluate initial mean for logging only (0 FEs spent on search yet).
    mean_x = _state_mean(state, xl, xu)
    mean_details = soo.evaluate_one(mean_x)
    mean_f = float(mean_details["f"])
    mean_f_history[0] = mean_f
    _log_soo_step(
        step=0,
        n_eval=n_eval,
        f=mean_f,
        details=mean_details,
        fitness_name=fitness_name,
        use_wandb=use_wandb,
        verbose=verbose,
        label="init",
    )
    if test_loader is not None and val_every > 0:
        _maybe_validate_mean(
            problem,
            mean_x,
            test_loader,
            n_classes=n_classes,
            n_eval=n_eval,
            opt_step=0,
            use_wandb=use_wandb,
            verbose=verbose,
            force=True,
        )

    for gen in range(1, steps + 1):
        key, key_ask, key_tell = jax.random.split(key, 3)
        population, state = es.ask(key_ask, state, params)
        X = np.asarray(population, dtype=np.float64)
        X = np.clip(X, xl, xu)

        fitness = soo.evaluate(X, details=False)
        n_eval = gen * popsize  # exactly λ FEs per generation
        # evosax minimizes; pass objective directly.
        state, _es_metrics = es.tell(
            key_tell, jnp.asarray(X, dtype=jnp.float32), jnp.asarray(fitness), state, params
        )

        # Report / validate the CMA center after the update (not counted as FE).
        mean_x = _state_mean(state, xl, xu)
        mean_details = soo.evaluate_one(mean_x)
        mean_f = float(mean_details["f"])
        mean_f_history[gen] = mean_f

        pop_best_f = float(np.min(fitness))
        _log_soo_step(
            step=gen,
            n_eval=n_eval,
            f=mean_f,
            details=mean_details,
            fitness_name=fitness_name,
            use_wandb=use_wandb,
            verbose=verbose,
            label=f"step {gen}",
            pop_best_f=pop_best_f,
        )

        if test_loader is not None and val_every > 0:
            _maybe_validate_mean(
                problem,
                mean_x,
                test_loader,
                n_classes=n_classes,
                n_eval=n_eval,
                opt_step=gen,
                use_wandb=use_wandb,
                verbose=verbose,
                force=(gen == 1) or (gen % val_every == 0),
            )

        if resample_every > 0 and gen % resample_every == 0 and gen < steps:
            problem.resample_batch()
            n_batches = len(getattr(problem, "eval_batch_pool", []) or [])
            print(
                f"[step {gen}] resampled eval pool "
                f"(batches={n_batches}, mode={getattr(problem, 'eval_mode', '?')}, "
                f"batch_version={problem.batch_version})"
            )

    return CMAESResult(
        X=mean_x,
        f=mean_f,
        mean_f_history=mean_f_history,
        steps=steps,
        popsize=popsize,
        fitness_name=fitness_name,
        details=mean_details,
    )


def _format_details(details: dict[str, float]) -> str:
    parts = []
    for key, value in details.items():
        if key == "f":
            continue
        parts.append(f"{key}={value:.6f}")
    return "  ".join(parts)


def _log_soo_step(
    *,
    step: int,
    n_eval: int,
    f: float,
    details: dict[str, float],
    fitness_name: str,
    use_wandb: bool,
    verbose: bool,
    label: str,
    pop_best_f: float | None = None,
) -> None:
    extra = _format_details(details)
    if verbose:
        msg = f"[{label}] n_eval={n_eval}  mean_f={f:.6f}"
        if extra:
            msg += f"  {extra}"
        if pop_best_f is not None:
            msg += f"  pop_best_f={pop_best_f:.6f}"
        print(msg)
    if use_wandb:
        payload: dict[str, Any] = {
            "train/step": step,
            "train/f": f,
            "train/mean_f": f,
            "train/fitness_name": fitness_name,
        }
        if pop_best_f is not None:
            payload["train/pop_best_f"] = pop_best_f
        for key, value in details.items():
            if key == "f":
                continue
            payload[f"train/{key}"] = value
        log_metrics(payload, n_eval=n_eval)


def _maybe_validate_mean(
    problem: WeightOptimizationProblem,
    mean_x: np.ndarray,
    test_loader: DataLoader,
    *,
    n_classes: int,
    n_eval: int,
    opt_step: int,
    use_wandb: bool,
    verbose: bool,
    force: bool,
) -> dict[str, float] | None:
    """Validate the CMA mean (center) on the test set."""
    if not force:
        return None
    acc, f1, prec, rec, _, _ = evaluate_weights_on_loader(
        problem, mean_x, test_loader, n_classes
    )
    metrics = {
        "train/step": int(opt_step),
        "val/acc": float(acc),
        "val/f1": float(f1),
        "val/precision": float(prec),
        "val/recall": float(rec),
        # Aliases matching NSGA report names (knee / best) for W&B overlays.
        "val/acc_best": float(acc),
        "val/f1_best": float(f1),
        "val/knee_acc": float(acc),
        "val/knee_f1": float(f1),
        "val/knee_precision": float(prec),
        "val/knee_recall": float(rec),
        "val/knee_pr_mean": float(0.5 * (prec + rec)),
    }
    if verbose:
        print(
            f"[val n_eval={n_eval} step={opt_step}] (CMA mean) "
            f"acc={acc:.4f}  f1={f1:.4f}  P={prec:.4f}  R={rec:.4f}"
        )
    if use_wandb:
        log_metrics(metrics, n_eval=n_eval)
    return metrics


def build_cma_wandb_config(
    args: Any,
    *,
    n_var: int,
    popsize: int,
    fitness_name: str,
) -> dict[str, Any]:
    """Minimal W&B config for CMA-ES SOO runs."""
    return {
        "dataset": args.dataset,
        "problem": args.problem,
        "activation": args.activation,
        "algo": "cmaes",
        "fitness": fitness_name,
        "init": args.init,
        "init_sigma": getattr(args, "init_sigma", 0.1),
        "steps": args.steps,
        "evals": getattr(args, "evals", None),
        "popsize": popsize,
        "batch_size": args.batch_size,
        "eval_mode": getattr(args, "eval_mode", "single"),
        "eval_batches": getattr(args, "eval_batches", 8),
        "resample_every": args.resample_every,
        "val_every": args.val_every,
        "val_batch_size": args.val_batch_size,
        "seed": args.seed,
        "device": args.device,
        "n_var": n_var,
        "n_obj": 1,
        "val_solution": "cma_mean",
        "wandb_x_axis": "Function Evaluations",
        "algo_config": {
            "name": "cmaes",
            "popsize": popsize,
            "std_init": getattr(args, "init_sigma", 0.1),
            "library": "evosax",
            "val_solution": "cma_mean",
        },
    }
