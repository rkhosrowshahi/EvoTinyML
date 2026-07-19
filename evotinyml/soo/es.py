"""Evosax single-objective runners (CMA-ES, SNES, xNES, OpenES, …)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from torch.utils.data import DataLoader

from evotinyml.fitness import SOOFitness
from evotinyml.problem import CE_PROBLEMS, WeightOptimizationProblem
from evotinyml.validation import (
    class_metric_fields,
    class_metric_log_dict,
    evaluate_weights_on_loader,
    per_class_acc_ce_on_eval_pool,
    per_class_acc_ce_on_loader,
)
from evotinyml.wandb_logger import log_metrics


def _maybe_ce_train_details(
    problem: WeightOptimizationProblem, flat: np.ndarray
) -> dict[str, float]:
    if getattr(problem, "problem_name", "") not in CE_PROBLEMS:
        return {}
    acc_c, ce_c = per_class_acc_ce_on_eval_pool(problem, flat)
    return class_metric_fields(acc_c, ce_c)

# CLI name -> (evosax class name, display name)
EVOSAX_SOO_ALGOS = {
    "cmaes": ("CMA_ES", "CMA-ES"),
    "snes": ("SNES", "SNES"),
    "xnes": ("xNES", "xNES"),
    "open_es": ("Open_ES", "OpenES"),
}

# OpenES mean-update optimizer (optax). Ignored for CMA / SNES / xNES.
ES_OPTIMS = ("sgd", "adam", "adamw")
ES_OPTIM_SCHEDULERS = ("constant", "cosine", "exponential")
DEFAULT_ES_OPTIM = "sgd"
DEFAULT_ES_OPTIM_LR = 1e-3
DEFAULT_ES_OPTIM_SCHEDULER = "constant"
DEFAULT_ES_OPTIM_MOMENTUM = 0.0
# OpenES / MO-OpenES sampling-noise (σ) schedule over steps.
ES_SIGMA_SCHEDULERS = ("constant", "cosine", "exponential")
DEFAULT_ES_SIGMA_SCHEDULER = "constant"


def default_es_popsize(n_var: int) -> int:
    """ES rule of thumb: ``4 + 3 * ln(n)`` (CMA / NES convention)."""
    return max(4, int(4 + 3 * math.log(max(n_var, 1))))


def default_soo_popsize(n_var: int, algo: str = "cmaes") -> int:
    """Default population size for evosax SOO algorithms."""
    pop = default_es_popsize(n_var)
    # OpenES antithetic sampling requires even population size.
    if algo == "open_es" and pop % 2 == 1:
        pop += 1
    return pop


def _init_mean(
    n_var: int,
    init: str,
    init_sigma: float,
    rng: np.random.Generator,
    theta0: np.ndarray | None = None,
) -> np.ndarray:
    """Draw ES mean with the same scheme as MOO population init."""
    sigma = float(init_sigma)
    name = init.lower()
    if name in {"zeros", "zero"}:
        return np.zeros(n_var, dtype=np.float64)
    if name in {"kaiming", "he", "theta0", "pytorch", "torch", "default"}:
        if theta0 is None:
            raise ValueError(
                "--init kaiming requires problem.theta0 (model default / Kaiming weights)."
            )
        theta0 = np.asarray(theta0, dtype=np.float64).ravel()
        if theta0.size != n_var:
            raise ValueError(f"theta0 length {theta0.size} != n_var {n_var}")
        return theta0.copy()
    if name == "uniform":
        return rng.uniform(-sigma, sigma, size=n_var).astype(np.float64)
    if name in {"both", "mixed"}:
        return rng.normal(0.0, sigma, size=n_var).astype(np.float64)
    return rng.normal(0.0, sigma, size=n_var).astype(np.float64)


def _state_mean(state, xl: float, xu: float) -> np.ndarray:
    """Distribution center (mean), clipped to box bounds."""
    return np.clip(np.asarray(state.mean, dtype=np.float64), xl, xu)


def _build_lr_schedule(scheduler: str, lr: float, steps: int):
    """Optax learning-rate schedule for the OpenES mean optimizer."""
    import optax

    name = scheduler.lower()
    lr = float(lr)
    decay_steps = max(int(steps), 1)
    if name == "constant":
        return optax.constant_schedule(lr)
    if name in {"cosine", "cosine_decay"}:
        return optax.cosine_decay_schedule(init_value=lr, decay_steps=decay_steps)
    if name in {"exponential", "exponential_decay"}:
        return optax.exponential_decay(
            init_value=lr,
            transition_steps=max(decay_steps // 10, 1),
            decay_rate=0.99,
            end_value=max(lr * 0.01, 1e-8),
        )
    raise ValueError(
        f"Unknown es_optim_scheduler: {scheduler!r}. Use one of {ES_OPTIM_SCHEDULERS}."
    )


def build_es_sigma_schedule(
    scheduler: str = DEFAULT_ES_SIGMA_SCHEDULER,
    sigma: float = 0.1,
    steps: int = 1,
    end: float | None = None,
):
    """Optax schedule for OpenES sampling std σ over optimization steps.

    ``end`` defaults to ``max(sigma * 0.01, 1e-6)`` for cosine / exponential
    (constant ignores ``end``). Cosine uses ``alpha = end / sigma``.
    """
    import optax

    name = scheduler.lower()
    sigma = float(sigma)
    decay_steps = max(int(steps), 1)
    if end is None:
        end_sigma = max(sigma * 0.01, 1e-6)
    else:
        end_sigma = max(float(end), 1e-12)
    if name == "constant":
        return optax.constant_schedule(sigma)
    if name in {"cosine", "cosine_decay"}:
        alpha = float(np.clip(end_sigma / max(sigma, 1e-12), 0.0, 1.0))
        return optax.cosine_decay_schedule(
            init_value=sigma, decay_steps=decay_steps, alpha=alpha
        )
    if name in {"exponential", "exponential_decay"}:
        return optax.exponential_decay(
            init_value=sigma,
            transition_steps=max(decay_steps // 10, 1),
            decay_rate=0.99,
            end_value=end_sigma,
        )
    raise ValueError(
        f"Unknown es_sigma_scheduler: {scheduler!r}. Use one of {ES_SIGMA_SCHEDULERS}."
    )


def sigma_at(schedule, step: int) -> float:
    """Evaluate an Optax σ schedule at an integer step index."""
    import jax.numpy as jnp

    return float(schedule(jnp.asarray(int(step))))


def build_open_es_optimizer(
    optim: str = DEFAULT_ES_OPTIM,
    lr: float = DEFAULT_ES_OPTIM_LR,
    scheduler: str = DEFAULT_ES_OPTIM_SCHEDULER,
    steps: int = 1,
    momentum: float = DEFAULT_ES_OPTIM_MOMENTUM,
):
    """Build the Optax transform used to update the OpenES mean."""
    import optax

    name = optim.lower()
    schedule = _build_lr_schedule(scheduler, lr, steps)
    mom = float(momentum)
    if name == "sgd":
        # optax: momentum=None disables the velocity buffer.
        return optax.sgd(
            learning_rate=schedule,
            momentum=None if mom <= 0.0 else mom,
        )
    if mom > 0.0:
        print(
            f"Note: --es-optim-momentum={mom} is ignored for --es-optim {name} "
            "(SGD only)."
        )
    if name == "adam":
        return optax.adam(learning_rate=schedule)
    if name == "adamw":
        return optax.adamw(learning_rate=schedule)
    raise ValueError(f"Unknown es_optim: {optim!r}. Use one of {ES_OPTIMS}.")


def _build_evosax(
    algo: str,
    population_size: int,
    n_var: int,
    init_sigma: float,
    *,
    steps: int = 1,
    es_optim: str = DEFAULT_ES_OPTIM,
    es_optim_lr: float = DEFAULT_ES_OPTIM_LR,
    es_optim_scheduler: str = DEFAULT_ES_OPTIM_SCHEDULER,
    es_optim_momentum: float = DEFAULT_ES_OPTIM_MOMENTUM,
    es_sigma_scheduler: str = DEFAULT_ES_SIGMA_SCHEDULER,
    es_sigma_end: float | None = None,
):
    """Construct an evosax ES and params; may bump popsize for OpenES antithetic."""
    from evosax import algorithms as evosax_algorithms

    if algo not in EVOSAX_SOO_ALGOS:
        raise ValueError(
            f"Unknown SOO algo: {algo!r}. Use one of {tuple(EVOSAX_SOO_ALGOS)}."
        )

    cls_name, display_name = EVOSAX_SOO_ALGOS[algo]
    Cls = getattr(evosax_algorithms, cls_name)
    sigma = float(init_sigma)

    if algo == "open_es":
        if population_size % 2 == 1:
            print(
                f"Note: OpenES antithetic sampling needs even popsize; "
                f"bumping {population_size} → {population_size + 1}."
            )
            population_size += 1
        optimizer = build_open_es_optimizer(
            es_optim,
            es_optim_lr,
            es_optim_scheduler,
            steps=steps,
            momentum=es_optim_momentum,
        )
        std_schedule = build_es_sigma_schedule(
            es_sigma_scheduler, sigma, steps=steps, end=es_sigma_end
        )
        es = Cls(
            population_size=population_size,
            solution=jnp.zeros(n_var),
            optimizer=optimizer,
            std_schedule=std_schedule,
        )
        params = es.default_params
    else:
        es = Cls(population_size=population_size, solution=jnp.zeros(n_var))
        params = es.default_params
        if hasattr(params, "std_init"):
            params = params.replace(std_init=sigma)

    return es, params, population_size, display_name


@dataclass
class SOOESResult:
    """Final SOO ES state for CLI / saving.

    ``X`` is the final distribution mean (center), not the best population member.
    """

    X: np.ndarray
    f: float
    mean_f_history: np.ndarray
    steps: int
    popsize: int
    fitness_name: str = "f"
    algo: str = "cmaes"
    details: dict[str, float] = field(default_factory=dict)


def run_soo_es(
    problem: WeightOptimizationProblem,
    *,
    algo: str = "cmaes",
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
    es_optim: str = DEFAULT_ES_OPTIM,
    es_optim_lr: float = DEFAULT_ES_OPTIM_LR,
    es_optim_scheduler: str = DEFAULT_ES_OPTIM_SCHEDULER,
    es_optim_momentum: float = DEFAULT_ES_OPTIM_MOMENTUM,
    es_sigma_scheduler: str = DEFAULT_ES_SIGMA_SCHEDULER,
    es_sigma_end: float | None = None,
) -> SOOESResult:
    """Ask / eval / tell loop for evosax SOO algorithms.

    Fitness is evaluated with torch (no JAX NN). evosax **minimizes** the
    fitness values passed to ``tell``. Train logging and test validation use
    the distribution **mean**, not the best sampled individual.

    Function Evaluations count as ``steps * popsize`` (population only).
    OpenES-only: ``es_optim`` / ``es_optim_lr`` / ``es_optim_scheduler`` /
    ``es_optim_momentum`` control the Optax mean update; ``es_sigma_scheduler``
    / ``es_sigma_end`` control the sampling-noise schedule (ignored for CMA /
    SNES / xNES).
    """
    algo = algo.lower()
    soo = getattr(problem, "soo_fitness", None)
    if not isinstance(soo, SOOFitness):
        raise TypeError(
            f"{algo} requires a problem with SOOFitness, got {type(problem).__name__}"
        )
    fitness_name = soo.fitness_name

    n_var = int(problem.n_var)
    if popsize is None:
        popsize = default_soo_popsize(n_var, algo)
    popsize = int(popsize)
    steps = int(steps)
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")

    es, params, popsize, display_name = _build_evosax(
        algo,
        popsize,
        n_var,
        init_sigma,
        steps=steps,
        es_optim=es_optim,
        es_optim_lr=es_optim_lr,
        es_optim_scheduler=es_optim_scheduler,
        es_optim_momentum=es_optim_momentum,
        es_sigma_scheduler=es_sigma_scheduler,
        es_sigma_end=es_sigma_end,
    )
    sigma_schedule = (
        build_es_sigma_schedule(
            es_sigma_scheduler, init_sigma, steps=steps, end=es_sigma_end
        )
        if algo == "open_es"
        else None
    )

    rng = np.random.default_rng(seed)
    mean0 = _init_mean(
        n_var,
        init,
        init_sigma,
        rng,
        theta0=getattr(problem, "theta0", None),
    )

    key = jax.random.key(int(seed))
    key, key_init = jax.random.split(key)
    state = es.init(key_init, jnp.asarray(mean0, dtype=jnp.float32), params)

    xl = float(np.asarray(problem.xl).reshape(-1)[0]) if problem.xl is not None else -1.0
    xu = float(np.asarray(problem.xu).reshape(-1)[0]) if problem.xu is not None else 1.0

    mean_f_history = np.empty(steps + 1, dtype=np.float64)
    n_eval = 0

    mean_x = _state_mean(state, xl, xu)
    mean_details = soo.evaluate_one(mean_x)
    mean_details.update(_maybe_ce_train_details(problem, mean_x))
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
        es_sigma=float(init_sigma) if algo == "open_es" else None,
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
        n_eval = gen * popsize
        state, _es_metrics = es.tell(
            key_tell, jnp.asarray(X, dtype=jnp.float32), jnp.asarray(fitness), state, params
        )

        mean_x = _state_mean(state, xl, xu)
        mean_details = soo.evaluate_one(mean_x)
        mean_details.update(_maybe_ce_train_details(problem, mean_x))
        mean_f = float(mean_details["f"])
        mean_f_history[gen] = mean_f

        pop_best_f = float(np.min(fitness))
        cur_sigma = (
            sigma_at(sigma_schedule, gen - 1) if sigma_schedule is not None else None
        )
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
            es_sigma=cur_sigma,
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

    return SOOESResult(
        X=mean_x,
        f=mean_f,
        mean_f_history=mean_f_history,
        steps=steps,
        popsize=popsize,
        fitness_name=fitness_name,
        algo=algo,
        details=mean_details,
    )


def _format_details(details: dict[str, float]) -> str:
    parts = []
    for key, value in details.items():
        if key == "f" or key.startswith("class_"):
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
    es_sigma: float | None = None,
) -> None:
    extra = _format_details(details)
    if verbose:
        msg = f"[{label}] n_eval={n_eval}  mean_f={f:.6f}"
        if extra:
            msg += f"  {extra}"
        if pop_best_f is not None:
            msg += f"  pop_best_f={pop_best_f:.6f}"
        if es_sigma is not None:
            msg += f"  es_sigma={es_sigma:.6f}"
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
        if es_sigma is not None:
            payload["train/es_sigma"] = float(es_sigma)
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
    """Validate the ES distribution mean (center) on the test set."""
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
    if getattr(problem, "problem_name", "") in CE_PROBLEMS:
        acc_c, ce_c = per_class_acc_ce_on_loader(
            problem, mean_x, test_loader, n_classes
        )
        metrics.update(class_metric_log_dict("val", acc_c, ce_c))
    if verbose:
        print(
            f"[val n_eval={n_eval} step={opt_step}] (ES mean) "
            f"acc={acc:.4f}  f1={f1:.4f}  P={prec:.4f}  R={rec:.4f}"
        )
    if use_wandb:
        log_metrics(metrics, n_eval=n_eval)
    return metrics


def build_soo_wandb_config(
    args: Any,
    *,
    n_var: int,
    popsize: int,
    fitness_name: str,
) -> dict[str, Any]:
    """Minimal W&B config for evosax SOO runs."""
    algo = getattr(args, "algo", "cmaes")
    display = EVOSAX_SOO_ALGOS.get(algo, (algo, algo))[1]
    es_optim = getattr(args, "es_optim", DEFAULT_ES_OPTIM)
    es_optim_lr = getattr(args, "es_optim_lr", DEFAULT_ES_OPTIM_LR)
    es_optim_scheduler = getattr(args, "es_optim_scheduler", DEFAULT_ES_OPTIM_SCHEDULER)
    es_optim_momentum = getattr(args, "es_optim_momentum", DEFAULT_ES_OPTIM_MOMENTUM)
    es_sigma_scheduler = getattr(args, "es_sigma_scheduler", DEFAULT_ES_SIGMA_SCHEDULER)
    es_sigma_end = getattr(args, "es_sigma_end", None)
    return {
        "dataset": args.dataset,
        "problem": args.problem,
        "activation": args.activation,
        "algo": algo,
        "fitness": fitness_name,
        "init": args.init,
        "init_sigma": getattr(args, "init_sigma", 0.1),
        "es_optim": es_optim,
        "es_optim_lr": es_optim_lr,
        "es_optim_scheduler": es_optim_scheduler,
        "es_optim_momentum": es_optim_momentum,
        "es_sigma_scheduler": es_sigma_scheduler,
        "es_sigma_end": es_sigma_end,
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
        "val_solution": "es_mean",
        "wandb_x_axis": "Function Evaluations",
        "algo_config": {
            "name": algo,
            "display_name": display,
            "popsize": popsize,
            "std_init": getattr(args, "init_sigma", 0.1),
            "es_optim": es_optim,
            "es_optim_lr": es_optim_lr,
            "es_optim_scheduler": es_optim_scheduler,
            "es_optim_momentum": es_optim_momentum,
            "es_sigma_scheduler": es_sigma_scheduler,
            "es_sigma_end": es_sigma_end,
            "library": "evosax",
            "val_solution": "es_mean",
        },
    }
