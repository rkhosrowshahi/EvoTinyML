"""Evosax single-objective runners (CMA-ES, SNES, xNES, OpenES, ASEBO, …)."""

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
    "sparse_open_es": ("SparseOpenES", "SparseOpenES"),
    "ada_open_es": ("AdaOpenES", "AdaOpenES"),
    "cr_fm_nes": ("CR_FM_NES", "CR-FM-NES"),
    "asebo": ("ASEBO", "ASEBO"),
    "lm_ma_es": ("LM_MA_ES", "LM-MA-ES"),
    "de": ("DifferentialEvolution", "DE"),
    "jde": ("JDE", "jDE"),
    "pso": ("PSO", "PSO"),
    "soft_m_pso": ("SoftMomentumPSO", "SoftMPSO"),
    "1m_pso": ("FirstMomentumPSO", "1MPSO"),
    "2m_pso": ("SecondMomentumPSO", "2MPSO"),
}

# Algos that require an even population size (antithetic / paired sampling).
EVEN_POPSIZE_ALGOS = frozenset(
    {"open_es", "sparse_open_es", "ada_open_es", "cr_fm_nes", "asebo"}
)
# Mean update via Optax (--es-optim*). Ignored for CMA / CR-FM-NES / LM-MA-ES / DE / jDE / PSO.
MEAN_OPTIMIZER_ALGOS = frozenset(
    {"open_es", "sparse_open_es", "ada_open_es", "snes", "xnes", "asebo"}
)
# Sampling-noise σ schedule (--es-sigma*).
SIGMA_SCHEDULE_ALGOS = frozenset({"open_es", "sparse_open_es", "asebo"})
# Self-adapted per-parameter σ (no schedule; --es-sigma-lr/-min/-max instead).
ADAPTIVE_SIGMA_ALGOS = frozenset({"ada_open_es"})
# Population-based (init with evaluated population; report best member).
POPULATION_BASED_ALGOS = frozenset(
    {"de", "jde", "pso", "soft_m_pso", "1m_pso", "2m_pso"}
)
# Algos that accept --pso-* swarm coefficients / velocity clamp.
PSO_ALGOS = frozenset({"pso", "soft_m_pso", "1m_pso", "2m_pso"})
# Soft-replacement momentum PSO variants (co-eval [x; p; g]).
MOM_PSO_ALGOS = frozenset({"soft_m_pso", "1m_pso", "2m_pso"})

# Momentum (--es-optim-momentum) applies to sgd and rmsprop only.
ES_OPTIMS = ("sgd", "adam", "adamw", "rmsprop")
ES_OPTIMS_WITH_MOMENTUM = frozenset({"sgd", "rmsprop"})
ES_OPTIM_SCHEDULERS = ("constant", "cosine", "exponential")
DEFAULT_ES_OPTIM = "adam"
DEFAULT_ES_OPTIM_LR = 0.01
DEFAULT_ES_OPTIM_SCHEDULER = "constant"
DEFAULT_ES_OPTIM_MOMENTUM = 0.9
# Weight decay for mean Optax update (0 = off). adamw uses Optax decoupled WD;
# sgd / adam / rmsprop chain optax.add_decayed_weights when wd > 0.
DEFAULT_ES_OPTIM_WD = 0.0
# Exponential LR floor (staircase ×decay_rate every data-epoch).
DEFAULT_ES_OPTIM_EXP_END = 1e-6
# OpenES / SparseOpenES / ASEBO / MO-OpenES sampling-noise (σ) schedule over steps.
ES_SIGMA_SCHEDULERS = ("constant", "cosine", "exponential")
DEFAULT_ES_SIGMA_SCHEDULER = "constant"
# Exponential σ / LR: staircase ×decay_rate every data-epoch (D // batch_size steps).
DEFAULT_ES_SIGMA_DECAY_RATE = 0.9
DEFAULT_ES_EPOCH_DECAY_RATE = DEFAULT_ES_SIGMA_DECAY_RATE
DEFAULT_ES_SIGMA_EXP_END = 1e-6
# AdaOpenES: --es-sigma-lr default is SNES's dimension-scaled rate
# (3 + ln d) / (5 sqrt d); --es-sigma-min/-max default to init_sigma × these ratios.
DEFAULT_ES_SIGMA_MIN_RATIO = 0.01
DEFAULT_ES_SIGMA_MAX_RATIO = 100.0
DEFAULT_ASEBO_SUBSPACE_DIMS = 1
# SparseOpenES: fraction of isotropic-noise dims zeroed before antithetic sampling.
DEFAULT_SPARSE_ES_MASK_PROB = 0.2
# Differential Evolution (evosax defaults).
DEFAULT_DE_F = 0.8
DEFAULT_DE_CR = 0.9
DEFAULT_DE_ELITISM = True
# jDE (Brest et al., 2006) self-adaptation defaults.
DEFAULT_JDE_F_INIT = 0.5
DEFAULT_JDE_CR_INIT = 0.9
DEFAULT_JDE_F_L = 0.1
DEFAULT_JDE_F_U = 0.9
DEFAULT_JDE_TAU_F = 0.1
DEFAULT_JDE_TAU_CR = 0.1
DEFAULT_JDE_ELITISM = False  # paper uses DE/rand/1/bin
# PSO (evosax defaults; uses FixedPSO from PR #109 seeding fix).
DEFAULT_PSO_INERTIA = 0.75
DEFAULT_PSO_COGNITIVE = 1.5
DEFAULT_PSO_SOCIAL = 2.0
# Clamp each velocity component to [-v_max, v_max].
DEFAULT_PSO_MAX_VELOCITY = 0.8
# When True, ask returns [offspring; personal-best archive] (2×popsize FEs/gen).
DEFAULT_EA_COEVAL = False
# Soft-replacement MomPSO anchor LRs. None → use each algo's class defaults
# (1m_pso: 0.3/0.1; 2m_pso: 1e-3/1e-3). Ignored by soft_m_pso (p ← p + m).
DEFAULT_MOM_PSO_ETA_PERSONAL = None
DEFAULT_MOM_PSO_ETA_GLOBAL = None
DEFAULT_MOM_PSO_BETA1 = 0.9
DEFAULT_MOM_PSO_BETA2 = 0.999
DEFAULT_MOM_PSO_GATE_TEMPERATURE = 0.75
DEFAULT_MOM_PSO_GATE_EMA_DECAY = 0.9
DEFAULT_MOM_PSO_GLOBAL_TOPK_FRACTION = 0.2


def default_es_popsize(n_var: int) -> int:
    """ES rule of thumb: ``4 + 3 * ln(n)`` (CMA / NES convention)."""
    return max(4, int(4 + 3 * math.log(max(n_var, 1))))


def default_soo_popsize(n_var: int, algo: str = "cmaes") -> int:
    """Default population size for evosax SOO algorithms."""
    pop = default_es_popsize(n_var)
    # OpenES / CR-FM-NES / ASEBO need even population size.
    if algo in EVEN_POPSIZE_ALGOS and pop % 2 == 1:
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


def _init_population(
    n_var: int,
    popsize: int,
    init: str,
    init_sigma: float,
    rng: np.random.Generator,
    theta0: np.ndarray | None = None,
) -> np.ndarray:
    """Draw an initial DE population (needs diversity across members)."""
    sigma = float(init_sigma)
    name = init.lower()
    popsize = int(popsize)
    if name in {"zeros", "zero"}:
        center = np.zeros(n_var, dtype=np.float64)
        pop = center + rng.normal(0.0, sigma, size=(popsize, n_var))
        pop[0] = center
        return pop.astype(np.float64)
    if name in {"kaiming", "he", "theta0", "pytorch", "torch", "default"}:
        if theta0 is None:
            raise ValueError(
                "--init kaiming requires problem.theta0 (model default / Kaiming weights)."
            )
        center = np.asarray(theta0, dtype=np.float64).ravel()
        if center.size != n_var:
            raise ValueError(f"theta0 length {center.size} != n_var {n_var}")
        pop = center + rng.normal(0.0, sigma, size=(popsize, n_var))
        pop[0] = center
        return pop.astype(np.float64)
    if name == "uniform":
        return rng.uniform(-sigma, sigma, size=(popsize, n_var)).astype(np.float64)
    return rng.normal(0.0, sigma, size=(popsize, n_var)).astype(np.float64)


def _state_best(state, xl: float, xu: float) -> np.ndarray:
    """Best population member for population-based algos (DE), clipped."""
    fitness = np.asarray(state.fitness, dtype=np.float64)
    population = np.asarray(state.population, dtype=np.float64)
    best = np.asarray(state.best_solution, dtype=np.float64)
    if np.isfinite(best).all() and np.isfinite(float(state.best_fitness)):
        return np.clip(best, xl, xu)
    idx = int(np.argmin(fitness))
    return np.clip(population[idx], xl, xu)


def _state_solution(algo: str, state, xl: float, xu: float) -> np.ndarray:
    """Solution used for logging / validation / saving."""
    if algo in POPULATION_BASED_ALGOS:
        return _state_best(state, xl, xu)
    return _state_mean(state, xl, xu)


def _jde_control_means(state) -> dict[str, float]:
    """Mean self-adapted F / CR across the jDE population."""
    f = np.asarray(state.differential_weights, dtype=np.float64)
    cr = np.asarray(state.crossover_rates, dtype=np.float64)
    return {
        "jde_mean_f": float(np.mean(f)),
        "jde_mean_cr": float(np.mean(cr)),
        "jde_std_f": float(np.std(f)),
        "jde_std_cr": float(np.std(cr)),
    }


def _ada_sigma_stats(state) -> dict[str, float]:
    """Summary of AdaOpenES's self-adapted per-parameter σ vector."""
    std = np.asarray(state.std, dtype=np.float64)
    return {
        "sigma_mean": float(np.mean(std)),
        "sigma_min": float(np.min(std)),
        "sigma_max": float(np.max(std)),
        "sigma_std": float(np.std(std)),
    }


def _current_sigma_scalar(
    algo: str,
    state,
    sigma_schedule,
    gen: int,
) -> float | None:
    """Representative scalar σ for verbose/W&B logging (schedule or adapted mean)."""
    if algo in ADAPTIVE_SIGMA_ALGOS:
        return float(np.mean(np.asarray(state.std, dtype=np.float64)))
    if sigma_schedule is not None:
        return sigma_at(sigma_schedule, gen)
    return None


def _build_lr_schedule(
    scheduler: str,
    lr: float,
    steps: int,
    steps_per_epoch: int | None = None,
    decay_rate: float = DEFAULT_ES_EPOCH_DECAY_RATE,
    end: float | None = None,
):
    """Optax learning-rate schedule for the OpenES mean optimizer.

    ``exponential`` matches σ: staircase ×``decay_rate`` (default 0.9) every
    data-epoch (``steps_per_epoch = n_train // batch_size``), floored at
    ``end`` (default ``1e-6``).
    """
    import optax

    name = scheduler.lower()
    lr = float(lr)
    decay_steps = max(int(steps), 1)
    if name == "constant":
        return optax.constant_schedule(lr)
    if name in {"cosine", "cosine_decay"}:
        return optax.cosine_decay_schedule(init_value=lr, decay_steps=decay_steps)
    if name in {"exponential", "exponential_decay"}:
        if steps_per_epoch is None:
            raise ValueError(
                "exponential es_optim_scheduler requires steps_per_epoch "
                "(n_train // batch_size)."
            )
        epoch_steps = max(int(steps_per_epoch), 1)
        rate = float(decay_rate)
        if not (0.0 < rate < 1.0):
            raise ValueError(
                f"es_optim decay_rate must be in (0, 1), got {rate}"
            )
        end_lr = (
            DEFAULT_ES_OPTIM_EXP_END if end is None else max(float(end), 1e-12)
        )
        return optax.exponential_decay(
            init_value=lr,
            transition_steps=epoch_steps,
            decay_rate=rate,
            staircase=True,
            end_value=end_lr,
        )
    raise ValueError(
        f"Unknown es_optim_scheduler: {scheduler!r}. Use one of {ES_OPTIM_SCHEDULERS}."
    )


def steps_per_data_epoch(n_train: int, batch_size: int) -> int:
    """ES gens per data-epoch: ``max(1, n_train // batch_size)``."""
    return max(1, int(n_train) // max(int(batch_size), 1))


def steps_per_data_epoch_from_problem(problem: Any) -> int:
    """Infer data-epoch length from the problem's eval batch sampler."""
    sampler = getattr(problem, "batch_sampler", None)
    if sampler is None:
        raise ValueError("problem has no batch_sampler; cannot infer steps_per_epoch")
    return steps_per_data_epoch(len(sampler.dataset), int(sampler.batch_size))


def build_es_sigma_schedule(
    scheduler: str = DEFAULT_ES_SIGMA_SCHEDULER,
    sigma: float = 0.1,
    steps: int = 1,
    end: float | None = None,
    steps_per_epoch: int | None = None,
    decay_rate: float = DEFAULT_ES_SIGMA_DECAY_RATE,
):
    """Optax schedule for OpenES sampling std σ over optimization steps.

    * ``constant`` — ignore ``end`` / ``steps_per_epoch``.
    * ``cosine`` — anneal to ``end`` (default ``max(sigma * 0.01, 1e-6)``)
      over ``steps``.
    * ``exponential`` — staircase: hold σ for one data-epoch
      (``steps_per_epoch = n_train // batch_size`` gens), then multiply by
      ``decay_rate`` (default 0.9). Floor at ``end`` (default ``1e-6``).
    """
    import optax

    name = scheduler.lower()
    sigma = float(sigma)
    decay_steps = max(int(steps), 1)
    if name == "constant":
        return optax.constant_schedule(sigma)
    if name in {"cosine", "cosine_decay"}:
        if end is None:
            end_sigma = max(sigma * 0.01, 1e-6)
        else:
            end_sigma = max(float(end), 1e-12)
        alpha = float(np.clip(end_sigma / max(sigma, 1e-12), 0.0, 1.0))
        return optax.cosine_decay_schedule(
            init_value=sigma, decay_steps=decay_steps, alpha=alpha
        )
    if name in {"exponential", "exponential_decay"}:
        if end is None:
            end_sigma = DEFAULT_ES_SIGMA_EXP_END
        else:
            end_sigma = max(float(end), 1e-12)
        if steps_per_epoch is None:
            raise ValueError(
                "exponential es_sigma_scheduler requires steps_per_epoch "
                "(n_train // batch_size)."
            )
        epoch_steps = max(int(steps_per_epoch), 1)
        rate = float(decay_rate)
        if not (0.0 < rate < 1.0):
            raise ValueError(
                f"es_sigma decay_rate must be in (0, 1), got {rate}"
            )
        return optax.exponential_decay(
            init_value=sigma,
            transition_steps=epoch_steps,
            decay_rate=rate,
            staircase=True,
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
    weight_decay: float = DEFAULT_ES_OPTIM_WD,
    steps_per_epoch: int | None = None,
):
    """Build the Optax transform used to update the OpenES mean."""
    import optax

    name = optim.lower()
    schedule = _build_lr_schedule(
        scheduler, lr, steps, steps_per_epoch=steps_per_epoch
    )
    mom = float(momentum)
    wd = float(weight_decay)
    # optax: momentum=None disables the velocity buffer (sgd / rmsprop).
    mom_arg = None if mom <= 0.0 else mom
    if name == "sgd":
        opt = optax.sgd(learning_rate=schedule, momentum=mom_arg)
    elif name == "rmsprop":
        opt = optax.rmsprop(learning_rate=schedule, momentum=mom_arg)
    else:
        if mom > 0.0:
            print(
                f"Note: --es-optim-momentum={mom} is ignored for --es-optim {name} "
                f"(only used with {' / '.join(sorted(ES_OPTIMS_WITH_MOMENTUM))})."
            )
        if name == "adam":
            opt = optax.adam(learning_rate=schedule)
        elif name == "adamw":
            # Decoupled WD is built into adamw (including wd == 0).
            return optax.adamw(learning_rate=schedule, weight_decay=wd)
        else:
            raise ValueError(f"Unknown es_optim: {optim!r}. Use one of {ES_OPTIMS}.")
    if wd > 0.0:
        opt = optax.chain(optax.add_decayed_weights(wd), opt)
    return opt


def _fix_lm_ma_es_c_c(es, params, population_size: int, n_var: int):
    """Work around evosax LM-MA-ES int32 overflow in ``c_c``.

    Upstream computes ``population_size / 4**arange(m) / n_var`` with integer
    ``4**k``, which overflows for ``k >= 16`` and yields ``inf`` / NaNs once
    those memory rows activate (~step 18). Recompute with float powers.
    """
    m = int(es.m)
    indices = jnp.arange(m, dtype=jnp.float32)
    c_c = float(population_size) / jnp.power(4.0, indices) / float(n_var)
    return params.replace(c_c=c_c)


def _seed_archive_from_init(state, population: np.ndarray, fitness: np.ndarray):
    """Seed archive best from the evaluated initial population (evosax PR #109).

    Upstream ``PopulationBasedAlgorithm.init`` left ``best_solution`` /
    ``best_fitness`` unset until the first ``tell``. Apply the same seeding
    here for algos that still use stock evosax init (e.g. DE).
    """
    if np.isfinite(float(state.best_fitness)) and np.isfinite(
        np.asarray(state.best_solution, dtype=np.float64)
    ).all():
        return state
    best_idx = int(np.argmin(fitness))
    return state.replace(
        best_solution=jnp.asarray(population[best_idx], dtype=jnp.float32),
        best_fitness=jnp.asarray(float(fitness[best_idx]), dtype=jnp.float32),
    )



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
    es_optim_wd: float = DEFAULT_ES_OPTIM_WD,
    es_optim_steps_per_epoch: int | None = None,
    es_sigma_scheduler: str = DEFAULT_ES_SIGMA_SCHEDULER,
    es_sigma_end: float | None = None,
    es_sigma_steps_per_epoch: int | None = None,
    es_sigma_lr: float | None = None,
    es_sigma_min: float | None = None,
    es_sigma_max: float | None = None,
    asebo_subspace_dims: int = DEFAULT_ASEBO_SUBSPACE_DIMS,
    sparse_es_mask_prob: float = DEFAULT_SPARSE_ES_MASK_PROB,
    de_f: float = DEFAULT_DE_F,
    de_cr: float = DEFAULT_DE_CR,
    de_elitism: bool = DEFAULT_DE_ELITISM,
    jde_f_init: float = DEFAULT_JDE_F_INIT,
    jde_cr_init: float = DEFAULT_JDE_CR_INIT,
    jde_f_l: float = DEFAULT_JDE_F_L,
    jde_f_u: float = DEFAULT_JDE_F_U,
    jde_tau_f: float = DEFAULT_JDE_TAU_F,
    jde_tau_cr: float = DEFAULT_JDE_TAU_CR,
    jde_elitism: bool = DEFAULT_JDE_ELITISM,
    pso_inertia: float = DEFAULT_PSO_INERTIA,
    pso_cognitive: float = DEFAULT_PSO_COGNITIVE,
    pso_social: float = DEFAULT_PSO_SOCIAL,
    pso_max_velocity: float = DEFAULT_PSO_MAX_VELOCITY,
    ea_coeval: bool = DEFAULT_EA_COEVAL,
    mom_pso_eta_personal: float | None = DEFAULT_MOM_PSO_ETA_PERSONAL,
    mom_pso_eta_global: float | None = DEFAULT_MOM_PSO_ETA_GLOBAL,
    mom_pso_beta1: float = DEFAULT_MOM_PSO_BETA1,
    mom_pso_beta2: float = DEFAULT_MOM_PSO_BETA2,
    mom_pso_gate_temperature: float = DEFAULT_MOM_PSO_GATE_TEMPERATURE,
    mom_pso_gate_ema_decay: float = DEFAULT_MOM_PSO_GATE_EMA_DECAY,
    mom_pso_global_topk_fraction: float = DEFAULT_MOM_PSO_GLOBAL_TOPK_FRACTION,
):
    """Construct an evosax ES and params; may bump popsize for even-pop algos."""
    from evosax import algorithms as evosax_algorithms

    from evotinyml.soo.asebo_stable import StableASEBO
    from evotinyml.soo.jde import JDE
    from evotinyml.soo.params_opt_es import (
        AdaOpenES,
        OpenES,
        SNES,
        SparseOpenES,
        xNES,
    )
    from evotinyml.soo.pso_fixed import FixedPSO
    from evotinyml.soo.pso_momentum import (
        FirstMomentumPSO,
        SecondMomentumPSO,
        SoftMomentumPSO,
    )

    if algo not in EVOSAX_SOO_ALGOS:
        raise ValueError(
            f"Unknown SOO algo: {algo!r}. Use one of {tuple(EVOSAX_SOO_ALGOS)}."
        )

    cls_name, display_name = EVOSAX_SOO_ALGOS[algo]
    if algo == "asebo":
        Cls = StableASEBO
    elif algo == "open_es":
        Cls = OpenES
    elif algo == "sparse_open_es":
        Cls = SparseOpenES
    elif algo == "ada_open_es":
        Cls = AdaOpenES
    elif algo == "snes":
        Cls = SNES
    elif algo == "xnes":
        Cls = xNES
    elif algo == "jde":
        Cls = JDE
    elif algo == "pso":
        Cls = FixedPSO
    elif algo == "soft_m_pso":
        Cls = SoftMomentumPSO
    elif algo == "1m_pso":
        Cls = FirstMomentumPSO
    elif algo == "2m_pso":
        Cls = SecondMomentumPSO
    else:
        Cls = getattr(evosax_algorithms, cls_name)
    sigma = float(init_sigma)
    uses_mean_optimizer = algo in MEAN_OPTIMIZER_ALGOS

    if algo in EVEN_POPSIZE_ALGOS and population_size % 2 == 1:
        print(
            f"Note: {display_name} needs even popsize; "
            f"bumping {population_size} → {population_size + 1}."
        )
        population_size += 1
    if algo in {"de", "jde"} and population_size < 4:
        raise ValueError(f"{display_name} requires popsize >= 4, got {population_size}")
    if algo in PSO_ALGOS and population_size < 2:
        raise ValueError(f"{display_name} requires popsize >= 2, got {population_size}")

    kwargs: dict[str, Any] = {
        "population_size": population_size,
        "solution": jnp.zeros(n_var),
    }
    if uses_mean_optimizer:
        kwargs["optimizer"] = build_open_es_optimizer(
            es_optim,
            es_optim_lr,
            es_optim_scheduler,
            steps=steps,
            momentum=es_optim_momentum,
            weight_decay=es_optim_wd,
            steps_per_epoch=es_optim_steps_per_epoch,
        )
    if algo in SIGMA_SCHEDULE_ALGOS:
        kwargs["std_schedule"] = build_es_sigma_schedule(
            es_sigma_scheduler,
            sigma,
            steps=steps,
            end=es_sigma_end,
            steps_per_epoch=es_sigma_steps_per_epoch,
        )
    if algo == "asebo":
        dims = int(asebo_subspace_dims)
        if dims < 1:
            raise ValueError(f"asebo_subspace_dims must be >= 1, got {dims}")
        if dims > n_var:
            raise ValueError(
                f"asebo_subspace_dims ({dims}) must be <= n_var ({n_var})"
            )
        kwargs["subspace_dims"] = dims
    if algo == "sparse_open_es":
        kwargs["mask_prob"] = float(sparse_es_mask_prob)
    if algo == "pso":
        kwargs["ea_coeval"] = bool(ea_coeval)
    # soft_m_pso / 1m_pso / 2m_pso always co-evaluate [x; p; g] (2*popsize+1).

    es = Cls(**kwargs)
    params = es.default_params
    if algo not in SIGMA_SCHEDULE_ALGOS and hasattr(params, "std_init"):
        params = params.replace(std_init=sigma)
    if algo == "ada_open_es":
        ada_sigma_lr = (
            float(es_sigma_lr) if es_sigma_lr is not None else float(params.sigma_lr_init)
        )
        ada_sigma_min = (
            float(es_sigma_min)
            if es_sigma_min is not None
            else sigma * DEFAULT_ES_SIGMA_MIN_RATIO
        )
        ada_sigma_max = (
            float(es_sigma_max)
            if es_sigma_max is not None
            else sigma * DEFAULT_ES_SIGMA_MAX_RATIO
        )
        if ada_sigma_lr <= 0.0:
            raise ValueError(f"es_sigma_lr must be > 0, got {ada_sigma_lr}")
        if ada_sigma_min <= 0.0:
            raise ValueError(f"es_sigma_min must be > 0, got {ada_sigma_min}")
        if ada_sigma_max <= ada_sigma_min:
            raise ValueError(
                f"es_sigma_max ({ada_sigma_max}) must be > es_sigma_min ({ada_sigma_min})"
            )
        params = params.replace(
            sigma_lr_init=ada_sigma_lr, sigma_min=ada_sigma_min, sigma_max=ada_sigma_max
        )
    if algo == "lm_ma_es":
        params = _fix_lm_ma_es_c_c(es, params, population_size, n_var)
    if algo == "de":
        params = params.replace(
            differential_weight=float(de_f),
            crossover_rate=float(de_cr),
            elitism=bool(de_elitism),
        )
    if algo == "jde":
        params = params.replace(
            f_init=float(jde_f_init),
            cr_init=float(jde_cr_init),
            f_l=float(jde_f_l),
            f_u=float(jde_f_u),
            tau_f=float(jde_tau_f),
            tau_cr=float(jde_tau_cr),
            elitism=bool(jde_elitism),
        )
    if algo in PSO_ALGOS:
        v_max = float(pso_max_velocity)
        if v_max <= 0.0:
            raise ValueError(f"pso_max_velocity must be > 0, got {v_max}")
        params = params.replace(
            inertia_coeff=float(pso_inertia),
            cognitive_coeff=float(pso_cognitive),
            social_coeff=float(pso_social),
            v_max=v_max,
        )
    if algo in MOM_PSO_ALGOS:
        beta1 = float(mom_pso_beta1)
        gate_temperature = float(mom_pso_gate_temperature)
        gate_ema_decay = float(mom_pso_gate_ema_decay)
        global_topk_fraction = float(mom_pso_global_topk_fraction)
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"mom_pso_beta1 must be in [0, 1), got {beta1}")
        if gate_temperature <= 0.0:
            raise ValueError(
                "mom_pso_gate_temperature must be > 0, "
                f"got {gate_temperature}"
            )
        if not (0.0 <= gate_ema_decay < 1.0):
            raise ValueError(
                "mom_pso_gate_ema_decay must be in [0, 1), "
                f"got {gate_ema_decay}"
            )
        if not (0.0 < global_topk_fraction <= 1.0):
            raise ValueError(
                "mom_pso_global_topk_fraction must be in (0, 1], "
                f"got {global_topk_fraction}"
            )
        replace_kw: dict[str, float] = {
            "beta1": beta1,
            "gate_temperature": gate_temperature,
            "gate_ema_decay": gate_ema_decay,
            "global_topk_fraction": global_topk_fraction,
        }
        # soft_m_pso uses p ← p + m (no η); keep class defaults.
        if algo in {"1m_pso", "2m_pso"}:
            eta_p = (
                float(params.eta_personal)
                if mom_pso_eta_personal is None
                else float(mom_pso_eta_personal)
            )
            eta_g = (
                float(params.eta_global)
                if mom_pso_eta_global is None
                else float(mom_pso_eta_global)
            )
            if eta_p <= 0.0:
                raise ValueError(f"mom_pso_eta_personal must be > 0, got {eta_p}")
            if eta_g <= 0.0:
                raise ValueError(f"mom_pso_eta_global must be > 0, got {eta_g}")
            replace_kw["eta_personal"] = eta_p
            replace_kw["eta_global"] = eta_g
        if algo == "2m_pso":
            beta2 = float(mom_pso_beta2)
            if not (0.0 <= beta2 < 1.0):
                raise ValueError(f"mom_pso_beta2 must be in [0, 1), got {beta2}")
            replace_kw["beta2"] = beta2
        params = params.replace(**replace_kw)

    return es, params, population_size, display_name


@dataclass
class SOOESResult:
    """Final SOO state for CLI / saving.

    ``X`` is the distribution mean for ES algos, or the best population member
    for population-based DE.
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
    max_evals: int | None = None,
    init: str = "gaussian",
    init_sigma: float = 0.1,
    seed: int = 1,
    val_every: int = 50,
    test_loader: DataLoader | None = None,
    n_classes: int = 10,
    verbose: bool = False,
    use_wandb: bool = True,
    es_optim: str = DEFAULT_ES_OPTIM,
    es_optim_lr: float = DEFAULT_ES_OPTIM_LR,
    es_optim_scheduler: str = DEFAULT_ES_OPTIM_SCHEDULER,
    es_optim_momentum: float = DEFAULT_ES_OPTIM_MOMENTUM,
    es_optim_wd: float = DEFAULT_ES_OPTIM_WD,
    es_optim_steps_per_epoch: int | None = None,
    es_sigma_scheduler: str = DEFAULT_ES_SIGMA_SCHEDULER,
    es_sigma_end: float | None = None,
    es_sigma_steps_per_epoch: int | None = None,
    es_sigma_lr: float | None = None,
    es_sigma_min: float | None = None,
    es_sigma_max: float | None = None,
    asebo_subspace_dims: int = DEFAULT_ASEBO_SUBSPACE_DIMS,
    sparse_es_mask_prob: float = DEFAULT_SPARSE_ES_MASK_PROB,
    de_f: float = DEFAULT_DE_F,
    de_cr: float = DEFAULT_DE_CR,
    de_elitism: bool = DEFAULT_DE_ELITISM,
    jde_f_init: float = DEFAULT_JDE_F_INIT,
    jde_cr_init: float = DEFAULT_JDE_CR_INIT,
    jde_f_l: float = DEFAULT_JDE_F_L,
    jde_f_u: float = DEFAULT_JDE_F_U,
    jde_tau_f: float = DEFAULT_JDE_TAU_F,
    jde_tau_cr: float = DEFAULT_JDE_TAU_CR,
    jde_elitism: bool = DEFAULT_JDE_ELITISM,
    pso_inertia: float = DEFAULT_PSO_INERTIA,
    pso_cognitive: float = DEFAULT_PSO_COGNITIVE,
    pso_social: float = DEFAULT_PSO_SOCIAL,
    pso_max_velocity: float = DEFAULT_PSO_MAX_VELOCITY,
    ea_coeval: bool = DEFAULT_EA_COEVAL,
    mom_pso_eta_personal: float | None = DEFAULT_MOM_PSO_ETA_PERSONAL,
    mom_pso_eta_global: float | None = DEFAULT_MOM_PSO_ETA_GLOBAL,
    mom_pso_beta1: float = DEFAULT_MOM_PSO_BETA1,
    mom_pso_beta2: float = DEFAULT_MOM_PSO_BETA2,
    mom_pso_gate_temperature: float = DEFAULT_MOM_PSO_GATE_TEMPERATURE,
    mom_pso_gate_ema_decay: float = DEFAULT_MOM_PSO_GATE_EMA_DECAY,
    mom_pso_global_topk_fraction: float = DEFAULT_MOM_PSO_GLOBAL_TOPK_FRACTION,
) -> SOOESResult:
    """Ask / eval / tell loop for evosax SOO algorithms.

    Fitness is evaluated with torch (no JAX NN). evosax **minimizes** the
    fitness values passed to ``tell``. Train logging and test validation use
    the distribution **mean** for ES algos, or the **best population member**
    for DE / jDE / PSO.

    Function Evaluations: each ``ask`` costs ``len(candidates)`` FEs (PSO with
    ``ea_coeval`` returns ``2 * popsize``; soft / 1m / 2m PSO return
    ``2 * popsize + 1``).
    Population-based algos also evaluate
    the initial population (``popsize``), then run ask/tell generations until
    the next ask would exceed ``max_evals`` (default ``steps * popsize``).
    """
    algo = algo.lower()
    soo = getattr(problem, "soo_fitness", None)
    if not isinstance(soo, SOOFitness):
        raise TypeError(
            f"{algo} requires a problem with SOOFitness, got {type(problem).__name__}"
        )
    fitness_name = soo.fitness_name
    population_based = algo in POPULATION_BASED_ALGOS

    n_var = int(problem.n_var)
    if popsize is None:
        popsize = default_soo_popsize(n_var, algo)
    popsize = int(popsize)
    steps = int(steps)
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")
    if max_evals is None:
        max_evals = steps * popsize
    max_evals = int(max_evals)
    if max_evals < 1:
        raise ValueError(f"max_evals must be >= 1, got {max_evals}")

    needs_epoch = False
    if algo in SIGMA_SCHEDULE_ALGOS and es_sigma_scheduler.lower() in {
        "exponential",
        "exponential_decay",
    }:
        needs_epoch = True
    if algo in MEAN_OPTIMIZER_ALGOS and es_optim_scheduler.lower() in {
        "exponential",
        "exponential_decay",
    }:
        needs_epoch = True
    epoch_steps = (
        steps_per_data_epoch_from_problem(problem) if needs_epoch else None
    )
    sigma_steps_per_epoch = (
        es_sigma_steps_per_epoch
        if es_sigma_steps_per_epoch is not None
        else epoch_steps
    )
    optim_steps_per_epoch = (
        es_optim_steps_per_epoch
        if es_optim_steps_per_epoch is not None
        else epoch_steps
    )

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
        es_optim_wd=es_optim_wd,
        es_optim_steps_per_epoch=optim_steps_per_epoch,
        es_sigma_scheduler=es_sigma_scheduler,
        es_sigma_end=es_sigma_end,
        es_sigma_steps_per_epoch=sigma_steps_per_epoch,
        es_sigma_lr=es_sigma_lr,
        es_sigma_min=es_sigma_min,
        es_sigma_max=es_sigma_max,
        asebo_subspace_dims=asebo_subspace_dims,
        sparse_es_mask_prob=sparse_es_mask_prob,
        de_f=de_f,
        de_cr=de_cr,
        de_elitism=de_elitism,
        jde_f_init=jde_f_init,
        jde_cr_init=jde_cr_init,
        jde_f_l=jde_f_l,
        jde_f_u=jde_f_u,
        jde_tau_f=jde_tau_f,
        jde_tau_cr=jde_tau_cr,
        jde_elitism=jde_elitism,
        pso_inertia=pso_inertia,
        pso_cognitive=pso_cognitive,
        pso_social=pso_social,
        pso_max_velocity=pso_max_velocity,
        ea_coeval=ea_coeval,
        mom_pso_eta_personal=mom_pso_eta_personal,
        mom_pso_eta_global=mom_pso_eta_global,
        mom_pso_beta1=mom_pso_beta1,
        mom_pso_beta2=mom_pso_beta2,
        mom_pso_gate_temperature=mom_pso_gate_temperature,
        mom_pso_gate_ema_decay=mom_pso_gate_ema_decay,
        mom_pso_global_topk_fraction=mom_pso_global_topk_fraction,
    )
    sigma_schedule = (
        build_es_sigma_schedule(
            es_sigma_scheduler,
            init_sigma,
            steps=steps,
            end=es_sigma_end,
            steps_per_epoch=sigma_steps_per_epoch,
        )
        if algo in SIGMA_SCHEDULE_ALGOS
        else None
    )
    lr_schedule = (
        _build_lr_schedule(
            es_optim_scheduler,
            es_optim_lr,
            steps=steps,
            steps_per_epoch=optim_steps_per_epoch,
        )
        if algo in MEAN_OPTIMIZER_ALGOS
        else None
    )

    rng = np.random.default_rng(seed)
    key = jax.random.key(int(seed))
    key, key_init = jax.random.split(key)

    xl = float(np.asarray(problem.xl).reshape(-1)[0]) if problem.xl is not None else -10.0
    xu = float(np.asarray(problem.xu).reshape(-1)[0]) if problem.xu is not None else 10.0

    # DE consumes one generation evaluating the initial population; remaining
    # ask/tell rounds use the rest of the FE budget (capped by max_evals).
    ask_steps = max(0, steps - 1) if population_based else steps
    mean_f_hist: list[float] = []

    if population_based:
        if popsize > max_evals:
            raise ValueError(
                f"Initial population ({popsize}) exceeds max_evals ({max_evals})"
            )
        pop0 = _init_population(
            n_var,
            popsize,
            init,
            init_sigma,
            rng,
            theta0=getattr(problem, "theta0", None),
        )
        pop0 = np.clip(pop0, xl, xu)
        fit0 = soo.evaluate(pop0, details=False)
        n_eval = int(pop0.shape[0])
        state = es.init(
            key_init,
            jnp.asarray(pop0, dtype=jnp.float32),
            jnp.asarray(fit0, dtype=jnp.float32),
            params,
        )
        # evosax PR #109: seed archive when stock init left it unset (DE).
        state = _seed_archive_from_init(state, pop0, fit0)
        sol_x = _state_solution(algo, state, xl, xu)
        # Prefer the evaluated init fitness for the reported best member.
        best_idx = int(np.argmin(fit0))
        sol_x = np.clip(pop0[best_idx], xl, xu)
        sol_details = soo.evaluate_one(sol_x)
        sol_details.update(_maybe_ce_train_details(problem, sol_x))
        if algo == "jde":
            sol_details.update(_jde_control_means(state))
        sol_f = float(sol_details["f"])
        init_pop_best = float(np.min(fit0))
    else:
        mean0 = _init_mean(
            n_var,
            init,
            init_sigma,
            rng,
            theta0=getattr(problem, "theta0", None),
        )
        state = es.init(key_init, jnp.asarray(mean0, dtype=jnp.float32), params)
        n_eval = 0
        sol_x = _state_solution(algo, state, xl, xu)
        sol_details = soo.evaluate_one(sol_x)
        sol_details.update(_maybe_ce_train_details(problem, sol_x))
        if algo in ADAPTIVE_SIGMA_ALGOS:
            sol_details.update(_ada_sigma_stats(state))
        sol_f = float(sol_details["f"])
        init_pop_best = None

    mean_f_hist.append(sol_f)
    completed_steps = 0
    _log_soo_step(
        step=0,
        n_eval=n_eval,
        f=sol_f,
        details=sol_details,
        fitness_name=fitness_name,
        use_wandb=use_wandb,
        verbose=verbose,
        label="init",
        pop_best_f=init_pop_best,
        es_sigma=_current_sigma_scalar(algo, state, sigma_schedule, 0),
        es_lr=sigma_at(lr_schedule, 0) if lr_schedule is not None else None,
        solution_label="best" if population_based else "mean",
    )
    if test_loader is not None and val_every > 0:
        _maybe_validate_mean(
            problem,
            sol_x,
            test_loader,
            n_classes=n_classes,
            n_eval=n_eval,
            opt_step=0,
            use_wandb=use_wandb,
            verbose=verbose,
            force=True,
            solution_label="best" if population_based else "mean",
        )

    for gen in range(1, ask_steps + 1):
        key, key_ask, key_tell = jax.random.split(key, 3)
        population, state = es.ask(key_ask, state, params)
        X = np.asarray(population, dtype=np.float64)
        X = np.clip(X, xl, xu)
        n_ask = int(X.shape[0])
        if n_eval + n_ask > max_evals:
            break

        # Sample minibatch(es) for this generation; shared by pop + reported solution.
        problem.sample_eval_pool()
        fitness = soo.evaluate(X, details=False)
        n_eval += n_ask
        state, _es_metrics = es.tell(
            key_tell, jnp.asarray(X, dtype=jnp.float32), jnp.asarray(fitness), state, params
        )

        sol_x = _state_solution(algo, state, xl, xu)
        sol_details = soo.evaluate_one(sol_x)
        sol_details.update(_maybe_ce_train_details(problem, sol_x))
        if algo == "jde":
            sol_details.update(_jde_control_means(state))
        if algo in ADAPTIVE_SIGMA_ALGOS:
            sol_details.update(_ada_sigma_stats(state))
        sol_f = float(sol_details["f"])
        mean_f_hist.append(sol_f)
        completed_steps = gen

        # PSO ea_coeval / soft_m_pso / 1m_pso / 2m_pso ask stacks anchors; report x best only.
        pop_best_f = float(np.min(fitness[:popsize]))
        cur_sigma = _current_sigma_scalar(algo, state, sigma_schedule, gen - 1)
        cur_lr = sigma_at(lr_schedule, gen - 1) if lr_schedule is not None else None
        _log_soo_step(
            step=gen,
            n_eval=n_eval,
            f=sol_f,
            details=sol_details,
            fitness_name=fitness_name,
            use_wandb=use_wandb,
            verbose=verbose,
            label=f"step {gen}",
            pop_best_f=pop_best_f,
            es_sigma=cur_sigma,
            es_lr=cur_lr,
            solution_label="best" if population_based else "mean",
        )

        if test_loader is not None and val_every > 0:
            _maybe_validate_mean(
                problem,
                sol_x,
                test_loader,
                n_classes=n_classes,
                n_eval=n_eval,
                opt_step=gen,
                use_wandb=use_wandb,
                verbose=verbose,
                force=(gen == 1) or (gen % val_every == 0),
                solution_label="best" if population_based else "mean",
            )

    if (
        test_loader is not None
        and val_every > 0
        and completed_steps > 0
        and completed_steps % val_every != 0
        and completed_steps != 1
    ):
        _maybe_validate_mean(
            problem,
            sol_x,
            test_loader,
            n_classes=n_classes,
            n_eval=n_eval,
            opt_step=completed_steps,
            use_wandb=use_wandb,
            verbose=verbose,
            force=True,
            solution_label="best" if population_based else "mean",
        )

    return SOOESResult(
        X=sol_x,
        f=sol_f,
        mean_f_history=np.asarray(mean_f_hist, dtype=np.float64),
        steps=completed_steps,
        popsize=popsize,
        fitness_name=fitness_name,
        algo=algo,
        details=sol_details,
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
    es_lr: float | None = None,
    solution_label: str = "mean",
) -> None:
    extra = _format_details(details)
    if verbose:
        msg = f"[{label}] n_eval={n_eval}  {solution_label}_f={f:.6f}"
        if extra:
            msg += f"  {extra}"
        if pop_best_f is not None:
            msg += f"  pop_best_f={pop_best_f:.6f}"
        if es_sigma is not None:
            msg += f"  es_sigma={es_sigma:.6f}"
        if es_lr is not None:
            msg += f"  es_lr={es_lr:.6f}"
        print(msg)
    if use_wandb:
        payload: dict[str, Any] = {
            "train/step": step,
            "train/f": f,
            "train/mean_f": f,
            "train/fitness_name": fitness_name,
            "train/solution": solution_label,
        }
        if pop_best_f is not None:
            payload["train/pop_best_f"] = pop_best_f
        if es_sigma is not None:
            payload["train/es_sigma"] = float(es_sigma)
        if es_lr is not None:
            payload["train/es_lr"] = float(es_lr)
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
    solution_label: str = "mean",
) -> dict[str, float] | None:
    """Validate the tracked SOO solution (ES mean or DE best) on the test set."""
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
        tag = "pop best" if solution_label == "best" else "ES mean"
        print(
            f"[val n_eval={n_eval} step={opt_step}] ({tag}) "
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
    es_optim_wd = getattr(args, "es_optim_wd", DEFAULT_ES_OPTIM_WD)
    es_sigma_scheduler = getattr(args, "es_sigma_scheduler", DEFAULT_ES_SIGMA_SCHEDULER)
    es_sigma_end = getattr(args, "es_sigma_end", None)
    es_sigma_lr = getattr(args, "es_sigma_lr", None)
    es_sigma_min = getattr(args, "es_sigma_min", None)
    es_sigma_max = getattr(args, "es_sigma_max", None)
    asebo_subspace_dims = getattr(
        args, "asebo_subspace_dims", DEFAULT_ASEBO_SUBSPACE_DIMS
    )
    sparse_es_mask_prob = getattr(
        args, "sparse_es_mask_prob", DEFAULT_SPARSE_ES_MASK_PROB
    )
    de_f = getattr(args, "de_f", DEFAULT_DE_F)
    de_cr = getattr(args, "de_cr", DEFAULT_DE_CR)
    de_elitism = getattr(args, "de_elitism", DEFAULT_DE_ELITISM)
    jde_f_init = getattr(args, "jde_f_init", DEFAULT_JDE_F_INIT)
    jde_cr_init = getattr(args, "jde_cr_init", DEFAULT_JDE_CR_INIT)
    jde_f_l = getattr(args, "jde_f_l", DEFAULT_JDE_F_L)
    jde_f_u = getattr(args, "jde_f_u", DEFAULT_JDE_F_U)
    jde_tau_f = getattr(args, "jde_tau_f", DEFAULT_JDE_TAU_F)
    jde_tau_cr = getattr(args, "jde_tau_cr", DEFAULT_JDE_TAU_CR)
    jde_elitism = getattr(args, "jde_elitism", DEFAULT_JDE_ELITISM)
    pso_inertia = getattr(args, "pso_inertia", DEFAULT_PSO_INERTIA)
    pso_cognitive = getattr(args, "pso_cognitive", DEFAULT_PSO_COGNITIVE)
    pso_social = getattr(args, "pso_social", DEFAULT_PSO_SOCIAL)
    pso_max_velocity = getattr(args, "pso_max_velocity", DEFAULT_PSO_MAX_VELOCITY)
    ea_coeval = getattr(args, "ea_coeval", DEFAULT_EA_COEVAL)
    mom_pso_eta_personal = getattr(
        args, "mom_pso_eta_personal", DEFAULT_MOM_PSO_ETA_PERSONAL
    )
    mom_pso_eta_global = getattr(
        args, "mom_pso_eta_global", DEFAULT_MOM_PSO_ETA_GLOBAL
    )
    mom_pso_beta1 = getattr(args, "mom_pso_beta1", DEFAULT_MOM_PSO_BETA1)
    mom_pso_beta2 = getattr(args, "mom_pso_beta2", DEFAULT_MOM_PSO_BETA2)
    mom_pso_gate_temperature = getattr(
        args, "mom_pso_gate_temperature", DEFAULT_MOM_PSO_GATE_TEMPERATURE
    )
    mom_pso_gate_ema_decay = getattr(
        args, "mom_pso_gate_ema_decay", DEFAULT_MOM_PSO_GATE_EMA_DECAY
    )
    mom_pso_global_topk_fraction = getattr(
        args,
        "mom_pso_global_topk_fraction",
        DEFAULT_MOM_PSO_GLOBAL_TOPK_FRACTION,
    )
    val_solution = "de_best" if algo in POPULATION_BASED_ALGOS else "es_mean"
    if algo in {"jde", "pso", "soft_m_pso", "1m_pso", "2m_pso"}:
        library = f"evosax+{algo}"
    else:
        library = "evosax"
    return {
        "dataset": args.dataset,
        "model": getattr(args, "model", None),
        "problem": args.problem,
        "activation": args.activation,
        "algo": algo,
        "fitness": fitness_name,
        "scalar_weights": list(
            getattr(args, "scalar_weights_resolved", None)
            or getattr(args, "scalar_weights", None)
            or []
        ),
        "init": args.init,
        "init_sigma": getattr(args, "init_sigma", 0.1),
        "xl": getattr(args, "xl", -10.0),
        "xu": getattr(args, "xu", 10.0),
        "es_optim": es_optim,
        "es_optim_lr": es_optim_lr,
        "es_optim_scheduler": es_optim_scheduler,
        "es_optim_momentum": es_optim_momentum,
        "es_optim_wd": es_optim_wd,
        "es_sigma_scheduler": es_sigma_scheduler,
        "es_sigma_end": es_sigma_end,
        "es_sigma_lr": es_sigma_lr,
        "es_sigma_min": es_sigma_min,
        "es_sigma_max": es_sigma_max,
        "asebo_subspace_dims": asebo_subspace_dims,
        "sparse_es_mask_prob": sparse_es_mask_prob,
        "de_f": de_f,
        "de_cr": de_cr,
        "de_elitism": de_elitism,
        "jde_f_init": jde_f_init,
        "jde_cr_init": jde_cr_init,
        "jde_f_l": jde_f_l,
        "jde_f_u": jde_f_u,
        "jde_tau_f": jde_tau_f,
        "jde_tau_cr": jde_tau_cr,
        "jde_elitism": jde_elitism,
        "pso_inertia": pso_inertia,
        "pso_cognitive": pso_cognitive,
        "pso_social": pso_social,
        "pso_max_velocity": pso_max_velocity,
        "ea_coeval": ea_coeval,
        "mom_pso_eta_personal": mom_pso_eta_personal,
        "mom_pso_eta_global": mom_pso_eta_global,
        "mom_pso_beta1": mom_pso_beta1,
        "mom_pso_beta2": mom_pso_beta2,
        "mom_pso_gate_temperature": mom_pso_gate_temperature,
        "mom_pso_gate_ema_decay": mom_pso_gate_ema_decay,
        "mom_pso_global_topk_fraction": mom_pso_global_topk_fraction,
        "steps": args.steps,
        "evals": getattr(args, "evals", None),
        "popsize": popsize,
        "batch_size": args.batch_size,
        "eval_mode": getattr(args, "eval_mode", "multi"),
        "eval_batches": getattr(args, "eval_batches", 50),
        "sampler": getattr(args, "sampler", "auto"),
        "val_every": args.val_every,
        "val_batch_size": args.val_batch_size,
        "seed": args.seed,
        "device": args.device,
        "n_var": n_var,
        "n_obj": 1,
        "val_solution": val_solution,
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
            "es_optim_wd": es_optim_wd,
            "es_sigma_scheduler": es_sigma_scheduler,
            "es_sigma_end": es_sigma_end,
            "es_sigma_lr": es_sigma_lr,
            "es_sigma_min": es_sigma_min,
            "es_sigma_max": es_sigma_max,
            "asebo_subspace_dims": asebo_subspace_dims,
            "sparse_es_mask_prob": sparse_es_mask_prob,
            "de_f": de_f,
            "de_cr": de_cr,
            "de_elitism": de_elitism,
            "jde_f_init": jde_f_init,
            "jde_cr_init": jde_cr_init,
            "jde_f_l": jde_f_l,
            "jde_f_u": jde_f_u,
            "jde_tau_f": jde_tau_f,
            "jde_tau_cr": jde_tau_cr,
            "jde_elitism": jde_elitism,
            "pso_inertia": pso_inertia,
            "pso_cognitive": pso_cognitive,
            "pso_social": pso_social,
            "pso_max_velocity": pso_max_velocity,
            "ea_coeval": ea_coeval,
            "mom_pso_eta_personal": mom_pso_eta_personal,
            "mom_pso_eta_global": mom_pso_eta_global,
            "mom_pso_beta1": mom_pso_beta1,
            "mom_pso_beta2": mom_pso_beta2,
            "mom_pso_gate_temperature": mom_pso_gate_temperature,
            "mom_pso_gate_ema_decay": mom_pso_gate_ema_decay,
            "mom_pso_global_topk_fraction": mom_pso_global_topk_fraction,

            "library": library,
            "val_solution": val_solution,
        },
    }
