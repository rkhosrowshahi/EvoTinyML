"""MOEA constructors (NSGA-II / NSGA-III)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.algorithm import Algorithm
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.util.display.output import Output
from pymoo.util.ref_dirs import get_reference_directions

from evotinyml.moo.crossover import NoCrossover
from evotinyml.moo.mutation import AbsoluteGaussianMutation, LayerWiseGaussianMutation


MOO_ALGORITHMS = ("nsga2", "nsga3")
CROSSOVERS = ("sbx", "none")
MUTATIONS = ("pm", "gaussian", "layerwise")


@dataclass(frozen=True)
class OperatorConfig:
    """Crossover + mutation settings shared by NSGA-II / NSGA-III."""

    crossover: str = "sbx"
    crossover_prob: float = 0.9
    crossover_eta: float = 15.0
    crossover_prob_var: float = 0.5
    mutation: str = "pm"
    mutation_prob: float = 0.9
    mutation_eta: float = 20.0
    mutation_sigma: float = 0.1
    mutation_prob_var: float | None = None

    def to_dict(self) -> dict:
        cx_name = self.crossover.lower()
        if cx_name == "none":
            crossover = {"type": "none"}
        else:
            crossover = {
                "type": "SBX",
                "prob": self.crossover_prob,
                "eta": self.crossover_eta,
                "prob_var": self.crossover_prob_var,
            }

        mut_name = self.mutation.lower()
        mutation: dict
        if mut_name == "gaussian":
            mutation = {
                "type": "gaussian",
                "prob": self.mutation_prob,
                "sigma": self.mutation_sigma,
                "prob_var": self.mutation_prob_var,
            }
        elif mut_name == "layerwise":
            mutation = {
                "type": "layerwise",
                "prob": self.mutation_prob,
                "sigma": self.mutation_sigma,
                "prob_var": self.mutation_prob_var,
                "scale": "he_fan_in_mean_norm",
            }
        else:
            mutation = {
                "type": "PM",
                "prob": self.mutation_prob,
                "eta": self.mutation_eta,
                "prob_var": self.mutation_prob_var,
            }
        return {
            "crossover": crossover,
            "mutation": mutation,
        }

    def build_crossover(self):
        cx_name = self.crossover.lower()
        if cx_name == "none":
            return NoCrossover()
        if cx_name == "sbx":
            return SBX(
                prob=self.crossover_prob,
                eta=self.crossover_eta,
                prob_var=self.crossover_prob_var,
            )
        raise ValueError(f"Unknown crossover: {self.crossover!r}. Use one of {CROSSOVERS}.")

    def build_mutation(self):
        mut_name = self.mutation.lower()
        if mut_name == "gaussian":
            return AbsoluteGaussianMutation(
                sigma=self.mutation_sigma,
                prob=self.mutation_prob,
                prob_var=self.mutation_prob_var,
            )
        if mut_name == "layerwise":
            return LayerWiseGaussianMutation(
                sigma=self.mutation_sigma,
                prob=self.mutation_prob,
                prob_var=self.mutation_prob_var,
            )
        if mut_name == "pm":
            return PM(
                prob=self.mutation_prob,
                eta=self.mutation_eta,
                prob_var=self.mutation_prob_var,
            )
        raise ValueError(f"Unknown mutation: {self.mutation!r}. Use one of {MUTATIONS}.")


def make_reference_directions(
    n_obj: int,
    pop_size: int,
    method: str = "energy",
    n_partitions: int | None = None,
    seed: int = 1,
) -> np.ndarray:
    """Build reference directions for NSGA-III.

    ``energy`` produces exactly ``pop_size`` directions (good default for
    many-objective). ``das-dennis`` uses uniform partitions (count depends on
    ``n_partitions`` and ``n_obj``).
    """
    method = method.lower()
    if method == "energy":
        return get_reference_directions("energy", n_obj, n_points=pop_size, seed=seed)
    if method == "das-dennis":
        if n_partitions is None:
            # Largest p with #dirs <= pop_size for Das-Dennis on the unit simplex.
            p = 1
            best = get_reference_directions("das-dennis", n_obj, n_partitions=1)
            while True:
                cand = get_reference_directions("das-dennis", n_obj, n_partitions=p + 1)
                if len(cand) > pop_size:
                    break
                best = cand
                p += 1
            return best
        return get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
    raise ValueError(f"Unknown ref-dirs method: {method!r}. Use 'energy' or 'das-dennis'.")


def build_algorithm(
    name: str,
    *,
    pop_size: int,
    n_obj: int,
    sampling: Sampling,
    output: Output,
    seed: int = 1,
    ref_dirs_method: str = "energy",
    n_partitions: int | None = None,
    operators: OperatorConfig | None = None,
) -> Algorithm:
    """Construct NSGA-II or NSGA-III with configurable crossover / mutation operators."""
    name = name.lower()
    ops = operators or OperatorConfig()
    crossover = ops.build_crossover()
    mutation = ops.build_mutation()

    if name == "nsga2":
        algo = NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            output=output,
        )
        algo.operator_config = ops
        algo.algo_config = {
            "name": "nsga2",
            "popsize": pop_size,
            **ops.to_dict(),
        }
        return algo

    if name == "nsga3":
        ref_dirs = make_reference_directions(
            n_obj=n_obj,
            pop_size=pop_size,
            method=ref_dirs_method,
            n_partitions=n_partitions,
            seed=seed,
        )
        algo = NSGA3(
            ref_dirs=ref_dirs,
            pop_size=pop_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            output=output,
        )
        algo.operator_config = ops
        algo.algo_config = {
            "name": "nsga3",
            "popsize": pop_size,
            "ref_dirs_method": ref_dirs_method,
            "n_partitions": n_partitions,
            "n_ref_dirs": int(len(ref_dirs)),
            **ops.to_dict(),
        }
        return algo

    raise ValueError(f"Unknown algorithm: {name!r}. Use one of {MOO_ALGORITHMS}.")
