"""Training callbacks."""

from __future__ import annotations

from pymoo.core.callback import Callback
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from evotinyml.problem import WeightOptimizationProblem


class ResampleBatchCallback(Callback):
    """Resample the evaluation batch every ``every`` steps.

    After resampling, re-evaluates the current population on the new batch so
    parent/offspring objectives stay comparable. The HV nadir from the initial
    step is left unchanged.
    """

    def __init__(self, every: int = 50) -> None:
        super().__init__()
        self.every = max(1, int(every))

    def notify(self, algorithm):
        problem = algorithm.problem
        if not isinstance(problem, WeightOptimizationProblem):
            return

        # Display runs before the callback, so after finishing step 50, 100, ...
        # we refresh the batch used by subsequent steps.
        if algorithm.n_gen <= 0 or algorithm.n_gen % self.every != 0:
            return

        problem.resample_batch()
        algorithm.evaluator.eval(problem, algorithm.pop, skip_already_evaluated=False)

        F = algorithm.pop.get("F")
        nd_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
        algorithm.opt = algorithm.pop[nd_idx]

        n_batches = len(getattr(problem, "eval_batch_pool", []) or [])
        print(
            f"[step {algorithm.n_gen}] resampled eval pool "
            f"(batches={n_batches}, mode={getattr(problem, 'eval_mode', '?')}, "
            f"batch_version={problem.batch_version}; HV nadir unchanged)"
        )
