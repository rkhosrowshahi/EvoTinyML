"""Step-based termination (replaces generation terminology)."""

from __future__ import annotations

from pymoo.core.termination import Termination


class MaximumStepTermination(Termination):
    """Terminate after a fixed number of optimization steps (excluding init).

    pymoo ``n_gen`` is 1 after the initial population; each further iteration
    is one optimization step. W&B logs step 0 at init and steps 1..N for
    optimization, so termination uses ``n_gen - 1 >= n_max_steps``.
    """

    def __init__(self, n_max_steps: int) -> None:
        super().__init__()
        self.n_max_steps = int(n_max_steps)

    def _update(self, algorithm):
        opt_step = max(0, int(algorithm.n_gen) - 1)
        return min(1.0, opt_step / self.n_max_steps)
