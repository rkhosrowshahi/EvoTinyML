"""Crossover operators for weight-space evolution."""

from __future__ import annotations

import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.util import default_random_state


class NoCrossover(Crossover):
    """Copy parents into fresh unevaluated offspring (mutation-only / ES-style).

    Unlike pymoo's ``nox.NoCrossover``, this does **not** reuse parent
    ``Individual`` objects. Reusing parents leaves ``evaluated`` set, so the
    evaluator skips offspring even after mutation mutates ``X`` — which stalls
    ``n_eval`` and can terminate the run almost immediately.
    """

    def __init__(self) -> None:
        # prob=0 → Crossover.do copies parent X into a new Population.
        super().__init__(n_parents=2, n_offsprings=2, prob=0.0)

    @default_random_state
    def _do(self, problem, X, random_state=None, **kwargs):
        # Unused when prob=0; keep a safe identity impl. if invoked.
        return np.copy(X)
