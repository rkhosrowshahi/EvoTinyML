"""Local on-disk history of train / val Pareto fronts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class FrontHistory:
    """Append P–R fronts and rewrite an NPZ on each snapshot."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.steps: list[int] = []
        self.n_nds: list[int] = []
        self._F_chunks: list[np.ndarray] = []
        self._precision_chunks: list[np.ndarray] = []
        self._recall_chunks: list[np.ndarray] = []
        self._scalar_keys: list[str] = []
        self._scalars: dict[str, list[float]] = {}

    def _pop_last(self) -> None:
        self.steps.pop()
        self.n_nds.pop()
        self._F_chunks.pop()
        self._precision_chunks.pop()
        self._recall_chunks.pop()
        for key in self._scalar_keys:
            self._scalars[key].pop()

    def append(
        self,
        step: int,
        precision: np.ndarray,
        recall: np.ndarray,
        scalars: dict[str, float] | None = None,
    ) -> None:
        p = np.asarray(precision, dtype=np.float64).ravel()
        r = np.asarray(recall, dtype=np.float64).ravel()
        if p.size == 0 or p.size != r.size:
            return
        order = np.argsort(p)
        p, r = p[order], r[order]
        f = np.column_stack([1.0 - p, 1.0 - r])

        # Replace snapshot if the same step is written twice (e.g. end-of-run).
        if self.steps and self.steps[-1] == int(step):
            self._pop_last()

        if scalars:
            if not self._scalar_keys:
                self._scalar_keys = sorted(scalars.keys())
                self._scalars = {k: [] for k in self._scalar_keys}
            for key in self._scalar_keys:
                if key not in scalars:
                    raise ValueError(f"Missing scalar {key!r} for history snapshot")
                self._scalars[key].append(float(scalars[key]))

        self.steps.append(int(step))
        self.n_nds.append(int(p.size))
        self._F_chunks.append(f)
        self._precision_chunks.append(p)
        self._recall_chunks.append(r)
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "steps": np.asarray(self.steps, dtype=np.int64)
            if self.steps
            else np.zeros(0, dtype=np.int64),
            "n_nds": np.asarray(self.n_nds, dtype=np.int64)
            if self.n_nds
            else np.zeros(0, dtype=np.int64),
            "F": np.concatenate(self._F_chunks, axis=0)
            if self._F_chunks
            else np.zeros((0, 2), dtype=np.float64),
            "precision": np.concatenate(self._precision_chunks, axis=0)
            if self._precision_chunks
            else np.zeros(0, dtype=np.float64),
            "recall": np.concatenate(self._recall_chunks, axis=0)
            if self._recall_chunks
            else np.zeros(0, dtype=np.float64),
        }
        for key in self._scalar_keys:
            payload[key] = np.asarray(self._scalars[key], dtype=np.float64)
        np.savez_compressed(self.path, **payload)


# Backward-compatible alias.
ParetoFrontHistory = FrontHistory
