"""Local on-disk history of train Pareto fronts."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class ParetoFrontHistory:
    """Append train P–R fronts and rewrite ``history.npz`` on each snapshot."""

    def __init__(self, path: str | Path = "history.npz") -> None:
        self.path = Path(path)
        self.steps: list[int] = []
        self.n_nds: list[int] = []
        self._F_chunks: list[np.ndarray] = []
        self._precision_chunks: list[np.ndarray] = []
        self._recall_chunks: list[np.ndarray] = []

    def append(self, step: int, precision: np.ndarray, recall: np.ndarray) -> None:
        p = np.asarray(precision, dtype=np.float64).ravel()
        r = np.asarray(recall, dtype=np.float64).ravel()
        if p.size == 0 or p.size != r.size:
            return
        order = np.argsort(p)
        p, r = p[order], r[order]
        f = np.column_stack([1.0 - p, 1.0 - r])

        # Replace snapshot if the same step is written twice (e.g. end-of-run).
        if self.steps and self.steps[-1] == int(step):
            self.steps.pop()
            self.n_nds.pop()
            self._F_chunks.pop()
            self._precision_chunks.pop()
            self._recall_chunks.pop()

        self.steps.append(int(step))
        self.n_nds.append(int(p.size))
        self._F_chunks.append(f)
        self._precision_chunks.append(p)
        self._recall_chunks.append(r)
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.steps:
            np.savez_compressed(
                self.path,
                steps=np.zeros(0, dtype=np.int64),
                n_nds=np.zeros(0, dtype=np.int64),
                F=np.zeros((0, 2), dtype=np.float64),
                precision=np.zeros(0, dtype=np.float64),
                recall=np.zeros(0, dtype=np.float64),
            )
            return
        np.savez_compressed(
            self.path,
            steps=np.asarray(self.steps, dtype=np.int64),
            n_nds=np.asarray(self.n_nds, dtype=np.int64),
            F=np.concatenate(self._F_chunks, axis=0),
            precision=np.concatenate(self._precision_chunks, axis=0),
            recall=np.concatenate(self._recall_chunks, axis=0),
        )
