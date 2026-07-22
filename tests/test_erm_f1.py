"""Unit tests for ERM hard/soft F1 scalar fitness."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from evotinyml.fitness import (
    F1Metrics,
    SoftF1Metrics,
    hard_macro_f1,
    soft_macro_f1,
)
from evotinyml.problem import PROBLEM_ALIASES, SOO_ONLY_PROBLEMS, SOO_PROBLEMS


class _IdentityLogits(nn.Module):
    """Model that ignores input and returns a fixed logits batch."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("logits", logits)
        # One dummy parameter so set_weights has something to write.
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits.expand(x.shape[0], -1)


def _mock_problem(logits: torch.Tensor, targets: torch.Tensor, n_classes: int):
    model = _IdentityLogits(logits)
    inputs = torch.zeros(targets.shape[0], 1)

    def set_weights(flat: np.ndarray) -> None:
        with torch.no_grad():
            model.w.copy_(torch.as_tensor(flat[:1], dtype=model.w.dtype))

    return SimpleNamespace(
        model=model,
        device=torch.device("cpu"),
        batch_sampler=SimpleNamespace(num_classes=n_classes),
        eval_batch_pool=[(inputs, targets)],
        set_weights=set_weights,
    )


def test_hard_macro_f1_perfect():
    pred = torch.tensor([0, 1, 0, 1])
    targets = torch.tensor([0, 1, 0, 1])
    assert np.isclose(hard_macro_f1(pred, targets, n_classes=2), 1.0)


def test_erm_f1_perfect_predictions():
    # Logits that argmax to the true labels.
    logits = torch.tensor(
        [
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 0.0],
            [0.0, 10.0],
        ]
    )
    targets = torch.tensor([0, 1, 0, 1])
    problem = _mock_problem(logits, targets, n_classes=2)
    metrics = F1Metrics(problem)
    details = metrics.scalar_details_for_weights(np.zeros(1, dtype=np.float64))
    assert np.isclose(details["f1"], 1.0)
    assert np.isclose(details["f"], 0.0)
    assert np.isclose(metrics.scalar_for_weights(np.zeros(1)), 0.0)


def test_soft_macro_f1_formula():
    probs = torch.tensor(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.7, 0.3],
            [0.4, 0.6],
        ],
        dtype=torch.float64,
    )
    targets = torch.tensor([0, 1, 0, 1])
    n_classes = 2
    y = torch.nn.functional.one_hot(targets, num_classes=n_classes).to(dtype=probs.dtype)
    soft_tp = (probs * y).sum(dim=0)
    soft_fp = (probs * (1.0 - y)).sum(dim=0)
    soft_fn = ((1.0 - probs) * y).sum(dim=0)
    prec = soft_tp / (soft_tp + soft_fp)
    rec = soft_tp / (soft_tp + soft_fn)
    expected = float((2.0 * prec * rec / (prec + rec)).mean().item())
    assert np.isclose(soft_macro_f1(probs, targets, n_classes), expected)


def test_erm_soft_f1_matches_helper():
    logits = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 0, 1])
    problem = _mock_problem(logits, targets, n_classes=2)
    metrics = SoftF1Metrics(problem, temperature=1.0)
    details = metrics.scalar_details_for_weights(np.zeros(1, dtype=np.float64))
    expected = soft_macro_f1(torch.softmax(logits, dim=1), targets, n_classes=2)
    assert np.isclose(details["f1"], expected)
    assert np.isclose(details["f"], 1.0 - expected)


def test_erm_f1_registered_as_soo_only():
    assert "erm_f1" in SOO_PROBLEMS
    assert "erm_soft_f1" in SOO_PROBLEMS
    assert "erm_f1" in SOO_ONLY_PROBLEMS
    assert "erm_soft_f1" in SOO_ONLY_PROBLEMS
    assert PROBLEM_ALIASES["f1"] == "erm_f1"
    assert PROBLEM_ALIASES["soft_f1"] == "erm_soft_f1"
