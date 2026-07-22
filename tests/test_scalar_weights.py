"""Weighted-sum SOO scalarization for MOO problems."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from evotinyml.fitness import (
    SOOFitness,
    SoftPrecisionRecallMetrics,
    default_scalar_weights,
    parse_scalar_weights,
    weighted_sum,
)
from evotinyml.problem import MOO_PROBLEMS, SOO_PROBLEMS, apply_scalar_weights


class _IdentityLogits(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("logits", logits)
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
        n_obj=2,
        problem_name="soft_precision_recall",
        _evaluate_individual=None,
    )


def test_default_scalar_weights_equal_ones():
    w = default_scalar_weights("soft_precision_recall", 2)
    assert w.shape == (2,)
    assert np.allclose(w, [1.0, 1.0])
    w10 = default_scalar_weights("cwrm_cross_entropy", 10)
    assert w10.shape == (10,)
    assert np.allclose(w10, np.ones(10))


def test_parse_scalar_weights_cli():
    w = parse_scalar_weights("1,0.1", 2, problem_name="cross_entropy_l1")
    assert np.allclose(w, [1.0, 0.1])


def test_parse_scalar_weights_rejects_bad_length():
    with pytest.raises(ValueError, match="length"):
        parse_scalar_weights("1,2,3", 2, problem_name="soft_precision_recall")


def test_weighted_sum_soft_pr():
    # Softmax that yields known soft P/R is hard; check SOOFitness weighting
    # against an explicit vector metric stub.
    class _Vec:
        n_obj = 2
        fitness_name = "1-P,1-R"

        def vector_for_weights(self, flat):
            return (np.asarray([0.4, 0.6], dtype=np.float64), 0.6, 0.4)

    soo = SOOFitness(_Vec(), scalar_weights=[1.0, 1.0])
    assert np.isclose(soo._scalar_one(np.zeros(1)), 1.0)
    soo.set_scalar_weights([2.0, 0.5])
    assert np.isclose(soo._scalar_one(np.zeros(1)), 2.0 * 0.4 + 0.5 * 0.6)
    details = soo.evaluate_one(np.zeros(1))
    assert np.isclose(details["f"], 2.0 * 0.4 + 0.5 * 0.6)
    assert np.isclose(details["w0"], 2.0)
    assert np.isclose(details["w1"], 0.5)
    assert np.isclose(details["precision"], 0.6)
    assert np.isclose(details["recall"], 0.4)


def test_equal_weights_match_sum_soft_pr():
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
    metrics = SoftPrecisionRecallMetrics(problem, temperature=1.0)
    soo = SOOFitness(metrics, scalar_weights=[1.0, 1.0])
    flat = np.zeros(1, dtype=np.float64)
    assert np.isclose(soo._scalar_one(flat), metrics.scalar_for_weights(flat))


def test_all_moo_in_soo_problems():
    assert MOO_PROBLEMS <= SOO_PROBLEMS
    assert "cwrm_cross_entropy" in SOO_PROBLEMS
    assert "cross_entropy_l1" in SOO_PROBLEMS


def test_apply_scalar_weights_on_stub():
    metrics = SimpleNamespace(
        n_obj=2,
        fitness_name="1-P,1-R",
        vector_for_weights=lambda flat: (np.array([1.0, 2.0]),),
        scalar_for_weights=lambda flat: 3.0,
    )
    soo = SOOFitness(metrics, scalar_weights=[1.0, 1.0])
    problem = SimpleNamespace(
        n_obj=2, problem_name="soft_precision_recall", soo_fitness=soo
    )
    w = apply_scalar_weights(problem, "3,1")
    assert np.allclose(w, [3.0, 1.0])
    assert np.isclose(soo._scalar_one(np.zeros(1)), weighted_sum([1.0, 2.0], [3.0, 1.0]))
