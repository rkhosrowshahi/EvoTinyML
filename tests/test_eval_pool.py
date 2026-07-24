"""Eval-batch sampling via the batch sampler, and MNIST CNN width."""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from evotinyml.data import RandomBatchSampler
from evotinyml.model import TinyCNN_MNIST_4K, build_model
from evotinyml.problem import ERMCrossEntropyProblem


def test_tinycnn_mnist_matches_esde_param_count():
    model = TinyCNN_MNIST_4K()
    assert model.num_parameters() == 4266
    assert build_model("mnist", 10).num_parameters() == 4266


def test_sample_eval_pool_draws_new_batch():
    images = torch.randn(512, 1, 28, 28)
    targets = torch.randint(0, 10, (512,))
    ds = TensorDataset(images, targets)
    sampler = RandomBatchSampler(ds, batch_size=64, num_classes=10, seed=0)
    model = TinyCNN_MNIST_4K()
    problem = ERMCrossEntropyProblem(
        model, sampler, eval_mode="single", eval_batches=1, device=torch.device("cpu")
    )

    x0, y0 = problem.eval_batch_pool[0]
    problem.sample_eval_pool()
    x1, y1 = problem.eval_batch_pool[0]

    assert x0.shape == x1.shape == (64, 1, 28, 28)
    # Same CRN batch for all individuals; different across sampler calls.
    assert not torch.equal(x0, x1) or not torch.equal(y0, y1)


def test_eval_mode_single_uses_one_batch():
    images = torch.randn(256, 1, 28, 28)
    targets = torch.randint(0, 10, (256,))
    ds = TensorDataset(images, targets)
    sampler = RandomBatchSampler(ds, batch_size=32, num_classes=10, seed=1)
    model = TinyCNN_MNIST_4K()
    problem = ERMCrossEntropyProblem(
        model, sampler, eval_mode="single", eval_batches=50, device=torch.device("cpu")
    )
    assert problem.eval_batches == 1
    assert len(problem.eval_batch_pool) == 1

    problem_multi = ERMCrossEntropyProblem(
        model,
        RandomBatchSampler(ds, batch_size=32, num_classes=10, seed=2),
        eval_mode="multi",
        eval_batches=3,
        device=torch.device("cpu"),
    )
    assert problem_multi.eval_batches == 3
    assert len(problem_multi.eval_batch_pool) == 3
