# EvoTinyML

Train tiny neural networks with **evolutionary algorithms** (or gradient-based SGD/Adam for a baseline).

The network weights are the thing being optimized. Fitness is measured on MNIST or CIFAR-10.

## Setup

```bash
pip install -r requirements.txt
```

## Models

- **MNIST** — small CNN (~4k parameters)
- **CIFAR-10** — slightly deeper CNN (~34k parameters)

## Quick start

**OpenES**:

```bash
python train.py --dataset mnist --algo open_es --problem erm_cross_entropy \
  --init kaiming --init-sigma 0.05 --popsize 256 --steps 300 \
  --batch-size 256 --device gpu --no-wandb --verbose
```

**SGD / Adam** baseline:

```bash
python train_sgd.py --dataset mnist --optimizer sgd --device gpu
python train_sgd.py --dataset mnist --optimizer adam --device gpu
```

**Multi-objective** (e.g. NSGA-II):

```bash
python train.py --dataset mnist --algo nsga2 --problem cwrm_cross_entropy \
  --device gpu --no-wandb --verbose
```

## Common options

| Flag | Meaning |
|------|---------|
| `--dataset` | `mnist`, `mnist_2cls`, or `cifar10` |
| `--algo` | Search method (`open_es`, `nsga2`, `cmaes`, …) |
| `--problem` | What to optimize (`erm_cross_entropy`, `erm_f1`, …) |
| `--steps` | Number of generations |
| `--popsize` | Population size |
| `--batch-size` | Training batch size (new batch each generation) |
| `--device` | `cpu`, `mps`, `cuda`, or `gpu` |
| `--wandb` / `--no-wandb` | Log to Weights & Biases (on by default) |

Use `--help` on either script for the full list.
