# EvoTinyML

Multi-objective evolutionary training of tiny CNNs with **NSGA-II / NSGA-III** ([pymoo](https://pymoo.org/)).

Weights are optimized directly in parameter space (no gradients) on MNIST or CIFAR-10, with optional [Weights & Biases](https://wandb.ai/) logging.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Quick start

```bash
python3 train.py --dataset mnist --problem soft_precision_recall --algo nsga2 --init gaussian --device mps --verbose --wandb
```

SGD baseline:

```bash
python3 train_sgd.py --dataset mnist --device mps
```

## Problems

| `--problem` | Objectives |
|-------------|------------|
| `per_class_ce` | Per-class cross-entropy (10-obj) |
| `precision_recall` | Hard macro precision / recall → minimize `(1−P, 1−R)` |
| `soft_precision_recall` | Soft (softmax) macro P / R → minimize `(1−P, 1−R)` |

## Operators

- **Init:** `uniform`, `gaussian`, `both`
- **Crossover:** `sbx`, `none`
- **Mutation:** `pm`, `gaussian` (absolute σ), `layerwise` (He fan-in scaled σ)

Example with weight-space mutation rates:

```bash
python3 train.py --dataset mnist --problem soft_precision_recall --algo nsga2 --init gaussian --init-sigma 0.1 --crossover sbx --crossover-prob 0.1 --mutation gaussian --mutation-prob 0.9 --mutation-sigma 0.1 --mutation-prob-var 0.2 --popsize 100 --batch-size 1024 --eval-mode multi --eval-batches 1 --resample-every 0 --val-every 50 --pareto-every 100 --device mps --verbose --wandb
```

## Outputs

- **W&B:** train/val scalars (knee metrics, HV, front extremes)
- **`history.npz`** (every `--pareto-every` steps): train Pareto front snapshots (`steps`, `n_nds`, `F`, `precision`, `recall`)
- **`--out path.npz`:** final ND weight vectors + objectives

## Model

`TinyCNN` (~1.4k params on MNIST):

```
Conv(→8, 3×3, s=2) → Act → Conv(→16, 3×3, s=2) → Act → GAP → Linear(→classes)
```

## License

Research code — use at your own risk.
