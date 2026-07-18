# EvoTinyML

Evolutionary training of tiny CNNs with **NSGA-II / NSGA-III** ([pymoo](https://pymoo.org/)) multi-objective search, or **CMA-ES** ([evosax](https://github.com/RobertTLange/evosax)) single-objective search.

Weights are optimized directly in parameter space (no gradients) on MNIST or CIFAR-10, with optional [Weights & Biases](https://wandb.ai/) logging.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Quick start

```bash
# MOO (soft precision–recall)
python3 train.py --dataset mnist --problem soft_precision_recall --algo nsga2 --init gaussian --device mps --verbose --wandb

# SOO CMA-ES (ERM cross-entropy)
python3 train.py --dataset mnist --problem erm_cross_entropy --algo cmaes --init zeros --evals 250000 --device mps --verbose --wandb
```

SGD baseline:

```bash
python3 train_sgd.py --dataset mnist --device mps
```

## Problems

| `--problem` | Type | Objectives |
|-------------|------|------------|
| `cwrm_cross_entropy` | MOO | Class-wise risk minimization + CE (one obj per class) |
| `precision_recall` | MOO | Hard macro P/R → minimize `(1−P, 1−R)` |
| `soft_precision_recall` | MOO / SOO | Soft macro P/R → MOO vector `(1−P, 1−R)` or CMA sum `(1−P)+(1−R)` |
| `erm_cross_entropy` | SOO | ERM mean CE on the eval pool (CMA only) |

Aliases: `cwce` / `per_class_ce` → `cwrm_cross_entropy`; `cross_entropy` → `erm_cross_entropy`.

## Algorithms

| `--algo` | Fitness | Notes |
|----------|---------|-------|
| `nsga2` / `nsga3` | MOO vector | pymoo; SBX / mutation apply |
| `cmaes` | SOO scalar | evosax; `soft_precision_recall` or `erm_cross_entropy`; operators ignored; default `--popsize` = `4+3·ln(n_var)` (~25 for TinyCNN) |

CMA validates and saves the **distribution mean** (not the best population sample).

```bash
python3 train.py --dataset mnist --problem soft_precision_recall --algo cmaes --init zeros --evals 250000 --device mps --verbose --wandb
python3 train.py --dataset mnist --problem erm_cross_entropy --algo cmaes --init zeros --evals 250000 --device mps --verbose --wandb
```

## Budget: steps vs evals

One generation costs **`popsize`** train fitness calls (Function Evaluations).

| Flag | Meaning |
|------|---------|
| `--steps N` | Run `N` generations → `evals = N × popsize` |
| `--evals M` | Budget `M` FEs → `steps = M // popsize` (wins if both set) |

Examples (CMA `popsize=25`): `--evals 250000` → 10 000 steps; `--steps 10000` → 250 000 evals.  
NSGA default `popsize=100`: `--evals 900000` → 9 000 steps.

Frozen 1024-sample train batch (never redraw):

```bash
--batch-size 1024 --eval-mode single --resample-every 0
```

## Operators (NSGA only)

- **Init:** `uniform`, `gaussian`, `both`, `zeros` (CMA uses the same scheme for the initial mean; `--init-sigma` is also CMA `std_init`)
- **Crossover:** `sbx`, `none`
- **Mutation:** `pm`, `gaussian` (absolute σ), `layerwise` (He fan-in scaled σ)

```bash
python3 train.py --dataset mnist --problem soft_precision_recall --algo nsga2 --init gaussian --init-sigma 0.1 --crossover sbx --crossover-prob 0.1 --mutation gaussian --mutation-prob 0.9 --mutation-sigma 0.1 --mutation-prob-var 0.2 --popsize 100 --batch-size 1024 --eval-mode single --resample-every 0 --val-every 50 --pareto-every 100 --evals 900000 --device mps --verbose --wandb
```

## W&B

```bash
--wandb / --no-wandb
--wandb-entity ENTITY
--wandb-project PROJECT
--wandb-name RUN_NAME          # default: {dataset}-{algo}-seed{seed}
```

- X-axis: **`Function Evaluations`** (`step × popsize`)
- Shared val keys for overlays: `val/acc`, `val/f1`, `val/knee_acc`, `val/acc_best`, …
- MOO also logs HV / front extremes; CMA logs `train/f` (mean fitness) and `train/step`

Compare runs locally:

```bash
python3 scripts/plot_wandb_compare.py --nsga <NSGA_RUN_ID> --cma <CMA_RUN_ID> --out plots/nsga2_vs_cmaes_val.png
```

## Outputs

- **W&B:** train/val scalars (see above)
- **`train_history.npz` / `val_history.npz`:** MOO P–R Pareto history only (skipped for CMA)
- **`--out path.npz`:** final ND set (MOO) or CMA **mean** weights

## Model

`TinyCNN` (~1.4k params on MNIST):

```
Conv(→8, 3×3, s=2) → Act → Conv(→16, 3×3, s=2) → Act → GAP → Linear(→classes)
```

## License

Research code — use at your own risk.
