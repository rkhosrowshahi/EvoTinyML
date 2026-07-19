# EvoTinyML

Evolutionary training of tiny CNNs with **NSGA-II / NSGA-III** ([pymoo](https://pymoo.org/)) multi-objective search, or **CMA-ES / SNES / xNES / OpenES** ([evosax](https://github.com/RobertTLange/evosax)) single-objective search.

Weights are optimized directly in parameter space (no gradients) on MNIST,
MNIST-2class (digits 0/1), or CIFAR-10, with optional [Weights & Biases](https://wandb.ai/) logging.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Quick start

```bash
# MOO (soft precision–recall)
python3 train.py --dataset mnist --problem soft_precision_recall --algo nsga2 --init gaussian --device mps --verbose --wandb

# SOO (ERM cross-entropy; CMA-ES / SNES / xNES / OpenES)
python3 train.py --dataset mnist --problem erm_cross_entropy --algo cmaes --init zeros --evals 250000 --device mps --verbose --wandb
python3 train.py --dataset mnist --problem erm_cross_entropy --algo snes --init zeros --evals 250000 --device mps --verbose --wandb
```

SGD baseline:

```bash
python3 train_sgd.py --dataset mnist --device mps
```

## Problems

| `--problem` | Type | Objectives |
|-------------|------|------------|
| `cwrm_cross_entropy` | MOO | Class-wise risk minimization + CE (one obj per class; `n_obj = num_classes`, so `mnist_2cls` → 2) |
| `precision_recall` | MOO | Hard macro P/R → minimize `(1−P, 1−R)` |
| `soft_precision_recall` | MOO / SOO | Soft macro P/R → MOO vector `(1−P, 1−R)` or SOO sum `(1−P)+(1−R)` |
| `erm_cross_entropy` | SOO | ERM mean CE on the eval pool (SOO ES only) |

Aliases: `cwce` / `per_class_ce` → `cwrm_cross_entropy`; `cross_entropy` → `erm_cross_entropy`.

## Algorithms

| `--algo` | Fitness | Notes |
|----------|---------|-------|
| `nsga2` / `nsga3` | MOO vector | pymoo; SBX / mutation apply |
| `cmaes` / `snes` / `xnes` / `open_es` | SOO scalar | evosax; `soft_precision_recall` or `erm_cross_entropy`; operators ignored; default `--popsize` = `4+3·ln(n_var)` (~25 for TinyCNN; even for OpenES) |
| `mgda_open_es` / `moead_open_es` | MOO vector | Multi-objective OpenES (see below); default problem `cwrm_cross_entropy`; operators ignored |

SOO validates and saves the **distribution mean** (not the best population sample).

## Multi-objective OpenES

Both algorithms keep the OpenES machinery (antithetic sampling, centered-rank
shaping, Optax mean update via `--es-optim*`) but consume the full objective
vector (e.g. one CE per class for `cwrm_cross_entropy`). Every evaluated
candidate feeds a non-dominated **archive** (`--archive-size`, pruned with
`--archive-selection nsga2|nsga3`), which is what gets validated on the test
set and saved with `--out`.

- **`mgda_open_es`** — single mean. Per-objective ES gradients are estimated
  from the same samples and combined into a common descent direction with
  MGDA (min-norm point in the convex hull; converges to one Pareto-stationary
  solution). Diagnostics: `train/mgda_dir_norm`, `train/mgda_w_max/min`.
- **`moead_open_es`** — `--moead-k` means, each tied to a weight vector on the
  objective simplex (`--ref-dirs`) and optimized on its scalarized subproblem
  (`--moead-scalarization tchebycheff|weighted_sum`, ideal point
  `--moead-ideal zero|adaptive`, augmentation `--moead-rho`). Weight vectors
  are shrunk toward uniform with `--moead-weight-shrink` (raw simplex corners
  ask for single-class specialists). `--moead-migrate-every N` restarts means
  at their best archive point. `--popsize` is the **total** evals per
  generation, split evenly (and forced even) across means.

```bash
python3 train.py --dataset mnist --algo mgda_open_es --init gaussian --evals 250000 \
  --es-optim sgd --es-optim-lr 0.1 --es-optim-momentum 0.9 --device mps --verbose --wandb

python3 train.py --dataset mnist --algo moead_open_es --init gaussian --evals 250000 \
  --moead-k 10 --popsize 80 --archive-selection nsga3 \
  --es-optim sgd --es-optim-lr 0.1 --es-optim-momentum 0.9 --device mps --verbose --wandb
```

OpenES mean update (ignored for other SOO algos):

| Flag | Default | Notes |
|------|---------|-------|
| `--es-optim` | `sgd` | Optax: `sgd` / `adam` / `adamw` |
| `--es-optim-lr` | `1e-3` | Initial LR (scheduled if not constant) |
| `--es-optim-scheduler` | `constant` | `constant` / `cosine` / `exponential` over `--steps` |
| `--es-optim-momentum` | `0` | SGD momentum (`0` = off; ignored for adam/adamw) |
| `--es-sigma-scheduler` | `constant` | Sampling noise σ schedule: `constant` / `cosine` / `exponential` (start = `--init-sigma`) |
| `--es-sigma-end` | `0.01·σ` | Final σ for cosine / exponential (ignored if constant) |

```bash
python3 train.py --dataset mnist --problem soft_precision_recall --algo cmaes --init zeros --evals 250000 --device mps --verbose --wandb
python3 train.py --dataset mnist --problem erm_cross_entropy --algo open_es --init zeros --evals 250000 \
  --es-optim sgd --es-optim-lr 1e-2 --es-optim-momentum 0.9 --es-optim-scheduler cosine \
  --es-sigma-scheduler cosine --device mps --verbose --wandb
```

## Budget: steps vs evals

One generation costs **`popsize`** train fitness calls (Function Evaluations).

| Flag | Meaning |
|------|---------|
| `--steps N` | Run `N` generations → `evals = N × popsize` |
| `--evals M` | Budget `M` FEs → `steps = M // popsize` (wins if both set) |

Examples (SOO ES default `popsize=25`): `--evals 250000` → 10 000 steps; `--steps 10000` → 250 000 evals.  
NSGA default `popsize=100`: `--evals 900000` → 9 000 steps.

Frozen 1024-sample train batch (never redraw):

```bash
--batch-size 1024 --eval-mode single --resample-every 0
```

## Operators (NSGA only)

- **Init:** `uniform`, `gaussian`, `both`, `zeros`, `kaiming` (ES center = PyTorch default / Kaiming-uniform `theta0`; NSGA: 1×`theta0` + `(popsize-1)×(theta0+N(0,σ))`; `--init-sigma` is also ES sampling std)
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
- MOO also logs HV / front extremes; SOO ES logs `train/f` (mean fitness) and `train/step`
- MOO Pareto plots (every `--pareto-every`): P–R → `train/pareto_front` /
  `val/pareto_front` (scatter); CWRM → train `*/pareto_radar` + `*/pareto_parallel`;
  val also logs CE and Acc fronts (`val/pareto_{radar|parallel}_{ce|acc}`)
  titled **Val Cross-entropy Pareto Front** / **Val Accuracy Pareto Front**

Compare NSGA vs a SOO ES run locally (script flag is still `--cma`):

```bash
python3 scripts/plot_wandb_compare.py --nsga <NSGA_RUN_ID> --cma <SOO_RUN_ID> --out plots/nsga2_vs_soo_val.png
```

## Outputs

- **W&B:** train/val scalars (see above) plus MOO Pareto images (`pareto_front` for P–R; `pareto_radar` + `pareto_parallel` for high-dim)
- **`train_history.npz` / `val_history.npz`:** MOO P–R Pareto history only (skipped for SOO ES)
- **`--out path.npz`:** final ND set (MOO) or SOO ES **mean** weights

## Model

`TinyCNN` (~1.4k params on MNIST):

```
Conv(→8, 3×3, s=2) → Act → Conv(→16, 3×3, s=2) → Act → GAP → Linear(→classes)
```

## License

Research code — use at your own risk.
