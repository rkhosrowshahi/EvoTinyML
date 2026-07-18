"""Plot NSGA-II vs CMA-ES metrics from W&B locally (x-axis = Function Evaluations).

Example:
  python3 scripts/plot_wandb_compare.py \
    --nsga mtph798q --cma xzqk9g5a --out plots/nsga2_vs_cmaes_val.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", default="rasa_research")
    p.add_argument("--project", default="EvoTinyML-Precision-Recall")
    p.add_argument("--nsga", required=True, help="NSGA run id")
    p.add_argument("--cma", required=True, help="CMA run id")
    p.add_argument("--samples", type=int, default=5000)
    p.add_argument("--out", type=str, default="plots/nsga2_vs_cmaes_val.png")
    return p.parse_args()


def _popsize(run) -> int:
    cfg = dict(run.config or {})
    pop = cfg.get("popsize")
    if pop is None and isinstance(cfg.get("algo_config"), dict):
        pop = cfg["algo_config"].get("popsize")
    return int(pop) if pop is not None else 1


def load_history(run, samples: int):
    """Load history and set ``_x`` = Function Evaluations.

    Prefer logged FE keys; otherwise convert opt ``step`` / ``_step`` via
    ``step * popsize`` (CMA λ / NSGA pop per generation).
    """
    df = run.history(samples=samples)
    pop = _popsize(run)
    fe_keys = ("Function Evaluations", "n_eval", "train/n_eval")
    for key in fe_keys:
        if key in df.columns and df[key].notna().any():
            df = df.sort_values(key)
            df["_x"] = df[key].astype(float)
            df["_xlabel"] = "Function Evaluations"
            df.attrs["popsize"] = pop
            df.attrs["fe_source"] = key
            return df

    step_col = None
    for cand in ("train/step", "step", "_step"):
        if cand in df.columns and df[cand].notna().any():
            step_col = cand
            break
    if step_col is not None:
        df = df.sort_values(step_col)
        # Old runs logged opt step; convert to FE = step * popsize.
        steps = df[step_col].astype(float)
        df["_x"] = steps * float(pop)
        df["_xlabel"] = "Function Evaluations"
        df.attrs["popsize"] = pop
        df.attrs["fe_source"] = f"{step_col}×{pop}"
        return df

    df["_x"] = np.arange(len(df), dtype=float)
    df["_xlabel"] = "index"
    df.attrs["popsize"] = pop
    df.attrs["fe_source"] = "index"
    return df


def series(df, *cands):
    for c in cands:
        if c in df.columns and df[c].notna().any():
            sub = df[["_x", c]].dropna()
            return c, sub["_x"].to_numpy(), sub[c].to_numpy()
    return None, None, None


def main() -> None:
    args = parse_args()
    api = wandb.Api()
    nsga = api.run(f"{args.entity}/{args.project}/{args.nsga}")
    cma = api.run(f"{args.entity}/{args.project}/{args.cma}")
    df_nsga = load_history(nsga, args.samples)
    df_cma = load_history(cma, args.samples)
    xlabel = "Function Evaluations"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fig.suptitle(
        "NSGA-II vs CMA-ES (x = Function Evaluations = step × popsize)\n"
        f"nsga2: {nsga.name} ({nsga.id}, pop={df_nsga.attrs.get('popsize')}, "
        f"{df_nsga.attrs.get('fe_source')})  |  "
        f"cmaes: {cma.name} ({cma.id}, pop={df_cma.attrs.get('popsize')}, "
        f"{df_cma.attrs.get('fe_source')})",
        fontsize=10,
    )

    ax = axes[0, 0]
    ax.set_title("val accuracy")
    for label, df, color in [("nsga2", df_nsga, "#1f77b4"), ("cmaes", df_cma, "#d62728")]:
        key, x, y = series(df, "val/knee_acc", "val/acc_best", "val/acc")
        if key is not None:
            ax.plot(x, y, label=f"{label} ({key})", color=color, lw=1.7)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.set_title("val F1")
    for label, df, color in [("nsga2", df_nsga, "#1f77b4"), ("cmaes", df_cma, "#d62728")]:
        key, x, y = series(df, "val/knee_f1", "val/f1_best", "val/f1")
        if key is not None:
            ax.plot(x, y, label=f"{label} ({key})", color=color, lw=1.7)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.set_title("train fitness proxy (lower better)")
    key, x, y = series(df_cma, "train/mean_f", "train/f", "train/best_f")
    if key:
        ax.plot(x, y, label=f"cmaes ({key})", color="#d62728", lw=1.7)
    if "train/f" in df_nsga.columns and df_nsga["train/f"].notna().any():
        sub = df_nsga[["_x", "train/f"]].dropna()
        ax.plot(sub["_x"], sub["train/f"], label="nsga2 (train/f)", color="#1f77b4", lw=1.7)
    elif "train/pf_pr_mean_max" in df_nsga.columns:
        sub = df_nsga[["_x", "train/pf_pr_mean_max"]].dropna()
        ax.plot(
            sub["_x"],
            2.0 * (1.0 - sub["train/pf_pr_mean_max"]),
            label="nsga2 2*(1-pf_pr_mean_max)",
            color="#1f77b4",
            lw=1.7,
        )
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.set_title("val P / R")
    for label, df, color in [("nsga2", df_nsga, "#1f77b4"), ("cmaes", df_cma, "#d62728")]:
        pairs = (
            [("val/knee_precision", "-"), ("val/knee_recall", "--")]
            if label == "nsga2"
            else [
                ("val/precision", "-"),
                ("val/recall", "--"),
                ("val/knee_precision", "-"),
                ("val/knee_recall", "--"),
            ]
        )
        seen = set()
        for metric, ls in pairs:
            short = "P" if "precision" in metric else "R"
            if short in seen:
                continue
            if metric not in df.columns or not df[metric].notna().any():
                continue
            sub = df[["_x", metric]].dropna()
            ax.plot(
                sub["_x"],
                sub[metric],
                label=f"{label} {short}",
                color=color,
                ls=ls,
                lw=1.4,
            )
            seen.add(short)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Shared x-limit so FE budgets are visually comparable.
    xmax = 0.0
    for df in (df_nsga, df_cma):
        if "_x" in df.columns and df["_x"].notna().any():
            xmax = max(xmax, float(np.nanmax(df["_x"].to_numpy())))
    if xmax > 0:
        for ax in axes.ravel():
            ax.set_xlim(0, xmax * 1.02)

    fig.savefig(out, dpi=150)
    print(f"saved {out.resolve()}")
    print(
        f"nsga FE source={df_nsga.attrs.get('fe_source')} "
        f"max_x={df_nsga['_x'].max() if '_x' in df_nsga else None}"
    )
    print(
        f"cma  FE source={df_cma.attrs.get('fe_source')} "
        f"max_x={df_cma['_x'].max() if '_x' in df_cma else None}"
    )


if __name__ == "__main__":
    main()
