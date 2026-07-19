"""Visualize a multi-objective Pareto front as a radar (spider) chart.

Each non-dominated solution becomes one closed polygon whose radius on each
spoke encodes how good that solution is on that objective. Because objectives
usually live on different scales (and some are minimized, some maximized),
the core trick is a per-objective normalization that maps every axis to a
common [0, 1] "goodness" scale where OUTER = BETTER.

Example (MGDA-OpenES class-wise CE archive):
  python3 scripts/plot_pareto_radar.py \\
    --npz results/mgda_front.npz --out figures/mgda_pareto_radar.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


def normalize(front: np.ndarray, directions: list[str]) -> np.ndarray:
    """Map each objective to [0, 1] with 1.0 = best seen, 0.0 = worst seen."""
    lo = front.min(axis=0)
    hi = front.max(axis=0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    scaled = (front - lo) / span
    for j, d in enumerate(directions):
        if d == "min":
            scaled[:, j] = 1.0 - scaled[:, j]
    return scaled


def thin_front(front: np.ndarray, max_n: int) -> np.ndarray:
    if len(front) <= max_n:
        return front
    idx = np.linspace(0, len(front) - 1, max_n).astype(int)
    return front[idx]


def radar_pareto(
    front: np.ndarray,
    labels: list[str],
    directions: list[str],
    color_by: np.ndarray | None = None,
    color_label: str = "",
    title: str = "Pareto front  (outer = better on every axis)",
    highlight: np.ndarray | None = None,
    highlight_label: str = "MGDA center",
):
    n_obj = front.shape[1]
    # Include highlight in the normalization pool so the center sits on the
    # same [0, 1] goodness scale as the archive.
    pool = front if highlight is None else np.vstack([front, highlight[None, :]])
    pool_norm = normalize(pool, directions)
    norm = pool_norm[: len(front)]
    highlight_norm = None if highlight is None else pool_norm[-1]

    angles = np.linspace(0, 2 * np.pi, n_obj, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    if color_by is None:
        color_by = norm.mean(axis=1)
        color_label = color_label or "mean goodness"
    cmap = cm.viridis
    cnorm = Normalize(vmin=float(color_by.min()), vmax=float(color_by.max()))

    # Draw worse solutions first so better ones sit on top.
    order = np.argsort(color_by)
    for i in order:
        row = norm[i]
        vals = row.tolist() + row[:1].tolist()
        c = cmap(cnorm(color_by[i]))
        ax.plot(angles, vals, color=c, linewidth=1.8, alpha=0.9)
        ax.fill(angles, vals, color=c, alpha=0.08)

    if highlight_norm is not None:
        vals = highlight_norm.tolist() + highlight_norm[:1].tolist()
        ax.plot(
            angles,
            vals,
            color="crimson",
            linewidth=2.6,
            marker="o",
            markersize=4,
            zorder=10,
            label=highlight_label,
        )
        ax.fill(angles, vals, color="crimson", alpha=0.12, zorder=9)
        ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.08), fontsize=9)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [f"{l}\n({d})" for l, d in zip(labels, directions)], fontsize=11
    )
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "best"], fontsize=8, color="grey")
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=14, pad=24)

    sm = cm.ScalarMappable(cmap=cmap, norm=cnorm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.75)
    cbar.set_label(color_label, fontsize=10)

    fig.tight_layout()
    return fig


def load_front(path: Path, *, use_acc: bool) -> tuple[np.ndarray, np.ndarray | None, list[str], list[str]]:
    data = np.load(path, allow_pickle=True)
    if use_acc:
        if "per_class_acc" not in data:
            raise SystemExit(
                f"{path} has no per_class_acc; pass results/mgda_front_acc.npz "
                "or omit --acc."
            )
        front = np.asarray(data["per_class_acc"], dtype=float)
        highlight = (
            np.asarray(data["center_per_class_acc"], dtype=float)
            if "center_per_class_acc" in data
            else None
        )
        directions = ["max"] * front.shape[1]
        kind = "per-class accuracy"
    else:
        front = np.asarray(data["F"], dtype=float)
        highlight = (
            np.asarray(data["means_F"][0], dtype=float) if "means_F" in data else None
        )
        directions = ["min"] * front.shape[1]
        kind = "class-wise CE"

    labels = [f"c{j}" for j in range(front.shape[1])]
    print(f"Loaded {len(front)} solutions × {front.shape[1]} objs ({kind}) from {path}")
    return front, highlight, labels, directions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--npz",
        type=Path,
        default=Path("results/mgda_front.npz"),
        help="Archive npz from --out (or mgda_front_acc.npz with --acc).",
    )
    p.add_argument(
        "--acc",
        action="store_true",
        help="Plot per-class accuracy (max) instead of class-wise CE (min).",
    )
    p.add_argument(
        "--max-n",
        type=int,
        default=0,
        help="Thin the front to this many polygons (0 = keep all).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("figures/mgda_pareto_radar.png"),
    )
    p.add_argument("--no-center", action="store_true", help="Hide MGDA center overlay.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    front, highlight, labels, directions = load_front(args.npz, use_acc=args.acc)
    if args.max_n > 0:
        # Keep color ranking on the full set before thinning for display.
        pref_full = normalize(front, directions).mean(axis=1)
        order = np.argsort(pref_full)
        # Prefer a spread of ranks rather than contiguous indices.
        keep = order[np.linspace(0, len(order) - 1, args.max_n).astype(int)]
        front = front[keep]

    pool = front if highlight is None else np.vstack([front, highlight[None, :]])
    pref = normalize(pool, directions)[: len(front)].mean(axis=1)

    kind = "accuracy" if args.acc else "CE"
    title = f"MGDA-OpenES Pareto archive  ({kind}; outer = better)"
    fig = radar_pareto(
        front,
        labels,
        directions,
        color_by=pref,
        color_label="mean goodness",
        title=title,
        highlight=None if args.no_center else highlight,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Plotted {len(front)} non-dominated solutions → {args.out}")


if __name__ == "__main__":
    main()
