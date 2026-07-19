"""Pareto-front figures for MOO runs (logged to W&B as images)."""

from __future__ import annotations

from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from evotinyml.problem import PR_PROBLEMS


def _knee_index_minimize(F: np.ndarray) -> int:
    F = np.asarray(F, dtype=float)
    if len(F) == 0:
        return 0
    return int(np.argmin(np.sum(np.square(F), axis=1)))


def _normalize_goodness(front: np.ndarray, directions: Sequence[str]) -> np.ndarray:
    """Map each objective to [0, 1] with 1 = best seen on that axis."""
    front = np.asarray(front, dtype=float)
    lo = front.min(axis=0)
    hi = front.max(axis=0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    scaled = (front - lo) / span
    for j, d in enumerate(directions):
        if d == "min":
            scaled[:, j] = 1.0 - scaled[:, j]
    return scaled


def figure_precision_recall_front(
    precision: np.ndarray,
    recall: np.ndarray,
    *,
    step: int,
    history: Sequence[tuple[int, np.ndarray, np.ndarray]] | None = None,
    title_prefix: str = "Train",
) -> Figure:
    """2D precision–recall ND front (matches the usual P–R scatter style)."""
    p = np.asarray(precision, dtype=float).ravel()
    r = np.asarray(recall, dtype=float).ravel()
    order = np.argsort(p)
    p, r = p[order], r[order]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    hist = list(history or [])
    # Drop duplicate of the current step if history already includes it.
    hist = [(s, hp, hr) for s, hp, hr in hist if int(s) != int(step)]
    if hist:
        cmap = cm.Blues
        norms = Normalize(vmin=0, vmax=max(len(hist) - 1, 1))
        for i, (s, hp, hr) in enumerate(hist):
            hp = np.asarray(hp, dtype=float).ravel()
            hr = np.asarray(hr, dtype=float).ravel()
            o = np.argsort(hp)
            color = cmap(0.25 + 0.65 * norms(i))
            ax.scatter(
                hp[o],
                hr[o],
                s=14,
                c=[color],
                alpha=0.55,
                label=f"step {int(s)} (n={len(hp)})",
                zorder=2,
            )

    ax.plot(p, r, color="0.55", linewidth=1.0, zorder=3)
    ax.scatter(p, r, s=22, c="black", zorder=4, label=f"step {int(step)} (n={len(p)})")

    if len(p) > 0:
        knee = int(np.argmin((1.0 - p) ** 2 + (1.0 - r) ** 2))
        ax.scatter(
            [p[knee]],
            [r[knee]],
            s=120,
            facecolors="none",
            edgecolors="crimson",
            linewidths=2.0,
            zorder=5,
            label=f"knee (P={p[knee]:.3f}, R={r[knee]:.3f})",
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("macro precision")
    ax.set_ylabel("macro recall")
    ax.set_title(f"{title_prefix} Pareto front (step {int(step)})")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


def figure_objective_front_2d(
    F: np.ndarray,
    *,
    step: int,
    x_label: str = "f1",
    y_label: str = "f2",
    title_prefix: str = "Train",
) -> Figure:
    """Generic 2-objective minimization scatter of the ND front."""
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[1] < 2:
        raise ValueError(f"Expected F with shape (n, >=2), got {F.shape}")
    x, y = F[:, 0], F[:, 1]
    order = np.argsort(x)
    x, y = x[order], y[order]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(x, y, color="0.55", linewidth=1.0, zorder=3)
    ax.scatter(x, y, s=22, c="black", zorder=4, label=f"ND (n={len(x)})")
    if len(F) > 0:
        knee = _knee_index_minimize(F)
        ax.scatter(
            [F[knee, 0]],
            [F[knee, 1]],
            s=120,
            facecolors="none",
            edgecolors="crimson",
            linewidths=2.0,
            zorder=5,
            label=f"knee ({F[knee, 0]:.3f}, {F[knee, 1]:.3f})",
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{title_prefix} Pareto front (step {int(step)})")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


def _prepare_normalized_front(
    F: np.ndarray,
    *,
    directions: Sequence[str],
    highlight: np.ndarray | None,
    max_n: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Thin front, normalize to goodness, optionally append highlight.

    Returns ``(norm, highlight_norm, color_by)`` where rows of ``norm`` are
    solutions and values are in [0, 1] with 1 = better.
    """
    F = np.asarray(F, dtype=float)
    if len(F) > max_n:
        pref = _normalize_goodness(F, directions).mean(axis=1)
        keep = np.argsort(pref)[-max_n:]
        F_plot = F[keep]
    else:
        F_plot = F

    pool = F_plot if highlight is None else np.vstack([F_plot, highlight.ravel()[None, :]])
    pool_norm = _normalize_goodness(pool, directions)
    norm = pool_norm[: len(F_plot)]
    highlight_norm = None if highlight is None else pool_norm[-1]
    color_by = norm.mean(axis=1)
    return norm, highlight_norm, color_by


def figure_radar_front(
    F: np.ndarray,
    *,
    step: int,
    labels: Sequence[str] | None = None,
    directions: Sequence[str] | None = None,
    title_prefix: str = "Train",
    highlight: np.ndarray | None = None,
    highlight_label: str = "center",
    max_n: int = 40,
) -> Figure:
    """Radar chart for n_obj > 2 (outer = better after per-axis normalization)."""
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[1] < 2:
        raise ValueError(f"Expected F with shape (n, >=2), got {F.shape}")
    n_obj = F.shape[1]
    if labels is None:
        labels = [f"f{j}" for j in range(n_obj)]
    if directions is None:
        directions = ["min"] * n_obj

    norm, highlight_norm, color_by = _prepare_normalized_front(
        F, directions=directions, highlight=highlight, max_n=max_n
    )

    angles = np.linspace(0, 2 * np.pi, n_obj, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    cmap = cm.viridis
    cnorm = Normalize(vmin=float(color_by.min()), vmax=float(color_by.max()))
    for i in np.argsort(color_by):
        vals = norm[i].tolist() + norm[i, :1].tolist()
        c = cmap(cnorm(color_by[i]))
        ax.plot(angles, vals, color=c, linewidth=1.6, alpha=0.85)
        ax.fill(angles, vals, color=c, alpha=0.06)

    if highlight_norm is not None:
        vals = highlight_norm.tolist() + highlight_norm[:1].tolist()
        ax.plot(
            angles,
            vals,
            color="crimson",
            linewidth=2.4,
            marker="o",
            markersize=4,
            zorder=10,
            label=highlight_label,
        )
        ax.fill(angles, vals, color="crimson", alpha=0.10, zorder=9)
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05), fontsize=9)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(list(labels), fontsize=10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "best"], fontsize=8, color="grey")
    ax.set_ylim(0, 1)
    ax.set_title(
        f"{title_prefix} Pareto radar (step {int(step)}; outer=better)",
        fontsize=13,
        pad=22,
    )
    sm = cm.ScalarMappable(cmap=cmap, norm=cnorm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.72)
    cbar.set_label("mean goodness", fontsize=10)
    fig.tight_layout()
    return fig


def figure_parallel_front(
    F: np.ndarray,
    *,
    step: int,
    labels: Sequence[str] | None = None,
    directions: Sequence[str] | None = None,
    title_prefix: str = "Train",
    highlight: np.ndarray | None = None,
    highlight_label: str = "center",
    max_n: int = 40,
) -> Figure:
    """Parallel-coordinates plot (top = better after per-axis normalization)."""
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[1] < 2:
        raise ValueError(f"Expected F with shape (n, >=2), got {F.shape}")
    n_obj = F.shape[1]
    if labels is None:
        labels = [f"f{j}" for j in range(n_obj)]
    if directions is None:
        directions = ["min"] * n_obj

    norm, highlight_norm, color_by = _prepare_normalized_front(
        F, directions=directions, highlight=highlight, max_n=max_n
    )
    xs = np.arange(n_obj)

    fig, ax = plt.subplots(figsize=(max(7.0, 0.7 * n_obj + 3.0), 5.5))
    cmap = cm.viridis
    cnorm = Normalize(vmin=float(color_by.min()), vmax=float(color_by.max()))
    for i in np.argsort(color_by):
        ax.plot(
            xs,
            norm[i],
            color=cmap(cnorm(color_by[i])),
            linewidth=1.4,
            alpha=0.75,
            zorder=2,
        )

    if highlight_norm is not None:
        ax.plot(
            xs,
            highlight_norm,
            color="crimson",
            linewidth=2.4,
            marker="o",
            markersize=5,
            zorder=10,
            label=highlight_label,
        )
        ax.legend(loc="upper right", fontsize=9)

    for x in xs:
        ax.axvline(x, color="0.85", linewidth=0.8, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(labels), fontsize=10)
    ax.set_xlim(-0.2, n_obj - 0.8)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("goodness (1 = best on axis)")
    ax.set_title(
        f"{title_prefix} Pareto parallel coords (step {int(step)}; top=better)",
        fontsize=13,
    )
    ax.grid(True, axis="y", alpha=0.35)
    sm = cm.ScalarMappable(cmap=cmap, norm=cnorm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("mean goodness", fontsize=10)
    fig.tight_layout()
    return fig


def figure_to_wandb_image(fig: Figure):
    """Convert a matplotlib figure to ``wandb.Image`` and close the figure."""
    import wandb

    image = wandb.Image(fig)
    plt.close(fig)
    return image


def pareto_front_images(
    F: np.ndarray,
    *,
    problem_name: str,
    step: int,
    key_prefix: str = "train",
    history: Sequence[tuple[int, np.ndarray, np.ndarray]] | None = None,
    highlight: np.ndarray | None = None,
    highlight_label: str = "center",
    obj_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build W&B image payload for the current ND objective matrix ``F``.

    - 2-obj P–R: ``{prefix}/pareto_front`` (precision–recall scatter)
    - 2-obj other: ``{prefix}/pareto_front`` (f1–f2 scatter)
    - n_obj > 2: ``{prefix}/pareto_radar`` and ``{prefix}/pareto_parallel``
    """
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or len(F) == 0 or F.shape[1] < 2:
        return {}

    if problem_name in PR_PROBLEMS:
        precision = 1.0 - F[:, 0]
        recall = 1.0 - F[:, 1]
        fig = figure_precision_recall_front(
            precision,
            recall,
            step=step,
            history=history,
            title_prefix=key_prefix.capitalize(),
        )
        return {f"{key_prefix}/pareto_front": figure_to_wandb_image(fig)}

    if F.shape[1] == 2:
        labels = list(obj_labels) if obj_labels is not None else ["f1", "f2"]
        fig = figure_objective_front_2d(
            F,
            step=step,
            x_label=labels[0],
            y_label=labels[1],
            title_prefix=key_prefix.capitalize(),
        )
        return {f"{key_prefix}/pareto_front": figure_to_wandb_image(fig)}

    labels = (
        list(obj_labels)
        if obj_labels is not None
        else [f"c{j}" for j in range(F.shape[1])]
    )
    directions = ["min"] * F.shape[1]
    title = key_prefix.capitalize()
    radar = figure_radar_front(
        F,
        step=step,
        labels=labels,
        directions=directions,
        title_prefix=title,
        highlight=highlight,
        highlight_label=highlight_label,
    )
    parallel = figure_parallel_front(
        F,
        step=step,
        labels=labels,
        directions=directions,
        title_prefix=title,
        highlight=highlight,
        highlight_label=highlight_label,
    )
    return {
        f"{key_prefix}/pareto_radar": figure_to_wandb_image(radar),
        f"{key_prefix}/pareto_parallel": figure_to_wandb_image(parallel),
    }
