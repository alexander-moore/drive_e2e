"""
Trajectory visualization for E2E driving experiments.

  plot_trajectory       — single sample: past (blue), GT future (green), pred future (red)
  plot_trajectory_batch — grid of the above for a batch
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch


def plot_trajectory(
    past_traj: np.ndarray,
    future_traj_gt: np.ndarray,
    future_traj_pred: np.ndarray | None = None,
    title: str = "",
    save_path: str | None = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot ego trajectories in the ego coordinate frame (x=forward, y=left).

    Args:
        past_traj:        (T_past, 2)   — history including anchor frame
        future_traj_gt:   (T_future, 2) — ground-truth future waypoints
        future_traj_pred: (T_future, 2) — model prediction (optional)
        title:            plot title string
        save_path:        if given, saves figure to this path
        show:             if True, calls plt.show()

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(6, 8))

    # Past trajectory — blue dots, fading older → newer
    T_past = len(past_traj)
    alphas = np.linspace(0.2, 1.0, T_past)
    for pt, a in zip(past_traj, alphas):
        ax.scatter(pt[1], pt[0], color="royalblue", alpha=a, s=18, zorder=3)
    ax.plot(past_traj[:, 1], past_traj[:, 0],
            color="royalblue", alpha=0.4, linewidth=1, zorder=2)

    # Ground-truth future — green dots
    gt_alphas = np.linspace(0.4, 1.0, len(future_traj_gt))
    for pt, a in zip(future_traj_gt, gt_alphas):
        ax.scatter(pt[1], pt[0], color="seagreen", alpha=a, s=18, zorder=3)
    ax.plot(future_traj_gt[:, 1], future_traj_gt[:, 0],
            color="seagreen", alpha=0.5, linewidth=1, zorder=2)

    # Predicted future — red dots
    if future_traj_pred is not None:
        pred_alphas = np.linspace(0.4, 1.0, len(future_traj_pred))
        for pt, a in zip(future_traj_pred, pred_alphas):
            ax.scatter(pt[1], pt[0], color="tomato", alpha=a, s=18, zorder=3)
        ax.plot(future_traj_pred[:, 1], future_traj_pred[:, 0],
                color="tomato", alpha=0.5, linewidth=1, zorder=2)

    # Anchor marker + heading arrow (forward = +x ego = up on plot)
    ax.scatter(0, 0, color="black", s=80, marker="*", zorder=5)
    ax.annotate("", xy=(0, 2), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    legend_items = [
        Line2D([0], [0], color="royalblue", marker="o", markersize=5, label="past"),
        Line2D([0], [0], color="seagreen",  marker="o", markersize=5, label="future (GT)"),
    ]
    if future_traj_pred is not None:
        legend_items.append(
            Line2D([0], [0], color="tomato", marker="o", markersize=5, label="future (pred)")
        )
    ax.legend(handles=legend_items, loc="upper right", fontsize=8)

    ax.set_xlabel("y  (left +)")
    ax.set_ylabel("x  (forward +)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=8, wrap=True)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_trajectory_batch(
    past_trajs: np.ndarray,
    future_trajs_gt: np.ndarray,
    future_trajs_pred: np.ndarray | None = None,
    titles: list[str] | None = None,
    max_cols: int = 4,
    save_path: str | None = None,
    show: bool = False,
) -> plt.Figure:
    """
    Grid of trajectory plots for a batch of samples.

    Args:
        past_trajs:        (B, T_past, 2)
        future_trajs_gt:   (B, T_future, 2)
        future_trajs_pred: (B, T_future, 2) or None
        max_cols:          columns in the grid
        save_path:         optional save path
        show:              call plt.show()
    """
    B = len(past_trajs)
    ncols = min(B, max_cols)
    nrows = (B + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 5))
    axes = np.array(axes).reshape(nrows, ncols)

    for idx in range(nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        if idx >= B:
            ax.axis("off")
            continue

        past = past_trajs[idx]
        gt   = future_trajs_gt[idx]
        pred = future_trajs_pred[idx] if future_trajs_pred is not None else None

        for pt, a in zip(past, np.linspace(0.2, 1.0, len(past))):
            ax.scatter(pt[1], pt[0], color="royalblue", alpha=a, s=12)
        ax.plot(past[:, 1], past[:, 0], color="royalblue", alpha=0.3, lw=1)

        for pt, a in zip(gt, np.linspace(0.4, 1.0, len(gt))):
            ax.scatter(pt[1], pt[0], color="seagreen", alpha=a, s=12)
        ax.plot(gt[:, 1], gt[:, 0], color="seagreen", alpha=0.4, lw=1)

        if pred is not None:
            for pt, a in zip(pred, np.linspace(0.4, 1.0, len(pred))):
                ax.scatter(pt[1], pt[0], color="tomato", alpha=a, s=12)
            ax.plot(pred[:, 1], pred[:, 0], color="tomato", alpha=0.4, lw=1)

        ax.scatter(0, 0, color="black", s=60, marker="*")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        label = titles[idx] if titles and idx < len(titles) else f"sample {idx}"
        ax.set_title(label, fontsize=7)
        ax.tick_params(labelsize=6)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig
