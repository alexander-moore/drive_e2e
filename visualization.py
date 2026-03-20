"""
Trajectory visualization for E2E driving experiments.

  plot_trajectory            — single sample: past (blue), GT future (green), pred future (red)
  plot_trajectory_batch      — grid of the above for a batch
  plot_trajectory_on_image   — overlay GT + pred on front camera image via perspective projection
  plot_video_prediction      — input context + GT vs predicted future frames per camera
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import torch


# ---------------------------------------------------------------------------
# Bench2Drive front camera constants (fixed across all scenarios)
#   Original resolution : 1600 × 900
#   Intrinsics          : fx = fy = 1142.52,  cx = 800,  cy = 450
#   Camera in ego frame : 0.8 m forward,  0 m lateral,  1.6 m above ground
#   Camera orientation  : identity (aligned with ego: X=forward, Y=right, Z=up)
# ---------------------------------------------------------------------------
_FX = 1142.5184053936916
_CX = 800.0
_CY = 450.0
_CAM_FWD    = 0.8   # metres forward from ego origin
_CAM_HEIGHT = 1.6   # metres above ground
_ORIG_W, _ORIG_H = 1600, 900


def _project_ego_traj_to_pixels(
    traj: np.ndarray,   # (T, 2)  x=forward, y=left  [metres, ego frame]
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """
    Project ego-frame ground-plane waypoints onto front-camera pixels.

    Coordinate path:
      ego frame   (x=fwd, y=left, z=0 ground)
        →  CARLA frame  (X=fwd, Y=right, Z=up)   : negate y
        →  camera-relative CARLA (subtract camera position)
        →  OpenCV camera frame (X=right, Y=down, Z=depth)
        →  pixel  (u, v)  at original 1600×900 resolution
        →  scale to displayed image size

    Returns (T, 2) float32 array of (u, v) pixel coords.
    NaN for points behind the camera or outside the image.
    """
    x_f = traj[:, 0]               # forward   [m]
    y_l = traj[:, 1]               # left      [m]

    # Vector from camera to ground point, in CARLA frame (X=fwd, Y=right, Z=up)
    # Z is constant: ground (z=0) is _CAM_HEIGHT below the camera
    Z_cv = x_f - _CAM_FWD          # OpenCV depth  = CARLA forward
    X_cv = -y_l                     # OpenCV right  = CARLA right = -ego left
    Y_cv = np.full_like(x_f, _CAM_HEIGHT)  # OpenCV down = camera height above ground

    valid = Z_cv > 0.1
    u = np.full(len(traj), np.nan, dtype=np.float32)
    v = np.full(len(traj), np.nan, dtype=np.float32)

    u[valid] = _FX * (X_cv[valid] / Z_cv[valid]) + _CX
    v[valid] = _FX * (Y_cv[valid] / Z_cv[valid]) + _CY

    # Scale from original resolution to displayed image
    u *= img_w / _ORIG_W
    v *= img_h / _ORIG_H

    # Mask out-of-frame points
    in_frame = valid & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u[~in_frame] = np.nan
    v[~in_frame] = np.nan

    return np.stack([u, v], axis=1)


def plot_trajectory_on_image(
    image: np.ndarray,                          # (H, W, 3) uint8 RGB
    future_traj_gt: np.ndarray,                 # (T, 2)  x=fwd, y=left  [m]
    future_traj_pred: np.ndarray | None = None, # (T, 2)
    title: str = "",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Overlay future trajectory on a front-camera image.

    Ground-plane waypoints in the ego frame are perspective-projected onto
    the image using Bench2Drive's fixed front-camera intrinsics.  Green dots =
    ground truth, red dots = model prediction.
    """
    H, W = image.shape[:2]
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(image)
    ax.axis("off")

    def _draw(traj, color, label):
        px = _project_ego_traj_to_pixels(traj, H, W)
        valid = ~np.isnan(px[:, 0])
        if not valid.any():
            return
        alphas = np.linspace(0.5, 1.0, valid.sum())
        pts = px[valid]
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2, alpha=0.8, zorder=3)
        for (u, v), a in zip(pts, alphas):
            ax.scatter(u, v, color=color, s=25, alpha=a, zorder=4,
                       edgecolors="white", linewidths=0.5)

    _draw(future_traj_gt, "limegreen", "GT")
    if future_traj_pred is not None:
        _draw(future_traj_pred, "tomato", "pred")

    legend_items = [Line2D([0], [0], color="limegreen", marker="o", markersize=5, label="GT future")]
    if future_traj_pred is not None:
        legend_items.append(Line2D([0], [0], color="tomato", marker="o", markersize=5, label="pred future"))
    ax.legend(handles=legend_items, loc="upper right", fontsize=7,
              framealpha=0.6, facecolor="black", labelcolor="white")

    if title:
        ax.set_title(title, fontsize=7, pad=3)

    plt.tight_layout(pad=0.1)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fig


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


# ---------------------------------------------------------------------------
# Video prediction visualization
# ---------------------------------------------------------------------------

def plot_video_prediction(
    input_frames:  np.ndarray,            # (n_cams, n_frames, H, W, 3)  float32 [0,1] or uint8
    target_frames: np.ndarray,            # (n_cams, m_frames, H, W, 3)
    pred_frames:   np.ndarray,            # (n_cams, m_frames, H, W, 3)
    cam_names:     Optional[List[str]] = None,
    frame_stride:  int = 1,
    title:         str = "",
    save_path:     Optional[str] = None,
    psnr_val:      Optional[float] = None,
) -> plt.Figure:
    """
    Visualise video prediction: input context alongside GT and predicted future.

    Layout — two rows per camera:
      Row 2c+0 (GT)  : [input_1 … input_n | gt_1 … gt_m]
      Row 2c+1 (Pred): [input_1 … input_n | pred_1 … pred_m]

    Colour coding:
      Blue border   — input context frames
      Green border  — ground-truth future frames
      Red border    — predicted future frames

    Args:
        input_frames:  (n_cams, n_frames, H, W, 3)  float32 [0,1] or uint8
        target_frames: (n_cams, m_frames, H, W, 3)
        pred_frames:   (n_cams, m_frames, H, W, 3)
        cam_names:     list of n_cams camera label strings
        frame_stride:  temporal stride used during sampling (for axis labels)
        title:         overall figure suptitle
        save_path:     if given, saves the figure here
        psnr_val:      optional scalar PSNR to include in the title
    """
    n_cams, n_frames = input_frames.shape[:2]
    m_frames = target_frames.shape[1]
    n_cols   = n_frames + m_frames
    n_rows   = n_cams * 2          # GT row + Pred row per camera

    if cam_names is None:
        cam_names = [f"cam {c}" for c in range(n_cams)]

    def _to_float(arr: np.ndarray) -> np.ndarray:
        if arr.dtype == np.uint8:
            return arr.astype(np.float32) / 255.0
        return arr.clip(0.0, 1.0)

    inp = _to_float(input_frames)
    gt  = _to_float(target_frames)
    pr  = _to_float(pred_frames)

    H, W = inp.shape[2], inp.shape[3]
    cell_w = max(1.4, W / 100)
    cell_h = max(1.0, H / 100)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell_w, n_rows * cell_h + 0.6),
        squeeze=False,
    )

    # Column headers (written on row 0 only)
    for col in range(n_cols):
        if col < n_frames:
            t_offset = col - (n_frames - 1)
            lbl = f"t{t_offset:+d}" if t_offset != 0 else "t"
        else:
            t_offset = col - n_frames + 1
            lbl = f"t+{t_offset}"
        if frame_stride != 1:
            lbl += f"\n(×{frame_stride})"
        axes[0, col].set_title(lbl, fontsize=7, pad=3, color="#333333")

    # Per-row content
    for cam in range(n_cams):
        for row_offset, (fut_frames, row_label, fut_color) in enumerate([
            (gt[cam],  "GT",   "#3daf3d"),   # green
            (pr[cam],  "pred", "#d64040"),   # red
        ]):
            row = cam * 2 + row_offset

            for col in range(n_cols):
                ax = axes[row, col]
                ax.axis("off")

                img    = inp[cam][col] if col < n_frames else fut_frames[col - n_frames]
                border = "#4878CF" if col < n_frames else fut_color

                ax.imshow(img, interpolation="bilinear", aspect="auto")
                rect = mpatches.FancyBboxPatch(
                    (0.01, 0.01), 0.98, 0.98,
                    boxstyle="square,pad=0",
                    linewidth=2.5,
                    edgecolor=border,
                    facecolor="none",
                    transform=ax.transAxes,
                    clip_on=False,
                )
                ax.add_patch(rect)

            # Row label on leftmost cell
            axes[row, 0].set_ylabel(
                f"{cam_names[cam]}\n{row_label}",
                fontsize=7, labelpad=4, rotation=0,
                ha="right", va="center",
            )
            axes[row, 0].yaxis.set_visible(True)

    fig.subplots_adjust(wspace=0.04, hspace=0.08, left=0.12, right=0.99,
                        top=0.90, bottom=0.04)

    # Dashed separator line between context and future columns
    try:
        x_ctx = axes[0, n_frames - 1].get_position().x1
        x_fut = axes[0, n_frames].get_position().x0
        x_sep = (x_ctx + x_fut) / 2
        fig.add_artist(plt.Line2D(
            [x_sep, x_sep], [0.04, 0.91],
            transform=fig.transFigure,
            color="#aaaaaa", linewidth=1.0, linestyle="--", zorder=10,
        ))
        fig.text(x_sep * 0.55, 0.925, "◀  context",
                 ha="center", va="bottom", fontsize=7.5, color="#4878CF")
        fig.text(x_sep + (1.0 - x_sep) * 0.45, 0.925, "future  ▶",
                 ha="center", va="bottom", fontsize=7.5, color="#777777")
    except Exception:
        pass

    legend_handles = [
        mpatches.Patch(color="#4878CF", label="input context"),
        mpatches.Patch(color="#3daf3d", label="GT future"),
        mpatches.Patch(color="#d64040", label="predicted future"),
    ]
    fig.legend(handles=legend_handles, loc="lower right",
               fontsize=7, ncol=3, framealpha=0.7,
               bbox_to_anchor=(0.99, 0.0))

    full_title = title
    if psnr_val is not None:
        full_title += f"   PSNR = {psnr_val:.1f} dB"
    if full_title:
        fig.suptitle(full_title, fontsize=9, y=0.98)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fig
