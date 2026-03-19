"""
Shared PyTorch Lightning module for E2E driving experiments.

Any model that accepts a batch dict and returns (B, T_future, 2) future
waypoints can be plugged in here.  Training, validation, metric logging,
and trajectory visualization callbacks all live here so individual model
files stay focused on architecture only.

Usage:
    model  = MLPPlanner(...)
    module = E2EDrivingModule(model, lr=1e-3)
    trainer.fit(module, datamodule)
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy

from visualization import plot_trajectory, plot_trajectory_batch
try:
    from losses import SILogLoss, DiceLoss, abs_rel        # when run from e2e/ dir
except ImportError:
    from e2e.losses import SILogLoss, DiceLoss, abs_rel   # when imported as e2e package


# ---------------------------------------------------------------------------
# Metrics
#
# Standard for SOTA E2E planning (UniAD, VAD, SparseDrive, BEV-Planner, etc.):
#   L2 displacement error at fixed future horizons: 1 s, 2 s, 3 s
#   Primary checkpoint metric: avg_l2 = mean(l2_1s, l2_2s, l2_3s)   [lower = better]
#
# At 10 Hz, horizon indices (0-based) are: 1s→9, 2s→19, 3s→29
# ---------------------------------------------------------------------------

HORIZONS = {
    "1s": 9,   # index of the 1-second-ahead waypoint (10th frame, 0-indexed)
    "2s": 19,
    "3s": 29,
}


def l2_at_horizon(pred: torch.Tensor, gt: torch.Tensor, t: int) -> torch.Tensor:
    """L2 displacement error at a single future timestep index t.

    This is the metric used by UniAD / VAD / SparseDrive: the Euclidean
    distance between predicted and GT position at exactly t seconds ahead,
    averaged over the batch.

    Args:
        pred: (B, T, 2)
        gt:   (B, T, 2)
        t:    timestep index (0-based)
    Returns:
        scalar tensor  [metres]
    """
    return torch.norm(pred[:, t] - gt[:, t], dim=-1).mean()


def avg_l2(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean L2 across the standard 1 s / 2 s / 3 s horizons."""
    return torch.stack([l2_at_horizon(pred, gt, t) for t in HORIZONS.values()]).mean()


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class E2EDrivingModule(pl.LightningModule):
    """
    Shared Lightning wrapper for any E2E driving model.

    Contract for wrapped models:
        forward(batch: dict) -> predicted_future_traj (B, T_future, 2)

    Primary validation metric: val/avg_l2  (lower is better).
    Trajectory plots are saved only when a new best val/avg_l2 is achieved.

    Args:
        model:        Any nn.Module obeying the contract above.
        lr:           Base learning rate.
        weight_decay: AdamW weight decay.
        viz_samples:  Number of samples to plot when a new best is reached.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        viz_samples: int = 4,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.viz_samples = viz_samples

        self._best_avg_l2: float = float("inf")
        self._viz_buffer: list[dict] = []

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, batch: dict) -> torch.Tensor:
        return self.model(batch)

    # ── shared step ──────────────────────────────────────────────────────────

    def _step(self, batch: dict, stage: str):
        pred = self(batch)          # (B, T, 2)
        gt   = batch["future_traj"] # (B, T, 2)

        # Training loss: avg L2 across horizons (same scale as val metric)
        loss = avg_l2(pred, gt)

        on_step = stage == "train"
        self.log(f"{stage}/loss",     loss, on_step=on_step, on_epoch=True,
                 prog_bar=True,  sync_dist=True)
        self.log(f"{stage}/avg_l2",   loss, on_step=False,   on_epoch=True,
                 prog_bar=True,  sync_dist=True)

        for name, t in HORIZONS.items():
            self.log(f"{stage}/l2_{name}", l2_at_horizon(pred, gt, t),
                     on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss, pred

    # ── train / val steps ────────────────────────────────────────────────────

    def training_step(self, batch: dict, batch_idx: int):
        loss, _ = self._step(batch, "train")
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        loss, pred = self._step(batch, "val")

        # Buffer a few samples for potential visualization
        if len(self._viz_buffer) < self.viz_samples:
            n = min(self.viz_samples - len(self._viz_buffer), pred.shape[0])
            for i in range(n):
                self._viz_buffer.append({
                    "past_traj":   batch["past_traj"][i].cpu(),
                    "future_traj": batch["future_traj"][i].cpu(),
                    "pred_traj":   pred[i].detach().cpu(),
                    "scenario":    batch["scenario"][i] if "scenario" in batch else "",
                    "anchor_idx":  int(batch["anchor_idx"][i]) if "anchor_idx" in batch else -1,
                })
        return loss

    def on_validation_epoch_end(self):
        # Read the metric that was just logged (gathered across all devices)
        current = self.trainer.callback_metrics.get("val/avg_l2")
        if current is None:
            self._viz_buffer.clear()
            return

        current = current.item()
        is_best = current < self._best_avg_l2

        if is_best:
            self._best_avg_l2 = current
            self._save_viz()

        self._viz_buffer.clear()

    def _save_viz(self):
        if not self._viz_buffer:
            return
        log_dir = Path(self.trainer.log_dir) if self.trainer.log_dir else Path("logs/e2e")
        viz_dir = log_dir / "best_trajectories"
        viz_dir.mkdir(parents=True, exist_ok=True)

        for i, s in enumerate(self._viz_buffer):
            title = f"best  epoch {self.current_epoch}  {s['scenario']}  frame {s['anchor_idx']}"
            plot_trajectory(
                past_traj=s["past_traj"].numpy(),
                future_traj_gt=s["future_traj"].numpy(),
                future_traj_pred=s["pred_traj"].numpy(),
                title=title,
                save_path=str(viz_dir / f"sample{i:02d}.png"),
            )

        # Also save a batch grid
        plot_trajectory_batch(
            past_trajs=torch.stack([s["past_traj"]   for s in self._viz_buffer]).numpy(),
            future_trajs_gt=torch.stack([s["future_traj"] for s in self._viz_buffer]).numpy(),
            future_trajs_pred=torch.stack([s["pred_traj"]  for s in self._viz_buffer]).numpy(),
            save_path=str(viz_dir / "grid.png"),
        )

    # ── optimizer ────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=self.lr * 0.01
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ---------------------------------------------------------------------------
# Multi-task module — trajectory + auxiliary depth + semantic
# ---------------------------------------------------------------------------

class MultiTaskE2EModule(E2EDrivingModule):
    """
    E2E driving module with auxiliary depth and semantic segmentation losses.

    Expects the model's forward() to return a dict:
        {
            "future_traj": (B, 50, 2),
            "depth":       (B, C, 1, H, W)   — only when depth labels in batch
            "semantic":    (B, C, 28, H, W)  — only when semantic labels in batch
        }

    Auxiliary losses:
        depth:    depth_weight × SILogLoss
        semantic: sem_weight × (CrossEntropy + 0.5 × DiceLoss)

    Depth metrics logged per val step: abs_rel, silog, depth_mse
    Semantic metrics accumulated: mIoU, mAcc, per-class IoU (logged at epoch end)
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        viz_samples: int = 4,
        depth_weight: float = 0.1,
        sem_weight: float = 0.1,
    ):
        super().__init__(model, lr, weight_decay, viz_samples)
        self.depth_weight = depth_weight
        self.sem_weight = sem_weight

        self.silog_loss = SILogLoss()
        self.dice_loss = DiceLoss()
        self.val_miou = MulticlassJaccardIndex(num_classes=28, average="macro")
        self.val_macc = MulticlassAccuracy(num_classes=28, average="macro")
        self.val_iou_per_class = MulticlassJaccardIndex(num_classes=28, average="none")
        self._has_sem_val_data = False

    def forward(self, batch: dict):
        return self.model(batch)

    def _step(self, batch: dict, stage: str):
        output = self.model(batch)
        pred_traj = output["future_traj"]   # (B, 50, 2)
        gt_traj = batch["future_traj"]      # (B, 50, 2)

        loss_traj = avg_l2(pred_traj, gt_traj)
        loss = loss_traj

        on_step = stage == "train"

        # ── Depth auxiliary loss ──────────────────────────────────────────────
        loss_depth = torch.zeros(1, device=loss.device)[0]
        if "depth" in batch and "depth" in output:
            pred_d = output["depth"]        # (B, C, 1, H, W)
            gt_d = batch["depth"]           # (B, C, 1, H, W)
            pred_d_flat = pred_d.flatten(0, 1)   # (B*C, 1, H, W)
            gt_d_flat = gt_d.flatten(0, 1)
            loss_depth = self.silog_loss(pred_d_flat, gt_d_flat)
            loss = loss + self.depth_weight * loss_depth

        # ── Semantic auxiliary loss ───────────────────────────────────────────
        loss_sem = torch.zeros(1, device=loss.device)[0]
        if "semantic" in batch and "semantic" in output:
            pred_s = output["semantic"]     # (B, C, 28, H, W)
            gt_s = batch["semantic"]        # (B, C, H, W) int64
            pred_s_flat = pred_s.flatten(0, 1)   # (B*C, 28, H, W)
            gt_s_flat = gt_s.flatten(0, 1)       # (B*C, H, W)
            loss_ce = F.cross_entropy(pred_s_flat, gt_s_flat)
            loss_dice = self.dice_loss(pred_s_flat, gt_s_flat)
            loss_sem = loss_ce + 0.5 * loss_dice
            loss = loss + self.sem_weight * loss_sem

        self.log(f"{stage}/loss",       loss,       on_step=on_step, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log(f"{stage}/loss_traj",  loss_traj,  on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/loss_depth", loss_depth, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/loss_sem",   loss_sem,   on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/avg_l2",     loss_traj,  on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

        for name, t in HORIZONS.items():
            self.log(f"{stage}/l2_{name}", l2_at_horizon(pred_traj, gt_traj, t),
                     on_step=False, on_epoch=True, sync_dist=True)

        return loss, output

    def training_step(self, batch: dict, batch_idx: int):
        loss, _ = self._step(batch, "train")
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        loss, output = self._step(batch, "val")
        pred_traj = output["future_traj"]

        # Buffer samples for trajectory visualization
        if len(self._viz_buffer) < self.viz_samples:
            n = min(self.viz_samples - len(self._viz_buffer), pred_traj.shape[0])
            for i in range(n):
                self._viz_buffer.append({
                    "past_traj":   batch["past_traj"][i].cpu(),
                    "future_traj": batch["future_traj"][i].cpu(),
                    "pred_traj":   pred_traj[i].detach().cpu(),
                    "scenario":    batch["scenario"][i] if "scenario" in batch else "",
                    "anchor_idx":  int(batch["anchor_idx"][i]) if "anchor_idx" in batch else -1,
                })

        # Depth quality metrics
        if "depth" in batch and "depth" in output:
            pred_d = output["depth"].flatten(0, 1)   # (B*C, 1, H, W)
            gt_d = batch["depth"].flatten(0, 1)
            self.log("val/abs_rel",   abs_rel(pred_d, gt_d),               on_epoch=True, sync_dist=True)
            self.log("val/silog",     self.silog_loss(pred_d, gt_d),        on_epoch=True, sync_dist=True)
            self.log("val/depth_mse", F.mse_loss(pred_d, gt_d),            on_epoch=True, sync_dist=True)

        # Semantic quality metrics (accumulated across batches)
        if "semantic" in batch and "semantic" in output:
            pred_s = output["semantic"].flatten(0, 1)  # (B*C, 28, H, W)
            gt_s = batch["semantic"].flatten(0, 1)     # (B*C, H, W)
            preds = pred_s.argmax(dim=1)               # (B*C, H, W)
            self.val_miou(preds, gt_s)
            self.val_macc(preds, gt_s)
            self.val_iou_per_class(preds, gt_s)
            self._has_sem_val_data = True

        return loss

    def on_validation_epoch_end(self):
        # Trajectory visualization (same logic as parent)
        current = self.trainer.callback_metrics.get("val/avg_l2")
        if current is not None:
            current = current.item()
            if current < self._best_avg_l2:
                self._best_avg_l2 = current
                self._save_viz()
        self._viz_buffer.clear()

        # Semantic metrics (only if any semantic data was seen this epoch)
        if self._has_sem_val_data:
            self.log("val/miou", self.val_miou.compute(), prog_bar=True)
            self.log("val/macc", self.val_macc.compute())
            for c, iou_val in enumerate(self.val_iou_per_class.compute()):
                self.log(f"val/iou_class{c}", iou_val)
        self.val_miou.reset()
        self.val_macc.reset()
        self.val_iou_per_class.reset()
        self._has_sem_val_data = False
