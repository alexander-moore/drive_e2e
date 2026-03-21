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

from visualization import (plot_trajectory, plot_trajectory_batch,
                           plot_trajectory_on_image, save_trajectory_video)
try:
    from losses import SILogLoss, DiceLoss, abs_rel, imitation_l1        # when run from e2e/ dir
except ImportError:
    from e2e.losses import SILogLoss, DiceLoss, abs_rel, imitation_l1   # when imported as e2e package


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
    """Mean L2 across the standard 1 s / 2 s / 3 s horizons.

    This is the primary *evaluation* metric used across SOTA papers
    (UniAD, VAD, SparseDrive) and is kept for checkpoint monitoring and
    cross-paper comparison.  It is NOT used as the training loss because it
    only supervises 3 of the 50 future timesteps — use imitation_l1 instead.
    """
    return torch.stack([l2_at_horizon(pred, gt, t) for t in HORIZONS.values()]).mean()


def ade(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Average Displacement Error: mean L2 over all future timesteps.

    Complements avg_l2 (which samples 3 fixed horizons) with a summary over
    the full trajectory.  Reported alongside FDE in motion-prediction literature
    (Argoverse, nuScenes, etc.) and increasingly in E2E planners.

    Args:
        pred: (B, T, 2)
        gt:   (B, T, 2)
    Returns:
        scalar tensor [metres]
    """
    return torch.norm(pred - gt, dim=-1).mean()


def fde(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Final Displacement Error: L2 at the last future timestep.

    Measures how far off the endpoint (5 s ahead at 10 Hz) the planner is.
    Reported as a standard metric in nuScenes, Argoverse, and UniAD.

    Args:
        pred: (B, T, 2)
        gt:   (B, T, 2)
    Returns:
        scalar tensor [metres]
    """
    return torch.norm(pred[:, -1] - gt[:, -1], dim=-1).mean()


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
        traj_loss:    Training loss for the trajectory head.
                      "l1"  — L1 imitation over all T timesteps (SOTA default,
                              matches VAD / UniAD / SparseDrive).
                      "l2"  — avg_l2 at the 3 evaluation horizons (legacy;
                              supervises only 3 of 50 steps).
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        viz_samples: int = 4,
        traj_loss: str = "l1",
    ):
        if traj_loss not in ("l1", "l2"):
            raise ValueError(f"traj_loss must be 'l1' or 'l2', got {traj_loss!r}")
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.viz_samples = viz_samples
        self.traj_loss = traj_loss

        self._best_avg_l2: float = float("inf")
        self._viz_buffer: list[dict] = []

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, batch: dict) -> torch.Tensor:
        return self.model(batch)

    # ── shared step ──────────────────────────────────────────────────────────

    def _step(self, batch: dict, stage: str):
        pred = self(batch)          # (B, T, 2)
        gt   = batch["future_traj"] # (B, T, 2)

        # Training loss.
        # "l1"  — L1 imitation over all T future timesteps (SOTA: VAD, UniAD, SparseDrive).
        #         Supervises every waypoint uniformly; L1 is more robust to outliers than L2.
        # "l2"  — avg L2 at the 3 horizon checkpoints only (legacy; 3 of 50 steps supervised).
        if self.traj_loss == "l1":
            loss = imitation_l1(pred, gt)
        else:
            loss = avg_l2(pred, gt)

        # Evaluation metrics (logged every epoch regardless of training loss).
        # avg_l2 is the primary checkpoint metric for cross-paper comparison.
        # ADE/FDE are standard motion-prediction metrics (nuScenes, Argoverse, UniAD).
        l2_metric = avg_l2(pred, gt)

        on_step = stage == "train"
        self.log(f"{stage}/loss",     loss,      on_step=on_step, on_epoch=True,
                 prog_bar=True,  sync_dist=True)
        self.log(f"{stage}/avg_l2",   l2_metric, on_step=False,   on_epoch=True,
                 prog_bar=True,  sync_dist=True)
        self.log(f"{stage}/ade",      ade(pred, gt), on_step=False, on_epoch=True,
                 prog_bar=False, sync_dist=True)
        self.log(f"{stage}/fde",      fde(pred, gt), on_step=False, on_epoch=True,
                 prog_bar=False, sync_dist=True)

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

        # Buffer samples for visualization, preferring frames with lateral curvature
        # (max absolute y-displacement in future_traj) so we show turns, not just
        # straight segments. Half the budget is reserved for curved frames; the rest
        # are filled with whatever is available.
        n_curved  = max(1, self.viz_samples // 2)
        n_any     = self.viz_samples

        curved  = [s for s in self._viz_buffer if s.get("curved")]
        straight = [s for s in self._viz_buffer if not s.get("curved")]
        buffered_scenes = {s["scenario"] for s in self._viz_buffer}

        for i in range(pred.shape[0]):
            if len(self._viz_buffer) >= n_any:
                break
            scenario = batch.get("scenario_viz", batch.get("scenario", [f"frame{int(batch['anchor_idx'][i])}"]))[i]
            if scenario in buffered_scenes:
                continue

            ft = batch["future_traj"][i]                     # (50, 2)
            max_lateral = ft[:, 1].abs().max().item()        # max |y| = lateral swing
            is_curved = max_lateral > 1.0                    # >1 m lateral = meaningful turn

            # Skip straight frames once curved quota is met but straight slots remain
            if is_curved and len(curved) >= n_curved:
                continue
            if not is_curved and len(straight) >= (n_any - n_curved):
                continue

            buffered_scenes.add(scenario)
            entry = {
                "past_traj":     batch["past_traj"][i].cpu(),
                "future_traj":   ft.cpu(),
                "pred_traj":     pred[i].detach().cpu(),
                "scenario":      scenario,
                "scenario_path": batch.get("scenario_path", [None] * pred.shape[0])[i],
                "anchor_idx":    int(batch["anchor_idx"][i]) if "anchor_idx" in batch else -1,
                "curved":        is_curved,
            }
            if "images" in batch:
                raw = batch["images"][i]
                img = (raw[-1, 0] if raw.dim() == 5 else raw[0]).cpu()   # (3, H, W)
                # Undo ImageNet normalisation if applied (values outside [0,1] indicate it was)
                if img.min() < 0 or img.max() > 1:
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img  = img * std + mean
                img = (img * 255).clamp(0, 255).byte()
                entry["image"] = img.permute(1, 2, 0).numpy()  # (H, W, 3)
            self._viz_buffer.append(entry)
            if is_curved:
                curved.append(entry)
            else:
                straight.append(entry)
        return loss

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self._viz_buffer.clear()
            return
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

        for s in self._viz_buffer:
            scene = s["scenario"] or f"frame{s['anchor_idx']}"
            title = f"epoch {self.current_epoch}  {scene}  frame {s['anchor_idx']}"
            plot_trajectory(
                past_traj=s["past_traj"].float().numpy(),
                future_traj_gt=s["future_traj"].float().numpy(),
                future_traj_pred=s["pred_traj"].float().numpy(),
                title=title,
                save_path=str(viz_dir / f"{scene}.png"),
            )
            if "image" in s:
                plot_trajectory_on_image(
                    image=s["image"],
                    future_traj_gt=s["future_traj"].float().numpy(),
                    future_traj_pred=s["pred_traj"].float().numpy(),
                    title=title,
                    save_path=str(viz_dir / f"{scene}_cam.png"),
                )

            if s.get("curved") and s.get("scenario_path") and s["anchor_idx"] >= 0:
                save_trajectory_video(
                    scenario_path=s["scenario_path"],
                    anchor_idx=s["anchor_idx"],
                    past_traj=s["past_traj"].float().numpy(),
                    future_traj_gt=s["future_traj"].float().numpy(),
                    future_traj_pred=s["pred_traj"].float().numpy(),
                    save_path=str(viz_dir / f"{scene}_video.mp4"),
                )

        # Also save a batch grid
        scene_titles = [s["scenario"] or f"frame{s['anchor_idx']}" for s in self._viz_buffer]
        plot_trajectory_batch(
            past_trajs=torch.stack([s["past_traj"]   for s in self._viz_buffer]).float().numpy(),
            future_trajs_gt=torch.stack([s["future_traj"] for s in self._viz_buffer]).float().numpy(),
            future_trajs_pred=torch.stack([s["pred_traj"]  for s in self._viz_buffer]).float().numpy(),
            titles=scene_titles,
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
        traj_loss: str = "l1",
    ):
        super().__init__(model, lr, weight_decay, viz_samples, traj_loss)
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

        # Trajectory training loss — same choice as the base module.
        if self.traj_loss == "l1":
            loss_traj = imitation_l1(pred_traj, gt_traj)
        else:
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

        # Evaluation metrics are always computed from avg_l2 / ADE / FDE regardless
        # of which training loss was selected, so val/avg_l2 remains comparable across runs.
        l2_metric = avg_l2(pred_traj, gt_traj)

        self.log(f"{stage}/loss",       loss,       on_step=on_step, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log(f"{stage}/loss_traj",  loss_traj,  on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/loss_depth", loss_depth, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/loss_sem",   loss_sem,   on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/avg_l2",     l2_metric,  on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log(f"{stage}/ade",        ade(pred_traj, gt_traj), on_step=False, on_epoch=True,
                 prog_bar=False, sync_dist=True)
        self.log(f"{stage}/fde",        fde(pred_traj, gt_traj), on_step=False, on_epoch=True,
                 prog_bar=False, sync_dist=True)

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

        # Buffer samples for visualization, preferring frames with lateral curvature
        n_curved  = max(1, self.viz_samples // 2)
        n_any     = self.viz_samples

        curved   = [s for s in self._viz_buffer if s.get("curved")]
        straight = [s for s in self._viz_buffer if not s.get("curved")]
        buffered_scenes = {s["scenario"] for s in self._viz_buffer}

        for i in range(pred_traj.shape[0]):
            if len(self._viz_buffer) >= n_any:
                break
            scenario = batch.get("scenario_viz", batch.get("scenario", [f"frame{int(batch['anchor_idx'][i])}"]))[i]
            if scenario in buffered_scenes:
                continue

            ft = batch["future_traj"][i]
            max_lateral = ft[:, 1].abs().max().item()
            is_curved = max_lateral > 1.0

            if is_curved and len(curved) >= n_curved:
                continue
            if not is_curved and len(straight) >= (n_any - n_curved):
                continue

            buffered_scenes.add(scenario)
            entry = {
                "past_traj":     batch["past_traj"][i].cpu(),
                "future_traj":   ft.cpu(),
                "pred_traj":     pred_traj[i].detach().cpu(),
                "scenario":      scenario,
                "scenario_path": batch.get("scenario_path", [None] * pred_traj.shape[0])[i],
                "anchor_idx":    int(batch["anchor_idx"][i]) if "anchor_idx" in batch else -1,
                "curved":        is_curved,
            }
            if "images" in batch:
                raw = batch["images"][i]
                img = (raw[-1, 0] if raw.dim() == 5 else raw[0]).cpu()   # (3, H, W)
                # Undo ImageNet normalisation if applied (values outside [0,1] indicate it was)
                if img.min() < 0 or img.max() > 1:
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img  = img * std + mean
                img = (img * 255).clamp(0, 255).byte()
                entry["image"] = img.permute(1, 2, 0).numpy()  # (H, W, 3)
            self._viz_buffer.append(entry)
            if is_curved:
                curved.append(entry)
            else:
                straight.append(entry)

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
        if self.trainer.sanity_checking:
            self._viz_buffer.clear()
            return
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
