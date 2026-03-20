"""
PyTorch Lightning module for self-supervised video autoencoder pretraining.

The model encodes ALL T = n_frames + m_frames frames jointly, then reconstructs
all T frames.  Loss is MSE over the full reconstruction.  Additional per-phase
losses (ctx_loss, future_loss) are logged separately for interpretability.

Saved visualisations show: context frames | GT future  (top row, GT)
                           context recon  | future recon (bottom row, Pred)
for each camera.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

try:
    from models.video_prediction_encoder import VideoPredictionModel
    from visualization import plot_video_prediction
    from dataset import CAMERAS
except ImportError:
    from e2e.models.video_prediction_encoder import VideoPredictionModel
    from e2e.visualization import plot_video_prediction
    from e2e.dataset import CAMERAS


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio in dB. pred/target assumed in [0, 1]."""
    mse = F.mse_loss(pred, target)
    return 10.0 * torch.log10(1.0 / (mse + 1e-8))


class VideoPredictionModule(pl.LightningModule):
    """
    Lightning wrapper for VideoPredictionModel autoencoder pretraining.

    Batch contract:
        batch["all_frames"]    — (B, n_cams, n_frames+m_frames, 3, H, W) in [0,1]
        batch["input_frames"]  — (B, n_cams, n_frames, 3, H, W)   (viz split)
        batch["target_frames"] — (B, n_cams, m_frames, 3, H, W)   (viz split)

    forward(batch) returns reconstructed frames (B, n_cams, n_frames+m_frames, 3, H, W).

    Args:
        model:        VideoPredictionModel instance.
        lr:           Base learning rate for AdamW.
        weight_decay: AdamW weight decay.
        viz_samples:  Max validation samples to visualise per best epoch.
        cam_names:    Camera name list for plot labels.
        frame_stride: Temporal stride (for axis labels only).
    """

    def __init__(
        self,
        model: VideoPredictionModel,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        viz_samples: int = 4,
        cam_names: list = None,
        frame_stride: int = 1,
    ):
        super().__init__()
        self.model        = model
        self.lr           = lr
        self.weight_decay = weight_decay
        self.viz_samples  = viz_samples
        self.cam_names    = cam_names or list(CAMERAS)
        self.frame_stride = frame_stride
        self.n_frames     = model.n_frames
        self.m_frames     = model.m_frames

        self._best_val_loss: float = float("inf")
        self._viz_buffer: list[dict] = []

    # ── forward ──────────────────────────────────────────────────────────

    def forward(self, batch: dict) -> torch.Tensor:
        """Returns reconstructed frames (B, n_cams, n_frames+m_frames, 3, H, W)."""
        return self.model(batch["all_frames"])

    # ── shared step ──────────────────────────────────────────────────────

    def _step(self, batch: dict, stage: str):
        recon  = self(batch)                       # (B, n_cams, T, 3, H, W)
        target = batch["all_frames"]               # (B, n_cams, T, 3, H, W)
        n      = self.n_frames

        loss = F.mse_loss(recon, target)

        # Per-phase losses for interpretability (not used for optimization)
        ctx_loss    = F.mse_loss(recon[:, :, :n],  target[:, :, :n])
        future_loss = F.mse_loss(recon[:, :, n:],  target[:, :, n:])

        on_step = (stage == "train")
        self.log(f"{stage}/loss",        loss,        on_step=on_step, on_epoch=True,
                 prog_bar=True,  sync_dist=True)
        self.log(f"{stage}/ctx_loss",    ctx_loss,    on_step=False,   on_epoch=True,
                 prog_bar=False, sync_dist=True)
        self.log(f"{stage}/future_loss", future_loss, on_step=False,   on_epoch=True,
                 prog_bar=False, sync_dist=True)

        if stage == "val":
            self.log("val/psnr", psnr(recon.detach(), target),
                     on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val/future_psnr", psnr(recon[:, :, n:].detach(), target[:, :, n:]),
                     on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss, recon

    # ── train / val steps ────────────────────────────────────────────────

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, _ = self._step(batch, "train")
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, recon = self._step(batch, "val")

        buffered = {s["scenario"] for s in self._viz_buffer}
        for i in range(recon.shape[0]):
            if len(self._viz_buffer) >= self.viz_samples:
                break
            scenario = (batch.get("scenario") or [f"sample{i}"])[i]
            if scenario in buffered:
                continue
            buffered.add(scenario)

            n = self.n_frames
            def _to_np(t):
                return t.detach().cpu().float().permute(0, 1, 3, 4, 2).numpy()

            self._viz_buffer.append({
                "scenario":       scenario,
                "anchor_idx":     int(batch["anchor_idx"][i]) if "anchor_idx" in batch else -1,
                # GT splits (for plot rows)
                "input_frames":   _to_np(batch["input_frames"][i]),   # (n_cams, n, H, W, 3)
                "target_frames":  _to_np(batch["target_frames"][i]),  # (n_cams, m, H, W, 3)
                # Reconstructions — same split
                "ctx_recon":      _to_np(recon[i, :, :n]),            # (n_cams, n, H, W, 3)
                "future_recon":   _to_np(recon[i, :, n:]),            # (n_cams, m, H, W, 3)
                "psnr":           psnr(recon[i:i+1].detach(),
                                       batch["all_frames"][i:i+1]).item(),
            })

        return loss

    def on_validation_epoch_end(self):
        current = self.trainer.callback_metrics.get("val/loss")
        if current is None:
            self._viz_buffer.clear()
            return
        current = current.item()
        if current < self._best_val_loss:
            self._best_val_loss = current
            self._save_viz()
        self._viz_buffer.clear()

    # ── visualisation ─────────────────────────────────────────────────────

    def _save_viz(self):
        if not self._viz_buffer:
            return

        log_dir = Path(self.trainer.log_dir) if self.trainer.log_dir else Path("logs/pretrain")
        viz_dir = log_dir / "best_predictions"
        viz_dir.mkdir(parents=True, exist_ok=True)

        n_cams    = self._viz_buffer[0]["input_frames"].shape[0]
        cam_names = self.cam_names[:n_cams]

        import numpy as np
        for s in self._viz_buffer:
            scene = s["scenario"]
            title = (f"epoch {self.current_epoch}  |  {scene}  frame {s['anchor_idx']}"
                     f"  |  stride={self.frame_stride}")

            # Concatenate GT splits for the "GT" rows: [input | future_gt]
            gt_all   = np.concatenate([s["input_frames"],  s["target_frames"]], axis=1)
            # Concatenate recon splits for the "Pred" rows: [ctx_recon | future_recon]
            recon_all = np.concatenate([s["ctx_recon"], s["future_recon"]],  axis=1)

            plot_video_prediction(
                input_frames  = s["input_frames"],
                target_frames = s["target_frames"],
                pred_frames   = s["future_recon"],
                cam_names     = cam_names,
                frame_stride  = self.frame_stride,
                title         = title,
                psnr_val      = s["psnr"],
                save_path     = str(viz_dir / f"{scene}.png"),
            )

            # Also save a second plot showing context reconstruction quality
            plot_video_prediction(
                input_frames  = s["input_frames"],
                target_frames = s["input_frames"],   # GT = original context
                pred_frames   = s["ctx_recon"],       # Pred = context reconstruction
                cam_names     = cam_names,
                frame_stride  = self.frame_stride,
                title         = title + "  [context reconstruction]",
                psnr_val      = s["psnr"],
                save_path     = str(viz_dir / f"{scene}_ctx.png"),
            )

    # ── optimizer ─────────────────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=self.lr * 0.01,
        )
        return {
            "optimizer":    optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
