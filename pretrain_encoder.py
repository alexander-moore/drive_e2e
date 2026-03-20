"""
pretrain_encoder.py — CLI script for video prediction pretraining.

Run from /workspace:
    python -m e2e.pretrain_encoder \\
        --data_root /workspace/bench2drive_mini \\
        --epochs 2 --batch_size 2 --front_cam_only

After training, saves encoder weights to {ckpt_dir}/encoder.pt so they can
be loaded into a downstream planner via model.encoder.load_state_dict(...).

frame_stride controls temporal subsampling:
    1 → consecutive frames (default)
    2 → every other frame (wider temporal window at lower resolution)
    3 → every third frame, etc.
"""

import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

try:
    from video_dataset import make_video_datasets
    from models.video_prediction_encoder import VideoPredictionModel
    from pretrain_module import VideoPredictionModule
    from dataset import CAMERAS
except ImportError:
    from e2e.video_dataset import make_video_datasets
    from e2e.models.video_prediction_encoder import VideoPredictionModel
    from e2e.pretrain_module import VideoPredictionModule
    from e2e.dataset import CAMERAS


def main():
    parser = argparse.ArgumentParser(
        description="Self-supervised video prediction pretraining on bench2drive.",
    )

    # ── Data ──────────────────────────────────────────────────────────────
    parser.add_argument("--data_root",      default="/workspace/bench2resize",
                        help="Path to dataset root (bench2resize or bench2drive_mini)")
    parser.add_argument("--n_frames",       type=int,   default=4,
                        help="Number of input context frames")
    parser.add_argument("--m_frames",       type=int,   default=4,
                        help="Number of future frames to predict")
    parser.add_argument("--frame_stride",   type=int,   default=1,
                        help="Step between sampled frames (1=consecutive, 2=every-other, etc.)")
    parser.add_argument("--image_size",     type=int,   nargs=2, default=[224, 224],
                        metavar=("H", "W"), help="Resize images to H W")
    parser.add_argument("--front_cam_only", action="store_true", default=False,
                        help="Use only the front camera (1 cam instead of 6)")
    parser.add_argument("--num_workers",     type=int,  default=8)
    parser.add_argument("--prefetch_factor", type=int,  default=4)
    parser.add_argument("--batch_size",      type=int,  default=8)

    # ── Model ─────────────────────────────────────────────────────────────
    parser.add_argument("--spatial_encoder",   default="resnet18",
                        choices=["resnet18", "resnet50", "tinyvit"],
                        help="Spatial backbone: resnet18 (default, trainable), "
                             "resnet50 (trainable), tinyvit (frozen, square images only)")
    parser.add_argument("--token_dim",         type=int, default=256,
                        help="Transformer token dimensionality (must be divisible by 4)")
    parser.add_argument("--n_encoder_layers",  type=int, default=4,
                        help="Number of (TemporalBlock, CameraBlock) pairs in encoder")
    parser.add_argument("--n_decoder_layers",  type=int, default=4,
                        help="Number of transformer decoder layers")
    parser.add_argument("--n_heads",           type=int, default=8,
                        help="Number of attention heads")

    # ── Optimiser ─────────────────────────────────────────────────────────
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # ── Training ──────────────────────────────────────────────────────────
    parser.add_argument("--epochs",    type=int,   default=100)
    parser.add_argument("--devices",   type=int,   default=1)
    parser.add_argument("--precision", default="32",
                        choices=["32", "16-mixed", "bf16-mixed"])
    parser.add_argument("--debug",     action="store_true", default=False,
                        help="Debug mode: 5 train batches, 2 val batches, 1 epoch")

    # ── Logging / checkpointing ───────────────────────────────────────────
    parser.add_argument("--name",      default="video_pretrain",
                        help="Experiment name shown in TensorBoard")
    parser.add_argument("--log_dir",   default="/workspace/e2e/logs")
    parser.add_argument("--ckpt_dir",  default="/workspace/e2e/checkpoints")
    parser.add_argument("--ckpt_path",   default=None,
                        help="Resume training from this Lightning checkpoint")
    parser.add_argument("--viz_samples", type=int, default=4,
                        help="Number of validation samples to visualise per best epoch")

    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    pl.seed_everything(42, workers=True)

    # ── Datasets & dataloaders ────────────────────────────────────────────
    image_size = tuple(args.image_size)
    n_cams     = 1 if args.front_cam_only else len(CAMERAS)

    train_ds, val_ds = make_video_datasets(
        root=args.data_root,
        n_frames=args.n_frames,
        m_frames=args.m_frames,
        frame_stride=args.frame_stride,
        image_size=image_size,
        front_cam_only=args.front_cam_only,
    )

    # persistent_workers requires num_workers > 0
    persist = args.num_workers > 0

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persist,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory=True,
        persistent_workers=persist,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = VideoPredictionModel(
        n_frames=args.n_frames,
        m_frames=args.m_frames,
        n_cams=n_cams,
        token_dim=args.token_dim,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        n_heads=args.n_heads,
        spatial_encoder=args.spatial_encoder,
        image_size=image_size,
    )
    cam_names = ["rgb_front"] if args.front_cam_only else list(CAMERAS)
    module = VideoPredictionModule(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        viz_samples=args.viz_samples,
        cam_names=cam_names,
        frame_stride=args.frame_stride,
    )

    # ── Callbacks & logger ────────────────────────────────────────────────
    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.name)

    callbacks = [
        ModelCheckpoint(
            dirpath=args.ckpt_dir,
            filename=f"{args.name}-{{epoch:03d}}-{{val/loss:.4f}}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=1 if args.debug else args.epochs,
        limit_train_batches=5 if args.debug else 1.0,
        limit_val_batches=2 if args.debug else 1.0,
        devices=args.devices,
        accelerator="auto",
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1 if args.debug else 10,
        val_check_interval=1.0,
        gradient_clip_val=1.0,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel    : VideoPredictionModel  ({n_params:,} params)")
    print(f"Backbone : {args.spatial_encoder}")
    print(f"Frames   : {args.n_frames} context → {args.m_frames} predicted"
          f"  (stride={args.frame_stride})")
    print(f"Cameras  : {n_cams}  ({'front only' if args.front_cam_only else 'all 6'})")
    print(f"Image    : {image_size[0]}×{image_size[1]}")
    print(f"Train    : {len(train_ds)} samples  ({args.data_root})")
    print(f"Val      : {len(val_ds)} samples")
    print(f"Logs     : {logger.log_dir}")
    if args.debug:
        print("[DEBUG] 5 train batches, 2 val batches, 1 epoch")
    print()

    trainer.fit(module, train_dl, val_dl, ckpt_path=args.ckpt_path)

    # ── Save encoder weights ──────────────────────────────────────────────
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = ckpt_dir / "encoder.pt"
    torch.save(model.encoder.state_dict(), encoder_path)
    print(f"\nEncoder weights saved → {encoder_path}")
    print("Load with: model.encoder.load_state_dict(torch.load('encoder.pt'))")

    # ── Log CLI / args to TensorBoard ─────────────────────────────────────
    import sys
    cli       = "python " + " ".join(sys.argv)
    arg_table = "| argument | value |\n|---|---|\n" + "\n".join(
        f"| `{k}` | `{v}` |" for k, v in sorted(vars(args).items())
    )
    logger.experiment.add_text("cli",  cli,       global_step=0)
    logger.experiment.add_text("args", arg_table, global_step=0)
    logger.experiment.add_text(
        "model_summary",
        f"**VideoPredictionModel** ({args.spatial_encoder}) — {n_params:,} parameters",
        global_step=0,
    )


if __name__ == "__main__":
    main()
