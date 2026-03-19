"""
train.py — CLI training script for E2E driving experiments.

Run from /workspace:
    python -m e2e.train --model mlp --data_root /workspace/bench2drive_mini

Examples:
    # MLP baseline, trajectory-only
    python -m e2e.train --model mlp

    # Larger MLP with custom hyperparams
    python -m e2e.train --model mlp --hidden_dim 512 --num_layers 6 --lr 3e-4 --epochs 50

    # Resume from checkpoint
    python -m e2e.train --model mlp --ckpt_path /workspace/checkpoints/e2e/last.ckpt
"""

import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split

from dataset import Bench2DriveDataset, make_datasets, PAST_STEPS_TOTAL, FUTURE_STEPS
from module import E2EDrivingModule, MultiTaskE2EModule


# ---------------------------------------------------------------------------
# Model registry — add new model classes here
# ---------------------------------------------------------------------------

def build_model(args):
    if args.model == "mlp":
        from models.mlp_planner import MLPPlanner
        return MLPPlanner(
            past_steps=PAST_STEPS_TOTAL,
            future_steps=FUTURE_STEPS,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    elif args.model == "transformer":
        from models.transformer_planner import TransformerPlanner
        return TransformerPlanner(
            past_steps=PAST_STEPS_TOTAL,
            future_steps=FUTURE_STEPS,
            d_model=args.d_model,
            nhead=args.nhead,
            enc_layers=args.enc_layers,
            dec_layers=args.dec_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
        )
    elif args.model == "vision_transformer":
        from models.vision_transformer_planner import VisionTransformerPlanner
        return VisionTransformerPlanner(
            token_dim=args.token_dim,
            num_heads=args.nhead,
            enc_layers=args.enc_layers,
            dec_layers=args.dec_layers,
            ffn_dim=args.token_dim * 4,
            multiscale=args.multiscale,
            front_cam_only=args.front_cam_only,
        )
    else:
        raise ValueError(f"Unknown model: {args.model!r}. "
                         f"Add it to the registry in train_e2e.py.")


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(args):
    if args.model == "vision_transformer":
        ds_kwargs = dict(
            load_images=True,
            image_size=(224, 224),
            load_depth=True,
            load_semantic=True,
            front_cam_only=args.front_cam_only,
        )
    else:
        ds_kwargs = dict(load_images=False, image_size=None)

    train_ds, val_ds = make_datasets(root=args.data_root, **ds_kwargs)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_dl, val_dl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train an E2E driving model on bench2drive.")

    # Data
    parser.add_argument("--data_root",   default="/workspace/bench2drive_mini",
                        help="Path to bench2drive_mini dataset root")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size",  type=int, default=32)

    # Model
    parser.add_argument("--model",       default="mlp",
                        choices=["mlp", "transformer", "vision_transformer"],
                        help="Model architecture to train")
    parser.add_argument("--hidden_dim",  type=int, default=256,
                        help="[mlp] hidden layer width")
    parser.add_argument("--num_layers",  type=int, default=4,
                        help="[mlp] number of hidden layers")
    parser.add_argument("--dropout",        type=float, default=0.1)
    # transformer args
    parser.add_argument("--d_model",        type=int,   default=128,
                        help="[transformer] model dimension")
    parser.add_argument("--nhead",          type=int,   default=4,
                        help="[transformer] attention heads")
    parser.add_argument("--enc_layers",     type=int,   default=3,
                        help="[transformer] encoder layers")
    parser.add_argument("--dec_layers",     type=int,   default=3,
                        help="[transformer] decoder layers")
    parser.add_argument("--dim_feedforward",type=int,   default=512,
                        help="[transformer] FFN hidden dim")
    # vision_transformer args
    parser.add_argument("--token_dim",      type=int,   default=256,
                        help="[vision_transformer] token dimensionality")
    parser.add_argument("--depth_weight",   type=float, default=0.1,
                        help="[vision_transformer] auxiliary depth loss weight")
    parser.add_argument("--sem_weight",     type=float, default=0.1,
                        help="[vision_transformer] auxiliary semantic loss weight")
    parser.add_argument("--multiscale",     dest="multiscale", action="store_true",  default=True,
                        help="[vision_transformer] use all 4 TinyViT scales (default: True)")
    parser.add_argument("--no_multiscale",  dest="multiscale", action="store_false",
                        help="[vision_transformer] use only TinyViT bottom level")
    parser.add_argument("--front_cam_only", action="store_true", default=False,
                        help="[vision_transformer] use only front camera")

    # Optimiser
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Training
    parser.add_argument("--epochs",      type=int, default=30)
    parser.add_argument("--devices",     type=int, default=1)
    parser.add_argument("--precision",   default="32",
                        choices=["32", "16-mixed", "bf16-mixed"])

    # Logging / checkpointing
    parser.add_argument("--log_dir",     default="/workspace/e2e/logs",
                        help="TensorBoard log root")
    parser.add_argument("--ckpt_dir",    default="/workspace/e2e/checkpoints",
                        help="Checkpoint output directory")
    parser.add_argument("--ckpt_path",   default=None,
                        help="Resume training from this checkpoint")
    parser.add_argument("--viz_samples", type=int, default=4,
                        help="Number of trajectories to visualise per val epoch")

    args = parser.parse_args()

    pl.seed_everything(42, workers=True)

    # ── build components ──────────────────────────────────────────────────
    model = build_model(args)
    if args.model == "vision_transformer":
        module = MultiTaskE2EModule(
            model=model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            viz_samples=args.viz_samples,
            depth_weight=args.depth_weight,
            sem_weight=args.sem_weight,
        )
    else:
        module = E2EDrivingModule(
            model=model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            viz_samples=args.viz_samples,
        )
    train_dl, val_dl = build_dataloaders(args)

    # ── callbacks & logger ────────────────────────────────────────────────
    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.model)

    callbacks = [
        ModelCheckpoint(
            dirpath=args.ckpt_dir,
            filename=f"{args.model}-{{epoch:03d}}-{{val/avg_l2:.4f}}",
            monitor="val/avg_l2",   # primary SOTA metric: mean L2 @ 1s/2s/3s
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # ── trainer ───────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=args.devices,
        accelerator="auto",
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        val_check_interval=1.0,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel : {args.model}  ({n_params:,} params)")
    print(f"Train : {len(train_dl.dataset)} samples   Val: {len(val_dl.dataset)} samples")
    print(f"Logs  : {logger.log_dir}\n")

    trainer.fit(module, train_dl, val_dl, ckpt_path=args.ckpt_path)

    # ── log CLI + args to TensorBoard for reproducibility ─────────────────
    cli = "python " + " ".join(__import__("sys").argv)
    arg_table = "| argument | value |\n|---|---|\n" + "\n".join(
        f"| `{k}` | `{v}` |" for k, v in sorted(vars(args).items())
    )
    logger.experiment.add_text("cli",  cli,       global_step=0)
    logger.experiment.add_text("args", arg_table, global_step=0)
    logger.experiment.add_text("model_summary",
        f"**{args.model}** — {n_params:,} parameters", global_step=0)


if __name__ == "__main__":
    main()
