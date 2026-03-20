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
            debug=args.debug,
        )
    elif args.model == "front_cam":
        from models.front_cam_planner import FrontCamPlanner
        return FrontCamPlanner(
            token_dim=args.token_dim,
            num_heads=args.nhead,
            enc_layers=args.enc_layers,
            dec_layers=args.dec_layers,
            multiscale=args.multiscale,
            debug=args.debug,
        )
    elif args.model == "front_cam_depth":
        from models.front_cam_depth_planner import FrontCamDepthPlanner
        return FrontCamDepthPlanner(
            token_dim=args.token_dim,
            num_heads=args.nhead,
            enc_layers=args.enc_layers,
            dec_layers=args.dec_layers,
            multiscale=args.multiscale,
            debug=args.debug,
        )
    elif args.model == "resnet":
        from models.resnet_planner import ResNetPlanner
        return ResNetPlanner(
            token_dim=args.token_dim,
            num_heads=args.nhead,
            enc_layers=args.enc_layers,
            dec_layers=args.dec_layers,
            multiscale=args.multiscale,
            backbone=args.resnet_variant,
            frozen=not args.trainable_backbone,
            debug=args.debug,
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
    elif args.model == "front_cam_depth":
        ds_kwargs = dict(
            load_images=True,
            image_size=(224, 224),
            load_depth=True,
            load_semantic=False,
            front_cam_only=True,
        )
    elif args.model in ("front_cam", "resnet"):
        ds_kwargs = dict(
            load_images=True,
            image_size=(224, 224),
            load_depth=False,
            load_semantic=False,
            front_cam_only=True,
        )
    else:
        ds_kwargs = dict(load_images=False, image_size=None)

    train_ds, _inner_val = make_datasets(root=args.data_root, **ds_kwargs)

    val_root = args.val_data_root or args.data_root
    if val_root != args.data_root:
        # Validate on a separate dataset (all scenarios used for val)
        val_ds = Bench2DriveDataset(root=val_root, **ds_kwargs)
    else:
        val_ds = _inner_val

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_dl, val_dl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train an E2E driving model on bench2drive.")

    # Data
    parser.add_argument("--data_root",     default="/workspace/bench2resize",
                        help="Path to training dataset root")
    parser.add_argument("--val_data_root", default="/workspace/bench2drive_mini",
                        help="Path to validation dataset root (default: bench2drive_mini). "
                             "Set to '' to val on a held-out split of --data_root instead.")
    parser.add_argument("--num_workers",     type=int, default=16)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--batch_size",      type=int, default=128)

    # Model
    parser.add_argument("--model",       default="mlp",
                        choices=["mlp", "transformer", "vision_transformer", "front_cam", "front_cam_depth", "resnet"],
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
    parser.add_argument("--resnet_variant", default="resnet50",
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
                        help="[resnet] which ResNet variant to use as backbone")
    parser.add_argument("--trainable_backbone", action="store_true", default=False,
                        help="[resnet] unfreeze backbone weights during training")

    # Debug
    parser.add_argument("--name", default=None,
                        help="Experiment name shown in TensorBoard (default: model name)")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Debug mode: print tensor shapes at vital stages and run only a few steps")

    # Loss
    parser.add_argument("--traj_loss",   default="l1", choices=["l1", "l2"],
                        help="Trajectory training loss. "
                             "'l1' = L1 imitation over all 50 steps (SOTA: VAD/UniAD/SparseDrive). "
                             "'l2' = avg L2 at 1s/2s/3s horizons only (legacy).")

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
    parser.add_argument("--viz_samples", type=int, default=5,
                        help="Number of trajectories to visualise per val epoch")

    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    pl.seed_everything(42, workers=True)

    # ── build components ──────────────────────────────────────────────────
    model = build_model(args)
    if args.model in ("vision_transformer", "front_cam_depth"):
        module = MultiTaskE2EModule(
            model=model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            viz_samples=args.viz_samples,
            depth_weight=args.depth_weight,
            sem_weight=0.0 if args.model == "front_cam_depth" else args.sem_weight,
            traj_loss=args.traj_loss,
        )
    else:
        module = E2EDrivingModule(  # front_cam and other models use E2EDrivingModule
            model=model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            viz_samples=args.viz_samples,
            traj_loss=args.traj_loss,
        )
    train_dl, val_dl = build_dataloaders(args)

    # ── callbacks & logger ────────────────────────────────────────────────
    exp_name = args.name or args.model
    logger = TensorBoardLogger(save_dir=args.log_dir, name=exp_name)

    callbacks = [
        ModelCheckpoint(
            dirpath=args.ckpt_dir,
            filename=f"{exp_name}-{{epoch:03d}}-{{val/avg_l2:.4f}}",
            monitor="val/avg_l2",   # primary SOTA metric: mean L2 @ 1s/2s/3s
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # ── trainer ───────────────────────────────────────────────────────────
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
    print(f"\nModel : {args.model}  ({n_params:,} params)  [{exp_name}]")
    print(f"Train : {len(train_dl.dataset)} samples  ({args.data_root})")
    print(f"Val   : {len(val_dl.dataset)} samples  ({args.val_data_root or args.data_root})")
    print(f"Logs  : {logger.log_dir}")
    if args.debug:
        print("[DEBUG] Debug mode enabled — 5 train batches, 2 val batches, 1 epoch, tensor shapes printed")
    print()

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
