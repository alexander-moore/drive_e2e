"""
ResNetPlanner — Single front-camera planner with a configurable ResNet backbone.

Identical to FrontCamPlanner but replaces the TinyViT backbone with a
pretrained ResNet, whose intermediate feature maps match TinyViT's spatial
resolutions at 224×224 input.  The backbone variant and whether it is frozen
are both configurable.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE  (multiscale=False default, token_dim D=128, backbone=resnet50)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  INPUTS
  images        (B, 1, 3, 224, 224)   — front camera only
  past_traj     (B, 41, 2)  ─┐
  speed         (B, 41)     ─┤─ concat → (B, 41, 10)
  acceleration  (B, 41, 3)  ─┤
  command       (B,)        ─┘  (one-hot dim 4, broadcast across time)

  ┌──────────────────────────────┐    ┌──────────────────────────────────────┐
  │  KinematicEncoder            │    │  ResNet  (pretrained, frozen or not) │
  │  Linear(10 → D)              │    │  in:  (B, 3, 224, 224)               │
  │  + 1D sin-cos pos enc        │    │  multiscale=True:  4 scales          │
  │  TransformerEncoder (pre-LN) │    │    s0: (B,  C1, 56, 56)  3136 tok    │
  │  enc_layers layers           │    │    s1: (B,  C2, 28, 28)   784 tok    │
  │  ──────────────────────────  │    │    s2: (B,  C3, 14, 14)   196 tok    │
  │  kin_mem  (B, 41, D)         │    │    s3: (B,  C4,  7,  7)    49 tok    │
  └──────────────┬───────────────┘    │  multiscale=False: bottleneck only   │
                 │                    │    s3: (B,  C4,  7,  7)    49 tok    │
                 │                    └──────────────┬───────────────────────┘
                 │                                   │  vis_projs[k]: Linear(C_k → D)
                 │                                   │  + 2D sin-cos pos enc per scale
                 │                    enc_feats:  list of (B, N_k, D)
                 │                                   │
                 └──────────────────────────────┬────┘
                                                │
                                     ┌──────────┴──────────────────────────┐
                                     │  Drive decoder                      │
                                     │  init: drive_embed  (B, 50, D)      │
                                     │  enc sources:                        │
                                     │    vis: (B, N_k, D) × num_vis_levels │
                                     │    kin: (B, 41, D)                   │
                                     │  dec_layers × FlexDecoderLayer       │
                                     │    cross-attn: vis × levels + kin    │
                                     └──────────────┬──────────────────────┘
                                                    │
                                             Linear(D → 2)
                                                    │
                                        future_traj (B, 50, 2)

  Backbone channel sizes at 224×224 input:
    resnet18 / resnet34  (basic block):      C1=64   C2=128  C3=256  C4=512
    resnet50 / resnet101 / resnet152  (bottleneck): C1=256 C2=512 C3=1024 C4=2048

  FlexDecoder layer (pre-norm):
    tokens = tokens + SelfAttn( LN(tokens) + token_pos )
    tokens = tokens + Σ_k CrossAttn_k( LN(tokens) + token_pos,  enc_feats[k] )
    tokens = tokens + FFN( LN(tokens) )
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch import Tensor
import torchvision.models as tv_models

from ._blocks import make_2d_sincos_pos_enc
from .vision_transformer_planner import (
    KinematicEncoder,
    FlexDecoderLayer,
    _make_1d_sincos_pos_enc,
    _dbg_header,
    _dbg_sec,
    _dbg_row,
    _DBG_W,
)


# ---------------------------------------------------------------------------
# Backbone registry
# ---------------------------------------------------------------------------

# (constructor, weights, [layer1_ch, layer2_ch, layer3_ch, layer4_ch])
_BACKBONES = {
    "resnet18":  (tv_models.resnet18,  tv_models.ResNet18_Weights.DEFAULT,  [64,  128,  256,  512]),
    "resnet34":  (tv_models.resnet34,  tv_models.ResNet34_Weights.DEFAULT,  [64,  128,  256,  512]),
    "resnet50":  (tv_models.resnet50,  tv_models.ResNet50_Weights.DEFAULT,  [256, 512, 1024, 2048]),
    "resnet101": (tv_models.resnet101, tv_models.ResNet101_Weights.DEFAULT, [256, 512, 1024, 2048]),
    "resnet152": (tv_models.resnet152, tv_models.ResNet152_Weights.DEFAULT, [256, 512, 1024, 2048]),
}


class ResNetEncoder(nn.Module):
    """
    Pretrained ResNet backbone that returns four intermediate feature maps.

    Args:
        variant: one of resnet18 / resnet34 / resnet50 / resnet101 / resnet152
        frozen:  if True, all backbone parameters are frozen (no gradients)

    Returns (layer1, layer2, layer3, layer4) for a (B, 3, 224, 224) input.
    Channel sizes depend on variant — see _BACKBONES table above.
    """

    def __init__(self, variant: str = "resnet50", frozen: bool = True,
                 grad_checkpoint: bool = False):
        super().__init__()
        if variant not in _BACKBONES:
            raise ValueError(f"Unknown ResNet variant {variant!r}. "
                             f"Choose from: {list(_BACKBONES)}")
        constructor, weights, self.out_channels = _BACKBONES[variant]
        backbone = constructor(weights=weights)
        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1,
                                    backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.frozen = frozen
        # Gradient checkpointing recomputes activations during backward to save
        # VRAM; only useful (and valid) when the backbone is trainable.
        self.grad_checkpoint = grad_checkpoint and not frozen
        if frozen:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, x: Tensor):
        x = self.stem(x)
        if self.grad_checkpoint:
            s0 = cp.checkpoint(self.layer1, x,  use_reentrant=False)
            s1 = cp.checkpoint(self.layer2, s0, use_reentrant=False)
            s2 = cp.checkpoint(self.layer3, s1, use_reentrant=False)
            s3 = cp.checkpoint(self.layer4, s2, use_reentrant=False)
        else:
            s0 = self.layer1(x)
            s1 = self.layer2(s0)
            s2 = self.layer3(s1)
            s3 = self.layer4(s2)
        return s0, s1, s2, s3


class ResNetPlanner(nn.Module):
    """
    Single front-camera planner with a configurable ResNet visual backbone.

    Drop-in replacement for FrontCamPlanner — identical decoder architecture,
    only the visual encoder differs.

    Args:
        token_dim:   token / model dimensionality D (default 128)
        num_heads:   attention heads (default 4)
        enc_layers:  kinematic encoder layers (default 2)
        dec_layers:  drive decoder layers (default 3)
        ffn_dim:     FFN hidden dim (defaults to token_dim * 4)
        multiscale:  if True, use all 4 ResNet scales; if False, layer4 only
        backbone:    ResNet variant — resnet18/34/50/101/152 (default resnet50)
        frozen:      if True, backbone weights are frozen (default True)
        debug:       if True, print tensor shapes during forward pass
    """

    def __init__(
        self,
        token_dim: int = 128,
        num_heads: int = 4,
        enc_layers: int = 2,
        dec_layers: int = 3,
        ffn_dim: int = None,
        multiscale: bool = False,
        backbone: str = "resnet50",
        frozen: bool = True,
        debug: bool = False,
    ):
        super().__init__()
        D = token_dim
        self.token_dim = D
        self.multiscale = multiscale
        self.frozen = frozen
        self.debug = debug
        ffn_dim = ffn_dim or D * 4

        # ── ResNet backbone ───────────────────────────────────────────────────
        self.encoder = ResNetEncoder(variant=backbone, frozen=frozen)
        all_chans = self.encoder.out_channels   # [C1, C2, C3, C4]

        # ── Visual scale configuration ────────────────────────────────────────
        if multiscale:
            self.num_vis_levels = 4
            self._vis_shapes = [(56, 56), (28, 28), (14, 14), (7, 7)]
            self._vis_chans = all_chans
        else:
            self.num_vis_levels = 1
            self._vis_shapes = [(7, 7)]
            self._vis_chans = [all_chans[-1]]

        # ── Visual feature projections (backbone channels → D) ────────────────
        self.vis_projs = nn.ModuleList([
            nn.Linear(c, D) for c in self._vis_chans
        ])

        # ── Kinematic encoder ─────────────────────────────────────────────────
        self.kin_encoder = KinematicEncoder(D, num_heads, enc_layers, ffn_dim)

        # ── Positional encodings (fixed, registered as buffers) ───────────────
        for k, (H, W) in enumerate(self._vis_shapes):
            pe = make_2d_sincos_pos_enc(H, W, D).unsqueeze(0)  # (1, H*W, D)
            self.register_buffer(f"enc_pos_{k}", pe)

        self.register_buffer("kin_pos",   _make_1d_sincos_pos_enc(41, D))  # (1, 41, D)
        self.register_buffer("drive_pos", _make_1d_sincos_pos_enc(50, D))  # (1, 50, D)

        # ── Drive decoder (50 tokens, num_vis_levels+1 encoder levels) ────────
        self.drive_embed = nn.Embedding(50, D)
        drive_enc_levels = self.num_vis_levels + 1  # +1 for kinematic memory
        self.drive_decoder = nn.ModuleList([
            FlexDecoderLayer(D, num_heads, drive_enc_levels, ffn_dim)
            for _ in range(dec_layers)
        ])
        self.drive_head = nn.Linear(D, 2)

    def _get_enc_pos(self, k: int) -> Tensor:
        return getattr(self, f"enc_pos_{k}")

    def _encode_visual(self, img: Tensor) -> list:
        """
        Run the ResNet backbone on (B, 3, 224, 224) and project each scale to D.
        Returns list of (B, N_k, D) tensors, one per vis level.
        """
        with torch.set_grad_enabled(not self.frozen):
            s0, s1, s2, s3 = self.encoder(img)

        all_feats = [s0, s1, s2, s3] if self.multiscale else [s3]

        enc_feats = []
        for feat, proj in zip(all_feats, self.vis_projs):
            # (B, C_k, H_k, W_k) → (B, N_k, C_k) → (B, N_k, D)
            x = feat.flatten(2).transpose(1, 2)
            enc_feats.append(proj(x))
        return enc_feats

    def forward(self, batch: dict) -> Tensor:
        dbg = self.debug

        if dbg:
            D = self.token_dim
            _dbg_header(f"TENSOR SHAPES  ·  ResNetPlanner  ·  "
                        f"B={batch['past_traj'].shape[0]}  D={D}")
            _dbg_sec("inputs")
            _dbg_row("past_traj",    batch["past_traj"])
            _dbg_row("speed",        batch["speed"])
            _dbg_row("acceleration", batch["acceleration"])
            _dbg_row("command",      batch["command"])
            _dbg_row("images",       batch["images"])

        kin_mem = self.kin_encoder(batch)   # (B, 41, D)
        B = kin_mem.shape[0]

        if dbg:
            _dbg_sec("kinematic encoding")
            _dbg_row("kin_mem", kin_mem, "B, past_steps, D")

        img = batch["images"][:, 0]         # (B, 3, 224, 224) — front cam only

        if dbg:
            frozen_str = "frozen" if self.frozen else "trainable"
            _dbg_sec(f"visual encoding  (ResNet {frozen_str}, front cam only  →  project to D)")
            _dbg_row("img (front cam)", img, "B, 3, H, W")

        enc_feats = self._encode_visual(img)   # list of (B, N_k, D)

        if dbg:
            for k, feat in enumerate(enc_feats):
                H, W = self._vis_shapes[k]
                _dbg_row(f"enc_feats[{k}]  ({H}×{W} → {H*W} tok)", feat,
                         f"scale {k}  (B, N_k, D)")

        enc_feats_drive = list(enc_feats) + [kin_mem]
        enc_pos_drive = [
            self._get_enc_pos(k) for k in range(self.num_vis_levels)
        ]
        enc_pos_drive.append(self.kin_pos)

        drive_tokens = self.drive_embed.weight.unsqueeze(0).expand(B, -1, -1)
        drive_pos = self.drive_pos

        if dbg:
            _dbg_sec(f"drive decoder  ({len(self.drive_decoder)} × FlexDecoder  "
                     f"self-attn + cross-attn × {len(enc_feats_drive)} srcs + FFN)")
            _dbg_row("drive_tokens  (init)", drive_tokens, "B, future_steps, D")

        for layer in self.drive_decoder:
            drive_tokens = layer(drive_tokens, enc_feats_drive, drive_pos, enc_pos_drive)

        if dbg:
            _dbg_row("drive_tokens  (decoded)", drive_tokens)

        future_traj = self.drive_head(drive_tokens)     # (B, 50, 2)

        if dbg:
            _dbg_row("future_traj", future_traj, "B, future_steps, 2")
            print("━" * _DBG_W)
            print(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")
            print("━" * _DBG_W)

        return future_traj


if __name__ == "__main__":
    import torch

    for variant in ["resnet18", "resnet50"]:
        for frozen in [True, False]:
            print(f"\n--- {variant}  frozen={frozen} ---")
            model = ResNetPlanner(backbone=variant, frozen=frozen, debug=False)
            model.eval()
            batch = {
                "past_traj":    torch.zeros(2, 41, 2),
                "speed":        torch.zeros(2, 41),
                "acceleration": torch.zeros(2, 41, 3),
                "command":      torch.zeros(2, dtype=torch.long),
                "images":       torch.zeros(2, 1, 3, 224, 224),
            }
            with torch.no_grad():
                out = model(batch)
            n = sum(p.numel() for p in model.parameters())
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  output: {tuple(out.shape)}  total={n:,}  trainable={n_trainable:,}")
