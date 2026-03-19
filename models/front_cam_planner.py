"""
FrontCamPlanner — Single front-camera ablation planner.

Ablation model answering: "how much does a single front camera help over
kinematics alone?"  Uses one camera instead of six, no depth/semantic
auxiliary heads — plugs directly into E2EDrivingModule.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE  (multiscale=False default, token_dim D=128)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  INPUTS
  images        (B, 1, 3, 224, 224)   — front camera only
  past_traj     (B, 41, 2)  ─┐
  speed         (B, 41)     ─┤─ concat → (B, 41, 7)
  acceleration  (B, 41, 3)  ─┤
  command       (B,)        ─┘

  ┌──────────────────────────────┐    ┌──────────────────────────────────────┐
  │  KinematicEncoder            │    │  TinyViT  (frozen)                   │
  │  Linear(7 → D)               │    │  in:  (B, 3, 224, 224)               │
  │  + 1D sin-cos pos enc        │    │  multiscale=True:  4 scales          │
  │  TransformerEncoder (pre-LN) │    │    s0: (B,  96, 56, 56)  3136 tok    │
  │  enc_layers layers           │    │    s1: (B, 192, 28, 28)   784 tok    │
  │  ──────────────────────────  │    │    s2: (B, 384, 14, 14)   196 tok    │
  │  kin_mem  (B, 41, D)         │    │    s3: (B, 576,  7,  7)    49 tok    │
  └──────────────┬───────────────┘    │  multiscale=False: bottleneck only   │
                 │                    │    s3: (B, 576,  7,  7)    49 tok    │
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

  FlexDecoder layer (pre-norm):
    tokens = tokens + SelfAttn( LN(tokens) + token_pos )
    tokens = tokens + Σ_k CrossAttn_k( LN(tokens) + token_pos,  enc_feats[k] )
    tokens = tokens + FFN( LN(tokens) )
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from ._tinyvit import TinyViTEncoder
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


class FrontCamPlanner(nn.Module):
    """
    Single front-camera ablation planner.

    Fuses a frozen TinyViT visual backbone (front camera only) with a
    kinematic transformer encoder to predict future trajectories.
    No auxiliary depth or semantic heads — use with E2EDrivingModule.

    Args:
        token_dim:   token / model dimensionality D (default 128)
        num_heads:   attention heads (default 4)
        enc_layers:  kinematic encoder layers (default 2)
        dec_layers:  drive decoder layers (default 3)
        ffn_dim:     FFN hidden dim (defaults to token_dim * 4)
        multiscale:  if True, use all 4 TinyViT scales; if False, bottom-only
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
        debug: bool = False,
    ):
        super().__init__()
        D = token_dim
        self.token_dim = D
        self.multiscale = multiscale
        self.debug = debug
        ffn_dim = ffn_dim or D * 4

        # ── Visual scale configuration ────────────────────────────────────────
        if multiscale:
            self.num_vis_levels = 4
            self._vis_shapes = [(56, 56), (28, 28), (14, 14), (7, 7)]
            self._vis_chans = [96, 192, 384, 576]
        else:
            self.num_vis_levels = 1
            self._vis_shapes = [(7, 7)]
            self._vis_chans = [576]

        # ── TinyViT encoder (frozen) ──────────────────────────────────────────
        self.encoder = TinyViTEncoder(img_h=224, img_w=224)

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
        Run TinyViT on (B, 3, 224, 224) front camera image and project each
        scale to D. Returns list of (B, N_k, D) tensors, one per vis level.
        """
        with torch.no_grad():
            skip0, skip1, skip2, bot = self.encoder(img)

        all_feats = [skip0, skip1, skip2, bot] if self.multiscale else [bot]

        enc_feats = []
        for feat, proj in zip(all_feats, self.vis_projs):
            # (B, C_k, H_k, W_k) → (B, N_k, C_k) → (B, N_k, D)
            x = feat.flatten(2).transpose(1, 2)
            enc_feats.append(proj(x))
        return enc_feats

    def forward(self, batch: dict) -> Tensor:
        dbg = self.debug

        # ── Kinematic encoding ────────────────────────────────────────────────
        if dbg:
            D = self.token_dim
            _dbg_header(f"TENSOR SHAPES  ·  FrontCamPlanner  ·  "
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

        # ── Visual encoding ───────────────────────────────────────────────────
        img = batch["images"][:, 0]         # (B, 3, 224, 224) — front cam only

        if dbg:
            _dbg_sec("visual encoding  (TinyViT frozen, front cam only  →  project to D)")
            _dbg_row("img (front cam)", img, "B, 3, H, W")

        enc_feats = self._encode_visual(img)   # list of (B, N_k, D)

        if dbg:
            for k, feat in enumerate(enc_feats):
                H, W = self._vis_shapes[k]
                _dbg_row(f"enc_feats[{k}]  ({H}×{W} → {H*W} tok)", feat,
                         f"scale {k}  (B, N_k, D)")

        # Encoder sources for drive decoder: vis scales + kinematic memory
        enc_feats_drive = list(enc_feats) + [kin_mem]

        enc_pos_drive = [
            self._get_enc_pos(k) for k in range(self.num_vis_levels)
        ]
        enc_pos_drive.append(self.kin_pos)  # (1, 41, D)

        # ── Drive decode ──────────────────────────────────────────────────────
        drive_tokens = self.drive_embed.weight.unsqueeze(0).expand(B, -1, -1)
        drive_pos = self.drive_pos          # (1, 50, D)

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
    print(__doc__)

    B = 2
    token_dim = 128

    model = FrontCamPlanner(
        token_dim=token_dim,
        num_heads=4,
        enc_layers=2,
        dec_layers=3,
        multiscale=False,
        debug=True,
    )
    model.eval()

    batch = {
        "past_traj":    torch.zeros(B, 41, 2),
        "speed":        torch.zeros(B, 41),
        "acceleration": torch.zeros(B, 41, 3),
        "command":      torch.zeros(B, dtype=torch.long),
        "images":       torch.zeros(B, 1, 3, 224, 224),
    }

    with torch.no_grad():
        out = model(batch)

    print(f"\nOutput shape: {tuple(out.shape)}  (expected: ({B}, 50, 2))")
