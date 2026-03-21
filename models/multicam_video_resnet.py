"""
MulticamVideoResNet — Multi-camera, multi-frame temporal planner.

Extends ResNetPlanner with:
  - All 6 cameras instead of front-only
  - Exponentially-sampled past frames (n_img_frames, default 8)
  - Three additive positional encodings per visual token:
      temporal (41-step sincos, shared with kinematic tokens)
      camera   (learned embedding per camera)
      spatial  (2D sincos per ResNet scale)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE  (multiscale=True, N_f=8 frames, C=6 cameras, token_dim D=256)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  INPUTS
  images              (B, N_f, C, 3, H, W)   N_f=8 frames, C=6 cameras
  img_frame_positions (B, N_f)               distances from anchor (0=present)
  past_traj           (B, 41, 2)  ─┐
  speed               (B, 41)     ─┤─ concat → (B, 41, 10)
  acceleration        (B, 41, 3)  ─┤
  command             (B,)        ─┘  (one-hot dim 4, broadcast across time)

  ┌──────────────────────────────┐    ┌──────────────────────────────────────────┐
  │  KinematicEncoder            │    │  ResNet  (pretrained, frozen or not)     │
  │  Linear(10 → D)              │    │  in:  (B*N_f*C, 3, H, W)                │
  │  + 1D sin-cos pos enc        │    │  multiscale=True:  4 scales              │
  │  TransformerEncoder (pre-LN) │    │    s0: (B*N_f*C, C1, 56, 56)            │
  │  enc_layers layers           │    │    s1: (B*N_f*C, C2, 28, 28)            │
  │  kin_mem  (B, 41, D)         │    │    s2: (B*N_f*C, C3, 14, 14)            │
  └──────────────┬───────────────┘    │    s3: (B*N_f*C, C4,  7,  7)            │
                 │                    └───────────────┬──────────────────────────┘
                 │                                    │ vis_projs[k]: Linear(C_k → D)
                 │                                    │ + temporal sincos PE  (B, N_f*C*N_k, D)
                 │                                    │ + camera learned PE   (B, N_f*C*N_k, D)
                 │                                    │ + spatial 2D sincos PE (B, N_f*C*N_k, D)
                 │                    enc_feats: list of (B, N_f*C*N_k, D)
                 │                    enc_pos:   list of (B, N_f*C*N_k, D)
                 └─────────────────────────────┬──────┘
                                               │
                                    ┌──────────┴──────────────────────────┐
                                    │  Drive decoder                      │
                                    │  init: drive_embed  (B, 50, D)      │
                                    │  enc sources:                        │
                                    │    vis: enc_feats × num_vis_levels   │
                                    │    kin: (B, 41, D)                   │
                                    │  dec_layers × FlexDecoderLayer       │
                                    └──────────────┬──────────────────────┘
                                                   │
                                            Linear(D → 2)
                                                   │
                                       future_traj (B, 50, 2)

  Positional encoding alignment (all share the 41-step sincos space):
    Kinematic tokens : index 0 (oldest) → 40 (present)
    Visual tokens    : sincos index = 40 - distance_from_anchor
    Drive tokens     : own 50-step sincos (independent)
"""

import torch
import torch.nn as nn
from torch import Tensor

from ._blocks import make_2d_sincos_pos_enc
from .resnet_planner import ResNetEncoder
from .vision_transformer_planner import (
    KinematicEncoder,
    FlexDecoderLayer,
    _make_1d_sincos_pos_enc,
    _dbg_header,
    _dbg_sec,
    _dbg_row,
    _DBG_W,
)

# Must match dataset.py PAST_STEPS / PAST_STEPS_TOTAL constants.
_PAST_STEPS       = 40   # history length (4 s @ 10 Hz)
_PAST_STEPS_TOTAL = 41   # _PAST_STEPS + anchor frame


class MulticamVideoResNet(nn.Module):
    """
    Multi-camera, multi-frame temporal planner with a ResNet visual backbone.

    Args:
        token_dim:    token / model dimensionality D (default 256)
        num_heads:    attention heads (default 4)
        enc_layers:   kinematic encoder layers (default 3)
        dec_layers:   drive decoder layers (default 3)
        ffn_dim:      FFN hidden dim (defaults to token_dim * 4)
        multiscale:   if True, use all 4 ResNet scales; if False, layer4 only
        backbone:     ResNet variant — resnet18/34/50/101/152 (default resnet18)
        frozen:       if True, backbone weights are frozen (default True)
        n_img_frames: number of past frames sampled per sample (default 8)
        n_cameras:    number of cameras (default 6)
        debug:        if True, print tensor shapes during forward pass
    """

    def __init__(
        self,
        token_dim: int = 256,
        num_heads: int = 4,
        enc_layers: int = 3,
        dec_layers: int = 3,
        ffn_dim: int = None,
        multiscale: bool = True,
        backbone: str = "resnet18",
        frozen: bool = False,
        n_img_frames: int = 8,
        n_cameras: int = 6,
        debug: bool = False,
    ):
        super().__init__()
        D = token_dim
        self.token_dim   = D
        self.multiscale  = multiscale
        self.frozen      = frozen
        self.n_img_frames = n_img_frames
        self.n_cameras   = n_cameras
        self.debug       = debug
        ffn_dim = ffn_dim or D * 4

        # ── ResNet backbone ───────────────────────────────────────────────────
        self.encoder = ResNetEncoder(variant=backbone, frozen=frozen)
        all_chans = self.encoder.out_channels   # [C1, C2, C3, C4]

        # ── Visual scale configuration ────────────────────────────────────────
        if multiscale:
            self.num_vis_levels = 4
            self._vis_shapes = [(56, 56), (28, 28), (14, 14), (7, 7)]
            self._vis_chans  = all_chans
        else:
            self.num_vis_levels = 1
            self._vis_shapes = [(7, 7)]
            self._vis_chans  = [all_chans[-1]]

        # ── Visual feature projections (backbone channels → D) ────────────────
        self.vis_projs = nn.ModuleList([
            nn.Linear(c, D) for c in self._vis_chans
        ])

        # ── Camera positional embedding (learned) ─────────────────────────────
        self.cam_embed = nn.Embedding(n_cameras, D)

        # ── Kinematic encoder ─────────────────────────────────────────────────
        self.kin_encoder = KinematicEncoder(D, num_heads, enc_layers, ffn_dim)

        # ── Positional encodings (fixed buffers) ──────────────────────────────
        # 2D spatial per scale
        for k, (H, W) in enumerate(self._vis_shapes):
            pe = make_2d_sincos_pos_enc(H, W, D).unsqueeze(0)  # (1, H*W, D)
            self.register_buffer(f"enc_pos_{k}", pe)

        # 1D temporal sincos: 41 steps, shared with kinematic tokens
        self.register_buffer("kin_pos",   _make_1d_sincos_pos_enc(_PAST_STEPS_TOTAL, D))  # (1, 41, D)
        self.register_buffer("drive_pos", _make_1d_sincos_pos_enc(50, D))                 # (1, 50, D)

        # ── Drive decoder (50 tokens, num_vis_levels+1 encoder levels) ────────
        self.drive_embed = nn.Embedding(50, D)
        drive_enc_levels = self.num_vis_levels + 1  # +1 for kinematic memory
        self.drive_decoder = nn.ModuleList([
            FlexDecoderLayer(D, num_heads, drive_enc_levels, ffn_dim)
            for _ in range(dec_layers)
        ])
        self.drive_head = nn.Linear(D, 2)

    def _encode_visual(self, batch: dict) -> tuple:
        """
        Run the ResNet backbone on all frames × cameras and project to D.

        Returns:
            enc_feats: list of (B, N_f*C*N_k, D)  — one per vis level
            enc_pos:   list of (B, N_f*C*N_k, D)  — additive temporal+camera+spatial PE
        """
        imgs        = batch["images"]               # (B, N_f, C, 3, H, W)
        frame_dists = batch["img_frame_positions"]  # (B, N_f) distances from anchor
        B, N_f, C   = imgs.shape[:3]

        # Flatten all frames and cameras for a single batched backbone pass
        flat = imgs.reshape(B * N_f * C, 3, *imgs.shape[-2:])
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            s0, s1, s2, s3 = self.encoder(flat)

        all_feats = [s0, s1, s2, s3] if self.multiscale else [s3]

        enc_feats, enc_pos = [], []
        for k, (feat, proj) in enumerate(zip(all_feats, self.vis_projs)):
            _, Ck, Hk, Wk = feat.shape
            N_k = Hk * Wk

            # Project to D: (B*N_f*C, Ck, Hk, Wk) → (B*N_f*C, N_k, D)
            x = proj(feat.flatten(2).transpose(1, 2))
            x = x.reshape(B, N_f, C, N_k, -1)  # (B, N_f, C, N_k, D)

            # ── Temporal PE: sincos index = PAST_STEPS - distance ─────────────
            # distance=0 (present) → index 40; distance=40 (oldest) → index 0
            sincos_idx = _PAST_STEPS - frame_dists          # (B, N_f)
            t_pe = self.kin_pos[0, sincos_idx, :]            # (B, N_f, D)
            t_pe = t_pe[:, :, None, None, :].expand_as(x)   # (B, N_f, C, N_k, D)

            # ── Camera PE: learned embedding ──────────────────────────────────
            c_ids = torch.arange(C, device=x.device)
            c_pe  = self.cam_embed(c_ids)                    # (C, D)
            c_pe  = c_pe[None, None, :, None, :].expand_as(x)

            # ── Spatial PE: 2D sincos per scale ───────────────────────────────
            s_pe = getattr(self, f"enc_pos_{k}")             # (1, N_k, D)
            s_pe = s_pe[:, None, None, :, :].expand_as(x)

            enc_feats.append(x.reshape(B, N_f * C * N_k, -1))
            enc_pos.append((t_pe + c_pe + s_pe).reshape(B, N_f * C * N_k, -1))

        return enc_feats, enc_pos

    def forward(self, batch: dict) -> Tensor:
        dbg = self.debug

        if dbg:
            D = self.token_dim
            _dbg_header(f"TENSOR SHAPES  ·  MulticamVideoResNet  ·  "
                        f"B={batch['past_traj'].shape[0]}  D={D}")
            _dbg_sec("inputs")
            _dbg_row("past_traj",            batch["past_traj"])
            _dbg_row("speed",                batch["speed"])
            _dbg_row("acceleration",         batch["acceleration"])
            _dbg_row("command",              batch["command"])
            _dbg_row("images",               batch["images"])
            _dbg_row("img_frame_positions",  batch["img_frame_positions"])

        kin_mem = self.kin_encoder(batch)   # (B, 41, D)
        B = kin_mem.shape[0]

        if dbg:
            _dbg_sec("kinematic encoding")
            _dbg_row("kin_mem", kin_mem, "B, past_steps, D")

        enc_feats, enc_pos_vis = self._encode_visual(batch)

        if dbg:
            frozen_str = "frozen" if self.frozen else "trainable"
            N_f = batch["images"].shape[1]
            C   = batch["images"].shape[2]
            _dbg_sec(f"visual encoding  (ResNet {frozen_str}, {N_f} frames × {C} cams)")
            for k, feat in enumerate(enc_feats):
                H, W = self._vis_shapes[k]
                _dbg_row(f"enc_feats[{k}]  ({H}×{W} → {H*W} tok × {N_f}f × {C}c)", feat,
                         f"scale {k}  (B, N_f*C*N_k, D)")

        enc_feats_drive = enc_feats + [kin_mem]
        enc_pos_drive   = enc_pos_vis + [self.kin_pos.expand(B, -1, -1)]

        drive_tokens = self.drive_embed.weight.unsqueeze(0).expand(B, -1, -1)

        if dbg:
            _dbg_sec(f"drive decoder  ({len(self.drive_decoder)} × FlexDecoder  "
                     f"self-attn + cross-attn × {len(enc_feats_drive)} srcs + FFN)")
            _dbg_row("drive_tokens  (init)", drive_tokens, "B, future_steps, D")

        for layer in self.drive_decoder:
            drive_tokens = layer(drive_tokens, enc_feats_drive, self.drive_pos, enc_pos_drive)

        if dbg:
            _dbg_row("drive_tokens  (decoded)", drive_tokens)

        future_traj = self.drive_head(drive_tokens)  # (B, 50, 2)

        if dbg:
            _dbg_row("future_traj", future_traj, "B, future_steps, 2")
            print("━" * _DBG_W)
            print(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")
            print("━" * _DBG_W)

        return future_traj


if __name__ == "__main__":
    import torch

    for variant in ["resnet18", "resnet50"]:
        print(f"\n--- {variant} ---")
        model = MulticamVideoResNet(
            backbone=variant, frozen=False, n_img_frames=8, n_cameras=6, debug=True
        )
        model.eval()
        B, N_f, C = 2, 8, 6
        batch = {
            "past_traj":           torch.zeros(B, 41, 2),
            "speed":               torch.zeros(B, 41),
            "acceleration":        torch.zeros(B, 41, 3),
            "command":             torch.zeros(B, dtype=torch.long),
            "images":              torch.zeros(B, N_f, C, 3, 224, 224),
            "img_frame_positions": torch.randint(0, 41, (B, N_f)),
        }
        with torch.no_grad():
            out = model(batch)
        n = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  output: {tuple(out.shape)}  total={n:,}  trainable={n_trainable:,}")
