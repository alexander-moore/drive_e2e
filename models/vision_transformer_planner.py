"""
VisionTransformerPlanner — Multimodal E2E driving planner.

Combines TinyViT visual backbone with kinematic encoder memory.
Three learned token sets: drive (50), depth (28×28 or 7×7), seg (28×28 or 7×7).

Drive tokens cross-attend to all cameras' visual features + kinematic memory.
Depth/seg tokens cross-attend to per-camera visual features for dense prediction.
Auxiliary SILog (depth) and CE+Dice (semantic) losses are computed during training.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from ._tinyvit import TinyViTEncoder
from ._transformer import TokenCNNHead
from ._blocks import make_2d_sincos_pos_enc


NUM_SEM_CLASSES = 28  # CARLA semantic class IDs 0-27 (R channel of instance PNGs)


def _make_1d_sincos_pos_enc(n: int, dim: int) -> Tensor:
    """Fixed 1D sinusoidal position encoding. Returns (1, n, dim)."""
    assert dim % 2 == 0, f"dim must be even, got {dim}"
    pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    enc = torch.zeros(n, dim)
    enc[:, 0::2] = torch.sin(pos * div)
    enc[:, 1::2] = torch.cos(pos * div)
    return enc.unsqueeze(0)  # (1, n, dim)


class FlexDecoderLayer(nn.Module):
    """
    Pre-norm transformer decoder layer with a configurable number of encoder levels.

    Self-attention (tokens attend to each other) followed by DPT-style
    cross-attention (tokens attend separately to each encoder scale, outputs
    summed) followed by a position-wise FFN.

    Pre-norm layout:
        tokens = tokens + SelfAttn  ( LayerNorm(tokens) + token_pos )
        tokens = tokens + Σ_k CrossAttn_k( LayerNorm(tokens) + token_pos,
                                            enc_feats[k]     + enc_pos[k],
                                            enc_feats[k] )
        tokens = tokens + FFN( LayerNorm(tokens) )

    Args:
        token_dim:      dimensionality D of all token vectors
        num_heads:      number of attention heads
        num_enc_levels: number of encoder feature levels to cross-attend to
        ffn_dim:        hidden dim of the FFN (defaults to 4 × token_dim)
    """

    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        num_enc_levels: int,
        ffn_dim: int = None,
    ):
        super().__init__()
        ffn_dim = ffn_dim or token_dim * 4

        self.norm1 = nn.LayerNorm(token_dim)
        self.self_attn = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(token_dim)
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
            for _ in range(num_enc_levels)
        ])

        self.norm3 = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, token_dim),
        )

    def forward(
        self,
        tokens: Tensor,       # (B, N_q, D)
        enc_feats: list,      # num_enc_levels × (B, N_k, D)
        token_pos: Tensor,    # (1, N_q, D)
        enc_pos: list,        # num_enc_levels × (1, N_k, D)
    ) -> Tensor:
        # 1. Self-attention (pre-norm; pos added to Q and K, not V)
        normed = self.norm1(tokens)
        q = normed + token_pos
        tokens = tokens + self.self_attn(q, q, normed)[0]

        # 2. DPT-style cross-attention (pre-norm; sum over encoder levels)
        normed = self.norm2(tokens)
        q = normed + token_pos
        cross_sum = torch.zeros_like(tokens)
        for i, ca in enumerate(self.cross_attns):
            k = enc_feats[i] + enc_pos[i]
            cross_sum = cross_sum + ca(q, k, enc_feats[i])[0]
        tokens = tokens + cross_sum

        # 3. FFN (pre-norm)
        tokens = tokens + self.ffn(self.norm3(tokens))

        return tokens


class KinematicEncoder(nn.Module):
    """
    Encodes past kinematic history into D-dimensional tokens.

    Input: [past_traj(2), speed(1), accel(3), cmd_scalar(1)] = 7 dims per step
    → Linear(7→D) + 1D sinusoidal pos enc
    → TransformerEncoder (pre-LN, enc_layers layers)
    → kinematic_mem: (B, 41, D)
    """

    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        enc_layers: int,
        ffn_dim: int = None,
    ):
        super().__init__()
        ffn_dim = ffn_dim or token_dim * 4
        self.proj = nn.Linear(7, token_dim)
        self.register_buffer("pos_enc", _make_1d_sincos_pos_enc(41, token_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=enc_layers, enable_nested_tensor=False
        )

    def forward(self, batch: dict) -> Tensor:
        B = batch["past_traj"].shape[0]
        past_traj = batch["past_traj"]          # (B, 41, 2)
        speed = batch["speed"].unsqueeze(-1)    # (B, 41, 1)
        accel = batch["acceleration"]           # (B, 41, 3)
        cmd = batch["command"].float() / 3.0    # normalise to [0, 1]
        cmd = cmd.view(B, 1, 1).expand(B, 41, 1)  # (B, 41, 1)

        x = torch.cat([past_traj, speed, accel, cmd], dim=-1)  # (B, 41, 7)
        x = self.proj(x) + self.pos_enc                        # (B, 41, D)
        return self.encoder(x)                                  # (B, 41, D)


class VisionTransformerPlanner(nn.Module):
    """
    Multimodal E2E driving planner that fuses TinyViT visual features with
    kinematic history to predict future trajectories, with optional auxiliary
    depth and semantic segmentation heads.

    Args:
        token_dim:      token / model dimensionality D (default 256)
        num_heads:      attention heads (default 8)
        enc_layers:     kinematic encoder layers (default 3)
        dec_layers:     decoder layers for all three token stacks (default 4)
        ffn_dim:        FFN hidden dim (default 1024)
        multiscale:     if True, use all 4 TinyViT scales; if False, bottom-only
        front_cam_only: if True, process only the front camera (C=1 instead of 6)
    """

    def __init__(
        self,
        token_dim: int = 256,
        num_heads: int = 8,
        enc_layers: int = 3,
        dec_layers: int = 4,
        ffn_dim: int = 1024,
        multiscale: bool = True,
        front_cam_only: bool = False,
    ):
        super().__init__()
        D = token_dim
        self.token_dim = D
        self.multiscale = multiscale
        self.num_cams = 1 if front_cam_only else 6

        # ── Visual scale configuration ────────────────────────────────────────
        if multiscale:
            self.num_vis_levels = 4
            self.token_stride = 8        # depth/seg token grid: 28×28 → stride 8 → 224
            self.n_q = 28 * 28           # 784 depth/seg tokens
            self._vis_shapes = [(56, 56), (28, 28), (14, 14), (7, 7)]
            self._vis_chans = [96, 192, 384, 576]
        else:
            self.num_vis_levels = 1
            self.token_stride = 32       # depth/seg token grid: 7×7 → stride 32 → 224
            self.n_q = 7 * 7             # 49 depth/seg tokens
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

        H_q = W_q = int(math.sqrt(self.n_q))
        aux_pe = make_2d_sincos_pos_enc(H_q, W_q, D).unsqueeze(0)  # (1, N_q, D)
        self.register_buffer("aux_token_pos", aux_pe)

        # ── Drive decoder (50 tokens, num_vis_levels+1 encoder levels) ────────
        self.drive_embed = nn.Embedding(50, D)
        drive_enc_levels = self.num_vis_levels + 1  # +1 for kinematic memory
        self.drive_decoder = nn.ModuleList([
            FlexDecoderLayer(D, num_heads, drive_enc_levels, ffn_dim)
            for _ in range(dec_layers)
        ])
        self.drive_head = nn.Linear(D, 2)

        # ── Depth decoder (N_q tokens per camera, num_vis_levels enc levels) ──
        self.depth_tokens = nn.Parameter(torch.randn(1, self.n_q, D) * 0.02)
        self.depth_decoder = nn.ModuleList([
            FlexDecoderLayer(D, num_heads, self.num_vis_levels, ffn_dim)
            for _ in range(dec_layers)
        ])
        self.depth_head = TokenCNNHead(D, token_stride=self.token_stride, out_channels=1)

        # ── Seg decoder (N_q tokens per camera, num_vis_levels enc levels) ────
        self.seg_tokens = nn.Parameter(torch.randn(1, self.n_q, D) * 0.02)
        self.seg_decoder = nn.ModuleList([
            FlexDecoderLayer(D, num_heads, self.num_vis_levels, ffn_dim)
            for _ in range(dec_layers)
        ])
        self.seg_head = TokenCNNHead(D, token_stride=self.token_stride, out_channels=NUM_SEM_CLASSES)

    def _get_enc_pos(self, k: int) -> Tensor:
        return getattr(self, f"enc_pos_{k}")

    def _encode_visual(self, imgs_flat: Tensor) -> list:
        """
        Run TinyViT on (B*C, 3, 224, 224) images and project each scale to D.
        Returns list of (B*C, N_k, D) tensors, one per vis level.
        """
        with torch.no_grad():
            skip0, skip1, skip2, bot = self.encoder(imgs_flat)

        all_feats = [skip0, skip1, skip2, bot] if self.multiscale else [bot]

        enc_feats_cam = []
        for feat, proj in zip(all_feats, self.vis_projs):
            # (B*C, C_k, H_k, W_k) → (B*C, N_k, C_k) → (B*C, N_k, D)
            x = feat.flatten(2).transpose(1, 2)
            enc_feats_cam.append(proj(x))
        return enc_feats_cam

    def _decode_aux(
        self,
        B: int,
        C: int,
        enc_feats_cam: list,
        tokens_param: Tensor,
        decoder: nn.ModuleList,
        head: nn.Module,
    ) -> Tensor:
        """
        Decode depth or semantic tokens for all cameras independently.
        Returns (B, C, out_ch, 224, 224).
        """
        BC = B * C
        tokens = tokens_param.expand(BC, -1, -1)          # (B*C, N_q, D)
        token_pos = self.aux_token_pos                      # (1, N_q, D)
        enc_pos = [self._get_enc_pos(k) for k in range(self.num_vis_levels)]

        for layer in decoder:
            tokens = layer(tokens, enc_feats_cam, token_pos, enc_pos)

        H_q = W_q = int(math.sqrt(self.n_q))
        # (B*C, N_q, D) → (B*C, D, H_q, W_q)
        tokens_2d = tokens.permute(0, 2, 1).reshape(BC, self.token_dim, H_q, W_q)

        out_flat = head(tokens_2d)                          # (B*C, out_ch, 224, 224)
        out_ch = out_flat.shape[1]
        return out_flat.view(B, C, out_ch, 224, 224)

    def forward(self, batch: dict) -> dict:
        # ── Kinematic encoding ────────────────────────────────────────────────
        kin_mem = self.kin_encoder(batch)   # (B, 41, D)
        B = kin_mem.shape[0]

        # ── Visual encoding ───────────────────────────────────────────────────
        C = self.num_cams
        imgs = batch["images"][:, :C]       # (B, C, 3, 224, 224)
        imgs_flat = imgs.flatten(0, 1)      # (B*C, 3, 224, 224)

        enc_feats_cam = self._encode_visual(imgs_flat)      # list of (B*C, N_k, D)

        # Multi-camera feats for drive: concat cameras along seq dim
        enc_feats_drive = []
        for feat in enc_feats_cam:
            # (B*C, N_k, D) → (B, C, N_k, D) → (B, C*N_k, D)
            feat_mc = feat.reshape(B, C, -1, self.token_dim).flatten(1, 2)
            enc_feats_drive.append(feat_mc)
        enc_feats_drive.append(kin_mem)     # (B, 41, D)

        # Positional encodings for drive decoder
        enc_pos_drive = [
            self._get_enc_pos(k).repeat(1, C, 1)   # (1, C*N_k, D)
            for k in range(self.num_vis_levels)
        ]
        enc_pos_drive.append(self.kin_pos)  # (1, 41, D)

        # ── Drive decode ──────────────────────────────────────────────────────
        drive_tokens = self.drive_embed.weight.unsqueeze(0).expand(B, -1, -1)
        drive_pos = self.drive_pos          # (1, 50, D)

        for layer in self.drive_decoder:
            drive_tokens = layer(drive_tokens, enc_feats_drive, drive_pos, enc_pos_drive)

        future_traj = self.drive_head(drive_tokens)         # (B, 50, 2)
        out = {"future_traj": future_traj}

        # ── Auxiliary decodes (only when labels are in batch) ─────────────────
        if "depth" in batch:
            out["depth"] = self._decode_aux(
                B, C, enc_feats_cam,
                self.depth_tokens, self.depth_decoder, self.depth_head,
            )

        if "semantic" in batch:
            out["semantic"] = self._decode_aux(
                B, C, enc_feats_cam,
                self.seg_tokens, self.seg_decoder, self.seg_head,
            )

        return out
