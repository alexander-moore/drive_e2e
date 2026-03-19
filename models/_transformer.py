"""
Shared transformer decoder components used by video_former_depth and
video_former_seg_depth.
"""
import math
import torch
import torch.nn as nn


class DepthDecoderLayer(nn.Module):
    """
    One pre-norm transformer decoder layer.

    Self-attention (tokens attend to each other) followed by DPT-style
    cross-attention (tokens attend separately to each of 4 encoder scales,
    outputs summed) followed by a position-wise FFN.

    Pre-norm layout:
        tokens = tokens + SelfAttn  ( LayerNorm(tokens) + token_pos )
        tokens = tokens + Σ_k CrossAttn_k( LayerNorm(tokens) + token_pos,
                                            enc_feats[k]     + enc_pos[k],
                                            enc_feats[k] )
        tokens = tokens + FFN( LayerNorm(tokens) )
    """

    NUM_ENC_LEVELS = 4

    def __init__(self, token_dim: int, num_heads: int, ffn_dim: int = None):
        super().__init__()
        ffn_dim = ffn_dim or token_dim * 4

        self.norm1     = nn.LayerNorm(token_dim)
        self.self_attn = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)

        self.norm2       = nn.LayerNorm(token_dim)
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
            for _ in range(self.NUM_ENC_LEVELS)
        ])

        self.norm3 = nn.LayerNorm(token_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(token_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, token_dim),
        )

    @staticmethod
    def _add_pos(t: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return t + pos

    def forward(
        self,
        tokens:    torch.Tensor,       # (B, N_q, D)
        enc_feats: list,               # 4 × (B, N_k, D)  — projected encoder features
        token_pos: torch.Tensor,       # (1, N_q, D)
        enc_pos:   list,               # 4 × (1, N_k, D)
    ) -> torch.Tensor:

        # 1. Self-attention (pre-norm; pos added to Q and K, not V)
        normed = self.norm1(tokens)
        q = self._add_pos(normed, token_pos)
        tokens = tokens + self.self_attn(q, q, normed)[0]

        # 2. DPT-style cross-attention (pre-norm; sum over 4 encoder levels)
        normed = self.norm2(tokens)
        q = self._add_pos(normed, token_pos)
        cross_sum = torch.zeros_like(tokens)
        for i, ca in enumerate(self.cross_attns):
            k = self._add_pos(enc_feats[i], enc_pos[i])
            cross_sum = cross_sum + ca(q, k, enc_feats[i])[0]
        tokens = tokens + cross_sum

        # 3. FFN (pre-norm)
        tokens = tokens + self.ffn(self.norm3(tokens))

        return tokens


class TokenCNNHead(nn.Module):
    """
    Progressive upsampler: (B, token_dim, H_q, W_q) → (B, out_channels, H, W).

    Uses log2(token_stride) ConvTranspose2d stages, each doubling spatial
    resolution and halving channel count (floor at 32).
    """

    def __init__(self, token_dim: int, token_stride: int, out_channels: int = 1):
        super().__init__()
        assert (token_stride & (token_stride - 1)) == 0, \
            "token_stride must be a power of 2"
        n_ups = int(math.log2(token_stride))

        dims = [max(32, token_dim >> i) for i in range(n_ups + 1)]

        layers = []
        for i in range(n_ups):
            layers += [
                nn.ConvTranspose2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                nn.BatchNorm2d(dims[i + 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(dims[i + 1], dims[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm2d(dims[i + 1]),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv2d(dims[-1], out_channels, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
