"""
Video autoencoder for self-supervised pretraining.

Architecture:
  VideoEncoder          — factorized ViViT-style (TemporalBlock + CameraBlock)
  VideoReconstructionDecoder — temporal query decoder → TokenCNNHead upsampler
  VideoPredictionModel  — encoder + decoder

Pretraining objective:
  Encode ALL T = n_frames + m_frames frames jointly, then reconstruct all T frames.
  This forces the encoder to learn motion- and scene-aware features covering the
  full temporal window.

Downstream planning:
  model.encoder.encode(x)  accepts (B, n_cams, T', 3, H, W) for any T' ≤ max_frames
  and returns (B, n_cams, token_dim, H/32, W/32) — compatible with cross-attention
  planners.  At inference, pass the n_frames context window; at pretraining, pass
  the full n_frames + m_frames window.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm

try:
    from models._blocks import make_2d_sincos_pos_enc, _RESNET_CHANNELS, _RESNET_FN
    from models._transformer import TokenCNNHead
    from models._tinyvit import TinyViTEncoder
except ImportError:
    from e2e.models._blocks import make_2d_sincos_pos_enc, _RESNET_CHANNELS, _RESNET_FN
    from e2e.models._transformer import TokenCNNHead
    from e2e.models._tinyvit import TinyViTEncoder


SPATIAL_STRIDE = 32  # all backbone choices downsample by this factor

_RESNET_WEIGHTS = {
    "resnet18": tvm.ResNet18_Weights.DEFAULT,
    "resnet50": tvm.ResNet50_Weights.DEFAULT,
}


# ── Spatial backbone wrappers ─────────────────────────────────────────────────

class _ResNetSpatialBackbone(nn.Module):
    """ResNet with avgpool + fc removed, returning (N, C, H/32, W/32)."""

    def __init__(self, name: str, pretrained: bool = True):
        super().__init__()
        weights = _RESNET_WEIGHTS[name] if pretrained else None
        backbone = _RESNET_FN[name](weights=weights)
        # children(): conv1, bn1, relu, maxpool, layer1..4, avgpool, fc
        self.net = nn.Sequential(*list(backbone.children())[:-2])
        self.out_channels = _RESNET_CHANNELS[name][-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Pre-norm transformer helpers ──────────────────────────────────────────────

class _PreNormAttnLayer(nn.Module):
    """Pre-norm self-attention + FFN layer."""

    def __init__(self, token_dim: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(token_dim)
        self.attn  = nn.MultiheadAttention(token_dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(token_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(token_dim, token_dim * 4),
            nn.GELU(),
            nn.Linear(token_dim * 4, token_dim),
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor = None) -> torch.Tensor:
        normed = self.norm1(x)
        q = k = (normed + pos) if pos is not None else normed
        x = x + self.attn(q, k, normed)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class _PreNormDecoderLayer(nn.Module):
    """Pre-norm transformer decoder layer (self-attn + cross-attn + FFN)."""

    def __init__(self, token_dim: int, n_heads: int):
        super().__init__()
        self.norm1      = nn.LayerNorm(token_dim)
        self.self_attn  = nn.MultiheadAttention(token_dim, n_heads, batch_first=True)
        self.norm2      = nn.LayerNorm(token_dim)
        self.cross_attn = nn.MultiheadAttention(token_dim, n_heads, batch_first=True)
        self.norm3      = nn.LayerNorm(token_dim)
        self.ffn        = nn.Sequential(
            nn.Linear(token_dim, token_dim * 4),
            nn.GELU(),
            nn.Linear(token_dim * 4, token_dim),
        )

    def forward(
        self,
        queries:    torch.Tensor,           # (B, N_q, D)
        memory:     torch.Tensor,           # (B, N_k, D)
        query_pos:  torch.Tensor = None,    # (1, N_q, D)
        memory_pos: torch.Tensor = None,    # (1, N_k, D)
    ) -> torch.Tensor:
        # Self-attention
        normed = self.norm1(queries)
        q = k = (normed + query_pos) if query_pos is not None else normed
        queries = queries + self.self_attn(q, k, normed)[0]
        # Cross-attention
        normed = self.norm2(queries)
        q = (normed + query_pos) if query_pos is not None else normed
        k = (memory + memory_pos) if memory_pos is not None else memory
        queries = queries + self.cross_attn(q, k, memory)[0]
        # FFN
        queries = queries + self.ffn(self.norm3(queries))
        return queries


# ── Factorized attention blocks ───────────────────────────────────────────────

class TemporalBlock(nn.Module):
    """Self-attention across n_frames per camera per spatial location.

    Pos enc is sized for max_frames (the pretraining window size) and sliced to
    the actual T at runtime, so the same block handles both the full pretraining
    sequence and shorter inference-time context windows.
    """

    def __init__(self, max_frames: int, token_dim: int, n_heads: int):
        super().__init__()
        self.temporal_pos = nn.Parameter(torch.zeros(1, max_frames, token_dim))
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        self.layer = _PreNormAttnLayer(token_dim, n_heads)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, n_cams, T, Hs_Ws, D)   T ≤ max_frames
        B, n_cams, T, Hs_Ws, D = tokens.shape
        # → (B*n_cams*Hs_Ws, T, D)
        x = tokens.permute(0, 1, 3, 2, 4).reshape(B * n_cams * Hs_Ws, T, D)
        x = self.layer(x, self.temporal_pos[:, :T, :])   # slice to actual T
        # → (B, n_cams, T, Hs_Ws, D)
        x = x.reshape(B, n_cams, Hs_Ws, T, D).permute(0, 1, 3, 2, 4)
        return x


class CameraBlock(nn.Module):
    """Self-attention across n_cams per frame per spatial location."""

    def __init__(self, n_cams: int, token_dim: int, n_heads: int):
        super().__init__()
        self.cam_embed = nn.Parameter(torch.zeros(1, n_cams, token_dim))
        nn.init.trunc_normal_(self.cam_embed, std=0.02)
        self.layer = _PreNormAttnLayer(token_dim, n_heads)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, n_cams, n_frames, Hs_Ws, D)
        B, n_cams, n_frames, Hs_Ws, D = tokens.shape
        # → (B*n_frames*Hs_Ws, n_cams, D)
        x = tokens.permute(0, 2, 3, 1, 4).reshape(B * n_frames * Hs_Ws, n_cams, D)
        x = self.layer(x, self.cam_embed)
        # → (B, n_cams, n_frames, Hs_Ws, D)
        x = x.reshape(B, n_frames, Hs_Ws, n_cams, D).permute(0, 3, 1, 2, 4)
        return x


# ── Encoder ───────────────────────────────────────────────────────────────────

class VideoEncoder(nn.Module):
    """
    Factorized ViViT-style encoder for multi-camera video.

    Input:  (B, n_cams, T, 3, H, W)   T ≤ max_frames
    Output: (B, n_cams, token_dim, H/32, W/32)   via encode()

    Steps:
      1. Flatten → spatial backbone → linear project
      2. Add 2D sincos pos enc → tokens (B, n_cams, T, H_s*W_s, token_dim)
      3. n_layers × (TemporalBlock, CameraBlock)
      4. Mean-pool over T → reshape to spatial map

    max_frames is the largest T this encoder will ever see (= n_frames + m_frames
    during pretraining).  The temporal pos enc is sized for max_frames and sliced
    at runtime, so encode() also accepts shorter sequences (e.g. n_frames context
    only) at inference time.
    """

    def __init__(
        self,
        max_frames:      int,
        n_cams:          int,
        token_dim:       int,
        n_layers:        int,
        n_heads:         int,
        spatial_encoder: str = "resnet18",
        image_size:      Tuple[int, int] = (224, 224),
    ):
        super().__init__()
        self.n_cams    = n_cams
        self.token_dim = token_dim

        H, W = image_size
        self.H_s = H // SPATIAL_STRIDE
        self.W_s = W // SPATIAL_STRIDE

        # Spatial backbone
        if spatial_encoder == "tinyvit":
            assert H == W, "TinyViT requires square images"
            self.backbone = TinyViTEncoder(img_h=H, img_w=W)
            backbone_channels = 576
            self._tinyvit = True
        else:
            self.backbone = _ResNetSpatialBackbone(spatial_encoder)
            backbone_channels = self.backbone.out_channels
            self._tinyvit = False

        self.proj = nn.Linear(backbone_channels, token_dim)

        # Fixed 2D sincos positional encoding for spatial tokens
        assert token_dim % 4 == 0, "token_dim must be divisible by 4"
        pos = make_2d_sincos_pos_enc(self.H_s, self.W_s, token_dim)  # (H_s*W_s, D)
        self.register_buffer("spatial_pos", pos.unsqueeze(0))         # (1, H_s*W_s, D)

        # Factorized attention: n_layers pairs of (TemporalBlock, CameraBlock)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(TemporalBlock(max_frames, token_dim, n_heads))
            self.blocks.append(CameraBlock(n_cams, token_dim, n_heads))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_cams, T, 3, H, W)   T ≤ max_frames
               Pass all n_frames+m_frames during pretraining;
               pass n_frames context only at inference/planning time.
        Returns:
            latent: (B, n_cams, token_dim, H_s, W_s)
        """
        B, n_cams, n_frames, _, H, W = x.shape

        # ── Spatial encoding ──────────────────────────────────────────────
        x_flat = x.flatten(0, 2)   # (B*n_cams*n_frames, 3, H, W)

        if self._tinyvit:
            feats = self.backbone(x_flat)[-1]   # bottleneck: (N, 576, H_s, W_s)
        else:
            feats = self.backbone(x_flat)        # (N, C, H_s, W_s)

        N, C, H_s, W_s = feats.shape
        # (N, H_s*W_s, C) → project → (N, H_s*W_s, token_dim)
        tokens = self.proj(feats.permute(0, 2, 3, 1).reshape(N, H_s * W_s, C))
        tokens = tokens + self.spatial_pos                              # add 2D pos enc

        # ── Factorized attention ──────────────────────────────────────────
        tokens = tokens.reshape(B, n_cams, n_frames, H_s * W_s, self.token_dim)
        for block in self.blocks:
            tokens = block(tokens)

        # ── Aggregate: mean over n_frames, reshape to spatial map ─────────
        tokens = tokens.mean(dim=2)                       # (B, n_cams, H_s*W_s, D)
        latent = (tokens
                  .permute(0, 1, 3, 2)                   # (B, n_cams, D, H_s*W_s)
                  .reshape(B, n_cams, self.token_dim, H_s, W_s))
        return latent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


# ── Decoder ───────────────────────────────────────────────────────────────────

class VideoReconstructionDecoder(nn.Module):
    """
    Temporal-query decoder: reconstructs n_out frames from a latent.

    During pretraining n_out = n_frames + m_frames (reconstruct everything).
    The same decoder could reconstruct any subset by changing n_out at init.

    Input:  latent (B, n_cams, token_dim, H_s, W_s)
    Output: pred   (B, n_cams, n_out, 3, H, W)

    Steps:
      1. Flatten cameras → (B*n_cams, token_dim, H_s, W_s)
      2. Flatten spatial → memory (B*n_cams, H_s*W_s, token_dim) + 2D sincos keys
      3. n_out learned temporal queries (B*n_cams, n_out, token_dim)
      4. TransformerDecoder (n_layers): self-attn over queries + cross-attn to memory
      5. Broadcast decoded frame embeddings over spatial memory tokens
         → (B*n_cams*n_out, token_dim, H_s, W_s)
      6. TokenCNNHead(stride=32) → (B*n_cams*n_out, 3, H, W)
      7. Sigmoid → [0, 1]
      8. Reshape → (B, n_cams, n_out, 3, H, W)
    """

    def __init__(
        self,
        n_cams:     int,
        n_out:      int,
        token_dim:  int,
        n_layers:   int,
        n_heads:    int,
        image_size: Tuple[int, int] = (224, 224),
    ):
        super().__init__()
        self.n_cams    = n_cams
        self.n_out     = n_out
        self.token_dim = token_dim

        H, W = image_size
        self.H_s = H // SPATIAL_STRIDE
        self.W_s = W // SPATIAL_STRIDE

        # 2D sincos positional encoding for memory (encoder spatial tokens)
        mem_pos = make_2d_sincos_pos_enc(self.H_s, self.W_s, token_dim)  # (H_s*W_s, D)
        self.register_buffer("memory_pos", mem_pos.unsqueeze(0))          # (1, H_s*W_s, D)

        # Learned temporal queries and their positional encoding
        self.temporal_queries = nn.Parameter(torch.zeros(n_out, token_dim))
        self.temporal_pos     = nn.Parameter(torch.zeros(1, n_out, token_dim))
        nn.init.trunc_normal_(self.temporal_queries, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos,     std=0.02)

        # Decoder layers
        self.layers = nn.ModuleList([
            _PreNormDecoderLayer(token_dim, n_heads) for _ in range(n_layers)
        ])

        # Upsample spatial map to full resolution
        self.head = TokenCNNHead(token_dim, SPATIAL_STRIDE, out_channels=3)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, n_cams, token_dim, H_s, W_s)
        Returns:
            pred_frames: (B, n_cams, n_out, 3, H, W)
        """
        B, n_cams, D, H_s, W_s = latent.shape

        # (B*n_cams, D, H_s, W_s) → flatten spatial → (B*n_cams, H_s*W_s, D)
        memory = (latent
                  .flatten(0, 1)                           # (B*n_cams, D, H_s, W_s)
                  .permute(0, 2, 3, 1)                     # (B*n_cams, H_s, W_s, D)
                  .reshape(B * n_cams, H_s * W_s, D))

        # Broadcast learned queries → (B*n_cams, n_out, D)
        queries = self.temporal_queries.unsqueeze(0).expand(B * n_cams, -1, -1)

        # Transformer decoder
        for layer in self.layers:
            queries = layer(queries, memory,
                            query_pos=self.temporal_pos,
                            memory_pos=self.memory_pos)
        # queries: (B*n_cams, n_out, D)

        # Broadcast frame embeddings over spatial positions, then add spatial context.
        # frame_emb: (B*n_cams, n_out, 1,       D)
        # memory:    (B*n_cams, 1,     H_s*W_s, D)
        # combined:  (B*n_cams, n_out, H_s*W_s, D)
        combined = queries.unsqueeze(2) + memory.unsqueeze(1)

        # Reshape to (B*n_cams*n_out, D, H_s, W_s) for the CNN head
        combined = (combined
                    .reshape(B * n_cams * self.n_out, H_s * W_s, D)
                    .permute(0, 2, 1)                      # (N, D, H_s*W_s)
                    .reshape(B * n_cams * self.n_out, D, H_s, W_s))

        # Upsample → (B*n_cams*n_out, 3, H, W), sigmoid, reshape
        pred = torch.sigmoid(self.head(combined))
        pred = pred.reshape(B, n_cams, self.n_out, 3, H_s * SPATIAL_STRIDE, W_s * SPATIAL_STRIDE)
        return pred


# ── Combined model ────────────────────────────────────────────────────────────

class VideoPredictionModel(nn.Module):
    """
    Video autoencoder: VideoEncoder + VideoReconstructionDecoder.

    Pretraining: encode all T = n_frames + m_frames frames jointly, reconstruct
    all T frames.  This shared objective forces the encoder to learn temporal
    structure across both context and future.

    Downstream planning: call encoder.encode(context_frames) where
    context_frames is (B, n_cams, n_frames, 3, H, W) — the encoder handles
    shorter sequences because TemporalBlock slices its pos enc to match T.

    Args:
        n_frames:          number of input context frames
        m_frames:          number of future frames; total = n_frames + m_frames
        n_cams:            number of cameras (1 if front_cam_only, else 6)
        token_dim:         transformer token dimensionality (must be divisible by 4)
        n_encoder_layers:  number of (TemporalBlock, CameraBlock) pairs
        n_decoder_layers:  number of transformer decoder layers
        n_heads:           attention heads
        spatial_encoder:   "resnet18" | "resnet50" | "tinyvit"
        image_size:        (H, W) — must match dataset image_size
    """

    def __init__(
        self,
        n_frames:         int = 4,
        m_frames:         int = 4,
        n_cams:           int = 6,
        token_dim:        int = 256,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        n_heads:          int = 8,
        spatial_encoder:  str = "resnet18",
        image_size:       Tuple[int, int] = (224, 224),
    ):
        super().__init__()
        self.n_frames = n_frames
        self.m_frames = m_frames
        self.n_cams   = n_cams
        total_frames  = n_frames + m_frames

        self.encoder = VideoEncoder(
            max_frames=total_frames,
            n_cams=n_cams,
            token_dim=token_dim,
            n_layers=n_encoder_layers,
            n_heads=n_heads,
            spatial_encoder=spatial_encoder,
            image_size=image_size,
        )
        self.decoder = VideoReconstructionDecoder(
            n_cams=n_cams,
            n_out=total_frames,
            token_dim=token_dim,
            n_layers=n_decoder_layers,
            n_heads=n_heads,
            image_size=image_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_cams, n_frames+m_frames, 3, H, W) all frames in [0, 1]
        Returns:
            recon: (B, n_cams, n_frames+m_frames, 3, H, W) reconstructed frames in [0, 1]
        """
        return self.decoder(self.encoder.encode(x))


if __name__ == "__main__":
    B, n_cams, n_frames, m_frames = 2, 1, 4, 4
    H, W = 224, 224
    total = n_frames + m_frames

    print("Instantiating VideoPredictionModel (resnet18, front_cam_only) ...")
    model = VideoPredictionModel(
        n_frames=n_frames, m_frames=m_frames, n_cams=n_cams,
        token_dim=256, n_encoder_layers=2, n_decoder_layers=2, n_heads=8,
        spatial_encoder="resnet18", image_size=(H, W),
    )
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Pretraining forward: all n+m frames in, all n+m frames reconstructed
    all_frames = torch.rand(B, n_cams, total, 3, H, W)
    with torch.no_grad():
        recon = model(all_frames)
    assert recon.shape == (B, n_cams, total, 3, H, W), recon.shape
    print(f"  Pretrain input  : {tuple(all_frames.shape)}")
    print(f"  Pretrain recon  : {tuple(recon.shape)}  ✓")

    # Downstream inference: only n_frames context → latent
    ctx = torch.rand(B, n_cams, n_frames, 3, H, W)
    with torch.no_grad():
        latent = model.encoder.encode(ctx)
    assert latent.shape == (B, n_cams, 256, H//32, W//32), latent.shape
    print(f"  Inference latent: {tuple(latent.shape)}  (B, n_cams, token_dim, H/32, W/32)  ✓")
