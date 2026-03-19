"""
Shared TinyViT-21M encoder used by LSTM and transformer video models.
"""
import importlib.util
import sys

import torch
import torch.nn as nn
from torch import Tensor


def _load_tiny_vit():
    """Import tiny_vit_21m_224 directly from the TinyViT repo by file path.

    Using a direct importlib load avoids conflicts with our own `models` package
    that would shadow TinyViT's `models/` directory if we relied on sys.path.
    """
    # Ensure TinyViT repo is on sys.path so tiny_vit.py can find its own deps
    _tinyvit_root = "/workspace/TinyViT"
    if _tinyvit_root not in sys.path:
        sys.path.insert(0, _tinyvit_root)

    mod_name = "_tinyvit_backbone"
    if mod_name in sys.modules:
        return sys.modules[mod_name].tiny_vit_21m_224

    spec = importlib.util.spec_from_file_location(
        mod_name,
        f"{_tinyvit_root}/models/tiny_vit.py",
    )
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec so that timm's @register_model decorator
    # can look it up by __module__ name during module initialization.
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod.tiny_vit_21m_224


class TinyViTEncoder(nn.Module):
    """
    Wraps timm TinyViT-21M and extracts 4 skip levels.
    Encoder is frozen (requires_grad=False on all params).

    Returns: (skip0, skip1, skip2, bottleneck)
      skip0:      (N,  96, H/4,  W/4)
      skip1:      (N, 192, H/8,  W/8)
      skip2:      (N, 384, H/16, W/16)
      bottleneck: (N, 576, H/32, W/32)

    N = B*S*C (flattened batch)
    """

    def __init__(self, img_h: int = 224, img_w: int = 224):
        super().__init__()
        tiny_vit_21m_224 = _load_tiny_vit()
        img_size = img_h  # TinyViT uses square images
        try:
            backbone = tiny_vit_21m_224(pretrained=True, img_size=img_size)
        except Exception:
            backbone = tiny_vit_21m_224(pretrained=False, img_size=img_size)

        self.patch_embed = backbone.patch_embed
        self.enc_layers  = backbone.layers

        for p in list(self.patch_embed.parameters()) + list(self.enc_layers.parameters()):
            p.requires_grad_(False)

    def forward(self, x: Tensor) -> tuple:
        """
        Args:
            x: (N, 3, H, W)

        Returns:
            (skip0, skip1, skip2, bottleneck) as spatial feature maps
        """
        x = self.patch_embed(x)           # (N, 96, H/4, W/4)
        H4, W4 = x.shape[-2], x.shape[-1]

        for blk in self.enc_layers[0].blocks:
            x = blk(x)
        skip0 = x                         # (N, 96, H/4, W/4)
        x = self.enc_layers[0].downsample(x)   # tokens (N, H/8·W/8, 192)

        for blk in self.enc_layers[1].blocks:
            x = blk(x)
        skip1_tok = x
        x = self.enc_layers[1].downsample(x)   # tokens (N, H/16·W/16, 384)

        for blk in self.enc_layers[2].blocks:
            x = blk(x)
        skip2_tok = x
        x = self.enc_layers[2].downsample(x)   # tokens (N, H/32·W/32, 576)

        for blk in self.enc_layers[3].blocks:
            x = blk(x)
        bot_tok = x

        N = x.shape[0]
        H8,  W8  = H4 // 2, W4 // 2
        H16, W16 = H4 // 4, W4 // 4
        H32, W32 = H4 // 8, W4 // 8

        skip1 = skip1_tok.view(N, H8,  W8,  192).permute(0, 3, 1, 2).contiguous()
        skip2 = skip2_tok.view(N, H16, W16, 384).permute(0, 3, 1, 2).contiguous()
        bot   = bot_tok  .view(N, H32, W32, 576).permute(0, 3, 1, 2).contiguous()

        return skip0, skip1, skip2, bot
