"""
Shared building blocks used by multiple models.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return x


# Encoder output channels per stage: (stem, layer1, layer2, layer3, layer4)
_RESNET_CHANNELS = {
    "resnet18": (64,  64,  128,  256,  512),
    "resnet34": (64,  64,  128,  256,  512),
    "resnet50": (64, 256,  512, 1024, 2048),
}

_RESNET_WEIGHTS = {
    "resnet18": tvm.ResNet18_Weights.DEFAULT,
    "resnet34": tvm.ResNet34_Weights.DEFAULT,
    "resnet50": tvm.ResNet50_Weights.DEFAULT,
}

_RESNET_FN = {
    "resnet18": tvm.resnet18,
    "resnet34": tvm.resnet34,
    "resnet50": tvm.resnet50,
}


def make_2d_sincos_pos_enc(H: int, W: int, dim: int) -> torch.Tensor:
    """
    Fixed 2D sinusoidal position encoding.

    Args:
        H, W : spatial grid size
        dim  : total encoding dimension (must be divisible by 4)

    Returns:
        (H * W, dim) float tensor
    """
    assert dim % 4 == 0, f"dim must be divisible by 4, got {dim}"
    half = dim // 2

    def sincos_1d(n: int, d: int) -> torch.Tensor:
        pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d)
        )
        enc = torch.zeros(n, d)
        enc[:, 0::2] = torch.sin(pos * div)
        enc[:, 1::2] = torch.cos(pos * div)
        return enc

    row_enc = sincos_1d(H, half)
    col_enc = sincos_1d(W, half)

    pe = torch.cat([
        row_enc.unsqueeze(1).expand(H, W, half),
        col_enc.unsqueeze(0).expand(H, W, half),
    ], dim=-1)
    return pe.reshape(H * W, dim)
