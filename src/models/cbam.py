"""
cbam.py — Convolutional Block Attention Module
Citation: Woo et al. (2018) "CBAM: Convolutional Block Attention Module" ECCV
https://arxiv.org/abs/1807.06521

Usage:
    from src.models.cbam import CBAM
    attention = CBAM(channels=256)
    x = attention(x)   # x: [B, C, H, W]
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Learns WHICH feature channels are relevant.

    Both avg-pool (diffuse semantic signal, e.g. flat potholes) and
    max-pool (peak distinctive features, e.g. hump edges) are passed
    through a shared MLP and summed before sigmoid gating.

    Parameters
    ----------
    channels  : int  — number of input/output channels
    reduction : int  — bottleneck ratio for the shared MLP (default 16)
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    """
    Learns WHERE in the feature map to focus.

    Computes channel-wise avg and max pools, concatenates them, then
    applies a 7×7 conv to produce a spatial saliency mask that
    suppresses background (sky, unmarked asphalt) and amplifies
    road-surface anomaly regions.

    Parameters
    ----------
    kernel_size : int — spatial conv kernel (7 recommended per paper)
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)       # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)     # [B, 1, H, W]
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    """
    Full CBAM: sequential channel → spatial attention.

    Forward pass:
        x' = x  ⊗  ChannelAttention(x)      # recalibrate channels
        x''= x' ⊗  SpatialAttention(x')     # recalibrate spatial locs

    Parameters
    ----------
    channels  : int
    reduction : int  — channel attention bottleneck (default 16)
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x
