import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """Average/max channel pooling followed by 7x7 conv and sigmoid."""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_map, max_map], dim=1)
        return self.sigmoid(self.conv(pooled))
