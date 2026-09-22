import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module

    Based on the attention formulation described in the paper:

        A_s = sigmoid(
            f_7x7(
                [F_avg ; F_max]
            )
        )

    where:
        F_avg = average pooling across channels
        F_max = max pooling across channels

    Input:
        Feature map F
        Shape: [B, C, H, W]

    Output:
        Spatial attention map A_s
        Shape: [B, 1, H, W]
    """

    def __init__(self, kernel_size=7):
        super().__init__()

        assert kernel_size in (3, 7), (
            "kernel_size must be 3 or 7"
        )

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Generator feature map.

            Shape:
                [B, C, H, W]

        Returns
        -------
        attention : torch.Tensor
            Spatial attention map.

            Shape:
                [B, 1, H, W]
        """

        # ----------------------------------------------------
        # Average pooling across channel dimension
        # ----------------------------------------------------

        avg_map = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        # ----------------------------------------------------
        # Max pooling across channel dimension
        # ----------------------------------------------------

        max_map, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        # ----------------------------------------------------
        # Concatenate average and maximum feature maps
        #
        # [B, 1, H, W] + [B, 1, H, W]
        #             ↓
        #       [B, 2, H, W]
        # ----------------------------------------------------

        pooled_features = torch.cat(
            [
                avg_map,
                max_map
            ],
            dim=1
        )

        # ----------------------------------------------------
        # 7x7 convolution + sigmoid
        # ----------------------------------------------------

        attention = self.conv(
            pooled_features
        )

        attention = self.sigmoid(
            attention
        )

        return attention