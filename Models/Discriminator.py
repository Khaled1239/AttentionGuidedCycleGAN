import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    70x70 PatchGAN Discriminator.

    Input:
        [B, 3, 128, 128]

    Output:
        PatchGAN score map.

    The discriminator uses a PatchGAN architecture
    where each output value corresponds to a local
    image patch.

    The output is a raw score map and is used with
    Least Squares GAN (LSGAN) loss.
    """

    def __init__(self, input_channels=3):
        super().__init__()

        self.model = nn.Sequential(

            # =================================================
            # Layer 1
            # =================================================
            # Input:
            #   [B, input_channels, 128, 128]
            #
            # Output:
            #   [B, 64, 64, 64]
            # =================================================

            nn.Conv2d(
                input_channels,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            # =================================================
            # Layer 2
            # =================================================
            # Output:
            #   [B, 128, 32, 32]
            # =================================================

            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.InstanceNorm2d(128),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            # =================================================
            # Layer 3
            # =================================================
            # Output:
            #   [B, 256, 16, 16]
            # =================================================

            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.InstanceNorm2d(256),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            # =================================================
            # Layer 4
            # =================================================
            # Output:
            #   [B, 512, 15, 15]
            # =================================================

            nn.Conv2d(
                256,
                512,
                kernel_size=4,
                stride=1,
                padding=1
            ),

            nn.InstanceNorm2d(512),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            # =================================================
            # Output Layer
            # =================================================
            # Output:
            #   [B, 1, 14, 14]
            #
            # Each value represents a local patch score.
            # =================================================

            nn.Conv2d(
                512,
                1,
                kernel_size=4,
                stride=1,
                padding=1
            )
        )

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image or masked image.

            Shape:
                [B, 3, 128, 128]

        Returns
        -------
        torch.Tensor
            PatchGAN score map.

            Shape:
                [B, 1, 14, 14]
        """

        return self.model(x)