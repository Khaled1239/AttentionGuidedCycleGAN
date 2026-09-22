import torch
import torch.nn as nn

from Models.Attention import SpatialAttention


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):
    """
    Residual block used in the CycleGAN generator.

    Structure:
        Conv 3x3
        InstanceNorm
        ReLU
        Conv 3x3
        InstanceNorm
        + skip connection
    """

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),

            nn.InstanceNorm2d(
                channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),

            nn.InstanceNorm2d(
                channels
            )
        )

    def forward(self, x):

        return x + self.block(x)


# ============================================================
# GENERATOR
# ============================================================

class Generator(nn.Module):
    """
    Attention-guided CycleGAN Generator.

    Input:
        [B, 3, 128, 128]

    Encoder:
        7x7, 64 channels
        3x3, 128 channels, stride 2
        3x3, 256 channels, stride 2

    Residual:
        6 residual blocks

    Decoder:
        3x3, 128 channels, stride 2
        3x3, 64 channels, stride 2
        7x7, 3 channels

    The spatial attention module receives the
    generator feature map rather than the raw RGB input.
    """

    def __init__(
        self,
        input_channels=3,
        output_channels=3,
        num_residual_blocks=6
    ):
        super().__init__()

        # ----------------------------------------------------
        # ENCODER
        # ----------------------------------------------------

        self.encoder = nn.Sequential(

            # 128x128 -> 128x128
            nn.Conv2d(
                input_channels,
                64,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=True
            ),

            nn.InstanceNorm2d(
                64
            ),

            nn.ReLU(
                inplace=True
            ),

            # 128x128 -> 64x64
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True
            ),

            nn.InstanceNorm2d(
                128
            ),

            nn.ReLU(
                inplace=True
            ),

            # 64x64 -> 32x32
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True
            ),

            nn.InstanceNorm2d(
                256
            ),

            nn.ReLU(
                inplace=True
            )
        )

        # ----------------------------------------------------
        # SPATIAL ATTENTION
        #
        # Feature map:
        # [B, 256, 32, 32]
        #
        # Attention:
        # [B, 1, 32, 32]
        # ----------------------------------------------------

        self.attention = SpatialAttention(
            kernel_size=7
        )

        # ----------------------------------------------------
        # RESIDUAL BLOCKS
        # ----------------------------------------------------

        residual_blocks = []

        for _ in range(num_residual_blocks):

            residual_blocks.append(
                ResidualBlock(
                    channels=256
                )
            )

        self.residual_blocks = nn.Sequential(
            *residual_blocks
        )

        # ----------------------------------------------------
        # DECODER
        # ----------------------------------------------------

        self.decoder = nn.Sequential(

            # 32x32 -> 64x64
            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=True
            ),

            nn.InstanceNorm2d(
                128
            ),

            nn.ReLU(
                inplace=True
            ),

            # 64x64 -> 128x128
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=True
            ),

            nn.InstanceNorm2d(
                64
            ),

            nn.ReLU(
                inplace=True
            ),

            # 128x128 -> 128x128
            nn.Conv2d(
                64,
                output_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=True
            ),

            nn.Tanh()
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x,
        return_attention=False
    ):

        # ----------------------------------------------------
        # ENCODER
        # ----------------------------------------------------

        features = self.encoder(x)

        # features:
        # [B, 256, 32, 32]

        # ----------------------------------------------------
        # SPATIAL ATTENTION
        # ----------------------------------------------------

        attention = self.attention(
            features
        )

        # attention:
        # [B, 1, 32, 32]

        # ----------------------------------------------------
        # APPLY ATTENTION TO FEATURE MAP
        # ----------------------------------------------------
        #
        # Broadcast:
        #
        # [B, 1, 32, 32]
        #          *
        # [B, 256, 32, 32]
        #
        # -> [B, 256, 32, 32]
        #
        # The attended feature map retains the
        # original spatial dimensions.
        # ----------------------------------------------------

        attended_features = (
            features * attention
        )

        # ----------------------------------------------------
        # RESIDUAL BLOCKS
        # ----------------------------------------------------

        features = self.residual_blocks(
            attended_features
        )

        # ----------------------------------------------------
        # DECODER
        # ----------------------------------------------------

        generated = self.decoder(
            features
        )

        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        if return_attention:

            return generated, attention

        return generated