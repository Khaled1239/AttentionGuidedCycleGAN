import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.Generator import Generator
from Models.Discriminator import Discriminator


class AttentionGuidedGenerator(nn.Module):
    """
    Wrapper around the Generator that applies
    the attention-guided image formulation:

        s' = s_a * G(s) + (1 - s_a) * s

    where:

        s   = original source image
        G(s) = generated image
        s_a = spatial attention mask

    The attention mask is generated from the
    generator feature maps.
    """

    def __init__(
        self,
        input_channels=3,
        output_channels=3
    ):
        super().__init__()

        self.generator = Generator(
            input_channels=input_channels,
            output_channels=output_channels
        )

        # ----------------------------------------------------
        # NOTE:
        # The actual attention module is inside Generator.
        #
        # This reference is useful for accessing it directly:
        #
        # self.generator.attention
        # ----------------------------------------------------

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor

            Input image.

            Shape:
                [B, 3, 128, 128]

        Returns
        -------
        output : torch.Tensor

            Attention-guided translated image.

            Shape:
                [B, 3, 128, 128]

        attention : torch.Tensor

            Full-resolution attention mask.

            Shape:
                [B, 1, 128, 128]
        """

        # ----------------------------------------------------
        # Generate image + low-resolution attention
        # ----------------------------------------------------

        generated, attention = self.generator(
            x,
            return_attention=True
        )

        # ----------------------------------------------------
        # Resize attention
        #
        # Generator attention:
        # [B, 1, 32, 32]
        #
        # Image:
        # [B, 3, 128, 128]
        #
        # Therefore:
        # [B, 1, 32, 32]
        #        ↓
        # [B, 1, 128, 128]
        # ----------------------------------------------------

        attention = F.interpolate(
            attention,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # ----------------------------------------------------
        # Attention-guided output
        #
        # s' = s_a * G(s)
        #      +
        #      (1 - s_a) * s
        #
        # Attention is automatically broadcast across
        # the RGB channels.
        # ----------------------------------------------------

        output = (
            attention * generated
            +
            (1.0 - attention) * x
        )

        return output, attention


class AttentionGuidedCycleGAN(nn.Module):
    """
    Attention-Guided CycleGAN.

    Domains:

        S = CNMC
        T = ALL-IDB

    Forward mapping:

        G_ST : CNMC -> ALL-IDB

    Reverse mapping:

        F_TS : ALL-IDB -> CNMC

    Discriminators:

        D_T : target / ALL-IDB discriminator
        D_S : source / CNMC discriminator
    """

    def __init__(
        self,
        input_channels=3,
        output_channels=3
    ):
        super().__init__()

        # ----------------------------------------------------
        # GENERATORS
        # ----------------------------------------------------

        self.G_ST = AttentionGuidedGenerator(
            input_channels=input_channels,
            output_channels=output_channels
        )

        self.F_TS = AttentionGuidedGenerator(
            input_channels=input_channels,
            output_channels=output_channels
        )

        # ----------------------------------------------------
        # DISCRIMINATORS
        # ----------------------------------------------------

        self.D_T = Discriminator(
            input_channels=output_channels
        )

        self.D_S = Discriminator(
            input_channels=input_channels
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, source, target):
        """
        Perform the CycleGAN forward process.

        Parameters
        ----------
        source : torch.Tensor

            Source-domain image.

            CNMC.

            Shape:
                [B, 3, 128, 128]

        target : torch.Tensor

            Target-domain image.

            ALL-IDB.

            Shape:
                [B, 3, 128, 128]

        Returns
        -------
        Dictionary containing:

            fake_target
            fake_source
            cycle_source
            cycle_target
            attention_source
            attention_target
        """

        # ----------------------------------------------------
        # SOURCE -> TARGET
        # ----------------------------------------------------

        fake_target, attention_source = self.G_ST(
            source
        )

        # ----------------------------------------------------
        # TARGET -> SOURCE
        # ----------------------------------------------------

        fake_source, attention_target = self.F_TS(
            target
        )

        # ----------------------------------------------------
        # TARGET -> SOURCE -> TARGET
        # ----------------------------------------------------

        cycle_target, _ = self.G_ST(
            fake_source
        )

        # ----------------------------------------------------
        # SOURCE -> TARGET -> SOURCE
        # ----------------------------------------------------

        cycle_source, _ = self.F_TS(
            fake_target
        )

        return {
            "fake_target": fake_target,
            "fake_source": fake_source,
            "cycle_source": cycle_source,
            "cycle_target": cycle_target,
            "attention_source": attention_source,
            "attention_target": attention_target
        }

    # ========================================================
    # DISCRIMINATOR INPUTS
    # ========================================================

    def get_target_discriminator_inputs(
        self,
        source,
        target
    ):
        """
        Prepare inputs for D_T.

        The discriminator receives attention-masked
        target-domain images.

        Returns:

            masked_real_target
            masked_fake_target
        """

        # ----------------------------------------------------
        # SOURCE -> TARGET
        # ----------------------------------------------------

        fake_target, attention_source = self.G_ST(
            source
        )

        # ----------------------------------------------------
        # MASK REAL TARGET
        #
        # The source attention mask is resized to
        # target image dimensions.
        # ----------------------------------------------------

        masked_real_target = (
            attention_source * target
        )

        # ----------------------------------------------------
        # MASK FAKE TARGET
        # ----------------------------------------------------

        masked_fake_target = (
            attention_source * fake_target
        )

        return (
            masked_real_target,
            masked_fake_target
        )

    # ========================================================
    # SOURCE DISCRIMINATOR INPUTS
    # ========================================================

    def get_source_discriminator_inputs(
        self,
        source,
        target
    ):
        """
        Prepare inputs for D_S.

        Returns:

            masked_real_source
            masked_fake_source
        """

        # ----------------------------------------------------
        # TARGET -> SOURCE
        # ----------------------------------------------------

        fake_source, attention_target = self.F_TS(
            target
        )

        # ----------------------------------------------------
        # MASK REAL SOURCE
        # ----------------------------------------------------

        masked_real_source = (
            attention_target * source
        )

        # ----------------------------------------------------
        # MASK FAKE SOURCE
        # ----------------------------------------------------

        masked_fake_source = (
            attention_target * fake_source
        )

        return (
            masked_real_source,
            masked_fake_source
        )