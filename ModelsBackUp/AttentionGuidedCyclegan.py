import torch.nn as nn
from .Attention import SpatialAttention
from .Generator import ResnetGenerator
from .Discriminator import PatchDiscriminator70


class AttentionGuidedGenerator(nn.Module):
    """
    output = attention * generated + (1 - attention) * original
    """

    def __init__(self, n_res_blocks=6):
        super().__init__()
        self.attention = SpatialAttention(7)
        self.generator = ResnetGenerator(n_res_blocks=n_res_blocks)

    def forward(self, x):
        attention = self.attention(x)
        generated = self.generator(x)
        output = attention * generated + (1.0 - attention) * x
        return output, attention


class AttentionGuidedCycleGAN(nn.Module):
    def __init__(self, n_res_blocks=6):
        super().__init__()
        self.G_ST = AttentionGuidedGenerator(n_res_blocks)
        self.F_TS = AttentionGuidedGenerator(n_res_blocks)
        self.D_T = PatchDiscriminator70()
        self.D_S = PatchDiscriminator70()
