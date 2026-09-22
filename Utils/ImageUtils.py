from pathlib import Path
import torch
from torchvision.utils import save_image


def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)


def save_triplet(source, fake, reconstructed, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    grid = torch.cat([denorm(source), denorm(fake), denorm(reconstructed)], 0)
    save_image(grid, path, nrow=source.size(0))
