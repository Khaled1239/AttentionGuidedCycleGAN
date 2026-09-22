import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image
from tqdm import tqdm

from Models.AttentionGuidedCyclegan import AttentionGuidedCycleGAN


EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".tif", ".tiff", ".webp"
}


def normalize_attention(attention):
    """
    Convert attention tensor into a displayable 1-channel image.

    Input:
        [B, C, H, W]

    Output:
        [B, 1, H, W]
        with values normalized to [0, 1]
    """

    # If attention has multiple feature channels,
    # average them into one attention map.
    if attention.dim() == 4 and attention.size(1) > 1:
        attention = attention.mean(dim=1, keepdim=True)

    # Make sure values are non-negative for visualization
    attention = attention.detach()

    min_val = attention.amin(dim=(1, 2, 3), keepdim=True)
    max_val = attention.amax(dim=(1, 2, 3), keepdim=True)

    attention = (attention - min_val) / (
        max_val - min_val + 1e-8
    )

    return attention


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input",
        default="Data/C_NMC"
    )

    p.add_argument(
        "--output",
        default="Outputs/Translated/CNMC_to_ALLIDB"
    )

    p.add_argument(
        "--checkpoint",
        default="Checkpoints/Latest.pt"
    )

    p.add_argument(
        "--image_size",
        type=int,
        default=128
    )

    p.add_argument(
        "--device",
        default="auto"
    )

    a = p.parse_args()

    # --------------------------------------------------
    # DEVICE
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if a.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if a.device == "auto"
        else a.device
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------

    model = AttentionGuidedCycleGAN().to(device)

    ckpt = torch.load(
        a.checkpoint,
        map_location=device
    )

    model.load_state_dict(
        ckpt["model"]
    )

    model.eval()

    # --------------------------------------------------
    # TRANSFORM
    # --------------------------------------------------

    transform = T.Compose([
        T.Resize((a.image_size, a.image_size)),
        T.ToTensor(),
        T.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        ),
    ])

    # --------------------------------------------------
    # INPUT / OUTPUT
    # --------------------------------------------------

    root = Path(a.input)
    out_root = Path(a.output)

    paths = [
        x for x in root.rglob("*")
        if x.is_file()
        and x.suffix.lower() in EXTENSIONS
    ]

    print("Images found:", len(paths))

    # --------------------------------------------------
    # TRANSLATION
    # --------------------------------------------------

    with torch.no_grad():

        for path in tqdm(
            paths,
            desc="Translating"
        ):

            # Load image
            x = transform(
                Image.open(path).convert("RGB")
            ).unsqueeze(0).to(device)

            # Generator
            fake, attention = model.G_ST(x)

            # --------------------------------------------------
            # DEBUG ATTENTION
            # --------------------------------------------------

            print("\n===================================")
            print("Image:", path.name)
            print("Attention shape:", tuple(attention.shape))
            print(
                "Attention min:",
                attention.min().item()
            )
            print(
                "Attention max:",
                attention.max().item()
            )
            print(
                "Attention mean:",
                attention.mean().item()
            )
            print(
                "Attention std:",
                attention.std().item()
            )
            print("===================================")

            # --------------------------------------------------
            # OUTPUT PATH
            # --------------------------------------------------

            rel = path.relative_to(root)

            out = (
                out_root /
                rel.with_suffix(".png")
            )

            att = (
                out_root /
                "Attention" /
                rel.with_suffix(".png")
            )

            out.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            att.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            # --------------------------------------------------
            # SAVE TRANSLATED IMAGE
            # --------------------------------------------------

            save_image(
                (fake * 0.5 + 0.5).clamp(0, 1),
                out
            )

            # --------------------------------------------------
            # SAVE ATTENTION MAP
            # --------------------------------------------------

            attention_vis = normalize_attention(
                attention
            )

            save_image(
                attention_vis,
                att
            )

    print("\nSaved:", out_root)


if __name__ == "__main__":
    main()