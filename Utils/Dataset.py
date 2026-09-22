from pathlib import Path
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(root):
    root = Path(root)
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )


class UnpairedImageDataset(Dataset):
    def __init__(self, source_root, target_root, image_size=128):
        self.source = collect_images(source_root)
        self.target = collect_images(target_root)

        if not self.source:
            raise RuntimeError(f"No images found in {source_root}")
        if not self.target:
            raise RuntimeError(f"No images found in {target_root}")

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5,)*3, (0.5,)*3),
        ])

    def __len__(self):
        return max(len(self.source), len(self.target))

    def __getitem__(self, idx):
        s_path = self.source[idx % len(self.source)]
        t_path = random.choice(self.target)

        s = self.transform(Image.open(s_path).convert("RGB"))
        t = self.transform(Image.open(t_path).convert("RGB"))

        return {"source": s, "target": t,
                "source_path": str(s_path), "target_path": str(t_path)}


def create_dataloader(source_root, target_root, image_size=128,
                      batch_size=4, num_workers=0):
    ds = UnpairedImageDataset(source_root, target_root, image_size)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
