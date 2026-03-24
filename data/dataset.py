"""PyTorch Dataset for MVTec Anomaly Detection."""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MVTecDataset(Dataset):
    """MVTec Anomaly Detection dataset.

    Args:
        root: Root directory containing the MVTec AD dataset.
        category: Product category (e.g. 'bottle', 'leather', 'cable').
        split: Dataset split ('train' or 'test').
        resize: Resize dimensions before cropping.
        cropsize: Center crop dimensions.
    """

    def __init__(
        self,
        root: Path,
        category: str,
        split: Literal["train", "test"] = "train",
        resize: int = 256,
        cropsize: int = 224,
    ) -> None:
        self.root = Path(root)
        self.category = category
        self.split = split
        self.resize = resize
        self.cropsize = cropsize

        self.category_dir = self.root / category
        if not self.category_dir.exists():
            raise FileNotFoundError(
                f"Category directory not found: {self.category_dir}. "
                "Run 'python -m data.download_mvtec' first."
            )

        self.transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(cropsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(resize, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(cropsize),
            transforms.ToTensor(),
        ])

        self.image_paths: list[Path] = []
        self.labels: list[int] = []
        self.mask_paths: list[Path | None] = []

        self._load_samples()

    def _load_samples(self) -> None:
        """Scan directory structure and populate sample lists."""
        split_dir = self.category_dir / self.split

        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for defect_dir in sorted(split_dir.iterdir()):
            if not defect_dir.is_dir():
                continue

            is_normal = defect_dir.name == "good"
            label = 0 if is_normal else 1

            for img_path in sorted(defect_dir.glob("*.png")):
                self.image_paths.append(img_path)
                self.labels.append(label)

                if is_normal or self.split == "train":
                    self.mask_paths.append(None)
                else:
                    mask_path = (
                        self.category_dir
                        / "ground_truth"
                        / defect_dir.name
                        / img_path.name.replace(".png", "_mask.png")
                    )
                    self.mask_paths.append(mask_path if mask_path.exists() else None)

        logger.info(
            "Loaded %d samples for %s/%s (normal: %d, anomalous: %d)",
            len(self.image_paths),
            self.category,
            self.split,
            self.labels.count(0),
            self.labels.count(1),
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, torch.Tensor, str]:
        """Get a sample.

        Returns:
            Tuple of (image_tensor, label, mask_tensor, category).
            - image_tensor: [3, cropsize, cropsize] normalized float tensor.
            - label: 0 for normal, 1 for anomalous.
            - mask_tensor: [1, cropsize, cropsize] binary mask (zeros if no mask).
            - category: Category string.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        mask_path = self.mask_paths[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            logger.exception("Failed to load image: %s", img_path)
            raise

        image_tensor = self.transform(image)

        if mask_path is not None and mask_path.exists():
            try:
                mask = Image.open(mask_path).convert("L")
                mask_tensor = self.mask_transform(mask)
                mask_tensor = (mask_tensor > 0.5).float()
            except Exception:
                logger.exception("Failed to load mask: %s", mask_path)
                mask_tensor = torch.zeros(1, self.cropsize, self.cropsize)
        else:
            mask_tensor = torch.zeros(1, self.cropsize, self.cropsize)

        return image_tensor, label, mask_tensor, self.category


def get_dataloaders(
    root: Path,
    category: str,
    batch_size: int = 32,
    resize: int = 256,
    cropsize: int = 224,
    num_workers: int = 4,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and test dataloaders for a category.

    Args:
        root: Root directory of MVTec AD dataset.
        category: Product category.
        batch_size: Batch size.
        resize: Resize dimension.
        cropsize: Center crop dimension.
        num_workers: Number of data loading workers.

    Returns:
        Tuple of (train_loader, test_loader).
    """
    train_dataset = MVTecDataset(root, category, split="train", resize=resize, cropsize=cropsize)
    test_dataset = MVTecDataset(root, category, split="test", resize=resize, cropsize=cropsize)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
