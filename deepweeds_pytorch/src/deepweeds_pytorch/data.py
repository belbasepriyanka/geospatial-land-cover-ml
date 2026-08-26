from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = 224):
    """Return train and evaluation transforms for RGB field imagery."""
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(
                brightness=0.20, contrast=0.20, saturation=0.20, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tfms, eval_tfms


class DeepWeedsDataset(Dataset):
    """DeepWeeds image dataset driven by the official CSV split files."""

    def __init__(
        self,
        csv_path: str | Path,
        image_dir: str | Path,
        transform: Optional[Callable] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform

        self.frame = pd.read_csv(self.csv_path)
        required = {"Filename", "Label", "Species"}
        missing = required.difference(self.frame.columns)
        if missing:
            raise ValueError(
                f"{self.csv_path} is missing required columns: {sorted(missing)}"
            )

        self.frame["Label"] = self.frame["Label"].astype(int)
        self.classes = (
            self.frame[["Label", "Species"]]
            .drop_duplicates()
            .sort_values("Label")
            .set_index("Label")["Species"]
            .to_dict()
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image_path = self.image_dir / str(row["Filename"])
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}. Run scripts/download_data.py first."
            )

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, int(row["Label"]), str(row["Filename"])


@dataclass(frozen=True)
class LoaderBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_names: list[str]


def load_class_names(labels_csv: str | Path) -> list[str]:
    frame = pd.read_csv(labels_csv)
    classes = (
        frame[["Label", "Species"]]
        .drop_duplicates()
        .sort_values("Label")
        .reset_index(drop=True)
    )
    return classes["Species"].tolist()


def build_loaders(
    data_root: str | Path,
    fold: int = 0,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
) -> LoaderBundle:
    """Build train/validation/test loaders from an official DeepWeeds fold."""
    if fold not in range(5):
        raise ValueError("fold must be one of 0, 1, 2, 3, 4")

    root = Path(data_root)
    image_dir = root / "images"
    label_dir = root / "labels"
    train_tfms, eval_tfms = build_transforms(image_size=image_size)

    train_ds = DeepWeedsDataset(
        label_dir / f"train_subset{fold}.csv", image_dir, transform=train_tfms
    )
    val_ds = DeepWeedsDataset(
        label_dir / f"val_subset{fold}.csv", image_dir, transform=eval_tfms
    )
    test_ds = DeepWeedsDataset(
        label_dir / f"test_subset{fold}.csv", image_dir, transform=eval_tfms
    )

    pin = torch.cuda.is_available()
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    class_names = [train_ds.classes[i] for i in sorted(train_ds.classes)]
    return LoaderBundle(train_loader, val_loader, test_loader, class_names)


def class_weights_from_csv(csv_path: str | Path, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights normalized to mean 1.0."""
    frame = pd.read_csv(csv_path)
    counts = frame["Label"].astype(int).value_counts().reindex(range(num_classes), fill_value=0)
    if (counts == 0).any():
        raise ValueError("Every class must appear in the training split.")
    weights = counts.sum() / (num_classes * counts.astype(float))
    return torch.tensor(weights.to_numpy(), dtype=torch.float32)
