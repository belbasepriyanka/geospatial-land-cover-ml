from pathlib import Path
import sys

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepweeds_pytorch.data import DeepWeedsDataset, build_transforms


def test_dataset_reads_rgb_image_and_label(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "sample.jpg"
    Image.new("RGB", (256, 256), (20, 120, 40)).save(image_path)

    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(
        [{"Filename": "sample.jpg", "Label": 0, "Species": "Example weed"}]
    ).to_csv(csv_path, index=False)

    _, eval_tfms = build_transforms(224)
    dataset = DeepWeedsDataset(csv_path, image_dir, transform=eval_tfms)
    image, label, filename = dataset[0]

    assert image.shape == (3, 224, 224)
    assert label == 0
    assert filename == "sample.jpg"
