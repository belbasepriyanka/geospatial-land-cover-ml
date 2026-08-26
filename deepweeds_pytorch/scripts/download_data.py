#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


ZENODO_IMAGES_URL = "https://zenodo.org/records/7939060/files/images.zip?download=1"
RAW_LABEL_BASE = "https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels"
LABEL_FILES = [
    "labels.csv",
    *[f"train_subset{i}.csv" for i in range(5)],
    *[f"val_subset{i}.csv" for i in range(5)],
    *[f"test_subset{i}.csv" for i in range(5)],
]


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"exists: {destination}")
        return
    print(f"downloading: {url}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the public DeepWeeds dataset.")
    parser.add_argument("--data-root", default="deepweeds_pytorch/data")
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep images.zip after extraction.",
    )
    args = parser.parse_args()

    root = Path(args.data_root)
    images_dir = root / "images"
    labels_dir = root / "labels"
    archive = root / "images.zip"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for filename in LABEL_FILES:
        download(f"{RAW_LABEL_BASE}/{filename}", labels_dir / filename)

    download(ZENODO_IMAGES_URL, archive)
    print(f"extracting: {archive}")
    with zipfile.ZipFile(archive, "r") as zip_ref:
        zip_ref.extractall(images_dir)

    nested = images_dir / "images"
    if nested.exists() and nested.is_dir():
        for source in nested.glob("*.jpg"):
            target = images_dir / source.name
            if not target.exists():
                source.replace(target)
        try:
            nested.rmdir()
        except OSError:
            pass

    n_images = len(list(images_dir.glob("*.jpg")))
    print(f"ready: {n_images:,} JPEG images in {images_dir}")

    if not args.keep_archive and archive.exists():
        archive.unlink()


if __name__ == "__main__":
    main()
