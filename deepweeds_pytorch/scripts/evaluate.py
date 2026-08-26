#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepweeds_pytorch.data import build_loaders
from deepweeds_pytorch.model import build_resnet18
from deepweeds_pytorch.reporting import collect_predictions, write_evaluation_outputs


def main():
    parser = argparse.ArgumentParser(description="Evaluate a DeepWeeds checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="deepweeds_pytorch/data")
    parser.add_argument("--output-dir", default="deepweeds_pytorch/results/evaluation")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    class_names = checkpoint["class_names"]
    fold = int(checkpoint.get("fold", 0))
    image_size = int(checkpoint.get("image_size", 224))

    loaders = build_loaders(
        args.data_root,
        fold=fold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=image_size,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])

    y_true, y_pred, names = collect_predictions(model, loaders.test, device)
    metrics = write_evaluation_outputs(
        y_true, y_pred, names, class_names, args.output_dir
    )
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
