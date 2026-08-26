#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepweeds_pytorch.data import build_transforms
from deepweeds_pytorch.model import build_resnet18


def main():
    parser = argparse.ArgumentParser(description="Predict a weed class for one RGB image.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    class_names = checkpoint["class_names"]
    image_size = int(checkpoint.get("image_size", 224))

    model = build_resnet18(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _, eval_tfms = build_transforms(image_size)
    image = Image.open(args.image).convert("RGB")
    tensor = eval_tfms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    top_k = min(args.top_k, len(class_names))
    values, indices = torch.topk(probabilities, top_k)
    for rank, (prob, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        print(f"{rank}. {class_names[idx]}: {prob:.4f}")


if __name__ == "__main__":
    main()
