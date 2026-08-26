from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepweeds_pytorch.model import build_resnet18


def test_model_output_shape():
    model = build_resnet18(num_classes=9, pretrained=False)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 224, 224))
    assert output.shape == (2, 9)
