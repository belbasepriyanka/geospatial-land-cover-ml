from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(
    num_classes: int = 9,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build a ResNet-18 classifier for multiclass vegetation imagery."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
