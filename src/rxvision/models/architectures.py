"""Model architectures used by the research pipeline."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s


class EfficientNetV2Classifier(nn.Module):
    """EfficientNetV2-S classifier with an explicit output contract.

    Pretrained weights are opt-in because requesting them may download an
    external artifact. Tests and offline workflows use randomly initialized
    weights.
    """

    architecture_name = "efficientnet_v2_s"

    def __init__(self, num_classes: int, *, pretrained: bool = False) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_v2_s(weights=weights)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return unnormalized class logits."""

        return self.backbone(inputs)
