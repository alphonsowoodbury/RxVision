from __future__ import annotations

import torch
from torch import nn

from rxvision.models import architectures


class FakeEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(4, 1000))

    def forward(self, inputs):
        features = torch.ones((inputs.shape[0], 4), device=inputs.device)
        return self.classifier(features)


def test_classifier_replaces_output_layer(monkeypatch):
    monkeypatch.setattr(architectures, "efficientnet_v2_s", lambda weights=None: FakeEfficientNet())
    model = architectures.EfficientNetV2Classifier(3, pretrained=False)

    output = model(torch.zeros(2, 3, 32, 32))

    assert output.shape == (2, 3)


def test_classifier_requires_multiple_classes():
    try:
        architectures.EfficientNetV2Classifier(1)
    except ValueError as exc:
        assert "at least 2" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("single-class classifier was accepted")
