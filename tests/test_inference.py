from __future__ import annotations

import math

import pytest
import torch
from PIL import Image
from torch import nn

from rxvision.inference import Predictor


class FixedLogits(nn.Module):
    def forward(self, inputs):
        return torch.tensor([[2.0, 1.0, 0.0]], device=inputs.device).repeat(inputs.shape[0], 1)


def test_predictor_applies_softmax_and_can_abstain():
    predictor = Predictor(
        FixedLogits(), ["candidate-a", "candidate-b", "candidate-c"], image_size=32, device="cpu"
    )
    image = Image.new("RGB", (40, 40), "white")

    result = predictor.predict(image, threshold=0.80, top_k=3)

    expected = math.exp(2) / (math.exp(2) + math.exp(1) + math.exp(0))
    assert result.candidate_label == "candidate-a"
    assert result.confidence == pytest.approx(expected)
    assert result.accepted is False
    assert [item["label"] for item in result.alternatives] == [
        "candidate-b",
        "candidate-c",
    ]


def test_predictor_validates_threshold():
    predictor = Predictor(FixedLogits(), ["a", "b", "c"], image_size=32, device="cpu")
    try:
        predictor.predict(Image.new("RGB", (32, 32)), threshold=1.1)
    except ValueError as exc:
        assert "threshold" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid threshold was accepted")
