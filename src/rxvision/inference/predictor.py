"""Calibrated-candidate inference with an explicit abstention path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn

from rxvision.checkpoint import load_checkpoint
from rxvision.data import build_inference_transform
from rxvision.device import select_device
from rxvision.models import EfficientNetV2Classifier


@dataclass(frozen=True)
class Prediction:
    """Ranked model output.

    ``accepted`` means only that the top score crossed a caller-selected
    threshold. It is not evidence that the candidate is medically correct.
    """

    candidate_label: str
    confidence: float
    accepted: bool
    alternatives: tuple[dict[str, float | str], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Predictor:
    """Run a classifier and expose ranked candidates with abstention."""

    def __init__(
        self,
        model: nn.Module,
        class_names: list[str],
        *,
        image_size: int = 224,
        device: str | torch.device | None = None,
    ) -> None:
        if len(class_names) < 2 or len(class_names) != len(set(class_names)):
            raise ValueError("class_names must contain at least two unique labels")
        self.class_names = list(class_names)
        self.device = select_device(str(device) if device is not None else None)
        self.model = model.to(self.device).eval()
        self.transform = build_inference_transform(image_size)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device | None = None,
    ) -> Predictor:
        selected = select_device(str(device) if device is not None else None)
        payload = load_checkpoint(path, map_location=selected)
        if payload["architecture"] != EfficientNetV2Classifier.architecture_name:
            raise ValueError(f"unsupported architecture: {payload['architecture']}")
        model = EfficientNetV2Classifier(len(payload["class_names"]), pretrained=False)
        model.load_state_dict(payload["state_dict"], strict=True)
        return cls(
            model,
            payload["class_names"],
            image_size=payload["image_size"],
            device=selected,
        )

    def predict(
        self,
        image: Image.Image,
        *,
        threshold: float = 0.80,
        top_k: int = 3,
    ) -> Prediction:
        """Return ranked candidates; abstain when the top score is too low."""

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            if logits.ndim != 2 or logits.shape[0] != 1:
                raise ValueError("model must return a [batch, classes] logits tensor")
            if logits.shape[1] != len(self.class_names):
                raise ValueError("model output count does not match class_names")
            probabilities = torch.softmax(logits[0], dim=0)

        count = min(top_k, len(self.class_names))
        scores, indices = torch.topk(probabilities, k=count)
        ranked = [
            {"label": self.class_names[index], "confidence": float(score)}
            for score, index in zip(scores.tolist(), indices.tolist(), strict=True)
        ]
        top = ranked[0]
        return Prediction(
            candidate_label=str(top["label"]),
            confidence=float(top["confidence"]),
            accepted=float(top["confidence"]) >= threshold,
            alternatives=tuple(ranked[1:]),
        )
