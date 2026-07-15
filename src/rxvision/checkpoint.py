"""Safe, explicit checkpoint serialization."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

CHECKPOINT_FORMAT = "rxvision-state-dict-v1"


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    class_names: Sequence[str],
    *,
    architecture: str,
    image_size: int = 224,
) -> Path:
    """Save weights plus the metadata required to interpret model outputs."""

    names = _validate_class_names(class_names)
    if image_size <= 0:
        raise ValueError("image_size must be positive")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "architecture": architecture,
            "image_size": image_size,
            "class_names": names,
            "state_dict": model.state_dict(),
        },
        destination,
    )
    return destination


def load_checkpoint(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Load and validate an RxVision state-dict checkpoint.

    ``weights_only=True`` prevents arbitrary Python object construction when
    supported by the installed PyTorch version.
    """

    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        payload = torch.load(path, map_location=map_location)

    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported or malformed RxVision checkpoint")
    if not isinstance(payload.get("state_dict"), dict):
        raise ValueError("checkpoint is missing a state_dict")
    payload["class_names"] = _validate_class_names(payload.get("class_names", []))
    if not isinstance(payload.get("architecture"), str):
        raise ValueError("checkpoint is missing architecture metadata")
    if not isinstance(payload.get("image_size"), int) or payload["image_size"] <= 0:
        raise ValueError("checkpoint has an invalid image_size")
    return payload


def _validate_class_names(class_names: Sequence[str]) -> list[str]:
    names = [name.strip() for name in class_names]
    if len(names) < 2 or any(not name for name in names):
        raise ValueError("class_names must contain at least two non-empty labels")
    if len(names) != len(set(names)):
        raise ValueError("class_names must be unique")
    return names
