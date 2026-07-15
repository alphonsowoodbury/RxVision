from __future__ import annotations

import torch
from torch import nn

from rxvision.checkpoint import CHECKPOINT_FORMAT, load_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path):
    model = nn.Linear(3, 2)
    path = save_checkpoint(
        tmp_path / "model.pt",
        model,
        ["class-a", "class-b"],
        architecture="tiny-test-model",
        image_size=64,
    )

    payload = load_checkpoint(path)

    assert payload["format"] == CHECKPOINT_FORMAT
    assert payload["class_names"] == ["class-a", "class-b"]
    assert payload["architecture"] == "tiny-test-model"
    assert payload["image_size"] == 64
    assert payload["state_dict"].keys() == model.state_dict().keys()


def test_checkpoint_rejects_unstructured_payload(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"state_dict": {}}, path)

    try:
        load_checkpoint(path)
    except ValueError as exc:
        assert "unsupported or malformed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("malformed checkpoint was accepted")
