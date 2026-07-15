from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rxvision.checkpoint import load_checkpoint
from rxvision.training import Trainer, TrainingConfig


def test_training_writes_best_state_dict_checkpoint(tmp_path):
    inputs = torch.randn(8, 3, 4, 4)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=4)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))
    trainer = Trainer(
        model,
        ["class-a", "class-b"],
        config=TrainingConfig(image_size=32, epochs=1, patience=1),
        device="cpu",
        architecture="tiny-test-model",
    )

    history = trainer.fit(loader, loader, tmp_path / "best.pt")
    payload = load_checkpoint(tmp_path / "best.pt")

    assert len(history) == 1
    assert 0.0 <= history[0]["validation_accuracy"] <= 1.0
    assert payload["architecture"] == "tiny-test-model"
