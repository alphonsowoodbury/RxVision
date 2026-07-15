"""Small, inspectable PyTorch training loop."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from rxvision.checkpoint import save_checkpoint
from rxvision.data import build_inference_transform, build_training_transform
from rxvision.device import select_device
from rxvision.models import EfficientNetV2Classifier


@dataclass(frozen=True)
class TrainingConfig:
    image_size: int = 224
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 10
    patience: int = 3
    seed: int = 42


class Trainer:
    """Train a classifier and persist the best validation-loss checkpoint."""

    def __init__(
        self,
        model: nn.Module,
        class_names: list[str],
        *,
        config: TrainingConfig | None = None,
        device: str | torch.device | None = None,
        architecture: str = EfficientNetV2Classifier.architecture_name,
    ) -> None:
        self.config = config or TrainingConfig()
        if self.config.epochs < 1 or self.config.patience < 1:
            raise ValueError("epochs and patience must be positive")
        self.class_names = list(class_names)
        self.device = select_device(str(device) if device is not None else None)
        self.model = model.to(self.device)
        self.architecture = architecture
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        _seed_everything(self.config.seed)

    @classmethod
    def from_directories(
        cls,
        train_dir: str | Path,
        validation_dir: str | Path,
        *,
        config: TrainingConfig | None = None,
        pretrained: bool = True,
        device: str | torch.device | None = None,
    ) -> tuple[Trainer, DataLoader[Any], DataLoader[Any]]:
        cfg = config or TrainingConfig()
        train_set = ImageFolder(train_dir, transform=build_training_transform(cfg.image_size))
        validation_set = ImageFolder(
            validation_dir, transform=build_inference_transform(cfg.image_size)
        )
        if train_set.class_to_idx != validation_set.class_to_idx:
            raise ValueError("training and validation class mappings differ")

        model = EfficientNetV2Classifier(len(train_set.classes), pretrained=pretrained)
        trainer = cls(model, train_set.classes, config=cfg, device=device)
        train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
        validation_loader = DataLoader(
            validation_set, batch_size=cfg.batch_size, shuffle=False, num_workers=0
        )
        return trainer, train_loader, validation_loader

    def fit(
        self,
        train_loader: DataLoader[Any],
        validation_loader: DataLoader[Any],
        output_path: str | Path,
    ) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0

        for epoch in range(self.config.epochs):
            train_loss, train_accuracy = self._run_epoch(train_loader, training=True)
            validation_loss, validation_accuracy = self._run_epoch(
                validation_loader, training=False
            )
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                }
            )

            if validation_loss < best_loss:
                best_loss = validation_loss
                stale_epochs = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    break

        if best_state is None:
            raise RuntimeError("training produced no checkpoint")
        self.model.load_state_dict(best_state)
        save_checkpoint(
            output_path,
            self.model,
            self.class_names,
            architecture=self.architecture,
            image_size=self.config.image_size,
        )
        return history

    def _run_epoch(self, loader: DataLoader[Any], *, training: bool) -> tuple[float, float]:
        self.model.train(training)
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        context = torch.enable_grad() if training else torch.inference_mode()
        with context:
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(inputs)
                loss = self.loss_fn(logits, labels)
                if training:
                    loss.backward()
                    self.optimizer.step()

                batch_size = labels.shape[0]
                total_loss += float(loss.detach()) * batch_size
                total_correct += int((logits.argmax(dim=1) == labels).sum())
                total_examples += batch_size

        if total_examples == 0:
            raise ValueError("data loader is empty")
        return total_loss / total_examples, total_correct / total_examples


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
