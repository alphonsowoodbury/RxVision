"""Consistent image transforms for training and inference."""

from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_training_transform(image_size: int = 224) -> transforms.Compose:
    """Build conservative augmentations for a research training run."""

    _validate_image_size(image_size)
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_inference_transform(image_size: int = 224) -> transforms.Compose:
    """Build the deterministic transform expected by inference."""

    _validate_image_size(image_size)
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _validate_image_size(image_size: int) -> None:
    if image_size < 32:
        raise ValueError("image_size must be at least 32 pixels")
