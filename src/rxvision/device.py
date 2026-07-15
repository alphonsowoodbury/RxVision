"""Runtime device selection."""

from __future__ import annotations

import torch


def select_device(preferred: str | None = None) -> torch.device:
    """Select an available PyTorch device.

    An explicit device is validated instead of silently falling back. When no
    preference is supplied, CUDA is preferred over Apple MPS and CPU.
    """

    if preferred:
        device = torch.device(preferred)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
