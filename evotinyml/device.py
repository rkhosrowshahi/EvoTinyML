"""Torch device resolution for CLI / training entry points."""

from __future__ import annotations

import torch


def resolve_device(name: str) -> torch.device:
    """Parse a device string; ``gpu`` picks CUDA then MPS when available."""
    key = str(name).strip().lower()
    if key == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        raise SystemExit(
            "--device gpu requested but neither CUDA nor MPS is available; "
            "use --device cpu or a specific device string."
        )
    return torch.device(name)
