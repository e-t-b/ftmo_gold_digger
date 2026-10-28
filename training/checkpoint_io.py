"""Restricted checkpoint loading for Gold Digger research artifacts."""

from pathlib import Path

import numpy as np
import torch


def safe_torch_load(path: str | Path) -> dict:
    """Load only tensors and explicitly allow-listed NumPy container types.

    This keeps PyTorch's restricted weights-only unpickler enabled.  It never
    falls back to the general-purpose Pickle loader.
    """

    from numpy._core.multiarray import _reconstruct, scalar

    allowed = [
        _reconstruct,
        scalar,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.float32)),
        type(np.dtype(np.float64)),
        type(np.dtype(np.uint32)),
    ]
    with torch.serialization.safe_globals(allowed):
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint root must be a dictionary")
    return payload
