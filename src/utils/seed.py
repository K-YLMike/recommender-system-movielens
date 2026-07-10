"""Reproducibility helpers: seeding and RNG-state capture/restore.

Multi-seed ablations are a first-class requirement of this project (we report
mean +/- std across seeds, and never call a difference a "result" unless it
clears the seed variance). To make a *resumed* training run bit-for-bit
continuous with an uninterrupted one, we also capture and restore the RNG states
of Python, NumPy and torch inside each training checkpoint.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and (if available) torch for reproducibility.

    Also sets ``PYTHONHASHSEED`` for good measure. We intentionally do *not*
    force fully-deterministic CUDA kernels: on an H100 that can slow training
    noticeably, and our variance is already measured across seeds, so the small
    residual kernel nondeterminism is accounted for rather than eliminated.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def capture_rng_state() -> dict[str, Any]:
    """Snapshot the RNG states so a resumed run continues the same stream."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    try:
        import torch

        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG states captured by :func:`capture_rng_state`."""
    if state is None:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    try:
        import torch

        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except ImportError:
        pass
