"""The device a test or benchmark run places its tensors on.

The run names it (``--tileops-device`` in ``tests/conftest.py``); every workload and every test
that is not ``cuda_only`` reads it here. Read it when a tensor is built: a module constant or a
default argument would keep the value from before the run set it.
"""

from __future__ import annotations

import torch

_device: torch.device | str = "cuda"


def run_device() -> torch.device | str:
    """The device this run places tensors on; ``"cuda"`` unless the run named another."""
    return _device


def set_run_device(device: torch.device | str) -> None:
    """Place this run's tensors on *device* from now on."""
    global _device
    _device = device
