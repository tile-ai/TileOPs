"""The nv backend: TileLang kernels for NVIDIA GPUs.

It ships in this distribution but joins through the same two callbacks and the same entry
point group as any third-party backend, and reads nothing a third party could not write.
Its op-to-kernel bindings are its own data in :mod:`._bindings`, not a field of the neutral
manifest.

Importing this module registers; it imports no kernel and no TileLang. That happens when an
op is first built, inside ``get_kernel``.
"""

from __future__ import annotations

import torch

from tileops.backend import register, register_detector

from ._builders import BUILDERS

#: This backend's target id.
TARGET = "nv"


def detect(device: torch.device) -> bool:
    """Is *device* an NVIDIA GPU?

    ``torch.device("cuda")`` is not enough: a ROCm build of torch calls its devices ``cuda``
    too, and these kernels do not run there. ``torch.version.hip`` is what separates them.
    """
    return device.type == "cuda" and torch.version.hip is None


def _register() -> None:
    register_detector(target=TARGET, detect=detect)
    for op, get_kernel in BUILDERS.items():
        register(op=op, target=TARGET, get_kernel=get_kernel)


_register()
