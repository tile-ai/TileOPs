"""A PrivateUse1 backend for tests, registered when this module is imported.

``pytest -p tests.privateuse1 --tileops-device=tileops_test`` runs a suite on it. It renames
PrivateUse1 as an out-of-tree backend does, has two devices and timing events, and keeps every
tensor in host memory, running each operator through torch's CPU fallback. It becomes the
process's accelerator, so it is loaded only by a run that names it, never by the test suite.
"""

from pathlib import Path

import torch
from torch.utils.cpp_extension import load

DEVICE_TYPE = "tileops_test"
DEVICE_COUNT = 2


class _DeviceModule:
    """What torch reads from ``torch.<device type>``; the current device lives in C++."""

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def is_initialized() -> bool:
        return True

    @staticmethod
    def _is_in_bad_fork() -> bool:
        return False

    @staticmethod
    def device_count() -> int:
        return DEVICE_COUNT

    @staticmethod
    def current_device() -> int:
        return torch.accelerator.current_device_index()

    @staticmethod
    def set_device(index: int) -> None:
        torch.accelerator.set_device_index(index)

    @staticmethod
    def synchronize(device=None) -> None:
        torch.accelerator.synchronize(device)

    @staticmethod
    def manual_seed_all(seed: int) -> None:
        """Every operator runs on the CPU, whose generator ``torch.manual_seed`` seeds."""


load(
    name="tileops_test_privateuse1",
    sources=[str(Path(__file__).with_name("backend.cpp"))],
    extra_cflags=[f"-DDEVICE_COUNT={DEVICE_COUNT}"],
    is_python_module=False,
)
torch.utils.rename_privateuse1_backend(DEVICE_TYPE)
torch._register_device_module(DEVICE_TYPE, _DeviceModule())
torch.utils.generate_methods_for_privateuse1_backend()
