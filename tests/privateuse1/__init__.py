"""Opt-in PrivateUse1 backend with synchronous CPU execution.

Importing it replaces the process accelerator; load only through the pytest plugin.
"""

from pathlib import Path

import torch
from torch.utils.cpp_extension import load

DEVICE_TYPE = "tileops_test"
DEVICE_COUNT = 2


class _DeviceModule:
    """PyTorch device-module hooks."""

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def _is_in_bad_fork() -> bool:
        return False

    @staticmethod
    def device_count() -> int:
        return DEVICE_COUNT

    @staticmethod
    def manual_seed_all(seed: int) -> None:
        """CPU fallback uses the generator seeded by torch.manual_seed."""


load(
    name="tileops_test_privateuse1",
    sources=[str(Path(__file__).with_name("backend.cpp"))],
    extra_cflags=[f"-DDEVICE_COUNT={DEVICE_COUNT}"],
    is_python_module=False,
)
torch.utils.rename_privateuse1_backend(DEVICE_TYPE)
torch._register_device_module(DEVICE_TYPE, _DeviceModule())
torch.utils.generate_methods_for_privateuse1_backend()
