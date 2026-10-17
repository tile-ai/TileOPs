"""The call's device, through torch's accelerator API rather than one device type's.

A process has at most one accelerator type, CUDA or a backend's, and torch names it. The
op layer and the benchmarks read the current index and enter a device through here, so
neither names CUDA where it means the call's device.
"""

import contextlib
import functools

import torch

__all__ = ["current_accelerator_type", "current_device", "device_context", "on_current_index"]


@functools.cache
def current_accelerator_type() -> "str | None":
    """The device type of the accelerator this process can run on, or ``None``.

    ``None`` too where torch was built with an accelerator the process has no device of.
    Read once: a call spec without a device index reads it on every call.
    ``tileops.utils.forget_device_properties`` drops it with the other device caches.
    """
    device = torch.accelerator.current_accelerator(check_available=True)
    return None if device is None else device.type


def current_device() -> "torch.device | None":
    """The accelerator's current device, where a call names none; ``None`` without one."""
    kind = current_accelerator_type()
    return None if kind is None else on_current_index(torch.device(kind))


def on_current_index(device: torch.device) -> torch.device:
    """*device* with its index filled in, when it has none and is the accelerator's type.

    Any other device comes back unchanged: a CPU device, or one of a type this process
    has no accelerator of.
    """
    if device.index is not None or device.type != current_accelerator_type():
        return device
    return torch.device(device.type, torch.accelerator.current_device_index())


def device_context(device: "torch.device | None") -> contextlib.AbstractContextManager:
    """Make *device* current for the block, when it is the accelerator's type.

    A null context for ``None``, a CPU device or a device of another type. It sets the
    current device only; the current stream is the one torch keeps for that device.
    """
    if device is None or device.type != current_accelerator_type():
        return contextlib.nullcontext()
    return torch.accelerator.device_index(on_current_index(device).index)
