"""What crosses the boundary between TileOPs and a backend. Types only."""

from __future__ import annotations

from typing import Callable, Final, NamedTuple, Union

import torch

# One call's result. A purely mutating op returns ``None``: ``torch.library.custom_op``
# cannot express a return value aliasing an input.
KernelResult = Union[torch.Tensor, tuple[torch.Tensor, ...], None]
# Called ``build_kernel(*inputs, **params)``: a `TensorSpec` per input in
# ``signature.inputs`` order — ``None`` for an ``optional: true`` input the call did not
# pass, so presence is read off the slot rather than off how many slots there are — then
# ``signature.params`` by keyword. Both lists are per-op, which the type system cannot
# express, hence ``...``. The kernel it returns may also define ``request_tune()`` and
# ``config``, as an in-tree ``Kernel`` does (``tileops.backend``).
BuildKernel = Callable[..., Callable[..., KernelResult]]
# "Is this the kind of device my kernels are written for" — ``False``, not an exception,
# for the rest. Per-call support is ``build_kernel``'s answer; it sees the dtypes too.
DetectFn = Callable[[torch.device], bool]


class DeviceFacts(NamedTuple):
    """What kernel selection reads about one device. A backend's ``ReadDeviceFacts`` returns it."""

    # The architecture, comparable only with the ``supported_archs`` of implementations
    # that run on the same device type.
    arch: int
    # The key of the calibrated board the device belongs to, or ``None``. Fitted tuning
    # data is keyed by it.
    calibration: "str | None"
    # Compute units: streaming multiprocessors on CUDA.
    sm_count: int
    # Shared memory one block may take after opting in, in bytes.
    smem_budget: int


# Called with a device of the type it was registered for, its index filled in where torch
# numbers that type's devices; a ``cpu`` device may carry none. Read on a dispatch miss,
# when selection first reads a fact of a call on that device.
ReadDeviceFacts = Callable[[torch.device], DeviceFacts]


class TensorSpec(NamedTuple):
    """What one tensor is, without the tensor. Handed to ``build_kernel``."""

    device: torch.device
    dtype: torch.dtype
    shape: tuple[int, ...]

    @staticmethod
    def of(tensor: torch.Tensor) -> "TensorSpec":
        """Describe *tensor*."""
        return TensorSpec(tensor.device, tensor.dtype, tuple(tensor.shape))


class _Builtin:
    """The type of :data:`BUILTIN`. One instance, compared by identity."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "BUILTIN"


# Ask for the in-tree implementation whatever is installed. Not a target name: unregistered
# and never in ``registered_targets()``.
BUILTIN: Final = _Builtin()

# What ``target=`` and the process default accept.
Target = Union[str, _Builtin, None]
