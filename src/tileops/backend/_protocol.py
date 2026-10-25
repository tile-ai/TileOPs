"""What crosses the boundary between TileOPs and a backend. Types only."""

from __future__ import annotations

import dataclasses
from typing import Callable, Final, Union

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


@dataclasses.dataclass(frozen=True, kw_only=True)
class DeviceFacts:
    """What kernel selection reads about one device. A backend's ``ReadDeviceFacts`` returns it.

    Constructed by keyword only. A field added in a later release comes with a default, so a
    reader written against this one keeps working.
    """

    # The architecture, comparable only with the ``supported_archs`` of implementations that
    # run on the same device type. CUDA encodes it as ``major * 10 + minor`` (90 for sm_90).
    # Another device type encodes it as its backend's ``supported_archs`` tables do, and that
    # backend documents the encoding.
    arch: int
    # Compute units: streaming multiprocessors on CUDA.
    sm_count: int
    # Shared memory one block may take after opting in, in bytes.
    smem_budget: int


# Called with a device of the type it was registered for, its index filled in where torch
# numbers that type's devices; a ``cpu`` device may carry none. Read on a dispatch miss,
# when selection first reads a fact of a call on that device.
ReadDeviceFacts = Callable[[torch.device], DeviceFacts]


@dataclasses.dataclass(frozen=True, kw_only=True)
class TensorSpec:
    """What one tensor is, without the tensor. Handed to ``build_kernel``.

    Constructed by keyword only. A field added in a later release comes with a default.
    """

    device: torch.device
    dtype: torch.dtype
    shape: tuple[int, ...]

    @staticmethod
    def of(tensor: torch.Tensor) -> "TensorSpec":
        """Describe *tensor*."""
        return TensorSpec(device=tensor.device, dtype=tensor.dtype, shape=tuple(tensor.shape))


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
