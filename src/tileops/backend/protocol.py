"""What crosses the boundary between TileOPs and a backend. Types only."""

from __future__ import annotations

from typing import Callable, Final, NamedTuple, Union

import torch


class TensorSpec(NamedTuple):
    """What one tensor is, without the tensor. Handed to ``build_kernel``."""

    device: torch.device
    dtype: torch.dtype
    shape: tuple[int, ...]

    @staticmethod
    def of(tensor: torch.Tensor) -> "TensorSpec":
        """Describe *tensor*."""
        return TensorSpec(tensor.device, tensor.dtype, tuple(tensor.shape))


#: One call's result. A purely mutating op returns ``None``: ``torch.library.custom_op``
#: cannot express a return value aliasing an input.
KernelResult = Union[torch.Tensor, tuple[torch.Tensor, ...], None]

#: Called ``build_kernel(*inputs, **params)``: a :class:`TensorSpec` per input in
#: ``signature.inputs`` order, then ``signature.params`` by keyword. Both lists are per-op,
#: which the type system cannot express, hence ``...``.
BuildKernel = Callable[..., Callable[..., KernelResult]]

#: "Is this the kind of device my kernels are written for" — ``False``, not an exception,
#: for the rest. Per-call support is ``build_kernel``'s answer; it sees the dtypes too.
DetectFn = Callable[[torch.device], bool]


class _Builtin:
    """The type of :data:`BUILTIN`. One instance, compared by identity."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "BUILTIN"


#: Ask for the in-tree implementation whatever is installed. Not a target name: unregistered
#: and never in ``registered_targets()``.
BUILTIN: Final = _Builtin()

#: What ``target=`` and the process default accept.
Target = Union[str, _Builtin, None]
