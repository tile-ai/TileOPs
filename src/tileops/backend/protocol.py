"""What crosses the boundary between TileOPs and a backend. Types only."""

from __future__ import annotations

from typing import Callable, Final, NamedTuple, Union

import torch


class TensorSpec(NamedTuple):
    """What one tensor is, without the tensor.

    ``build_kernel`` is handed these instead of real tensors, so "do not read the data, do
    not keep a reference" needs no rule: neither is possible.
    """

    device: torch.device
    dtype: torch.dtype
    shape: tuple[int, ...]

    @staticmethod
    def of(tensor: torch.Tensor) -> "TensorSpec":
        """Describe *tensor*."""
        return TensorSpec(tensor.device, tensor.dtype, tuple(tensor.shape))


#: One call's result. A purely mutating op returns ``None``: ``torch.library.custom_op``
#: cannot express a return value aliasing an input, so the op layer adds the chaining
#: convenience above this boundary.
KernelResult = Union[torch.Tensor, tuple[torch.Tensor, ...], None]

#: Called ``build_kernel(*inputs, **params)``: a :class:`TensorSpec` per input in the op's
#: manifest ``signature.inputs`` order, then its ``signature.params`` by keyword. Returns
#: something callable with the tensors those specs describe. Both lists are per-op, which
#: the type system cannot express, hence ``...``.
BuildKernel = Callable[..., Callable[..., KernelResult]]

#: "Is this the kind of device my kernels are written for" — ``False``, not an exception,
#: for devices it does not serve. Whether a particular call is supported is ``build_kernel``'s
#: answer, which sees the dtypes and shapes too.
DetectFn = Callable[[torch.device], bool]


class _Builtin:
    """The type of :data:`BUILTIN`. One instance, compared by identity."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "BUILTIN"


#: Ask for the in-tree implementation whatever is installed. Not a target name: unregistered,
#: no table entry, never in ``registered_targets()``. ``None`` means "decide for me", which
#: cannot say this once a third-party backend claims the device.
BUILTIN: Final = _Builtin()

#: What ``target=`` and the process default accept: a target name, :data:`BUILTIN`, or
#: ``None`` for "decide from the device".
Target = Union[str, _Builtin, None]
