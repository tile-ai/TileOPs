"""What crosses the boundary between TileOPs and a backend. Types only."""

from __future__ import annotations

from typing import Callable, Final, NamedTuple, Union

import torch


class TensorSpec(NamedTuple):
    """What one tensor is, without the tensor.

    A backend's ``build_kernel`` is handed these rather than real tensors. That removes the
    need for a rule the neutral layer cannot check — "do not read the data, do not keep a
    reference": reading data would make the built kernel depend on values the memo does not
    key on, and keeping a reference would hold a tensor alive for as long as the kernel is
    cached, which is the whole process.
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

#: What a backend hands over: called ``build_kernel(*inputs, **params)`` with a
#: :class:`TensorSpec` per input in the op's manifest ``signature.inputs`` order, then its
#: ``signature.params`` by keyword with the op's normalized values. Returns something
#: callable with the tensors those specs describe. Both argument lists are per-op, which
#: the type system cannot express, hence ``...``.
BuildKernel = Callable[..., Callable[..., KernelResult]]

#: How a target recognizes its devices. Answers "is this the kind of device my kernels are
#: written for"; ``False`` rather than an exception for devices it does not serve. Whether a
#: particular call is supported — dtype, shape, parameter combination — is answered by
#: ``build_kernel``, which sees all of it.
DetectFn = Callable[[torch.device], bool]


class _Builtin:
    """The type of :data:`BUILTIN`. One instance, compared by identity."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "BUILTIN"


#: Ask for the in-tree implementation, whatever is installed. Not a target name: it is not
#: registered, holds no table entry, and never appears in ``registered_targets()``. It
#: exists because ``None`` means "decide for me", which cannot say "this time, run the
#: kernels that ship with TileOPs" once a third-party backend claims the device.
BUILTIN: Final = _Builtin()

#: What ``target=`` and the process default accept: a target name, :data:`BUILTIN`, or
#: ``None`` for "decide from the device".
Target = Union[str, _Builtin, None]
