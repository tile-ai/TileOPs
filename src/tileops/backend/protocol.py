"""What crosses the boundary between TileOPs and a backend. Types only."""

from __future__ import annotations

from typing import Callable, NamedTuple, Protocol, Union

import torch


class InputSpec(NamedTuple):
    """What one input tensor is, without the tensor.

    Also the op layer's memo key, so comparing it costs nothing extra. Device, dtype and
    shape suffice only because the calling convention hands kernels contiguous tensors —
    a kernel wanting strided input needs that convention extended first.
    """

    device: torch.device
    dtype: torch.dtype
    shape: tuple[int, ...]

    @staticmethod
    def of(tensor: torch.Tensor) -> "InputSpec":
        """Describe *tensor*."""
        return InputSpec(tensor.device, tensor.dtype, tuple(tensor.shape))


#: One call's result. A purely mutating op returns ``None``: ``torch.library.custom_op``
#: cannot express a return value aliasing an input, so the op layer adds the chaining
#: convenience above this boundary.
KernelResult = Union[torch.Tensor, tuple[torch.Tensor, ...], None]


class Kernel(Protocol):
    """What ``get_kernel`` returns. A structural convention — do not subclass.

    Positional order is the manifest's ``signature.inputs`` order. An optional
    ``autotune(*example: torch.Tensor) -> None`` may be present; ``Op.autotune`` calls it
    when it is. Tuning needs real tensors, which exist only once the kernel does, which is
    why it is a method here rather than a third registered callback.
    """

    def __call__(self, *tensors: torch.Tensor) -> KernelResult: ...


#: Callback one: is this device my target? Answer ``False`` rather than raising for
#: devices this target does not serve. Asked once per device, then memoized.
DetectFn = Callable[[torch.device], bool]

#: Callback two: something that computes this op. Called as ``get_kernel(*inputs,
#: **params)`` — :class:`InputSpec` values in the manifest's ``signature.inputs`` order,
#: then its ``signature.params`` by keyword, already validated. Keyword names are per-op,
#: which the type system cannot express.
GetKernelFn = Callable[..., Kernel]
