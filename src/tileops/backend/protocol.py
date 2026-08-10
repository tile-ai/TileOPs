"""What crosses the boundary between TileOPs and a backend.

Types only. Nothing here names a kernel class, a dispatch key or a priority: a backend
chooses among its own implementations inside ``get_kernel``, so dispatch answers *who*
does it, never *how*.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Protocol, Union

import torch


class InputSpec(NamedTuple):
    """What one input tensor *is*, without the tensor.

    A backend builds a kernel from the properties of the call, never from the data. This
    description is also the op layer's memo key, so comparing it costs nothing beyond the
    lookup that has to happen anyway.

    Device, dtype and shape are the whole description because the calling convention
    hands kernels contiguous tensors only. A kernel wanting strided input needs that
    convention extended first.
    """

    device: torch.device
    dtype: torch.dtype
    shape: tuple[int, ...]

    @staticmethod
    def of(tensor: torch.Tensor) -> "InputSpec":
        """Describe *tensor*."""
        return InputSpec(tensor.device, tensor.dtype, tuple(tensor.shape))


#: One call's result: a tensor, several, or nothing. ``torch.library.custom_op`` cannot
#: express a return value aliasing an input, so a purely mutating op returns ``None`` and
#: the op layer adds the chaining convenience above this boundary.
KernelResult = Union[torch.Tensor, tuple[torch.Tensor, ...], None]


class Kernel(Protocol):
    """What ``get_kernel`` returns. A structural convention — do not subclass.

    Positional order is the manifest's ``signature.inputs`` declaration order. An
    optional ``autotune(*example: torch.Tensor) -> None`` may be present: ``Op.autotune``
    calls it when it is and does nothing when it is not. Tuning measures, measurement
    needs real tensors, and tensors exist only once the kernel does — which is why it is a
    method here rather than a third registered callback.
    """

    def __call__(self, *tensors: torch.Tensor) -> KernelResult: ...


#: Callback one: is this device my target? Answer, do not raise — ``False`` for devices
#: this target does not serve. Called once per distinct device, then cached.
DetectFn = Callable[[torch.device], bool]

#: Callback two: give me something that computes this op. Called as
#: ``get_kernel(*inputs, **params)`` — :class:`InputSpec` values in the manifest's
#: ``signature.inputs`` order, then its ``signature.params`` by keyword, already
#: validated. Keyword names are per-op, which the type system cannot express.
GetKernelFn = Callable[..., Kernel]
