"""NanToNum op: replace NaN, +Inf, -Inf with specified values."""

from typing import ClassVar, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.elementwise import NanToNumFwdKernel
from tileops.kernels.elementwise.call_spec import NanToNumCall, NanToNumFwdInterface
from tileops.kernels.kernel_base import KernelInterface
from tileops.ops.elementwise._base import ELEMENTWISE, UnaryOp


class NanToNumFwdOp(UnaryOp):
    """NanToNum: replace NaN, +Inf, -Inf with specified values."""

    kernel_types = {"nan_to_num": NanToNumFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {ELEMENTWISE: NanToNumFwdInterface}

    def __init__(
        self,
        *,
        nan: float = 0.0,
        posinf: Optional[float] = None,
        neginf: Optional[float] = None,
        target: Target = None,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            nan: Replacement for NaN (default 0.0).
            posinf: Replacement for +Inf. Manifest default ``None`` resolves
                to the largest finite value representable in the element type of the
                call (matches ``torch.nan_to_num``). An explicit value outside
                that element type's finite range is stored as Inf, as torch does.
            neginf: Replacement for -Inf. Manifest default ``None`` resolves
                to the smallest (most negative) finite value representable
                in the element type of the call.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for
                the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.nan = nan
        self.posinf = posinf
        self.neginf = neginf
        super().__init__(target=target)

    def _call_spec(self, input: torch.Tensor) -> NanToNumCall:
        return NanToNumCall(
            device=input.device,
            n_total=input.numel(),
            dtype=input.dtype,
            nan=self.nan,
            posinf=self.posinf,
            neginf=self.neginf,
        )
