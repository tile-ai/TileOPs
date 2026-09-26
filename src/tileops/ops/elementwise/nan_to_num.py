"""NanToNum op: replace NaN, +Inf, -Inf with specified values."""

from typing import ClassVar, Dict, Optional

import torch

from tileops.backend import Target
from tileops.kernels.elementwise import NanToNumFwdKernel
from tileops.kernels.kernel_base import Kernel

from ..op_base import Op
from ._base import _PerDtypeKernels


class NanToNumFwdOp(_PerDtypeKernels, Op):
    """NanToNum: replace NaN, +Inf, -Inf with specified values."""

    compile_boundary: ClassVar[bool] = True
    kernel_types = {"nan_to_num": NanToNumFwdKernel}

    def __init__(
        self,
        *,
        nan: float = 0.0,
        posinf: Optional[float] = None,
        neginf: Optional[float] = None,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
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
            kernel_map: Optional kernel dispatch override.
            tune: Whether to autotune the kernel.
        """
        self.nan = nan
        self.posinf = posinf
        self.neginf = neginf
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def _build(self, dtype: torch.dtype, n_total: int):
        """Resolve the replacement values against *dtype*, then build.

        A ``None`` bound means "this dtype's largest finite value", so it
        cannot be resolved before the element type is known. Picking
        ``finfo(dtype).max`` matches ``torch.nan_to_num``; forwarding ``+inf``
        would write back the infinity the op was called to replace. A given value is
        cast to *dtype* as torch casts it, through float32, so one past the dtype's
        range becomes Inf.
        """

        def cast(value: float) -> float:
            return torch.tensor(value, dtype=torch.float32).to(dtype).item()

        posinf = torch.finfo(dtype).max if self.posinf is None else cast(self.posinf)
        neginf = torch.finfo(dtype).min if self.neginf is None else cast(self.neginf)
        # Replacement values are positional; the kernel constructor's
        # parameter naming is encapsulated below the Op layer.
        impl, ctor_dtype = self._selected_kernel_cls().specialize(dtype)
        return impl(n_total, ctor_dtype, cast(self.nan), posinf, neginf, tune=self.tune)

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.contiguous()
        return self._kernel((input,), input.dtype, input.numel())(input)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run the op on ``input``."""
        return self._call_boundary(input)
