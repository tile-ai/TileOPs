"""Cumulative scan operators (cumsum, cumprod)."""

from typing import ClassVar, Mapping, Optional

import torch

from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._kernels.reduction.call_spec import (
    CumprodFwdInterface,
    CumsumFwdInterface,
    CumulativeCall,
)
from tileops._kernels.reduction.cumulative import (
    CumsumParallelScanKernel,
    CumulativeKernel,
    CumulativeRowScanKernel,
)
from tileops._ops.op_base import Op
from tileops.backend import Target
from tileops.manifest.primitives import normalize_axis

__all__ = ["CumprodFwdOp", "CumsumFwdOp", "CumulativeOp"]


class CumulativeOp(Op):
    """Abstract base for cumulative scan operators with a user-selectable axis.

    Subclasses must override `_op_kind` (class attribute) — the kernel's
    op-kind dispatch string (`"sum"` or `"prod"`).

    """

    _op_kind: str

    def __init__(
        self,
        dim: int,
        *,
        dtype: Optional[torch.dtype] = None,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            dim: The axis to scan, required as in torch. Negative values count from the
                last axis.
            dtype: The dtype the input is cast to before the scan, and the output's;
                ``None`` keeps the input's, as in torch.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.dim = dim
        self.dtype = dtype
        super().__init__(target=target)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run the scan.

        One call to the operator this op registers: this is as far as dynamo traces.
        """
        if self.dtype is not None:
            # torch casts the input before it scans.
            input = input.to(self.dtype)
        input = input.contiguous()  # handed over as the manifest declares it
        return self.kernel_for(
            "cumulative_fwd",
            lambda: CumulativeCall(
                device=input.device,
                shape=tuple(input.shape),
                axis=normalize_axis(self.dim, input.ndim),
                op_kind=self._op_kind,
                dtype=input.dtype,
            ),
        )(input)


class CumsumFwdOp(CumulativeOp):
    """Cumulative sum operator: ``output = cumsum(input, dim)``.

    Output has the shape of ``input``, and its dtype or ``dtype`` when one is passed.
    Alignment padding is handled inside the kernel via masked loads.

    A row one thread block can stage in shared memory takes the whole-row scan;
    other rows with ``M < 128 and N > 8192`` take a three-pass parallel scan for SM
    utilization; every other shape takes the tiled scan.

    Example:
        ```python linenums="1"
        op = CumsumFwdOp(dim=-1)
        x = torch.randn(1024, 4096, dtype=torch.float16, device="cuda")
        y = op(x)  # shape: (1024, 4096)
        ```
    """

    _op_kind = "sum"
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "cumulative_row_scan": CumulativeRowScanKernel,
        "cumulative_parallel_scan": CumsumParallelScanKernel,
        "cumulative_fwd": CumulativeKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "cumulative_fwd": CumsumFwdInterface
    }


class CumprodFwdOp(CumulativeOp):
    """Cumulative product operator: ``output = cumprod(input, dim)``.

    Output has the shape of ``input``, and its dtype or ``dtype`` when one is passed.
    Alignment padding is handled inside the kernel via masked loads.

    Example:
        ```python linenums="1"
        op = CumprodFwdOp(dim=-1)
        x = torch.randn(1024, 4096, dtype=torch.float16, device="cuda") * 0.01 + 0.99
        y = op(x)  # shape: (1024, 4096)
        ```
    """

    _op_kind = "prod"
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "cumulative_row_scan": CumulativeRowScanKernel,
        "cumulative_fwd": CumulativeKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "cumulative_fwd": CumprodFwdInterface
    }
