"""Cumulative scan operators (cumsum, cumprod)."""

from math import prod
from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.reduction.cumulative import CumulativeKernel
from tileops.manifest.primitives import normalize_axis

from ..op_base import Op

__all__ = ["CumprodFwdOp", "CumsumFwdOp", "CumulativeOp"]


class CumulativeOp(Op):
    """Abstract base for cumulative scan operators with a user-selectable axis.

    Subclasses must override `_op_kind` (class attribute) — the kernel's
    op-kind dispatch string (`"sum"` or `"prod"`).

    """

    _op_kind: str

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"cumulative_fwd": CumulativeKernel}

    def __init__(
        self,
        dim: int = -1,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            dim: Reduction axis (default -1). Negative values are normalized at
                forward time (`dim % x.ndim`).
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: If True, autotune tile configs.
        """
        self.dim = dim
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the scan.

        One call to the operator this op registers: this is as far as dynamo traces.
        """
        return self._call_boundary(x)

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        axis = normalize_axis(self.dim, x.ndim)
        x = x.contiguous()  # handed over as the manifest declares it
        n = x.shape[axis]
        # From the shape, not from ``numel``: an empty scanned axis makes ``n`` zero.
        m = prod(d for i, d in enumerate(x.shape) if i != axis)
        kernel = self.kernel_for(
            "cumulative_fwd", (x,), (tuple(x.shape), axis, x.dtype, x.device.index, m, n)
        )
        return kernel(x)

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built from the whole shape and the axis it scans.

        The kernel owns the permute, so the whole shape decides which kernel it is.
        """
        _shape, axis, dtype, device_index, m, n = call
        return call, lambda: self.kernel_map["cumulative_fwd"](
            m,
            n,
            self._op_kind,
            dtype,
            scan_axis=axis,
            tune=self.tune,
            device_index=device_index,
        )


class CumsumFwdOp(CumulativeOp):
    """Cumulative sum operator: ``y = cumsum(x, dim)``.

    Output has the same shape and dtype as ``x``. Alignment padding is
    handled inside the kernel via masked loads.

    A row one thread block can stage in shared memory takes the whole-row scan.
    Of what is left, shapes with ``M < 128 and N > 8192`` take a three-pass
    parallel scan for SM utilization; every other shape takes the tiled scan.

    Args:
        dim: Reduction axis (default -1). Negative values are normalized
            at forward time.
        target: Which set of kernels serves this op — a target name, ``BUILTIN``
            for the in-tree kernels, or ``None`` to decide from the input device.
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).

    Example:
        ```python linenums="1"
        op = CumsumFwdOp()
        x = torch.randn(1024, 4096, dtype=torch.float16, device="cuda")
        y = op(x)  # shape: (1024, 4096)
        ```
    """

    _op_kind = "sum"


class CumprodFwdOp(CumulativeOp):
    """Cumulative product operator: ``y = cumprod(x, dim)``.

    Output has the same shape and dtype as ``x``. Alignment padding is
    handled inside the kernel via masked loads.

    Args:
        dim: Reduction axis (default -1). Negative values are normalized
            at forward time.
        target: Which set of kernels serves this op — a target name, ``BUILTIN``
            for the in-tree kernels, or ``None`` to decide from the input device.
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).

    Example:
        ```python linenums="1"
        op = CumprodFwdOp()
        x = torch.randn(1024, 4096, dtype=torch.float16, device="cuda") * 0.01 + 0.99
        y = op(x)  # shape: (1024, 4096)
        ```
    """

    _op_kind = "prod"
