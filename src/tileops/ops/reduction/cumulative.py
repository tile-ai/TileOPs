"""Cumulative scan operators (cumsum, cumprod)."""

from math import prod
from typing import Dict, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.cumulative import CumulativeKernel

from ..op_base import Op
from ._boundary import register_reduction_op

__all__ = ["CumprodFwdOp", "CumsumFwdOp", "CumulativeOp"]


class CumulativeOp(Op):
    """Abstract base for cumulative scan operators with a user-selectable axis.

    Subclasses must override `_op_kind` (class attribute) — the kernel's
    op-kind dispatch string (`"sum"` or `"prod"`).

    """

    _op_kind: str

    #: Set by ``register_reduction_op`` on each concrete op; a base registers none.
    _wrapped = None

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
        self.N = None
        self.dim = dim
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self._last_roofline_mn: Optional[Tuple[int, int]] = None

    def _infer_output_shapes(self, x_shape: Tuple[int, ...]) -> Dict[str, Tuple[int, ...]]:
        """Manifest ``shape_rules``: a scan writes one element per input element."""
        return {"y": tuple(x_shape)}

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"cumulative_fwd": CumulativeKernel}

    def eval_roofline(self) -> Tuple[int, int]:
        if self._last_roofline_mn is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior "
                "forward() call to bind dynamic input shape (M)"
            )
        M, N = self._last_roofline_mn
        if self.dtype is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior "
                "forward() call to bind dtype"
            )
        elem_bytes = self.dtype.itemsize
        # Per row: N-1 ops (running sum/prod) ≈ M*N flops total.
        # Read x + write y = 2 * M * N elements.
        return (M * N, 2 * M * N * elem_bytes)

    def _validate_and_normalize_dim(self, x: torch.Tensor) -> int:
        """Validate the input and return the non-negative axis to scan.

        Which devices a set of kernels runs on is the kernel's own statement, so no
        device kind is checked here.

        Raises:
            ValueError: ``dim`` names an axis this rank does not have.
        """
        self._validate_dtypes(x)
        ndim = x.ndim
        if not (-ndim <= self.dim < ndim):
            raise ValueError(f"dim={self.dim} out of range for {ndim}-D input")
        self.dtype = x.dtype
        return self.dim % ndim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the scan.

        One call to the operator this op registers: this is as far as dynamo traces.
        """
        return type(self)._wrapped(x, self._instance_key)

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Validate, resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        axis = self._validate_and_normalize_dim(x)
        x = x.contiguous()  # handed over as the manifest declares it
        n = x.shape[axis]
        self.N = n
        # From the shape, not from ``numel``: an empty scanned axis makes ``n`` zero.
        m = prod(d for i, d in enumerate(x.shape) if i != axis)
        self._last_roofline_mn = (m, n)
        kernel = self.get_or_build_kernel(
            "cumulative_fwd",
            (x,),
            # The kernel owns the permute, so the whole shape decides which kernel it is.
            key=(tuple(x.shape), axis, x.dtype, x.device.index),
            build=lambda: self.kernel_map["cumulative_fwd"](
                m,
                n,
                self._op_kind,
                x.dtype,
                scan_axis=axis,
                tune=self.tune,
                device_index=x.device.index,
            ),
        )
        return kernel(x)


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


for _op_cls in (CumsumFwdOp, CumprodFwdOp):
    register_reduction_op(_op_cls)
