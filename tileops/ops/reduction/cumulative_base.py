"""CumulativeOp base class for scan operators (cumsum, cumprod).

Implements manifest-driven dynamic shape binding: ctor binds only
semantic/config params (`dim`, optional committed `N` for strict
compatibility); `M` and the effective `N` are derived at forward time, kernels
are built lazily and cached by `(M, N, dtype, device)`.

Forward pipeline: validate -> movedim(dim → -1) -> reshape (M, N) -> kernel
(handles alignment via masked loads) -> trim -> reshape -> movedim(-1 → dim).
"""

from abc import abstractmethod
from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import DEFAULT_ALIGNMENT, align_up
from tileops.kernels.reduction.cumulative import CumulativeKernel

from ..op_base import Op

__all__ = ["CumulativeOp"]


class CumulativeOp(Op):
    """Abstract base for cumulative scan operators with a user-selectable axis.

    Subclasses must override `_op_kind` (class attribute) — the kernel's
    op-kind dispatch string (`"sum"` or `"prod"`).

    Args:
        dtype: Data type (float32, float16, or bfloat16). If omitted,
            inferred from the first input tensor.
        N: Optional reduction dimension size. When provided, forward validates
            it against ``x.shape[dim]`` for backward compatibility.
        dim: Reduction axis (default -1). Negative values are normalized at
            forward time (`dim % x.ndim`).
        kernel_map: Optional kernel override dict.
        tune: If True, autotune tile configs.
    """

    _op_kind: str

    # `static_dims.N = x.shape[dim]` is param-dependent (depends on `dim`),
    # so the static-axis frozenset is bound at forward time after dim
    # normalization, not at the class level (per docs/design/ops-design.md § Step 3).
    _static_axes: frozenset = frozenset()

    def __init__(
        self,
        N: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        dim: int = -1,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.N = N
        self.dtype = dtype
        self._committed_N = N
        self._committed_dtype = dtype
        self.dim = dim
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple[int, int, torch.dtype, int | None], Kernel] = {}
        self._last_roofline_mn: Optional[Tuple[int, int]] = None

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

    def _get_kernel(
        self, M: int, N: int, dtype: torch.dtype, device_index: int | None,
    ) -> Kernel:
        """Return a kernel built for (M, N, dtype), caching by specialization.

        Note: Not decorated with @torch.compiler.disable to allow torch.compile(fullgraph=True).
        Pre-warming the cache before compilation ensures kernel construction happens outside
        the compiled region.
        """
        key = (M, N, dtype, device_index)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["cumulative_fwd"](
                M, N, self._op_kind, dtype, tune=self.tune,
            )
        return self._kernel_cache[key]

    def _validate_and_normalize_dim(self, x: torch.Tensor) -> tuple[int, int, torch.dtype]:
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if self._committed_dtype is not None and x.dtype != self._committed_dtype:
            raise ValueError(
                f"Expected x.dtype {self._committed_dtype}, got {x.dtype}"
            )
        ndim = x.ndim
        if not (-ndim <= self.dim < ndim):
            raise ValueError(
                f"dim={self.dim} out of range for {ndim}-D input"
            )
        dim_norm = self.dim % ndim
        N = x.shape[dim_norm]
        if self._committed_N is not None and self._committed_N != N:
            raise ValueError(
                f"Expected x.shape[{self.dim}]={self._committed_N}, "
                f"got {N}"
            )
        self.N = N
        self.dtype = x.dtype
        # Bind the dynamic static-axis (param-dependent N axis) so
        # Op-layer cache-key / introspection consumers see the committed axis.
        self._static_axes = frozenset({(0, dim_norm)})
        return dim_norm, N, x.dtype

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Subclasses implement: validate, movedim, kernel call, restore."""
        raise NotImplementedError

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        """Shared forward implementation. Subclasses call this from `forward`."""
        ndim = x.ndim
        dim_norm, N, dtype = self._validate_and_normalize_dim(x)

        if dim_norm != ndim - 1:
            x = x.movedim(dim_norm, -1)
        post_move_shape = tuple(x.shape)
        x = x.contiguous().reshape(-1, N)
        M = x.shape[0]

        # Alignment padding is handled inside the kernel via masked loads.
        y = self._get_kernel(M, N, dtype, x.device.index)(x)
        self._last_roofline_mn = (M, N)

        # Kernel output is N_padded-wide along last dim; trim to N.
        N_padded = align_up(N, DEFAULT_ALIGNMENT)
        if N_padded != N:
            y = y[:, :N]
        y = y.reshape(post_move_shape)
        if dim_norm != ndim - 1:
            y = y.movedim(-1, dim_norm)
        return y
