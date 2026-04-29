"""Base class for softmax-family operators (L2 Op layer).

Provides the shared validate -> reshape -> kernel -> trim -> reshape
pattern for softmax, log_softmax, and logsumexp ops.  Alignment padding
is handled inside the kernel via masked loads, not on the host.

Construction: ``op(dtype=..., dim=-1)``.  M and N are derived
from the input tensor at forward time, and kernels are cached by
``(M, N)`` to avoid rebuilds.
"""

import warnings
from math import prod
from typing import Dict, List, Optional, Union

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import DEFAULT_ALIGNMENT, align_up

from ..op_base import Op
from ._multidim import EmptyDimPolicy, flatten_for_multidim, normalize_dim, restore_multidim_shape

__all__ = ["_SoftmaxBaseOp"]


def _resolve_implicit_softmax_dim(name: str, ndim: int) -> int:
    """Resolve ``dim=None`` to a scalar axis matching PyTorch's implicit choice.

    Mirrors ``torch.nn.functional._get_softmax_dim``: pick 0 for
    ``ndim in {0, 1, 3}`` else 1, and emit the same deprecation
    ``UserWarning`` PyTorch emits.

    Args:
        name: Op name used in the warning message (e.g. ``"softmax"``).
        ndim: Rank of the input tensor.

    Returns:
        Resolved scalar reduction axis (non-negative).
    """
    warnings.warn(
        f"Implicit dimension choice for {name} has been deprecated. "
        "Change the call to include dim=X as an argument.",
        UserWarning,
        stacklevel=3,
    )
    if ndim == 0 or ndim == 1 or ndim == 3:
        return 0
    return 1


class _SoftmaxBaseOp(Op):
    """Base class for softmax-family ops.

    Handles shared validation, reshape, pad/trim logic. Subclasses only
    need to set ``_op_kind``, ``_kernel_key``, ``_kernel_class`` and
    override output reshaping if needed.

    Args:
        dtype: Data type (float32, float16, or bfloat16).
        dim: Reduction dimension (default -1).
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    _op_kind: str  # set by subclass
    _kernel_key: str  # set by subclass
    _kernel_class: type  # set by subclass
    _supports_multidim: bool = False  # override to True in reduced-dim ops (e.g. LogSumExpFwdOp)
    _empty_dim_policy: EmptyDimPolicy = "reject"

    # `static_dims.N = x.shape[dim]` is param-dependent (depends on `dim`),
    # so the static-axis frozenset is bound at forward time after dim
    # normalization, not at the class level (per docs/design/ops-design.md § Step 3).
    _static_axes: frozenset = frozenset()

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        dim: Union[int, List[int]] = -1,
        N: Optional[int] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.dtype = dtype
        self.dim = dim
        self.N = N
        self.keepdim = False
        self._tune = tune
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, object] = {}
        self._last_roofline_mn: tuple[int, int] | None = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._kernel_key: self._kernel_class}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, x: torch.Tensor) -> None:
        """Validate input tensor."""
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.ndim == 0:
            raise ValueError("Input tensor must be at least 1D")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the softmax-family op.

        Accepts arbitrary-dim input along the configured dim.
        Supports ``dim=list[int]`` for multi-dim reduction (logsumexp).
        """
        self._validate(x)
        orig_shape = x.shape

        # Resolve dim=None per call (don't mutate self.dim) so the same op
        # instance accepts inputs of different ranks, matching F.softmax.
        effective_dim: Union[int, List[int], None] = self.dim
        if effective_dim is None and not self._supports_multidim:
            effective_dim = _resolve_implicit_softmax_dim(self._op_kind, x.ndim)

        if isinstance(effective_dim, (list, tuple)) or effective_dim is None:
            if not self._supports_multidim:
                raise ValueError(
                    f"{type(self).__name__} does not support multi-dim reduction. "
                    "Use a scalar dim."
                )
            dims = normalize_dim(
                effective_dim, x.ndim, empty_dim_policy=self._empty_dim_policy,
            )
            # Bind the dynamic static-axes (param-dependent reduction axes) so
            # the Op-layer cache-key / introspection consumers see the
            # committed axes. Mirrors the single-dim path below.
            self._static_axes = frozenset((0, d) for d in dims)
            x, orig_shape, _kept = flatten_for_multidim(x, dims)
            N = x.shape[-1]
            M = prod(x.shape[:-1])
            self._last_roofline_mn = (M, N)
            x = x.reshape(M, N)
            kernel = self._get_or_create_kernel(M, N, device_index=x.device.index)
            # Alignment padding is handled by the kernel's forward().
            y = kernel(x)
            N_padded = align_up(N, DEFAULT_ALIGNMENT)
            if N_padded != N:
                y = y[:, :N] if y.ndim == 2 else y
            return restore_multidim_shape(y, orig_shape, dims, self.keepdim)

        # --- single-dim path ---
        # Validate and normalize dim (match PyTorch IndexError behavior).
        assert isinstance(effective_dim, int)
        if effective_dim < -x.ndim or effective_dim >= x.ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-x.ndim}, {x.ndim - 1}], but got {effective_dim})"
            )
        dim = effective_dim % x.ndim

        # N = size along reduction dim, M = product of all other dims.
        N = x.shape[dim]
        if self.N is not None and N != self.N:
            raise ValueError(
                f"{type(self).__name__}: committed N={self.N} does not match "
                f"x.shape[{effective_dim}]={N}"
            )
        # Bind the dynamic static-axis (param-dependent N axis) so the
        # Op-layer cache-key / introspection consumers see the committed axis.
        self._static_axes = frozenset({(0, dim)})
        M = prod(s for i, s in enumerate(x.shape) if i != dim)
        self._last_roofline_mn = (M, N)

        # If reduction dim is not the last, move it to the end.
        needs_transpose = dim != x.ndim - 1
        if needs_transpose:
            x = x.movedim(dim, -1)

        x = x.contiguous().reshape(M, N)

        # Get or create cached kernel for this (M, N, device).
        kernel = self._get_or_create_kernel(M, N, device_index=x.device.index)

        # Alignment padding is handled by the kernel's forward().
        y = kernel(x)

        # Trim padding (kernel output may still be N_padded-wide).
        N_padded = align_up(N, DEFAULT_ALIGNMENT)
        if N_padded != N:
            y = y[:, :N] if y.ndim == 2 else y

        return self._reshape_output(y, orig_shape, dim, needs_transpose)

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_mn is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() "
                "call to bind dynamic input shape"
            )
        M, N = self._last_roofline_mn
        elem_bytes = self.dtype.itemsize
        if self._op_kind == "softmax":
            return 5 * M * N, 2 * M * N * elem_bytes
        if self._op_kind == "log_softmax":
            return 6 * M * N, 2 * M * N * elem_bytes
        if self._op_kind == "logsumexp":
            return 4 * M * N, (M * N + M) * elem_bytes
        raise NotImplementedError(
            f"{type(self).__name__} has unknown roofline op kind {self._op_kind!r}"
        )

    def _get_or_create_kernel(self, M: int, N: int, device_index: int | None = None) -> object:
        """Return a cached kernel for (M, N, device_index), creating one if needed."""
        key = (M, N, device_index)
        if key not in self._kernel_cache:
            kernel_cls = self.kernel_map[self._kernel_key]
            self._kernel_cache[key] = kernel_cls(
                M, N, self._op_kind, self.dtype, tune=self._tune,
                device_index=device_index,
            )
        return self._kernel_cache[key]

    # ------------------------------------------------------------------
    # Output reshaping
    # ------------------------------------------------------------------

    def _reshape_output(
        self,
        y: torch.Tensor,
        orig_shape: torch.Size,
        dim: int,
        needs_transpose: bool,
    ) -> torch.Tensor:
        """Restore original shape.

        Default (softmax/log_softmax): output same shape as input.
        Reduced-dim ops (logsumexp): remove or keep dim based on keepdim.
        """
        # y is (M, N) or (M,) depending on op kind.
        if y.ndim == 2:
            # Same-shape ops: rebuild the transposed shape, then move dim back.
            if needs_transpose:
                transposed_shape = list(orig_shape)
                transposed_shape.append(transposed_shape.pop(dim))
                y = y.reshape(transposed_shape)
                y = y.movedim(-1, dim)
            else:
                y = y.reshape(orig_shape)
        else:
            # Reduced-dim ops (logsumexp): (M,) -> remove or keep dim.
            if self.keepdim:
                kept_shape = list(orig_shape)
                kept_shape[dim] = 1
                y = y.reshape(kept_shape)
            else:
                reduced_shape = [s for i, s in enumerate(orig_shape) if i != dim]
                y = y.squeeze() if len(reduced_shape) == 0 else y.reshape(reduced_shape)

        return y
