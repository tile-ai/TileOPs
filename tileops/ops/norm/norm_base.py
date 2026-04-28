"""RowNormOp base class for row-wise normalization operators.

Implements the canonical static_dims pattern from docs/design/ops-design.md: ctor
binds only what the manifest declares as `static_dims` (`N`) plus
signature.params (`dim`, `eps`); `M` (the leading-dims product) is derived
at forward time and used to lazily build / cache kernels keyed by `M`.

Forward pipeline: validate -> movedim(dim → -1) -> reshape (M, N) -> pad ->
kernel -> trim -> reshape -> movedim(-1 → dim).

BatchNorm uses spatial reduction (not a single axis), so it does NOT inherit
from this.
"""

from abc import abstractmethod
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from tileops.kernels.kernel_base import Kernel

from ..op_base import Op

__all__ = ["ALIGNMENT", "RowNormOp"]

ALIGNMENT = 256


class RowNormOp(Op):
    """Abstract base class for row-wise normalization operators with a
    user-selectable reduction axis.

    Subclasses must override:
    - ``_kernel_key`` (class attribute): kernel-map key string.
    - ``_kernel_cls`` (class attribute): kernel class.
    - ``forward()``: the user-facing call (use the helpers below).

    Args:
        N: Hidden / reduction dimension size (statically committed at ctor).
        dtype: Data type (float16 or bfloat16).
        dim: Reduction axis (default -1). Negative values are normalized at
            forward time (``dim % x.ndim``).
        eps: Epsilon for numerical stability.
        kernel_map: Optional kernel override dict.
        tune: If True, autotune tile configs.
    """

    _kernel_key: str
    _kernel_cls: type

    # `static_dims.N = x.shape[dim]` is param-dependent (depends on `dim`),
    # so the static-axis frozenset is bound at forward time after dim
    # normalization, not at the class level (per docs/design/ops-design.md § Step 3).
    _static_axes: frozenset = frozenset()

    def __init__(
        self,
        *,
        N: int,
        dtype: torch.dtype,
        dim: int = -1,
        eps: float = 1e-6,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.N = N
        self.dtype = dtype
        self.dim = dim
        self.eps = eps
        self.tune = tune
        self.N_padded = self._align_up(N, ALIGNMENT)
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[int, Kernel] = {}
        self._last_roofline_mn: Optional[Tuple[int, int]] = None

    @staticmethod
    def _align_up(n: int, alignment: int) -> int:
        """Round *n* up to the nearest multiple of *alignment*."""
        return ((n + alignment - 1) // alignment) * alignment

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._kernel_key: self._kernel_cls}

    def eval_roofline(self) -> Tuple[int, int]:
        if self._last_roofline_mn is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior "
                "forward() call to bind dynamic input shape (M)"
            )
        M, N = self._last_roofline_mn
        elem_bytes = self.dtype.itemsize
        return (4 * M * N, (2 * M * N + N) * elem_bytes)

    @property
    def _needs_pad(self) -> bool:
        return self.N_padded != self.N

    def _get_kernel(self, M: int) -> Kernel:
        """Return a kernel built for (M, self.N), caching by M."""
        if M not in self._kernel_cache:
            self._kernel_cache[M] = self.kernel_map[self._kernel_key](
                M, self.N, self.eps, self.dtype, tune=self.tune,
            )
        return self._kernel_cache[M]

    # -- Forward pipeline helpers --------------------------------------------

    def _validate_and_normalize_dim(
        self, x: torch.Tensor, weight: torch.Tensor
    ) -> int:
        """Validate input/weight, return the non-negative reduction axis."""
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if not weight.is_cuda or weight.dtype != self.dtype:
            raise ValueError(
                f"weight must be a CUDA tensor of dtype {self.dtype}"
            )
        if weight.ndim != 1 or weight.shape[0] != self.N:
            raise ValueError(
                f"weight must be 1-D with shape ({self.N},), "
                f"got {tuple(weight.shape)}"
            )
        ndim = x.ndim
        if not (-ndim <= self.dim < ndim):
            raise ValueError(
                f"dim={self.dim} out of range for {ndim}-D input"
            )
        dim_norm = self.dim % ndim
        if x.shape[dim_norm] != self.N:
            raise ValueError(
                f"Expected x.shape[{self.dim}]={self.N}, "
                f"got {x.shape[dim_norm]}"
            )
        # Bind the dynamic static-axis (param-dependent N axis) so
        # Op-layer cache-key/introspection consumers see the committed axis.
        self._static_axes = frozenset({(0, dim_norm)})
        return dim_norm

    def _flatten_to_2d(
        self, x: torch.Tensor, dim_norm: int
    ) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        """Move target axis to last, flatten leading dims to (M, N).

        Returns (2-D x, shape after movedim) so the caller can restore
        leading dims after the kernel.
        """
        if dim_norm != x.ndim - 1:
            x = x.movedim(dim_norm, -1)
        post_move_shape = tuple(x.shape)
        x = x.contiguous().reshape(-1, self.N)
        return x, post_move_shape

    def _pad_row(self, t: torch.Tensor) -> torch.Tensor:
        """Pad a 2-D row tensor along dim=-1 to N_padded."""
        return F.pad(t, (0, self.N_padded - self.N))

    def _pad_vec(self, t: torch.Tensor) -> torch.Tensor:
        """Pad a 1-D vector to N_padded."""
        return F.pad(t, (0, self.N_padded - self.N))

    def _trim_and_unflatten(
        self,
        y: torch.Tensor,
        post_move_shape: Tuple[int, ...],
        dim_norm: int,
        ndim: int,
    ) -> torch.Tensor:
        """Inverse of `_flatten_to_2d`: trim padding, restore shape, move axis back."""
        if self._needs_pad:
            y = y[:, : self.N]
        y = y.reshape(post_move_shape)
        if dim_norm != ndim - 1:
            y = y.movedim(-1, dim_norm)
        return y

    @abstractmethod
    def forward(
        self, *args: object, **kwargs: object
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Subclasses must implement their own forward with concrete args."""
        raise NotImplementedError("Subclasses must implement forward()")
