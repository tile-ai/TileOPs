"""Batched GEMM op (BmmFwdOp).

Strict 3D-3D batched matrix multiplication matching ``torch.bmm``: every
batch item is an independent GEMM, no broadcasting.
"""

from typing import Dict, Hashable, Optional, Tuple

import torch

from tileops.kernels.bmm import BmmKernel
from tileops.kernels.kernel_base import Kernel

from .op_base import Op

__all__ = ["BmmFwdOp"]


class BmmFwdOp(Op):
    """Batched dense GEMM: ``d[i] = a[i] @ b[i]``.

    Input shapes are strict 3D — ``a: [B, M, K]``, ``b: [B, K, N]``,
    ``d: [B, M, N]``. Batch and contraction dims are checked at
    ``forward()`` time;  The kernel is compiled on first use for each ``(batch, m, n, k, dtype)`` combo
    and cached.

    Args:
        kernel_map: Optional kernel override dict.
        tune: Whether to autotune (applied when a kernel is first built).

    Example:
        >>> op = BmmFwdOp()
        >>> d = op(a, b)                          # a=[B,M,K], b=[B,K,N] -> d=[B,M,N]
        >>> flops, nbytes = op.eval_roofline()    # valid after the forward
    """

    def __init__(
        self,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self._tune = tune
        self.dispatch_kernel(kernel_map)
        # (batch, m, n, k, dtype) -> Kernel instance; built lazily on first use.
        self._kernel_cache: Dict[Hashable, Kernel] = {}
        # Fast path: skip re-inference when the input signature is unchanged.
        self._active_sig: Optional[tuple] = None
        self._active_kernel: Optional[Kernel] = None
        # Roofline / dtype bindings, populated on the first forward().
        self.batch: Optional[int] = None
        self.m: Optional[int] = None
        self.n: Optional[int] = None
        self.k: Optional[int] = None
        self.dtype: Optional[torch.dtype] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"bmm_kernel": BmmKernel}

    def _infer_bmnk(
        self, a: torch.Tensor, b: torch.Tensor,
    ) -> Tuple[int, int, int, int]:
        """Derive logical ``(batch, m, n, k)`` from ``[B,M,K]`` and ``[B,K,N]``.

        Raises:
            ValueError: If ranks are wrong, batch dims mismatch, or the
                k dim disagrees between ``a`` and ``b``.
        """
        if a.dim() != 3 or b.dim() != 3:
            raise ValueError(
                f"BmmFwdOp expects strict 3D inputs a=[B,M,K] and b=[B,K,N] "
                f"(got a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}); "
            )
        batch_a, m, k_a = a.shape
        batch_b, k_b, n = b.shape
        if batch_a != batch_b:
            raise ValueError(
                f"BmmFwdOp batch dim mismatch: a.shape[0]={batch_a} vs "
                f"b.shape[0]={batch_b}"
            )
        if k_a != k_b:
            raise ValueError(
                f"BmmFwdOp contraction dim mismatch: a contributes K={k_a}, "
                f"b contributes K={k_b} (a.shape={tuple(a.shape)}, "
                f"b.shape={tuple(b.shape)})."
            )
        return batch_a, m, n, k_a

    def _cache_key(self, *input_shapes: Tuple[int, ...]) -> Hashable:
        """Project onto the dims the kernel actually specializes on."""
        return (self.batch, self.m, self.n, self.k,
                None if self.dtype is None else str(self.dtype))

    def _get_kernel(
        self, batch: int, m: int, n: int, k: int, dtype: torch.dtype,
    ) -> Kernel:
        """Return the cached BmmKernel for the given dims, building lazily."""
        key = (batch, m, n, k, dtype)
        kernel = self._kernel_cache.get(key)
        if kernel is None:
            kernel = self.kernel_map["bmm_kernel"](
                batch, m, n, k, dtype, tune=self._tune)
            self._kernel_cache[key] = kernel
        return kernel

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Fast path: same input signature as the last call → reuse the already
        # built/JIT'd kernel directly.
        sig = (a.shape, b.shape, a.dtype)
        if sig != self._active_sig:
            self._validate_dtypes(a, b)
            batch, m, n, k = self._infer_bmnk(a, b)
            # Bind dims/dtype for the manifest func-mode roofline.
            self.batch, self.m, self.n, self.k = batch, m, n, k
            self.dtype = a.dtype
            self.a_shape = tuple(a.shape)
            self.b_shape = tuple(b.shape)
            kernel = self._get_kernel(batch, m, n, k, a.dtype)
            # Expose the active kernel so autotune()/introspection can find it.
            self.kernel = kernel
            self._active_kernel = kernel
            self._active_sig = sig

        return self._active_kernel(a, b)

    def autotune(self) -> None:
        """Autotune every kernel built so far.

        ``BmmFwdOp`` caches kernels lazily in ``self._kernel_cache`` rather
        than as direct attributes, so the base ``Op.autotune`` (which scans
        ``dir(self)``) would miss them. Tune each cached kernel instead.
        """
        for kernel in self._kernel_cache.values():
            kernel.autotune()
