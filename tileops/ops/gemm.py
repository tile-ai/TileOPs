from typing import Dict, Hashable, Optional, Tuple

import torch

from tileops.kernels.gemm import GemmKernel, GemvKernel
from tileops.kernels.kernel_base import Kernel
from tileops.utils import get_sm_version

from .op_base import Op

__all__ = ["GemmOp"]


class GemmOp(Op):
    """Dense GEMM, input-inferred and aligned to DeepGEMM's call-time JIT.

    The logical dims ``m, n, k`` and the dtype are derived from the ``forward``
    inputs; nothing is committed at construction. The dtype-specialized kernel
    is built (and cached) on first use for each ``(m, n, k, dtype)`` — mirroring
    DeepGEMM's compile-on-first-call + per-config cache.

    Layouts via ``(trans_a, trans_b)`` (== DeepGEMM ``nt``/``nn``/``tn``/``tt``):
      - ``(False, True)``  NT (default): ``A @ Bᵀ``
      - ``(False, False)`` NN:           ``A @ B``
      - ``(True,  False)`` TN:           ``Aᵀ @ B``
      - ``(True,  True)``  TT:           ``Aᵀ @ Bᵀ``

    Args:
        trans_a: Whether ``a`` is stored transposed (``[K, M]``).
        trans_b: Whether ``b`` is stored transposed (``[N, K]``). Default ``True`` (NT).
        kernel_map: Optional kernel override dict.
        tune: Whether to autotune (applied when a kernel is first built).

    Example:
        >>> op = GemmOp()                       # NT by default
        >>> d = op(a, b)                         # a=[M,K], b=[N,K] -> d=[M,N]
        >>> flops, nbytes = op.eval_roofline()   # valid after the forward
    """

    def __init__(
        self,
        trans_a: bool = False,
        trans_b: bool = True,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.trans_a = trans_a
        self.trans_b = trans_b
        self._tune = tune
        self.dispatch_kernel(kernel_map)
        # (m, n, k, dtype) -> Kernel instance; built lazily on first use.
        self._kernel_cache: Dict[Hashable, Kernel] = {}
        # Fast path: skip re-inference when the input signature is unchanged.
        # _active_sig = (a.shape, b.shape, dtype); _active = (mode, kernel, n, m).
        self._active_sig: Optional[tuple] = None
        self._active: Optional[tuple] = None
        # Roofline / dtype bindings, populated on the first forward().
        self.m: Optional[int] = None
        self.n: Optional[int] = None
        self.k: Optional[int] = None
        self.dtype: Optional[torch.dtype] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        kernels: Dict[str, Kernel] = {"gemm_kernel": GemmKernel}
        # GemvKernel is SM90-only; only advertise it where it can install.
        if get_sm_version() in (GemvKernel.supported_archs or []):
            kernels["gemv_kernel"] = GemvKernel
        return kernels

    def _infer_mnk(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[int, int, int]:
        """Derive logical ``(m, n, k)`` from input shapes per the trans flags."""
        k_a, m = (a.shape[0], a.shape[1]) if self.trans_a else (a.shape[1], a.shape[0])
        n, k_b = (b.shape[0], b.shape[1]) if self.trans_b else (b.shape[1], b.shape[0])
        if k_a != k_b:
            raise ValueError(
                f"GEMM contraction dim mismatch: a contributes K={k_a}, b contributes K={k_b} "
                f"(a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}, "
                f"trans_a={self.trans_a}, trans_b={self.trans_b})"
            )
        return m, n, k_a

    def _cache_key(self, *input_shapes: Tuple[int, ...]) -> Hashable:
        """Project onto the dims the kernel actually specializes on."""
        return (self.m, self.n, self.k, self.trans_a, self.trans_b,
                None if self.dtype is None else str(self.dtype))

    def _get_kernel(self, m: int, n: int, k: int, dtype: torch.dtype) -> Tuple[str, Kernel]:
        """Return ``(mode, kernel)`` for the given dims, building/caching lazily.

        ``mode`` is ``"lhs_row"``/``"rhs_col"`` for the GEMV fast path, else
        ``"gemm"`` — the hand-written warp-specialized ``GemmKernel`` (SM90),
        covering all four ``(trans_a, trans_b)`` layouts.
        """
        gemv_lhs_row = (m == 1 and not self.trans_a and self.trans_b)
        gemv_rhs_col = (n == 1 and not self.trans_a and not self.trans_b)
        gemv_cls = self.kernel_map.get("gemv_kernel")
        if (gemv_lhs_row or gemv_rhs_col) and gemv_cls is not None:
            mode = "lhs_row" if gemv_lhs_row else "rhs_col"
            key = (mode, m, n, k, dtype)
            kernel = self._kernel_cache.get(key)
            if kernel is None:
                # lhs_row: a is [1, K], reduce over K -> use (n, k); rhs_col uses (m, k).
                kernel = gemv_cls(n if mode == "lhs_row" else m, k, dtype, tune=self._tune)
                self._kernel_cache[key] = kernel
            return mode, kernel

        key = ("gemm", m, n, k, dtype)
        kernel = self._kernel_cache.get(key)
        if kernel is None:
            kernel = self.kernel_map["gemm_kernel"](
                m, n, k, dtype, tune=self._tune, trans_a=self.trans_a, trans_b=self.trans_b)
            self._kernel_cache[key] = kernel
        return "gemm", kernel

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Fast path: same input signature as the last call → reuse the already
        # built/JIT'd kernel directly, skipping dtype validation, shape
        # inference, and the cache lookup (this is the steady state in
        # benchmarking / serving, where per-call Python overhead matters).
        sig = (a.shape, b.shape, a.dtype)
        if sig != self._active_sig:
            self._validate_dtypes(a, b)
            m, n, k = self._infer_mnk(a, b)
            # Bind dims/dtype for the manifest func-mode roofline (read post-forward).
            self.m, self.n, self.k = m, n, k
            self.dtype = a.dtype
            self.a_shape = tuple(a.shape)
            self.b_shape = tuple(b.shape)
            mode, kernel = self._get_kernel(m, n, k, a.dtype)
            # Expose the active kernel so autotune()/introspection can find it.
            self.kernel = kernel
            self._active = (mode, kernel, n, m)
            self._active_sig = sig

        mode, kernel, n, m = self._active
        if mode == "lhs_row":
            return kernel(a.reshape(-1), b).reshape(1, n)
        if mode == "rhs_col":
            return kernel(b.reshape(-1), a).reshape(m, 1)
        return kernel(a, b)

    def autotune(self) -> None:
        """Autotune every kernel built so far.

        ``GemmOp`` caches kernels lazily in ``self._kernel_cache`` rather than as
        direct attributes, so the base ``Op.autotune`` (which scans ``dir(self)``)
        would miss them. Tune each cached kernel instead.
        """
        for kernel in self._kernel_cache.values():
            kernel.autotune()
