"""Batched GEMM op (BmmFwdOp).

Strict 3D-3D batched matrix multiplication matching ``torch.bmm``: every
batch item is an independent GEMM, no broadcasting.
"""

import warnings
from typing import Dict, Hashable, Optional, Set, Tuple

import torch

from tileops.kernels.bmm import BmmFp8Kernel, BmmKernel
from tileops.kernels.kernel_base import Kernel

from .op_base import Op

__all__ = ["BmmFp8Op", "BmmFwdOp"]


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
        if k_a % 16 != 0:
            raise ValueError(
                f"BmmFwdOp requires contraction dim K to be a multiple of 16 "
                f"(WGMMA alignment; see manifest shape_rules), got K={k_a}")
        return batch_a, m, n, k_a

    def _cache_key(self, *input_shapes: Tuple[int, ...]) -> Hashable:
        """Project onto the dims the kernel actually specializes on."""
        if len(input_shapes) == 2:
            a_shape, b_shape = input_shapes
            if len(a_shape) == 3 and len(b_shape) == 3:
                batch, m, k = a_shape
                _, _, n = b_shape
                return (batch, m, n, k, None if self.dtype is None else str(self.dtype))
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
        sig = (a.shape, b.shape, a.dtype, b.dtype)
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


class BmmFp8Op(Op):
    """Batched FP8 GEMM: ``d[i] = (a[i] @ b[i]) * scale_a * scale_b``.
    Args:
        out_dtype: Output tensor dtype (``torch.float16`` or ``torch.bfloat16``).
        kernel_map: Optional kernel override dict.
        tune: Whether to autotune (applied when a kernel is first built).

    Example:
        >>> op = BmmFp8Op(out_dtype=torch.bfloat16)
        >>> # Default path: b is [B,K,N] (torch.bmm layout).
        >>> d = op(a, b_kn, scale_a, scale_b)
        >>> # Fast path: b is [B,N,K] (K-innermost), zero-copy into kernel.
        >>> d = op(a, b_nk, scale_a, scale_b)
        >>> flops, nbytes = op.eval_roofline()    # valid after the forward
    """

    def __init__(
        self,
        out_dtype: torch.dtype | str = "bfloat16",
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
        b_layout: str = "kn",
    ) -> None:
        if isinstance(out_dtype, str):
            out_dtype = getattr(torch, out_dtype)
        if out_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"BmmFp8Op outputs torch.float16 or torch.bfloat16, "
                f"got {out_dtype}")
        if b_layout not in ("kn", "nk"):
            raise ValueError(
                f"BmmFp8Op b_layout must be 'kn' or 'nk', got {b_layout!r}")
        self.out_dtype = out_dtype
        self._tune = tune
        self.b_layout = b_layout
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[Hashable, Kernel] = {}
        self._active_sig: Optional[tuple] = None
        self._active: Optional[Kernel] = None
        # Shape-signatures for which we've already emitted the "slow path"
        # warning; keeps a single BmmFp8Op from spamming the log on every
        # forward when a caller consistently passes b in [B,K,N] layout.
        self._kn_warned: Set[Tuple[int, int, int, int]] = set()
        self.batch: Optional[int] = None
        self.m: Optional[int] = None
        self.n: Optional[int] = None
        self.k: Optional[int] = None
        self.dtype: Optional[torch.dtype] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "bmm_fp8_kernel": BmmFp8Kernel,
        }

    def _validate_dtypes(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
    ) -> None:
        if a.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"BmmFp8Op only supports torch.float8_e4m3fn, got {a.dtype}")
        if b.dtype != a.dtype:
            raise ValueError(f"BmmFp8Op expects b dtype {a.dtype}, got {b.dtype}")
        if scale_a.dtype != torch.float32 or scale_b.dtype != torch.float32:
            raise ValueError("BmmFp8Op expects scale_a and scale_b to be torch.float32")

    def _infer_bmnk(
        self, a: torch.Tensor, b: torch.Tensor,
    ) -> Tuple[int, int, int, int, bool]:
        """Derive logical ``(batch, m, n, k, b_is_nk)`` from ``a`` and ``b``.
        """
        if a.dim() != 3 or b.dim() != 3:
            raise ValueError(
                f"BmmFp8Op expects strict 3D inputs a=[B,M,K] and "
                f"b=[B,K,N] or [B,N,K] (got a.shape={tuple(a.shape)}, "
                f"b.shape={tuple(b.shape)})"
            )
        batch_a, m, k = a.shape
        batch_b, b1, b2 = b.shape
        if batch_a != batch_b:
            raise ValueError(
                f"BmmFp8Op batch dim mismatch: a.shape[0]={batch_a} vs "
                f"b.shape[0]={batch_b}"
            )
        if self.b_layout == "nk":
            if b2 != k:
                raise ValueError(
                    f"BmmFp8Op b_layout='nk' but b={tuple(b.shape)} is not a "
                    f"valid [B,N,K] (needs b.shape[2]==K={k})"
                )
            n, b_is_nk = b1, True
        else:  # 'kn'
            if b1 != k:
                raise ValueError(
                    f"BmmFp8Op contraction dim mismatch: a contributes K={k}, "
                    f"but b.shape={tuple(b.shape)} is not a valid [B,K,N] "
                    f"(needs b.shape[1]==K={k})."
                )
            n, b_is_nk = b2, False
        if k % 32 != 0:
            raise ValueError(
                f"BmmFp8Op requires contraction dim K to be a multiple of "
                f"32 (FP8 WGMMA K-step), got K={k}")
        return batch_a, m, n, k, b_is_nk

    def _validate_shapes(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
    ) -> Tuple[int, int, int, int, bool]:
        if not a.is_cuda:
            raise ValueError(
                f"BmmFp8Op expects all inputs to be on CUDA, got device {a.device}"
            )
        if (b.device != a.device or scale_a.device != a.device
                or scale_b.device != a.device):
            raise ValueError(
                f"BmmFp8Op expects all inputs to be on the same CUDA device, got "
                f"a: {a.device}, b: {b.device}, scale_a: {scale_a.device}, "
                f"scale_b: {scale_b.device}"
            )
        batch, m, n, k, b_is_nk = self._infer_bmnk(a, b)
        if scale_a.dim() != 0 or scale_b.dim() != 0:
            raise ValueError(
                "BmmFp8Op supports scale shapes ()/() only (per-tensor, "
                "global fp32 scalar shared across the batch, matching "
                "flashinfer.bmm_fp8's A_scale/B_scale), got "
                f"{tuple(scale_a.shape)}/{tuple(scale_b.shape)}"
            )
        return batch, m, n, k, b_is_nk

    def _get_kernel(
        self,
        batch: int,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Kernel:
        key = (batch, m, n, k, dtype, self.out_dtype, device)
        kernel = self._kernel_cache.get(key)
        if kernel is None:
            kernel = self.kernel_map["bmm_fp8_kernel"](
                batch, m, n, k, dtype, self.out_dtype, device=device,
                tune=self._tune)
            self._kernel_cache[key] = kernel
        return kernel

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
    ) -> torch.Tensor:
        sig = (
            a.shape,
            b.shape,
            b.stride(),
            scale_a.shape,
            scale_b.shape,
            a.dtype,
            b.dtype,
            scale_a.dtype,
            scale_b.dtype,
            self.out_dtype,
        )
        if sig != self._active_sig:
            self._validate_dtypes(a, b, scale_a, scale_b)
            batch, m, n, k, b_is_nk = self._validate_shapes(
                a, b, scale_a, scale_b)
            self.batch, self.m, self.n, self.k = batch, m, n, k
            self.dtype = a.dtype
            self.a_shape = tuple(a.shape)
            self.b_shape = tuple(b.shape)
            self.scale_a_shape = tuple(scale_a.shape)
            self.scale_b_shape = tuple(scale_b.shape)
            self._b_is_nk = b_is_nk
            kernel = self._get_kernel(batch, m, n, k, a.dtype, device=a.device)
            self._active = kernel
            self._active_sig = sig
        if self._b_is_nk:
            b = b.contiguous()
        else:
            # Slow path: [B,K,N] layout requires an extra DtoD transpose
            # before the fp8-TN WGMMA kernel can consume it. Emit a one-shot
            # warning per (B,M,N,K) shape so users know how to opt into the
            # zero-copy fast path (pass b as [B,N,K]).
            shape_key = (self.batch, self.m, self.n, self.k)
            if shape_key not in self._kn_warned:
                self._kn_warned.add(shape_key)
                warnings.warn(
                    f"BmmFp8Op: b has layout [B,K,N] (shape={self.b_shape}); "
                    f"triggering an extra transpose(-2,-1).contiguous() DtoD "
                    f"copy before the fp8-TN WGMMA kernel. For best "
                    f"performance pass b as [B,N,K] (K-innermost) for the "
                    f"zero-copy fast path.",
                    stacklevel=2,
                )
            b = b.transpose(-2, -1).contiguous()
        scale_a = scale_a.reshape(1)
        scale_b = scale_b.reshape(1)
        return self._active(a, b, scale_a, scale_b)

    def autotune(self) -> None:
        for kernel in self._kernel_cache.values():
            kernel.autotune()
