from typing import Dict, Hashable, Optional, Tuple

import torch

from tileops.kernels.gemm import (
    GemmFp8BlockScaledKernel,
    GemmFp8EpilogueKernel,
    GemmKernel,
    GemvKernel,
    SmallBatchGemmKernel,
)
from tileops.kernels.gemm_call import GemmCall
from tileops.kernels.gemm_w4a16 import GROUP_SIZE, GemmW4A16Kernel
from tileops.kernels.kernel_base import Kernel

from .op_base import Op

__all__ = ["GemmFp8FwdOp", "GemmFwdOp", "GemmW4A16FwdOp"]


class GemmFwdOp(Op):
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
        >>> op = GemmFwdOp()                       # NT by default
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
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        # (m, n, k, dtype) -> Kernel instance; built lazily on first use.
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
        return {
            "gemm_kernel": GemmKernel,
            "gemv_kernel": GemvKernel,
            "small_batch_kernel": SmallBatchGemmKernel,
        }

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
        return (
            self.m,
            self.n,
            self.k,
            self.trans_a,
            self.trans_b,
            None if self.dtype is None else str(self.dtype),
        )

    def _get_kernel(self, m: int, n: int, k: int, dtype: torch.dtype) -> Tuple[str, Kernel]:
        """Return ``(mode, kernel)`` for the given dims, building/caching lazily.

        Each candidate states the region it serves (``gemm_call.gemv_region``
        / ``small_batch_region``, read through ``Kernel.applies``); this
        method owns only mechanism: mapping the selected key to a kernel
        instance and caching it.
        """
        call = GemmCall(m=m, n=n, k=k, dtype=dtype, trans_a=self.trans_a, trans_b=self.trans_b)
        key = self.select_kernel_key(("gemv_kernel", "small_batch_kernel", "gemm_kernel"), call)
        if key == "gemv_kernel":
            # lhs_row: a is [1, K], reduce over K -> use (n, k); rhs_col uses (m, k).
            mode = "lhs_row" if m == 1 and self.trans_b else "rhs_col"
            gemv_cls = self.kernel_map["gemv_kernel"]
            kernel = self.get_or_build_kernel(
                "gemv_kernel",
                key=(mode, m, n, k, dtype),
                build=lambda: gemv_cls(n if mode == "lhs_row" else m, k, dtype, tune=self.tune),
            )
            return mode, kernel

        if key == "small_batch_kernel":
            sb_cls = self.kernel_map["small_batch_kernel"]
            kernel = self.get_or_build_kernel(
                "small_batch_kernel",
                (m, n, k, dtype),
                lambda: sb_cls(m, n, k, dtype, tune=self.tune),
            )
            return "small_batch", kernel

        kernel = self.get_or_build_kernel(
            "gemm_kernel",
            key=(m, n, k, dtype),
            build=lambda: self.kernel_map["gemm_kernel"](
                m, n, k, dtype, tune=self.tune, trans_a=self.trans_a, trans_b=self.trans_b
            ),
        )
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


class GemmFp8FwdOp(Op):
    """Dense FP8 NT GEMM, input-inferred.

    Public layout is ``a=[M, K]`` and ``b=[N, K]``. ``scale_a`` and
    ``scale_b`` must be either per-tensor ``(1, 1)`` scales or block128
    scales with shapes ``(M, ceil(K / 128))`` and ``(N, ceil(K / 128))``.
    """

    def __init__(
        self,
        out_dtype: torch.dtype | str = "bfloat16",
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        if isinstance(out_dtype, str):
            out_dtype = getattr(torch, out_dtype)
        if out_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"GemmFp8FwdOp outputs torch.float16 or torch.bfloat16, got {out_dtype}"
            )
        self.out_dtype = out_dtype
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self._active_sig: Optional[tuple] = None
        self._active: Optional[Kernel] = None
        self.m: Optional[int] = None
        self.n: Optional[int] = None
        self.k: Optional[int] = None
        self.dtype: Optional[torch.dtype] = None
        self.has_bias = False

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "gemm_fp8_epilogue_kernel": GemmFp8EpilogueKernel,
            "gemm_fp8_block_scaled_kernel": GemmFp8BlockScaledKernel,
        }

    def _validate_dtypes(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        if a.dtype != torch.float8_e4m3fn:
            raise ValueError(f"GemmFp8FwdOp only supports torch.float8_e4m3fn, got {a.dtype}")
        if b.dtype != a.dtype:
            raise ValueError(f"GemmFp8FwdOp expects b dtype {a.dtype}, got {b.dtype}")
        if scale_a.dtype != torch.float32 or scale_b.dtype != torch.float32:
            raise ValueError("GemmFp8FwdOp expects scale_a and scale_b to be torch.float32")
        out_dtype = (
            getattr(torch, self.out_dtype) if isinstance(self.out_dtype, str) else self.out_dtype
        )
        if bias is not None and bias.dtype != out_dtype:
            raise ValueError(f"GemmFp8FwdOp expects bias dtype {out_dtype}, got {bias.dtype}")

    def _infer_mnk(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[int, int, int]:
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"GemmFp8FwdOp expects 2D a/b, got a.ndim={a.ndim}, b.ndim={b.ndim}")
        m, k = a.shape
        n, k_b = b.shape
        if k != k_b:
            raise ValueError(
                f"FP8 GEMM contraction dim mismatch: a.shape={tuple(a.shape)}, "
                f"b.shape={tuple(b.shape)}"
            )
        return m, n, k

    def _infer_output_shapes(
        self,
        a_shape: Tuple[int, ...],
        b_shape: Tuple[int, ...],
        scale_a_shape: Tuple[int, ...],
        scale_b_shape: Tuple[int, ...],
        bias_shape: Optional[Tuple[int, ...]] = None,
    ) -> dict[str, Tuple[int, int]]:
        return {"d": (a_shape[0], b_shape[0])}

    def _validate_shapes(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> tuple[int, int, int]:
        m, n, k = self._infer_mnk(a, b)
        if scale_a.ndim != 2 or scale_b.ndim != 2:
            raise ValueError(
                f"GemmFp8FwdOp expects 2D scales, got {tuple(scale_a.shape)} and "
                f"{tuple(scale_b.shape)}"
            )
        per_tensor = (tuple(scale_a.shape), tuple(scale_b.shape)) == ((1, 1), (1, 1))
        scale_k = (k + 127) // 128
        block128 = tuple(scale_a.shape) == (m, scale_k) and tuple(scale_b.shape) == (n, scale_k)
        if not per_tensor and not block128:
            raise ValueError(
                "GemmFp8FwdOp supports scale shapes (1, 1)/(1, 1) or "
                f"{(m, scale_k)}/{(n, scale_k)}, got "
                f"{tuple(scale_a.shape)}/{tuple(scale_b.shape)}"
            )
        if bias is not None and tuple(bias.shape) != (n,):
            raise ValueError(f"GemmFp8FwdOp bias must have shape {(n,)}, got {tuple(bias.shape)}")
        return m, n, k

    def _select_kernel_name(
        self,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        m: int,
        n: int,
        k: int,
    ) -> str:
        if (tuple(scale_a.shape), tuple(scale_b.shape)) == ((1, 1), (1, 1)):
            return "gemm_fp8_epilogue_kernel"
        scale_k = (k + 127) // 128
        if tuple(scale_a.shape) == (m, scale_k) and tuple(scale_b.shape) == (n, scale_k):
            return "gemm_fp8_block_scaled_kernel"
        raise ValueError(
            "GemmFp8FwdOp supports scale shapes (1, 1)/(1, 1) or "
            f"{(m, scale_k)}/{(n, scale_k)}, got "
            f"{tuple(scale_a.shape)}/{tuple(scale_b.shape)}"
        )

    def _get_kernel(
        self,
        kernel_name: str,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        scale_a_shape: Tuple[int, ...],
        scale_b_shape: Tuple[int, ...],
    ) -> Kernel:
        return self.get_or_build_kernel(
            kernel_name,
            key=(m, n, k, dtype, scale_a_shape, scale_b_shape, self.out_dtype),
            build=lambda: self.kernel_map[kernel_name](
                m, n, k, dtype, self.out_dtype, tune=self.tune
            ),
        )

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sig = (
            a.shape,
            b.shape,
            scale_a.shape,
            scale_b.shape,
            a.dtype,
            b.dtype,
            scale_a.dtype,
            scale_b.dtype,
            self.out_dtype,
            (bias.shape, bias.dtype) if bias is not None else None,
        )
        if sig != self._active_sig:
            self._validate_dtypes(a, b, scale_a, scale_b, bias)
            m, n, k = self._validate_shapes(a, b, scale_a, scale_b, bias)
            self.m, self.n, self.k = m, n, k
            self.dtype = a.dtype
            self.scale_a_shape = tuple(scale_a.shape)
            self.scale_b_shape = tuple(scale_b.shape)
            self.has_bias = bias is not None
            kernel_name = self._select_kernel_name(scale_a, scale_b, m, n, k)
            kernel = self._get_kernel(
                kernel_name, m, n, k, a.dtype, tuple(scale_a.shape), tuple(scale_b.shape)
            )
            self.kernel = kernel
            self._active = kernel
            self._active_sig = sig

        return self._active(a, b, scale_a, scale_b, bias)


class GemmW4A16FwdOp(Op):
    """Dense W4A16 NT GEMM with group-wise affine weight dequantization.

    Public layout is ``activation=[M, K]`` and ``packed_weight=[N, K / 2]``.
    Two unsigned INT4 values are packed per byte: the low nibble stores even K
    and the high nibble stores odd K. ``weight_scale`` and ``weight_zero`` are
    group128 metadata with shape ``[N, K / 128]``. The kernel dequantizes the
    current W4 tile into A16 shared memory and computes ``activation @ W.T``.
    """

    def __init__(
        self,
        group_size: int = GROUP_SIZE,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        if group_size != GROUP_SIZE:
            raise ValueError(
                f"GemmW4A16FwdOp currently supports group_size={GROUP_SIZE}, got {group_size}"
            )
        self.group_size = group_size
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self._active_sig: Optional[tuple] = None
        self._active: Optional[Kernel] = None
        self.m: Optional[int] = None
        self.n: Optional[int] = None
        self.k: Optional[int] = None
        self.dtype: Optional[torch.dtype] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"gemm_w4a16_kernel": GemmW4A16Kernel}

    def _validate_dtypes(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zero: torch.Tensor,
    ) -> None:
        if activation.dtype != torch.float16:
            raise ValueError(
                f"GemmW4A16FwdOp currently supports float16 activation, got {activation.dtype}"
            )
        if packed_weight.dtype != torch.uint8:
            raise ValueError(
                f"GemmW4A16FwdOp expects uint8 packed_weight, got {packed_weight.dtype}"
            )
        if weight_scale.dtype != torch.float32:
            raise ValueError(
                f"GemmW4A16FwdOp expects float32 weight_scale, got {weight_scale.dtype}"
            )
        if weight_zero.dtype != torch.uint8:
            raise ValueError(f"GemmW4A16FwdOp expects uint8 weight_zero, got {weight_zero.dtype}")

    def _infer_mnk(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
    ) -> Tuple[int, int, int]:
        if activation.ndim != 2 or packed_weight.ndim != 2:
            raise ValueError(
                "GemmW4A16FwdOp expects rank-2 activation and packed_weight, got "
                f"{activation.ndim} and {packed_weight.ndim}"
            )
        m, k = activation.shape
        n, packed_k = packed_weight.shape
        if k % 2 != 0:
            raise ValueError(f"GemmW4A16FwdOp expects even K for W4 packing, got {k}")
        if packed_k != k // 2:
            raise ValueError(
                "GemmW4A16FwdOp packed_weight shape mismatch: expected second dim "
                f"{k // 2}, got {packed_k}"
            )
        if k % self.group_size != 0:
            raise ValueError(
                f"GemmW4A16FwdOp expects K divisible by group_size={self.group_size}, got {k}"
            )
        return m, n, k

    def _infer_output_shapes(
        self,
        activation_shape: Tuple[int, ...],
        packed_weight_shape: Tuple[int, ...],
        weight_scale_shape: Tuple[int, ...],
        weight_zero_shape: Tuple[int, ...],
    ) -> dict[str, Tuple[int, int]]:
        return {"output": (activation_shape[0], packed_weight_shape[0])}

    def _validate_shapes(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zero: torch.Tensor,
    ) -> tuple[int, int, int]:
        m, n, k = self._infer_mnk(activation, packed_weight)
        groups = k // self.group_size
        metadata_shape = (n, groups)
        if tuple(weight_scale.shape) != metadata_shape:
            raise ValueError(
                f"GemmW4A16FwdOp weight_scale must have shape {metadata_shape}, "
                f"got {tuple(weight_scale.shape)}"
            )
        if tuple(weight_zero.shape) != metadata_shape:
            raise ValueError(
                f"GemmW4A16FwdOp weight_zero must have shape {metadata_shape}, "
                f"got {tuple(weight_zero.shape)}"
            )
        return m, n, k

    def _get_kernel(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
    ) -> Kernel:
        return self.get_or_build_kernel(
            "gemm_w4a16_kernel",
            key=(m, n, k, dtype, self.group_size),
            build=lambda: self.kernel_map["gemm_w4a16_kernel"](
                m, n, k, dtype, tune=self.tune, group_size=self.group_size
            ),
        )

    def forward(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zero: torch.Tensor,
    ) -> torch.Tensor:
        sig = (
            activation.shape,
            packed_weight.shape,
            weight_scale.shape,
            weight_zero.shape,
            activation.dtype,
            packed_weight.dtype,
            weight_scale.dtype,
            weight_zero.dtype,
            self.group_size,
        )
        if sig != self._active_sig:
            self._validate_dtypes(activation, packed_weight, weight_scale, weight_zero)
            m, n, k = self._validate_shapes(activation, packed_weight, weight_scale, weight_zero)
            self.m, self.n, self.k = m, n, k
            self.dtype = activation.dtype
            self.packed_weight_shape = tuple(packed_weight.shape)
            self.weight_scale_shape = tuple(weight_scale.shape)
            self.weight_zero_shape = tuple(weight_zero.shape)
            kernel = self._get_kernel(m, n, k, activation.dtype)
            self.kernel = kernel
            self._active = kernel
            self._active_sig = sig

        return self._active(activation, packed_weight, weight_scale, weight_zero)
