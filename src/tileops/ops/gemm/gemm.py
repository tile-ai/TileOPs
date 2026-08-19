from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.gemm.call_spec import GemmCall
from tileops.kernels.gemm.dense import (
    GemmFp8BlockScaledKernel,
    GemmFp8EpilogueKernel,
    GemmKernel,
    GemvKernel,
    SmallBatchGemmKernel,
)
from tileops.kernels.gemm.w4a16 import GROUP_SIZE, GemmW4A16Kernel
from tileops.kernels.gemm.w4a16_decode import GemmW4A16DecodeKernel
from tileops.kernels.kernel_base import Kernel
from tileops.perf.profile import tensor_core_roof

from ..op_base import Op

__all__ = ["GemmFp8FwdOp", "GemmFwdOp", "GemmW4A16FwdOp"]


class GemmFwdOp(Op):
    """Dense GEMM, input-inferred and aligned to DeepGEMM's call-time JIT.

    The logical dims ``m, n, k`` and the dtype are derived from the ``forward``
    inputs; nothing is committed at construction. The dtype-specialized kernel
    is built (and cached) on first use for each ``(m, n, k, dtype)`` — mirroring
    DeepGEMM's compile-on-first-call + per-config cache.

    The ``(trans_a, trans_b)`` pair selects one of four layouts, matching DeepGEMM's
    ``nt`` / ``nn`` / ``tn`` / ``tt``:

    | Flags | Layout | Product |
    | --- | --- | --- |
    | ``(False, True)`` | NT, the default | $d = a \\mathbin{@} b^{\\top}$ |
    | ``(False, False)`` | NN | $d = a \\mathbin{@} b$ |
    | ``(True, False)`` | TN | $d = a^{\\top} \\mathbin{@} b$ |
    | ``(True, True)`` | TT | $d = a^{\\top} \\mathbin{@} b^{\\top}$ |

    """

    def __init__(
        self,
        trans_a: bool = False,
        trans_b: bool = True,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            trans_a: Whether ``a`` is stored transposed ($[K \\times M]$).
            trans_b: Whether ``b`` is stored transposed ($[N \\times K]$). Default ``True`` (NT).
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune (applied when a kernel is first built).
        """
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
        """Derive logical ``(m, n, k)`` from input shapes per the trans flags.

        Rank is checked first: an extra axis is otherwise dropped silently, and
        the dims read out of the remaining axes reach the kernel builder, which
        compiles for a shape the call does not have before TileLang rejects the
        arguments.
        """
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(
                f"GemmFwdOp contracts two matrices, got a.ndim={a.ndim}, b.ndim={b.ndim}"
            )
        m, n = self._infer_output_shapes(a.shape, b.shape)["d"]
        k_a = a.shape[0] if self.trans_a else a.shape[1]
        k_b = b.shape[1] if self.trans_b else b.shape[0]
        if k_a != k_b:
            raise ValueError(
                f"GEMM contraction dim mismatch: a contributes K={k_a}, b contributes K={k_b} "
                f"(a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}, "
                f"trans_a={self.trans_a}, trans_b={self.trans_b})"
            )
        return m, n, k_a

    def _get_kernel(
        self, inputs: "tuple[torch.Tensor | None, ...]", m: int, n: int, k: int, dtype: torch.dtype
    ) -> Tuple[str, Kernel]:
        """Return ``(mode, kernel)`` for the given dims, building/caching lazily.

        ``mode`` is ``"lhs_row"``/``"rhs_col"`` for the GEMV fast path (the two
        differ in which operand is the vector, so ``forward`` reshapes
        accordingly), ``"small_batch"`` for the low-``m`` NT bandwidth kernel,
        else ``"gemm"`` — ``GemmKernel`` (SM90), covering all four
        ``(trans_a, trans_b)`` layouts.

        Which one serves the call is stated by the candidates themselves
        (``call_spec.gemv_region`` / ``small_batch_region``, read through
        ``Kernel.applies``); this method owns only mechanism: mapping the
        selected key to a kernel instance and caching it.
        """
        call = GemmCall(m=m, n=n, k=k, dtype=dtype, trans_a=self.trans_a, trans_b=self.trans_b)
        key = self.select_kernel_key(("gemv_kernel", "small_batch_kernel", "gemm_kernel"), call)
        if key == "gemv_kernel":
            # lhs_row: a is [1, K], reduce over K -> use (n, k); rhs_col uses (m, k).
            mode = "lhs_row" if m == 1 and self.trans_b else "rhs_col"
            gemv_cls = self.kernel_map["gemv_kernel"]
            kernel = self.get_or_build_kernel(
                "gemv_kernel",
                inputs,
                key=(mode, m, n, k, dtype),
                build=lambda: gemv_cls(n if mode == "lhs_row" else m, k, dtype, tune=self.tune),
            )
            return mode, kernel

        if key == "small_batch_kernel":
            sb_cls = self.kernel_map["small_batch_kernel"]
            kernel = self.get_or_build_kernel(
                "small_batch_kernel",
                key=(m, n, k, dtype),
                build=lambda: sb_cls(m, n, k, dtype, tune=self.tune),
            )
            return "small_batch", kernel

        kernel = self.get_or_build_kernel(
            "gemm_kernel",
            inputs,
            key=(m, n, k, dtype),
            build=lambda: self.kernel_map["gemm_kernel"](
                m, n, k, dtype, tune=self.tune, trans_a=self.trans_a, trans_b=self.trans_b
            ),
        )
        return "gemm", kernel

    def _infer_output_shapes(
        self,
        a_shape: tuple[int, ...],
        b_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        """Manifest ``shape_rules``: which axis carries ``M`` and ``N`` follows the layout flags."""
        m = a_shape[1] if self.trans_a else a_shape[0]
        n = b_shape[0] if self.trans_b else b_shape[1]
        return {"d": (m, n)}

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multiply the two matrices under the layout the constructor selected.

        Args:
            a: Left operand, $[M \\times K]$, or $[K \\times M]$ when ``trans_a``.
            b: Right operand, $[N \\times K]$ under the default NT layout, or
                $[K \\times N]$ when ``trans_b`` is false.

        Returns:
            The product, $[M \\times N]$, in the dtype of the inputs.

        Raises:
            ValueError: The contraction dims the two operands contribute do not match.

        Example:
            ```python linenums="1"
            op = GemmFwdOp()                      # NT by default
            d = op(a, b)                          # a=[M,K], b=[N,K] -> d=[M,N]
            flops, nbytes = op.eval_roofline()    # valid after the forward
            ```
        """
        # Fast path: same input signature as the last call → reuse the already
        # built/JIT'd kernel directly, skipping dtype validation, shape
        # inference, and the cache lookup (this is the steady state in
        # benchmarking / serving, where per-call Python overhead matters).
        # Because it skips the gate, the signature carries every dtype that gate
        # reads — a further input dtype has to be added here as well.
        sig = (a.shape, b.shape, a.dtype, b.dtype)
        if sig != self._active_sig:
            self._validate_dtypes(a, b)
            m, n, k = self._infer_mnk(a, b)
            # Bind dims/dtype for the manifest func-mode roofline (read post-forward).
            self.m, self.n, self.k = m, n, k
            self.dtype = a.dtype
            self.a_shape = tuple(a.shape)
            self.b_shape = tuple(b.shape)
            mode, kernel = self._get_kernel((a, b), m, n, k, a.dtype)
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

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.dtype)


class GemmFp8FwdOp(Op):
    """Dense FP8 NT GEMM, input-inferred.

    Public layout is ``a``: $[M \\times K]$ and ``b``: $[N \\times K]$. ``scale_a`` and
    ``scale_b`` must be either per-tensor $[1 \\times 1]$ scales or block128
    scales with shapes $[M \\times \\lceil K/128 \\rceil]$ and $[N \\times \\lceil K/128 \\rceil]$.
    """

    def __init__(
        self,
        out_dtype: torch.dtype | str = "bfloat16",
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            out_dtype: Output dtype.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
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
        inputs: "tuple[torch.Tensor | None, ...]",
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
            inputs,
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
        """Multiply the two FP8 matrices, apply the scales, and add the bias.

        Args:
            a: Left operand, $[M \\times K]$, ``torch.float8_e4m3fn``.
            b: Right operand, $[N \\times K]$, same dtype as ``a``.
            scale_a: ``torch.float32`` scales for ``a``: per-tensor $[1 \\times 1]$, or
                block128 $[M \\times \\lceil K/128 \\rceil]$.
            scale_b: The same for ``b``: $[1 \\times 1]$ or
                $[N \\times \\lceil K/128 \\rceil]$. Both scales take the same form.
            bias: Optional bias, $[N]$, in ``out_dtype``.

        Returns:
            The scaled product plus bias, $[M \\times N]$, in ``out_dtype``.

        Raises:
            ValueError: A dtype is not one of those listed above, ``a`` or ``b`` is not
                2D, the contraction dims do not match, the two scales are not both
                per-tensor or both block128, or the bias is not $[N]$.

        Example:
            ```python linenums="1"
            op = GemmFp8FwdOp(out_dtype=torch.bfloat16)
            d = op(a, b, scale_a, scale_b)        # per-tensor scales
            flops, nbytes = op.eval_roofline()    # valid after the forward
            ```
        """
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
                (a, b, scale_a, scale_b, bias),
                kernel_name,
                m,
                n,
                k,
                a.dtype,
                tuple(scale_a.shape),
                tuple(scale_b.shape),
            )
            self.kernel = kernel
            self._active = kernel
            self._active_sig = sig

        return self._active(a, b, scale_a, scale_b, bias)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.dtype)


class GemmW4A16FwdOp(Op):
    """Dense W4A16 NT GEMM with group-wise affine weight dequantization.

    Public layout is ``activation``: $[M \\times K]$ and ``packed_weight``: $[N \\times K/2]$.
    Two unsigned INT4 values are packed per byte: the low nibble stores even K
    and the high nibble stores odd K. ``weight_scale`` and ``weight_zero`` are
    group128 metadata with shape $[N \\times K/128]$. The kernel dequantizes the
    current W4 tile into A16 shared memory and computes ``activation @ W.T``.
    """

    def __init__(
        self,
        group_size: int = GROUP_SIZE,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            group_size: Manifest ``params.group_size``, ``int``, default ``128``.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
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
        inputs: "tuple[torch.Tensor | None, ...]",
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
    ) -> Kernel:
        return self.get_or_build_kernel(
            key_name,
            inputs,
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
        """Dequantize the INT4 weight tile by tile and multiply.

        Args:
            activation: Activations, $[M \\times K]$, ``torch.float16``.
            packed_weight: Weights, $[N \\times K/2]$, ``torch.uint8`` — two INT4
                values per byte, even $K$ in the low nibble.
            weight_scale: Group scales, $[N \\times K/128]$, ``torch.float32``.
            weight_zero: Group zero points, $[N \\times K/128]$, ``torch.uint8``.

        Returns:
            The product, $[M \\times N]$, in ``torch.float16``.

        Raises:
            ValueError: A dtype is not one of those listed above, ``activation`` or
                ``packed_weight`` is not 2D, $K$ is odd, $K$ is not divisible by
                ``group_size``, or a packed or metadata shape disagrees with $K$.

        Example:
            ```python linenums="1"
            op = GemmW4A16FwdOp()
            d = op(activation, packed_weight, weight_scale, weight_zero)
            flops, nbytes = op.eval_roofline()    # valid after the forward
            ```
        """
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
            kernel = self._get_kernel(
                (activation, packed_weight, weight_scale, weight_zero), m, n, k, activation.dtype
            )
            self.kernel = kernel
            self._active = kernel
            self._active_sig = sig

        return self._active(activation, packed_weight, weight_scale, weight_zero)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.dtype)
