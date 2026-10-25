import warnings
from typing import ClassVar, Mapping, Optional

import torch

from tileops._kernels.gemm import (
    GEMMCall,
    GEMMCpAsyncKernel,
    GEMMFP8Block128WaveFwdKernel,
    GEMMFP8BlockScaleKernel,
    GEMMFP8Call,
    GEMMFP8FwdInterface,
    GEMMFP8SwapABKernel,
    GEMMFP8TensorScaleKernel,
    GEMMFP81D2DFwdKernel,
    GEMMFwdInterface,
    GEMMINT8W8A8Call,
    GEMMINT8W8A8FwdInterface,
    GEMMINT8W8A8Kernel,
    GEMMINT8W8A8PingPongKernel,
    GEMMINT8W8A8SimpleKernel,
    GEMMTMAKernel,
    GEMMW4A8Call,
    GEMMW4A8DecodeKernel,
    GEMMW4A8FwdInterface,
    GEMMW4A8PrefillKernel,
    GEMMW4A16Call,
    GEMMW4A16FwdInterface,
    GEMMW4A16Kernel,
    GEMMW4A16MMAKernel,
    GemvKernel,
    W4A8RepackCall,
    W4A8RepackFwdInterface,
    W4A8RepackKernel,
    W4A16RepackCall,
    W4A16RepackFwdInterface,
    W4A16RepackKernel,
)
from tileops._kernels.gemm.w4a8_repack import W4A8_LAYOUT
from tileops._kernels.gemm.w4a16 import W4A16_LAYOUT
from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._ops.op_base import Op
from tileops._perf.profile import tensor_core_roof
from tileops.backend import Target

__all__ = ["GEMMFP8FwdOp", "GEMMFwdOp", "GEMMINT8W8A8FwdOp", "GEMMW4A8FwdOp", "GEMMW4A16FwdOp"]


class GEMMFwdOp(Op):
    """Dense GEMM. Nothing is committed at construction: ``m, n, k`` and the dtype
    come from the ``forward`` inputs, so ``eval_roofline()`` is valid only after a call.

    The ``(trans_a, trans_b)`` pair selects one of four layouts:

    | Flags | Layout | Product |
    | --- | --- | --- |
    | ``(False, True)`` | NT, the default | $d = a \\mathbin{@} b^{\\top}$ |
    | ``(False, False)`` | NN | $d = a \\mathbin{@} b$ |
    | ``(True, False)`` | TN | $d = a^{\\top} \\mathbin{@} b$ |
    | ``(True, True)`` | TT | $d = a^{\\top} \\mathbin{@} b^{\\top}$ |
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gemm_tma": GEMMTMAKernel,
        "gemm_cp_async": GEMMCpAsyncKernel,
        "gemv": GemvKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"gemm": GEMMFwdInterface}

    def __init__(
        self,
        trans_a: bool = False,
        trans_b: bool = True,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtypes are taken from the first call.

        Args:
            trans_a: Whether ``a`` is stored transposed ($[K \\times M]$).
            trans_b: Whether ``b`` is stored transposed ($[N \\times K]$). Default ``True`` (NT).
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.trans_a = trans_a
        self.trans_b = trans_b
        super().__init__(target=target)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multiply the two matrices under the layout the constructor selected.

        Args:
            a: Left operand, $[M \\times K]$, or $[K \\times M]$ when ``trans_a``.
            b: Right operand, $[N \\times K]$ under the default NT layout, or
                $[K \\times N]$ when ``trans_b`` is false.

        Returns:
            The product, $[M \\times N]$, in the dtype of the inputs.

        Example:
            ```python linenums="1"
            op = GEMMFwdOp()                      # NT by default
            d = op(a, b)                          # a=[M,K], b=[N,K] -> d=[M,N]
            flops, nbytes = op.eval_roofline()    # valid after the forward
            ```
        """
        a, b = a.contiguous(), b.contiguous()

        def call() -> GEMMCall:
            m, k = (a.shape[1], a.shape[0]) if self.trans_a else a.shape
            return GEMMCall(
                m=m,
                n=b.shape[0] if self.trans_b else b.shape[1],
                k=k,
                dtype=a.dtype,
                trans_a=self.trans_a,
                trans_b=self.trans_b,
                device=a.device,
            )

        return self.kernel_for("gemm", call)(a, b)

    def roof_key(self) -> str:
        return tensor_core_roof(self.last_call.indices["T"])


class GEMMFP8FwdOp(Op):
    """Dense FP8 NT GEMM, input-inferred: $d = (a \\cdot s_a) \\mathbin{@} (b \\cdot s_b)^{\\top} + \\text{bias}$.

    ``a`` is $[M \\times K]$ and ``b`` is $[N \\times K]$, the operand ``torch._scaled_mm``
    receives as ``b.T``. ``scale_a`` and ``scale_b`` are both per-tensor $[1 \\times 1]$
    scales, or ``scale_a`` is per 1x128 block along K, $[M \\times \\lceil K/128 \\rceil]$,
    with ``scale_b`` per 1x128 block, $[N \\times \\lceil K/128 \\rceil]$, or per 128x128
    block, $[\\lceil N/128 \\rceil \\times \\lceil K/128 \\rceil]$.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gemm_fp8_tensor_scale": GEMMFP8TensorScaleKernel,
        "gemm_fp8_block_scale": GEMMFP8BlockScaleKernel,
        "gemm_fp8_1d2d": GEMMFP81D2DFwdKernel,
        "gemm_fp8_block128_wave": GEMMFP8Block128WaveFwdKernel,
        "gemm_fp8_swap_ab": GEMMFP8SwapABKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"gemm_fp8": GEMMFP8FwdInterface}

    def __init__(
        self,
        out_dtype: torch.dtype = torch.bfloat16,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtypes are taken from the first call.

        Args:
            out_dtype: Output dtype, ``torch.bfloat16`` or ``torch.float16``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
        """
        self.out_dtype = out_dtype
        super().__init__(target=target)

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
                per 1x128 block $[M \\times \\lceil K/128 \\rceil]$.
            scale_b: ``torch.float32`` scales for ``b``: $[1 \\times 1]$ when ``scale_a`` is,
                else per 1x128 block $[N \\times \\lceil K/128 \\rceil]$ or per 128x128
                block $[\\lceil N/128 \\rceil \\times \\lceil K/128 \\rceil]$.
            bias: Optional bias, $[N]$, in ``out_dtype``.

        Returns:
            The scaled product plus bias, $[M \\times N]$, in ``out_dtype``.

        Example:
            ```python linenums="1"
            op = GEMMFP8FwdOp(out_dtype=torch.bfloat16)
            d = op(a, b, scale_a, scale_b)        # per-tensor scales
            flops, nbytes = op.eval_roofline()    # valid after the forward
            ```
        """
        a, b, scale_a, scale_b = (t.contiguous() for t in (a, b, scale_a, scale_b))
        bias = None if bias is None else bias.contiguous()
        (m, k), n = a.shape, b.shape[0]
        kernel = self.kernel_for(
            "gemm_fp8",
            lambda: GEMMFP8Call(
                m=m,
                n=n,
                k=k,
                dtype=a.dtype,
                scale_a_shape=tuple(scale_a.shape),
                scale_b_shape=tuple(scale_b.shape),
                out_dtype=self.out_dtype,
                has_bias=bias is not None,
                device=a.device,
            ),
        )
        return kernel(a, b, scale_a, scale_b, bias)

    def roof_key(self) -> str:
        return tensor_core_roof(self.last_call.indices["T"])


class GEMMINT8W8A8FwdOp(Op):
    """INT8 NT GEMM with per-token and per-channel dequantization, input-inferred:
    $d = (a \\mathbin{@} b^{\\top}) \\cdot s_a \\cdot s_b + \\text{bias}$.

    ``a`` is an $[M \\times K]$ activation and ``b`` an $[N \\times K]$ weight, both
    ``torch.int8``. ``scale_a`` is one ``float32`` scale for ``a`` ($[1 \\times 1]$) or one
    per token ($[M \\times 1]$); ``scale_b`` one for ``b`` ($[1 \\times 1]$) or one per
    output channel ($[1 \\times N]$).

    Accuracy: the product is accumulated in int32, as ``cutlass_scaled_mm`` accumulates it,
    and is exact while $K \\le 131071$: an int8 product is at most $2^{14}$ in magnitude, so
    no partial sum leaves the int32 range below that; past it a sum can wrap. Each result is
    converted to float32, multiplied by its token scale and then its channel scale, the bias
    is added in float32, and the sum is rounded once to ``out_dtype``. A result can differ
    from ``vllm._custom_ops.cutlass_scaled_mm`` by one unit in the last place of
    ``out_dtype``, where the two order the float32 multiplies differently.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gemm_int8_w8a8": GEMMINT8W8A8Kernel,
        "gemm_int8_w8a8_pingpong": GEMMINT8W8A8PingPongKernel,
        "gemm_int8_w8a8_simple": GEMMINT8W8A8SimpleKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "gemm_int8_w8a8": GEMMINT8W8A8FwdInterface
    }

    def __init__(
        self,
        out_dtype: torch.dtype = torch.bfloat16,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and scale granularities are taken from each call.

        Args:
            out_dtype: Output dtype, ``torch.bfloat16`` or ``torch.float16``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
        """
        self.out_dtype = out_dtype
        super().__init__(target=target)

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Multiply the two INT8 matrices, dequantize, and add the bias.

        Args:
            a: Activation, $[M \\times K]$, ``torch.int8``.
            b: Weight, $[N \\times K]$, ``torch.int8``.
            scale_a: ``torch.float32`` scale of ``a``: $[1 \\times 1]$, or one per row,
                $[M \\times 1]$.
            scale_b: ``torch.float32`` scale of ``b``: $[1 \\times 1]$, or one per row of
                ``b``, $[1 \\times N]$.
            bias: Optional bias, $[N]$, in ``out_dtype``.

        Returns:
            The dequantized product plus bias, $[M \\times N]$, in ``out_dtype``.

        Example:
            ```python linenums="1"
            op = GEMMINT8W8A8FwdOp(out_dtype=torch.bfloat16)
            d = op(a, b, scale_a, scale_b)        # a=[M,K], b=[N,K] -> d=[M,N]
            flops, nbytes = op.eval_roofline()    # valid after the forward
            ```
        """
        a, b, scale_a, scale_b = (t.contiguous() for t in (a, b, scale_a, scale_b))
        bias = None if bias is None else bias.contiguous()
        (m, k), n = a.shape, b.shape[0]
        kernel = self.kernel_for(
            "gemm_int8_w8a8",
            lambda: GEMMINT8W8A8Call(
                m=m,
                n=n,
                k=k,
                per_token=tuple(scale_a.shape) == (m, 1),
                per_channel=tuple(scale_b.shape) == (1, n),
                out_dtype=self.out_dtype,
                has_bias=bias is not None,
                device=a.device,
            ),
        )
        return kernel(a, b, scale_a, scale_b, bias)

    def roof_key(self) -> str:
        return tensor_core_roof(torch.int8)


class GEMMW4A16FwdOp(Op):
    """Dense W4A16 NT GEMM with group-wise affine weight dequantization.

    Inputs are activation ``[M, K]``, prepacked weight ``[N, K/2]``, and scale
    and zero-point tensors ``[N, K/group_size]``. Weight ``(n, k)`` is
    ``(q - zero[n, g]) * scale[n, g]`` with ``g = k // group_size`` and ``q`` its
    INT4 value. ``repack`` converts a row-major packed weight once at load time; the
    two layouts have the same shape and dtype and cannot be distinguished at runtime.
    The output is ``activation @ W.T`` with shape ``[M, N]``. The in-tree kernel serves
    ``group_size = 128`` only and refuses any other value when it is built.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gemm_w4a16": GEMMW4A16Kernel,
        "gemm_w4a16_mma": GEMMW4A16MMAKernel,
        "w4a16_repack": W4A16RepackKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "gemm_w4a16": GEMMW4A16FwdInterface,
        "w4a16_repack": W4A16RepackFwdInterface,
    }

    def __init__(
        self,
        group_size: int = 128,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtypes are taken from the first call.

        Args:
            group_size: Weights per dequantization group along K (default 128).
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
        """
        self.group_size = group_size
        super().__init__(target=target)

    def request_tune(self) -> None:
        """Warn and stay out of tuned mode: W4A16 uses its calibrated selector, because
        generic autotuning cannot time the composite path."""
        warnings.warn(
            "GEMMW4A16FwdOp does not support generic autotuning; using the calibrated selector",
            UserWarning,
            stacklevel=2,
        )

    def repack(self, packed_weight: torch.Tensor) -> torch.Tensor:
        """Put a row-major packed weight into the order ``forward`` reads.

        The order is a contract between this op's repack and its GEMM, so both come
        from the same set of kernels.

        Args:
            packed_weight: Row-major packed weights, $[N \\times K/2]$, ``torch.uint8``:
                two INT4 per byte, even $K$ in the low nibble.

        Returns:
            A prepacked tensor with the same shape and dtype.

        Raises:
            ValueError: The input is not a rank-2 uint8 tensor whose K dimension
                contains whole MMA steps.
        """
        if packed_weight.dtype != torch.uint8:
            raise ValueError(f"repack expects uint8 packed_weight, got {packed_weight.dtype}")
        if packed_weight.ndim != 2:
            raise ValueError(f"repack expects a rank-2 weight, got {packed_weight.ndim}")
        n, packed_k = packed_weight.shape
        if packed_k % (W4A16_LAYOUT.mma_step_k // 2):
            raise ValueError(
                f"repack needs K/2={packed_k} to be a multiple of {W4A16_LAYOUT.mma_step_k // 2}, the"
                " packed width of one MMA K step"
            )
        packed_weight = packed_weight.contiguous()
        return self.kernel_for(
            "w4a16_repack",
            lambda: W4A16RepackCall(n=n, packed_k=packed_k, device=packed_weight.device),
        )(packed_weight)

    def forward(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zero: torch.Tensor,
    ) -> torch.Tensor:
        """Multiply the activations by the dequantized INT4 weight.

        Args:
            activation: Activations, $[M \\times K]$, ``torch.float16``.
            packed_weight: Weights, $[N \\times K/2]$, ``torch.uint8``, in the order
                ``repack`` returns.
            weight_scale: Group scales, $[N \\times K/\\text{group\\_size}]$, in the
                activation dtype.
            weight_zero: Group zero points, $[N \\times K/\\text{group\\_size}]$,
                ``torch.uint8``.

        Returns:
            The product, $[M \\times N]$, in ``torch.float16``.

        Example:
            ```python linenums="1"
            op = GEMMW4A16FwdOp()
            d = op(activation, packed_weight, weight_scale, weight_zero)
            flops, nbytes = op.eval_roofline()    # valid after the forward
            ```
        """
        inputs = tuple(
            t.contiguous() for t in (activation, packed_weight, weight_scale, weight_zero)
        )
        (m, k), n = activation.shape, packed_weight.shape[0]
        kernel = self.kernel_for(
            "gemm_w4a16",
            lambda: GEMMW4A16Call(
                m=m,
                n=n,
                k=k,
                dtype=activation.dtype,
                group_size=self.group_size,
                device=activation.device,
            ),
        )
        return kernel(*inputs)

    def roof_key(self) -> str:
        return tensor_core_roof(self.last_call.indices["T"])


class GEMMW4A8FwdOp(Op):
    """Dense W4A8 NT GEMM: FP8 activations times a group-scaled signed INT4 weight.

    Inputs are activation ``[M, K]`` in ``float8_e4m3fn``, prepacked weight ``[N, K/2]``,
    ``float8_e4m3fn`` group scales ``[N, K/group_size]``, and ``float32`` channel scales
    ``[N]`` and token scales ``[M]``; the output is ``[M, N]`` in ``bfloat16``.

    Weight ``(n, k)`` enters the product as ``q * weight_group_scale[n, k // group_size]``
    rounded to ``float8_e4m3fn``, nearest even, with ``q`` its signed INT4 value in
    ``[-8, 7]``. Output ``(m, n)`` is that FP8 product summed over K in FP32, times
    ``activation_scale[m] * weight_channel_scale[n]``, rounded to ``bfloat16``.

    Accuracy: for finite inputs each output is within ``5e-5 * K + 2e-2 * |r|`` of ``r``,
    the FP32 contraction with token-first scaling and BF16 rounding, and the squared error
    of the whole output is at most ``5e-5`` of its and ``r``'s combined energy. Where a zero
    channel or token scale makes ``r`` zero, the output is exactly zero. Infinities and NaNs
    appear where ``r`` has them; NaN payloads and the sign of zero are unspecified.
    Results are not bitwise reproducible across calls.

    ``forward`` reads the weight in the order ``repack`` returns, which has the shape and
    dtype of the row-major packing, so passing the row-major weight gives wrong results
    rather than an error. M must be positive, N a positive multiple of 16 and K a positive
    multiple of 128. The in-tree kernels serve ``group_size = 128`` only and refuse any other
    value when one is selected. One instance must not run on two streams at once.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gemm_w4a8_decode": GEMMW4A8DecodeKernel,
        "gemm_w4a8_prefill": GEMMW4A8PrefillKernel,
        "w4a8_repack": W4A8RepackKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "gemm_w4a8": GEMMW4A8FwdInterface,
        "w4a8_repack": W4A8RepackFwdInterface,
    }

    def __init__(
        self,
        group_size: int = 128,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes are taken from the first call.

        Args:
            group_size: Weights per group scale along K (default 128).
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
        """
        self.group_size = group_size
        super().__init__(target=target)

    def repack(self, packed_weight: torch.Tensor) -> torch.Tensor:
        """Put a row-major packed weight into the order ``forward`` reads.

        The order is a contract between this op's repack and its GEMMs, so they come from
        the same set of kernels.

        Args:
            packed_weight: Row-major packed weights, $[N \\times K/2]$, ``torch.uint8``: two
                signed two's-complement INT4 per byte, even $K$ in the low nibble.

        Returns:
            A prepacked tensor with the same shape and dtype.

        Raises:
            ValueError: The input is not a rank-2 uint8 tensor with N a positive multiple of 16 and
                K a positive multiple of 128.
        """
        if packed_weight.dtype != torch.uint8:
            raise ValueError(f"repack expects uint8 packed_weight, got {packed_weight.dtype}")
        if packed_weight.ndim != 2:
            raise ValueError(f"repack expects a rank-2 weight, got {packed_weight.ndim}")
        n, k = packed_weight.shape[0], 2 * packed_weight.shape[1]
        if n <= 0 or k <= 0:
            raise ValueError(f"repack needs positive N and K, got N={n}, K={k}")
        if n % W4A8_LAYOUT.unit_rows or k % W4A8_LAYOUT.tile_k:
            raise ValueError(
                f"repack needs N={n} to be a multiple of {W4A8_LAYOUT.unit_rows} and K={k} of"
                f" {W4A8_LAYOUT.tile_k}, the extents of one packed unit"
            )
        packed_weight = packed_weight.contiguous()
        return self.kernel_for(
            "w4a8_repack", lambda: W4A8RepackCall(n=n, k=k, device=packed_weight.device)
        )(packed_weight)

    def forward(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_group_scale: torch.Tensor,
        weight_channel_scale: torch.Tensor,
        activation_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Multiply the FP8 activations by the scaled INT4 weight.

        Args:
            activation: Activations, $[M \\times K]$, ``torch.float8_e4m3fn``.
            packed_weight: Weights, $[N \\times K/2]$, ``torch.uint8``, in the order
                ``repack`` returns.
            weight_group_scale: Group scales, $[N \\times K/\\text{group\\_size}]$,
                ``torch.float8_e4m3fn``.
            weight_channel_scale: Channel scales, $[N]$, ``torch.float32``.
            activation_scale: Token scales, $[M]$, ``torch.float32``.

        Returns:
            The product, $[M \\times N]$, in ``torch.bfloat16``.

        Example:
            ```python linenums="1"
            op = GEMMW4A8FwdOp()
            prepacked = op.repack(packed_weight)  # once, at load time
            d = op(activation, prepacked, group_scale, channel_scale, token_scale)
            flops, nbytes = op.eval_roofline()    # valid after the forward
            ```
        """
        inputs = tuple(
            t.contiguous()
            for t in (
                activation,
                packed_weight,
                weight_group_scale,
                weight_channel_scale,
                activation_scale,
            )
        )
        (m, k), n = activation.shape, packed_weight.shape[0]
        return self.kernel_for(
            "gemm_w4a8",
            lambda: GEMMW4A8Call(
                m=m, n=n, k=k, group_size=self.group_size, device=activation.device
            ),
        )(*inputs)

    def roof_key(self) -> str:
        return tensor_core_roof(torch.float8_e4m3fn)
