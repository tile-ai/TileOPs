import warnings
from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.gemm import (
    GemmCall,
    GemmCpAsyncKernel,
    GemmFP8BlockScaleKernel,
    GemmFP8Call,
    GemmFP8FwdInterface,
    GemmFP8TensorScaleKernel,
    GemmFP81D2DFwdKernel,
    GemmFP81D2DWaveFwdKernel,
    GemmFwdInterface,
    GemmTMAKernel,
    GemmW4A16Call,
    GemmW4A16FwdInterface,
    GemmW4A16Kernel,
    GemmW4A16MMAKernel,
    GemvKernel,
    W4A16RepackCall,
    W4A16RepackFwdInterface,
    W4A16RepackKernel,
)
from tileops.kernels.gemm.w4a16 import W4A16_LAYOUT
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.ops.op_base import Op
from tileops.perf.profile import tensor_core_roof

__all__ = ["GemmFP8FwdOp", "GemmFwdOp", "GemmW4A16FwdOp"]


class GemmFwdOp(Op):
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
        "gemm_tma": GemmTMAKernel,
        "gemm_cp_async": GemmCpAsyncKernel,
        "gemv": GemvKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"gemm": GemmFwdInterface}

    def __init__(
        self,
        trans_a: bool = False,
        trans_b: bool = True,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtypes are taken from the first call.

        Args:
            trans_a: Whether ``a`` is stored transposed ($[K \\times M]$).
            trans_b: Whether ``b`` is stored transposed ($[N \\times K]$). Default ``True`` (NT).
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune (applied when a kernel is first built).
        """
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

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
            op = GemmFwdOp()                      # NT by default
            d = op(a, b)                          # a=[M,K], b=[N,K] -> d=[M,N]
            flops, nbytes = op.eval_roofline()    # valid after the forward
            ```
        """
        a, b = a.contiguous(), b.contiguous()
        m, k = (a.shape[1], a.shape[0]) if self.trans_a else a.shape
        call = GemmCall(
            m=m,
            n=b.shape[0] if self.trans_b else b.shape[1],
            k=k,
            dtype=a.dtype,
            trans_a=self.trans_a,
            trans_b=self.trans_b,
            device=a.device,
        )
        return self.kernel_for("gemm", call)(a, b)

    def compute_roof(self) -> str:
        return tensor_core_roof(self.last_call.ix["T"])


class GemmFP8FwdOp(Op):
    """Dense FP8 NT GEMM, input-inferred: $d = (a \\cdot s_a) \\mathbin{@} (b \\cdot s_b)^{\\top} + \\text{bias}$.

    ``a`` is $[M \\times K]$ and ``b`` is $[N \\times K]$, the operand ``torch._scaled_mm``
    receives as ``b.T``. ``scale_a`` and ``scale_b`` are both per-tensor $[1 \\times 1]$
    scales, or ``scale_a`` is per 1x128 block along K, $[M \\times \\lceil K/128 \\rceil]$,
    with ``scale_b`` per 1x128 block, $[N \\times \\lceil K/128 \\rceil]$, or per 128x128
    block, $[\\lceil N/128 \\rceil \\times \\lceil K/128 \\rceil]$.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gemm_fp8_tensor_scale": GemmFP8TensorScaleKernel,
        "gemm_fp8_block_scale": GemmFP8BlockScaleKernel,
        "gemm_fp8_1d2d": GemmFP81D2DFwdKernel,
        "gemm_fp8_1d2d_wave": GemmFP81D2DWaveFwdKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"gemm_fp8": GemmFP8FwdInterface}

    def __init__(
        self,
        out_dtype: torch.dtype = torch.bfloat16,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtypes are taken from the first call.

        Args:
            out_dtype: Output dtype, ``torch.bfloat16`` or ``torch.float16``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.out_dtype = out_dtype
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

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
            op = GemmFP8FwdOp(out_dtype=torch.bfloat16)
            d = op(a, b, scale_a, scale_b)        # per-tensor scales
            flops, nbytes = op.eval_roofline()    # valid after the forward
            ```
        """
        a, b, scale_a, scale_b = (t.contiguous() for t in (a, b, scale_a, scale_b))
        bias = None if bias is None else bias.contiguous()
        (m, k), n = a.shape, b.shape[0]
        call = GemmFP8Call(
            m=m,
            n=n,
            k=k,
            dtype=a.dtype,
            scale_a_shape=tuple(scale_a.shape),
            scale_b_shape=tuple(scale_b.shape),
            out_dtype=self.out_dtype,
            has_bias=bias is not None,
            device=a.device,
        )
        kernel = self.kernel_for("gemm_fp8", call)
        return kernel(a, b, scale_a, scale_b, bias)

    def compute_roof(self) -> str:
        return tensor_core_roof(self.last_call.ix["T"])


class GemmW4A16FwdOp(Op):
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
        "gemm_w4a16": GemmW4A16Kernel,
        "gemm_w4a16_mma": GemmW4A16MMAKernel,
        "w4a16_repack": W4A16RepackKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "gemm_w4a16": GemmW4A16FwdInterface,
        "w4a16_repack": W4A16RepackFwdInterface,
    }

    def __init__(
        self,
        group_size: int = 128,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtypes are taken from the first call.

        Args:
            group_size: Weights per dequantization group along K (default 128).
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Accepted for the common op interface and ignored with a warning.
                W4A16 uses its calibrated selector because generic autotuning cannot
                time the composite path.
        """
        self.group_size = group_size
        if tune:
            warnings.warn(
                "GemmW4A16FwdOp does not support generic autotuning; using the calibrated selector",
                UserWarning,
                stacklevel=2,
            )
        self.tune = False
        self.target = target
        self.dispatch_kernel(kernel_map)

    def autotune(self) -> None:
        """Keep the op out of tuned mode until composite-path tuning is supported."""
        warnings.warn(
            "GemmW4A16FwdOp does not support generic autotuning; using the calibrated selector",
            UserWarning,
            stacklevel=2,
        )

    def repack(self, packed_weight: torch.Tensor) -> torch.Tensor:
        """Put a row-major packed weight into the order ``forward`` reads.

        The order is a contract between this op's repack and its GEMM, so both come
        from the same set of kernels: replacing one through ``kernel_map=`` replaces
        the other with it.

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
        call = W4A16RepackCall(n=n, packed_k=packed_k, device=packed_weight.device)
        return self.kernel_for("w4a16_repack", call)(packed_weight)

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
            op = GemmW4A16FwdOp()
            d = op(activation, packed_weight, weight_scale, weight_zero)
            flops, nbytes = op.eval_roofline()    # valid after the forward
            ```
        """
        inputs = tuple(
            t.contiguous() for t in (activation, packed_weight, weight_scale, weight_zero)
        )
        (m, k), n = activation.shape, packed_weight.shape[0]
        call = GemmW4A16Call(
            m=m,
            n=n,
            k=k,
            dtype=activation.dtype,
            group_size=self.group_size,
            device=activation.device,
        )
        kernel = self.kernel_for("gemm_w4a16", call)
        return kernel(*inputs)

    def compute_roof(self) -> str:
        return tensor_core_roof(self.last_call.ix["T"])
