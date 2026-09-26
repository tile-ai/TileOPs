import warnings
from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.gemm.call_spec import GemmCall
from tileops.kernels.gemm.dense import (
    GemmCpAsyncKernel,
    GemmFp8BlockScaleKernel,
    GemmFp8TensorScaleKernel,
    GemmTmaKernel,
    GemvKernel,
)
from tileops.kernels.gemm.w4a16 import _LAYOUT, GemmW4A16Kernel
from tileops.kernels.gemm.w4a16_repack import W4A16RepackKernel
from tileops.kernels.kernel_base import Kernel
from tileops.perf.profile import tensor_core_roof

from ..op_base import Op

__all__ = ["GemmFp8FwdOp", "GemmFwdOp", "GemmW4A16FwdOp"]


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

    compile_boundary: ClassVar[bool] = True

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gemm_tma_kernel": GemmTmaKernel,
        "gemm_cp_async_kernel": GemmCpAsyncKernel,
        "gemv_kernel": GemvKernel,
    }

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

    def _call_spec(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        device: Optional[torch.device] = None,
    ) -> GemmCall:
        """State this call, for selection to filter candidates against."""
        return GemmCall(
            m=m,
            n=n,
            k=k,
            dtype=dtype,
            trans_a=self.trans_a,
            trans_b=self.trans_b,
            device=device,
            tune=self.tune,
        )

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
        return self._call_boundary(a, b)

    def _eager_forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        a, b = a.contiguous(), b.contiguous()
        m, k = (a.shape[1], a.shape[0]) if self.trans_a else a.shape
        n = b.shape[0] if self.trans_b else b.shape[1]
        kernel = self.kernel_for("gemm", (a, b), self._call_spec(m, n, k, a.dtype, a.device))
        return kernel(a, b)

    def compute_roof(self) -> str:
        return tensor_core_roof(self.last_call.ix["T"])


class GemmFp8FwdOp(Op):
    """Dense FP8 NT GEMM, input-inferred: $d = (a \\cdot s_a) \\mathbin{@} (b \\cdot s_b)^{\\top} + \\text{bias}$.

    ``a`` is $[M \\times K]$ and ``b`` is $[N \\times K]$, the operand ``torch._scaled_mm``
    receives as ``b.T``. ``scale_a`` and ``scale_b`` are both per-tensor $[1 \\times 1]$
    scales, or both per 1x128 block along K: $[M \\times \\lceil K/128 \\rceil]$ and
    $[N \\times \\lceil K/128 \\rceil]$.
    """

    compile_boundary: ClassVar[bool] = True

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gemm_fp8_tensor_scale_kernel": GemmFp8TensorScaleKernel,
        "gemm_fp8_block_scale_kernel": GemmFp8BlockScaleKernel,
    }

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
                block128 $[M \\times \\lceil K/128 \\rceil]$.
            scale_b: The same for ``b``: $[1 \\times 1]$ or
                $[N \\times \\lceil K/128 \\rceil]$. Both scales take the same form.
            bias: Optional bias, $[N]$, in ``out_dtype``.

        Returns:
            The scaled product plus bias, $[M \\times N]$, in ``out_dtype``.

        Example:
            ```python linenums="1"
            op = GemmFp8FwdOp(out_dtype=torch.bfloat16)
            d = op(a, b, scale_a, scale_b)        # per-tensor scales
            flops, nbytes = op.eval_roofline()    # valid after the forward
            ```
        """
        return self._call_boundary(a, b, scale_a, scale_b, bias)

    def _eager_forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        a, b, scale_a, scale_b = (t.contiguous() for t in (a, b, scale_a, scale_b))
        bias = None if bias is None else bias.contiguous()
        (m, k), n = a.shape, b.shape[0]
        call = GemmCall(
            m=m,
            n=n,
            k=k,
            dtype=a.dtype,
            trans_b=True,
            scale_a_shape=tuple(scale_a.shape),
            scale_b_shape=tuple(scale_b.shape),
            out_dtype=self.out_dtype,
            device=a.device,
            tune=self.tune,
        )
        self.kernel = self.kernel_for("gemm_fp8", (a, b, scale_a, scale_b, bias), call)
        return self.kernel(a, b, scale_a, scale_b, bias)

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

    compile_boundary: ClassVar[bool] = True

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"gemm_w4a16_kernel": GemmW4A16Kernel}

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

    @staticmethod
    def repack(packed_weight: torch.Tensor) -> torch.Tensor:
        """Put a row-major packed weight into the order ``forward`` reads.

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
        if packed_k % (_LAYOUT.mma_step_k // 2):
            raise ValueError(
                f"repack needs K/2={packed_k} to be a multiple of {_LAYOUT.mma_step_k // 2}, the"
                " packed width of one MMA K step"
            )
        kernel = W4A16RepackKernel(n, packed_k, device_index=packed_weight.device.index)
        return kernel(packed_weight)

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
        return self._call_boundary(activation, packed_weight, weight_scale, weight_zero)

    def _eager_forward(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zero: torch.Tensor,
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        inputs = tuple(
            t.contiguous() for t in (activation, packed_weight, weight_scale, weight_zero)
        )
        (m, k), n = activation.shape, packed_weight.shape[0]
        call = GemmCall(
            m=m,
            n=n,
            k=k,
            dtype=activation.dtype,
            trans_b=True,
            group_size=self.group_size,
            device=activation.device,
            tune=self.tune,
        )
        self.kernel = self.kernel_for("gemm_w4a16", inputs, call)
        return self.kernel(*inputs)

    def compute_roof(self) -> str:
        return tensor_core_roof(self.last_call.ix["T"])
