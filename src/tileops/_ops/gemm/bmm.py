"""Batched GEMM ops (BMMFwdOp, BMMFP8FwdOp).

Strict 3D-3D batched matrix multiplication matching ``torch.bmm``: every
batch item is an independent GEMM, no broadcasting.
"""

import warnings
from collections.abc import Mapping
from typing import ClassVar

import torch

from tileops._kernels.gemm import (
    BMMCall,
    BMMFP8Call,
    BMMFP8FwdInterface,
    BMMFP8Kernel,
    BMMFP8PersistentKernel,
    BMMFP8TransposeCall,
    BMMFP8TransposeFwdInterface,
    BMMFP8TransposeKernel,
    BMMFP8WSKernel,
    BMMFwdInterface,
    BMMKernel,
    BMMPersistentKernel,
    BMMPingPongKernel,
)
from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._ops.op_base import Op
from tileops._perf.profile import tensor_core_roof
from tileops.backend import Target

__all__ = ["BMMFP8FwdOp", "BMMFwdOp"]


class BMMFwdOp(Op):
    """Batched dense GEMM: $d_i = a_i \\mathbin{@} b_i \\quad \\text{for } i \\in [0, B)$.

    Shapes are strictly 3D: ``a`` is $[B \\times M \\times K]$, ``b`` is
    $[B \\times K \\times N]$ as in ``torch.bmm`` (no transpose flag), and ``d`` is
    $[B \\times M \\times N]$. ``a`` and ``out`` may be strided: a batch-strided view, such
    as the transpose of an $[M \\times B \\times K]$ tensor, is read or written in place.
    Each new ``(batch, m, n, k, dtype)`` and each new strided layout of ``a`` or ``out``
    compiles a kernel on first use. The in-tree kernels need $K$ to be a multiple of 16.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "bmm_pingpong": BMMPingPongKernel,
        "bmm_persistent": BMMPersistentKernel,
        "bmm": BMMKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"bmm": BMMFwdInterface}

    def __init__(
        self,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtypes are taken from the first call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        super().__init__(target=target)

    def forward(
        self, a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Multiply the two batches, one GEMM per batch item.

        Args:
            a: Left operand, $[B \\times M \\times K]$; any strides.
            b: Right operand, $[B \\times K \\times N]$.
            out: $[B \\times M \\times N]$ buffer the product is written to, any strides,
                or None.

        Returns:
            The product, $[B \\times M \\times N]$, in the dtype of the inputs: ``out``
            where given.

        Example:
            ```python linenums="1"
            op = BMMFwdOp()
            d = op(a, b)                          # a=[B,M,K], b=[B,K,N] -> d=[B,M,N]
            flops, nbytes = op.eval_roofline()    # valid after the forward
            ```
        """
        a = a if a.stride(-1) == 1 else a.contiguous()
        b = b.contiguous()
        batch, m, k = a.shape
        kernel = self.kernel_for(
            "bmm",
            lambda: BMMCall(batch=batch, m=m, n=b.shape[2], k=k, dtype=a.dtype, device=a.device),
        )
        return kernel(a, b, out)

    def roof_key(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.indices["T"])


class BMMFP8FwdOp(Op):
    """Batched FP8 GEMM: ``d[i] = (a[i] @ b[i]) * a_scale * b_scale``.

    ``trans_b`` states which axis order ``b`` arrives in: ``False``, the default, is
    ``torch.bmm``'s $[B \\times K \\times N]$; ``True`` is $[B \\times N \\times K]$.

    A ``b`` that does not lie K-innermost in memory, under either value of ``trans_b``, is
    first transposed into a new buffer (one extra read and write of ``b``), because FP8
    WGMMA reads ``b`` only K-innermost. The in-tree kernel needs $K$ to be a multiple of 32.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "bmm_fp8_ws": BMMFP8WSKernel,
        "bmm_fp8_persistent": BMMFP8PersistentKernel,
        "bmm_fp8": BMMFP8Kernel,
        "bmm_fp8_transpose": BMMFP8TransposeKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "bmm_fp8": BMMFP8FwdInterface,
        "bmm_fp8_transpose": BMMFP8TransposeFwdInterface,
    }

    def __init__(
        self,
        out_dtype: torch.dtype = torch.bfloat16,
        trans_b: bool = False,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtypes are taken from the first call.

        Args:
            out_dtype: Output tensor dtype (``torch.float16`` or ``torch.bfloat16``).
            trans_b: Whether ``b``'s axes are $[B \\times N \\times K]$ rather than
                $[B \\times K \\times N]$. Default ``False``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
        """
        self.out_dtype = out_dtype
        self.trans_b = trans_b
        super().__init__(target=target)
        # ``b`` shapes already warned about, so one op warns once per shape.
        self._kn_warned: set[tuple[int, int, int]] = set()

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Multiply the two FP8 batches and apply the two scales.

        Args:
            a: Left operand, $[B \\times M \\times K]$, ``torch.float8_e4m3fn``.
            b: Right operand, same dtype as ``a``: $[B \\times K \\times N]$ under the
                default ``trans_b``, or $[B \\times N \\times K]$ when ``trans_b`` is true.
            a_scale: Per-tensor scale for ``a``, a 0-dim ``torch.float32`` tensor.
            b_scale: Per-tensor scale for ``b``, a 0-dim ``torch.float32`` tensor.

        Returns:
            The scaled product, $[B \\times M \\times N]$, in ``out_dtype``.

        Example:
            ```python linenums="1"
            op = BMMFP8FwdOp(trans_b=True)      # b as [B, N, K]
            d = op(a, b_nk, a_scale, b_scale)
            flops, nbytes = op.eval_roofline()    # valid after the forward
            ```
        """
        a = a.contiguous()
        b = self._as_k_innermost(b, a.dtype, a.device)
        a_scale, b_scale = a_scale.reshape(1), b_scale.reshape(1)
        batch, m, k = a.shape
        kernel = self.kernel_for(
            "bmm_fp8",
            lambda: BMMFP8Call(
                batch=batch,
                m=m,
                n=b.shape[1],
                k=k,
                dtype=a.dtype,
                out_dtype=self.out_dtype,
                device=a.device,
            ),
        )
        return kernel(a, b, a_scale, b_scale)

    def _as_k_innermost(
        self, b: torch.Tensor, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Return ``b`` as a contiguous $[B \\times N \\times K]$ tensor.

        ``trans_b`` says which axis carries K; ``b``'s strides say whether any
        movement is owed.

        Args:
            b: The right operand as the caller passed it.
            dtype: Element dtype, part of the transpose kernel's identity.
            device: Device the transpose would be built for.

        Returns:
            ``b`` itself when it already lies K-innermost, else a transposed copy.
        """
        b_nk = b if self.trans_b else b.transpose(-2, -1)
        if b_nk.is_contiguous():
            return b_nk
        batch, n, k = b_nk.shape
        if b_nk.stride() == (n * k, 1, n):
            # A contiguous [B, K, N] seen the other way round, which is the only
            # stride pattern the transpose kernel reads.
            shape_key = (batch, n, k)
            if shape_key not in self._kn_warned:
                self._kn_warned.add(shape_key)
                warnings.warn(
                    f"BMMFP8FwdOp: b (shape={tuple(b.shape)}) does not lie "
                    f"K-innermost, so it is transposed into a new buffer before "
                    f"the FP8 WGMMA kernel, which reads only that order. Passing "
                    f"b K-innermost skips the copy and is the faster call.",
                    stacklevel=2,
                )
            kernel = self.kernel_for(
                "bmm_fp8_transpose",
                lambda: BMMFP8TransposeCall(
                    batch=batch, rows=k, cols=n, dtype=dtype, device=device
                ),
            )
            return kernel(b_nk.transpose(-2, -1))
        return b_nk.contiguous()

    def roof_key(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.indices["T"])
