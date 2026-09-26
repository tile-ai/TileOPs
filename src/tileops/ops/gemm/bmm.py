"""Batched GEMM ops (BmmFwdOp, BmmFp8FwdOp).

Strict 3D-3D batched matrix multiplication matching ``torch.bmm``: every
batch item is an independent GEMM, no broadcasting.
"""

import warnings
from typing import ClassVar, Dict, Mapping, Optional, Set, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.gemm.bmm import (
    BmmFp8Kernel,
    BmmFp8TransposeKernel,
    BmmKernel,
    BmmPersistentKernel,
)
from tileops.kernels.gemm.call_spec import BmmCall
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.perf.profile import tensor_core_roof

from ..op_base import Op

__all__ = ["BmmFp8FwdOp", "BmmFwdOp"]


class BmmFwdOp(Op):
    """Batched dense GEMM: $d_i = a_i \\mathbin{@} b_i \\quad \\text{for } i \\in [0, B)$.

    Shapes are strictly 3D: ``a`` is $[B \\times M \\times K]$, ``b`` is
    $[B \\times K \\times N]$, and ``d`` is $[B \\times M \\times N]$. A kernel is
    compiled on first use for each ``(batch, m, n, k, dtype)`` combination and cached.
    The in-tree kernels need $K$ to be a multiple of 16 and refuse other calls when built.
    """

    compile_boundary: ClassVar[bool] = True

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "bmm_persistent_kernel": BmmPersistentKernel,
        "bmm_kernel": BmmKernel,
    }

    def __init__(
        self,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtypes are taken from the first call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multiply the two batches, one GEMM per batch item.

        Args:
            a: Left operand, $[B \\times M \\times K]$.
            b: Right operand, $[B \\times K \\times N]$.

        Returns:
            The product, $[B \\times M \\times N]$, in the dtype of the inputs.

        Example:
            ```python linenums="1"
            op = BmmFwdOp()
            d = op(a, b)                          # a=[B,M,K], b=[B,K,N] -> d=[B,M,N]
            flops, nbytes = op.eval_roofline()    # valid after the forward
            ```
        """
        return self._call_boundary(a, b)

    def _eager_forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        a, b = a.contiguous(), b.contiguous()
        batch, m, k = a.shape
        call = BmmCall(
            batch=batch, m=m, n=b.shape[2], k=k, dtype=a.dtype, device=a.device, tune=self.tune
        )
        # Expose the active kernel so autotune()/introspection can find it.
        self.kernel = self.kernel_for("bmm", (a, b), call)
        return self.kernel(a, b)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.ix["T"])


class BmmFp8FwdOp(Op):
    """Batched FP8 GEMM: ``d[i] = (a[i] @ b[i]) * scale_a * scale_b``.

    ``trans_b`` states which axis order ``b`` arrives in: ``False`` is torch.bmm's
    $[B \\times K \\times N]$, ``True`` is $[B \\times N \\times K]$.

    FP8 WGMMA reads ``b`` K-innermost and has no transposed operand mode, so a
    ``b`` that does not already lie that way is transposed into a new buffer,
    costing one extra read and write of it. Which calls pay is decided by ``b``'s
    strides, not by ``trans_b``: passing ``b`` K-innermost avoids the copy under
    either value of the flag, and is the faster call. The in-tree kernel needs $K$
    to be a multiple of 32 and refuses other calls when built.
    """

    compile_boundary: ClassVar[bool] = True

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "bmm_fp8_kernel": BmmFp8Kernel,
        "bmm_fp8_transpose_kernel": BmmFp8TransposeKernel,
    }

    def __init__(
        self,
        out_dtype: torch.dtype = torch.bfloat16,
        trans_b: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtypes are taken from the first call.

        Args:
            out_dtype: Output tensor dtype (``torch.float16`` or ``torch.bfloat16``).
            trans_b: Whether ``b``'s axes are $[B \\times N \\times K]$ rather than
                $[B \\times K \\times N]$. Which of the two is faster is decided by
                ``b``'s strides, not by this flag.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune (applied when a kernel is first built).
        """
        self.out_dtype = out_dtype
        self.trans_b = trans_b
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        # ``b`` shapes already warned about, so one op warns once per shape.
        self._kn_warned: Set[Tuple[int, int, int]] = set()

    def _get_transpose_kernel(
        self,
        inputs: "tuple[torch.Tensor | None, ...]",
        batch: int,
        rows: int,
        cols: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Kernel:
        return self.kernel_for(
            "bmm_fp8_transpose_kernel", inputs, (batch, rows, cols, dtype, device)
        )

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation per role, built per shape, dtype and device."""
        if role == "bmm_fp8_transpose_kernel":
            batch, rows, cols, dtype, device = call
            return call, lambda: self.kernel_map["bmm_fp8_transpose_kernel"](
                batch, rows, cols, dtype, device=device, tune=self.tune
            )
        batch, m, n, k, dtype, out_dtype, device = call
        return call, lambda: self.kernel_map["bmm_fp8_kernel"](
            batch, m, n, k, dtype, out_dtype, device=device, tune=self.tune
        )

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
    ) -> torch.Tensor:
        """Multiply the two FP8 batches and apply the two scales.

        Args:
            a: Left operand, $[B \\times M \\times K]$, ``torch.float8_e4m3fn``.
            b: Right operand, same dtype as ``a``: $[B \\times K \\times N]$, or
                $[B \\times N \\times K]$ when ``trans_b``.
            scale_a: Per-tensor scale for ``a``, a 0-dim ``torch.float32`` tensor.
            scale_b: Per-tensor scale for ``b``, a 0-dim ``torch.float32`` tensor.

        Returns:
            The scaled product, $[B \\times M \\times N]$, in ``out_dtype``.

        Example:
            ```python linenums="1"
            op = BmmFp8FwdOp(out_dtype=torch.bfloat16)      # b as [B, K, N]
            d = op(a, b_kn, scale_a, scale_b)
            flops, nbytes = op.eval_roofline()    # valid after the forward
            ```
        """
        return self._call_boundary(a, b, scale_a, scale_b)

    def _eager_forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        a = a.contiguous()
        b = self._as_k_innermost(b, a.dtype, a.device)
        scale_a, scale_b = scale_a.reshape(1), scale_b.reshape(1)
        batch, m, k = a.shape
        call = (batch, m, b.shape[1], k, a.dtype, self.out_dtype, a.device)
        self.kernel = self.kernel_for("bmm_fp8_kernel", (a, b, scale_a, scale_b), call)
        return self.kernel(a, b, scale_a, scale_b)

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
                    f"BmmFp8FwdOp: b (shape={tuple(b.shape)}) does not lie "
                    f"K-innermost, so it is transposed into a new buffer before "
                    f"the FP8 WGMMA kernel, which reads only that order. Passing "
                    f"b K-innermost skips the copy and is the faster call.",
                    stacklevel=2,
                )
            kernel = self._get_transpose_kernel((b,), batch, k, n, dtype, device)
            return kernel(b_nk.transpose(-2, -1))
        return b_nk.contiguous()

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.ix["T"])
