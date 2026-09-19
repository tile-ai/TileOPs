"""Batched GEMM op (BmmFwdOp).

Strict 3D-3D batched matrix multiplication matching ``torch.bmm``: every
batch item is an independent GEMM, no broadcasting.
"""

import warnings
from typing import ClassVar, Dict, Optional, Set, Tuple

import torch

from tileops.kernels.gemm.bmm import (
    BmmFp8Kernel,
    BmmFp8TransposeKernel,
    BmmKernel,
    BmmPersistentKernel,
)
from tileops.kernels.gemm.call_spec import BmmCall
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.perf.profile import tensor_core_roof

from .._compile_boundary_codegen import OperatorSpec
from ..op_base import Op

__all__ = ["BmmFp8FwdOp", "BmmFwdOp"]


class BmmFwdOp(Op):
    """Batched dense GEMM: $d_i = a_i \\mathbin{@} b_i \\quad \\text{for } i \\in [0, B)$.

    Shapes are strictly 3D: ``a`` is $[B \\times M \\times K]$, ``b`` is
    $[B \\times K \\times N]$, and ``d`` is $[B \\times M \\times N]$. The batch and
    contraction dims are checked at ``forward()`` time. A kernel is compiled on first
    use for each ``(batch, m, n, k, dtype)`` combination and cached.

    """

    compile_boundary: ClassVar[tuple[OperatorSpec, ...]] = (OperatorSpec(),)

    def __init__(
        self,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. No shape or dtype is bound until the first call.

        Args:
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        # (batch, m, n, k, dtype) -> Kernel instance; built lazily on first use.
        # Fast path: skip re-inference when the input signature is unchanged.
        self._active_sig: Optional[tuple] = None
        self._active_kernel: Optional[Kernel] = None
        # Roofline / dtype bindings, populated on the first forward().

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "bmm_persistent_kernel": BmmPersistentKernel,
            "bmm_kernel": BmmKernel,
        }

    def _infer_bmnk(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> Tuple[int, int, int, int]:
        """Derive logical ``(batch, m, n, k)`` from $[B \\times M \\times K]$ and $[B \\times K \\times N]$.

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
                f"BmmFwdOp batch dim mismatch: a.shape[0]={batch_a} vs b.shape[0]={batch_b}"
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
                f"(WGMMA alignment; see manifest shape_rules), got K={k_a}"
            )
        return batch_a, m, n, k_a

    def _call_spec(
        self,
        batch: int,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> BmmCall:
        """State the inferred BMM call for implementation selection."""
        return BmmCall(
            batch=batch,
            m=m,
            n=n,
            k=k,
            dtype=dtype,
            device=device,
            tune=self.tune,
        )

    def _get_kernel(
        self,
        inputs: "tuple[torch.Tensor | None, ...]",
        call: BmmCall,
    ) -> Kernel:
        """Return what serves *call*, building and caching on a miss."""
        return self.kernel_for("bmm", inputs, call)

    def _infer_output_shapes(
        self,
        a_shape: tuple[int, ...],
        b_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        """Manifest ``shape_rules``: ``d.shape == (B, M, N)``."""
        return {"d": (a_shape[0], a_shape[1], b_shape[2])}

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multiply the two batches, one GEMM per batch item.

        Args:
            a: Left operand, $[B \\times M \\times K]$.
            b: Right operand, $[B \\times K \\times N]$.

        Returns:
            The product, $[B \\times M \\times N]$, in the dtype of the inputs.

        Raises:
            ValueError: The operands disagree on dtype or device, either is not 3D,
                or their batch or contraction dims do not match.

        Example:
            ```python linenums="1"
            op = BmmFwdOp()
            d = op(a, b)                          # a=[B,M,K], b=[B,K,N] -> d=[B,M,N]
            flops, nbytes = op.eval_roofline()    # valid after the forward
            ```
        """
        return self._wrapped(a, b, self._instance_key)

    def _eager_forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Validate, resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        sig = (a.shape, b.shape, a.dtype, b.dtype, a.device)
        if sig != self._active_sig:
            self._validate_dtypes(a, b)
            batch, m, n, k = self._infer_bmnk(a, b)
            # Bind dims/dtype for the manifest func-mode roofline.
            self.batch, self.m, self.n, self.k = batch, m, n, k
            self.dtype = a.dtype
            self.a_shape = tuple(a.shape)
            self.b_shape = tuple(b.shape)
            call = self._call_spec(batch, m, n, k, a.dtype, a.device)
            kernel = self._get_kernel((a, b), call)
            # Expose the active kernel so autotune()/introspection can find it.
            self.kernel = kernel
            self._active_kernel = kernel
            self._active_sig = sig

        return self._active_kernel(a, b)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.dtype)


class BmmFp8FwdOp(Op):
    """Batched FP8 GEMM: ``d[i] = (a[i] @ b[i]) * scale_a * scale_b``.

    ``trans_b`` states which axis order ``b`` arrives in: ``False`` is torch.bmm's
    $[B \\times K \\times N]$, ``True`` is $[B \\times N \\times K]$.

    FP8 WGMMA reads ``b`` K-innermost and has no transposed operand mode, so a
    ``b`` that does not already lie that way is transposed into a new buffer,
    costing one extra read and write of it. Which calls pay is decided by ``b``'s
    strides, not by ``trans_b``: passing ``b`` K-innermost avoids the copy under
    either value of the flag, and is the faster call.
    """

    compile_boundary: ClassVar[tuple[OperatorSpec, ...]] = (OperatorSpec(),)

    def __init__(
        self,
        out_dtype: torch.dtype = torch.bfloat16,
        trans_b: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            out_dtype: Output tensor dtype (``torch.float16`` or ``torch.bfloat16``).
            trans_b: Whether ``b``'s axes are $[B \\times N \\times K]$ rather than
                $[B \\times K \\times N]$. Which of the two is faster is decided by
                ``b``'s strides, not by this flag.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune (applied when a kernel is first built).
        """
        if out_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"BmmFp8FwdOp outputs torch.float16 or torch.bfloat16, got {out_dtype}"
            )
        self.out_dtype = out_dtype
        self.trans_b = trans_b
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self._active_sig: Optional[tuple] = None
        self._active: Optional[Kernel] = None
        # ``b`` shapes already warned about, so one op warns once per shape.
        self._kn_warned: Set[Tuple[int, int, int]] = set()

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "bmm_fp8_kernel": BmmFp8Kernel,
            "bmm_fp8_transpose_kernel": BmmFp8TransposeKernel,
        }

    def _validate_dtypes(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
    ) -> None:
        if a.dtype != torch.float8_e4m3fn:
            raise ValueError(f"BmmFp8FwdOp only supports torch.float8_e4m3fn, got {a.dtype}")
        if b.dtype != a.dtype:
            raise ValueError(f"BmmFp8FwdOp expects b dtype {a.dtype}, got {b.dtype}")
        if scale_a.dtype != torch.float32 or scale_b.dtype != torch.float32:
            raise ValueError("BmmFp8FwdOp expects scale_a and scale_b to be torch.float32")

    def _infer_bmnk(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> Tuple[int, int, int, int]:
        """Derive logical ``(batch, m, n, k)`` from ``a`` and ``b``."""
        if a.dim() != 3 or b.dim() != 3:
            raise ValueError(
                f"BmmFp8FwdOp expects strict 3D inputs a=[B,M,K] and "
                f"b=[B,K,N] or [B,N,K] (got a.shape={tuple(a.shape)}, "
                f"b.shape={tuple(b.shape)})"
            )
        batch_a, m, k = a.shape
        batch_b, b1, b2 = b.shape
        if batch_a != batch_b:
            raise ValueError(
                f"BmmFp8FwdOp batch dim mismatch: a.shape[0]={batch_a} vs b.shape[0]={batch_b}"
            )
        if self.trans_b:
            if b2 != k:
                raise ValueError(
                    f"{type(self).__name__} takes b as [B,N,K], but "
                    f"b={tuple(b.shape)} needs b.shape[2]==K={k}"
                )
            n = b1
        else:  # 'kn'
            if b1 != k:
                raise ValueError(
                    f"BmmFp8FwdOp contraction dim mismatch: a contributes K={k}, "
                    f"but b.shape={tuple(b.shape)} is not a valid [B,K,N] "
                    f"(needs b.shape[1]==K={k})."
                )
            n = b2
        if k % 32 != 0:
            raise ValueError(
                f"BmmFp8FwdOp requires contraction dim K to be a multiple of "
                f"32 (FP8 WGMMA K-step), got K={k}"
            )
        return batch_a, m, n, k

    def _validate_shapes(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
    ) -> Tuple[int, int, int, int]:
        if not a.is_cuda:
            raise ValueError(f"BmmFp8FwdOp expects all inputs to be on CUDA, got device {a.device}")
        if b.device != a.device or scale_a.device != a.device or scale_b.device != a.device:
            raise ValueError(
                f"BmmFp8FwdOp expects all inputs to be on the same CUDA device, got "
                f"a: {a.device}, b: {b.device}, scale_a: {scale_a.device}, "
                f"scale_b: {scale_b.device}"
            )
        batch, m, n, k = self._infer_bmnk(a, b)
        if scale_a.dim() != 0 or scale_b.dim() != 0:
            raise ValueError(
                "BmmFp8FwdOp supports scale shapes ()/() only (per-tensor, "
                "global fp32 scalar shared across the batch, matching "
                "flashinfer.bmm_fp8's A_scale/B_scale), got "
                f"{tuple(scale_a.shape)}/{tuple(scale_b.shape)}"
            )
        return batch, m, n, k

    def _get_kernel(
        self,
        inputs: "tuple[torch.Tensor | None, ...]",
        batch: int,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Kernel:
        return self.kernel_for(
            "bmm_fp8_kernel", inputs, (batch, m, n, k, dtype, self.out_dtype, device)
        )

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

    def _infer_output_shapes(
        self,
        a_shape: tuple[int, ...],
        b_shape: tuple[int, ...],
        scale_a_shape: tuple[int, ...],
        scale_b_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        return {"d": (a_shape[0], a_shape[1], b_shape[1 if self.trans_b else 2])}

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

        Raises:
            ValueError: An input is not on CUDA, a dtype is not the one listed above,
                a scale is not 0-dim, the batch or contraction dims do not match, or
                $K$ is not a multiple of 32 — the FP8 WGMMA K-step.

        Example:
            ```python linenums="1"
            op = BmmFp8FwdOp(out_dtype=torch.bfloat16)      # b as [B, K, N]
            d = op(a, b_kn, scale_a, scale_b)
            flops, nbytes = op.eval_roofline()    # valid after the forward
            ```
        """
        return self._wrapped(a, b, scale_a, scale_b, self._instance_key)

    def _eager_forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
    ) -> torch.Tensor:
        """Validate, resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        sig = (
            a.device,
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
            batch, m, n, k = self._validate_shapes(a, b, scale_a, scale_b)
            self.batch, self.m, self.n, self.k = batch, m, n, k
            self.dtype = a.dtype
            self.a_shape = tuple(a.shape)
            self.b_shape = tuple(b.shape)
            self.scale_a_shape = tuple(scale_a.shape)
            self.scale_b_shape = tuple(scale_b.shape)
            kernel = self._get_kernel(
                (a, b, scale_a, scale_b), batch, m, n, k, a.dtype, device=a.device
            )
            self._active = kernel
            self._active_sig = sig
        b = self._as_k_innermost(b, a.dtype, a.device)
        scale_a = scale_a.reshape(1)
        scale_b = scale_b.reshape(1)
        return self._active(a, b, scale_a, scale_b)

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
        return tensor_core_roof(self.dtype)
