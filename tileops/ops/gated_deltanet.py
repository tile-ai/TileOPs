from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.gated_deltanet import (
    GatedDeltaNetBwdKernel,
    GatedDeltaNetFwdKernel,
    GatedDeltaNetPrefillFwdKernel,
)
from tileops.kernels.gated_deltanet_recurrence import (
    GatedDeltaNetDecodeFP32Kernel,
    GatedDeltaNetDecodeKernel,
    GatedDeltaNetDecodeRawCudaFlaStyleKernel,
)
from tileops.kernels.kernel_base import Kernel
from tileops.utils import get_sm_version

from .op_base import Op

__all__ = [
    "GatedDeltaNetBwdOp",
    "GatedDeltaNetDecodeOp",
    "GatedDeltaNetFwdOp",
    "GatedDeltaNetOp",
    "GatedDeltaNetPrefillFwdOp",
]


class GatedDeltaNetFwdOp(Op):
    """Gated DeltaNet forward operator.

    Pipeline: prepare_wy_repr(k, g, beta) -> (Aw, Au) -> gated_deltanet_fwd(q, k, v, g, beta, Aw, Au) -> o.

    Layout: BHSD (batch, head, seq_len, dim).

    .. note:: Layout convention difference with FLA

        TileOPs uses **BHSD** layout: ``q/k [B, H, S, DK]``, ``v [B, H, S, DV]``,
        ``g/beta [B, H, S]``.

        FLA (``fla.ops.gated_delta_rule.chunk_gated_delta_rule``) uses **BTHN**
        layout: ``q/k [B, T, H, K]``, ``v [B, T, H, V]``, ``g/beta [B, T, H]``.

        When comparing against FLA, tensors must be transposed::

            # TileOPs BHSD -> FLA BTHK
            q_fla = q.permute(0, 2, 1, 3)   # [B, H, S, DK] -> [B, S, H, DK]
            g_fla = g.permute(0, 2, 1)       # [B, H, S]     -> [B, S, H]

            # FLA BTHV -> TileOPs BHSD
            o_tileops = o_fla.permute(0, 2, 1, 3)  # [B, S, H, DV] -> [B, H, S, DV]

    Args:
        batch: Batch size.
        heads: Number of attention heads.
        seq_len: Sequence length (must be divisible by chunk_size).
        dim_k: Key/query dimension.
        dim_v: Value dimension.
        chunk_size: Chunk size for chunked linear attention.
        dtype: Data type for computation.
        kernel_map: Optional kernel overrides.
        tune: Whether to autotune kernels.
    """

    def __init__(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        dim_k: int,
        dim_v: int,
        chunk_size: int = 64,
        dtype: torch.dtype = torch.float32,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.chunk_size = chunk_size
        self.dtype = dtype

        if seq_len % chunk_size != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})"
            )

        self.dispatch_kernel(kernel_map)

        fwd_kernel_cls = self.kernel_map["GatedDeltaNetFwdKernel"]

        kernel_dtype = Kernel.dtype_to_str(dtype)
        self.kernel = fwd_kernel_cls(
            batch, heads, seq_len, chunk_size, dim_k, dim_v,
            dtype=kernel_dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "GatedDeltaNetFwdKernel": GatedDeltaNetFwdKernel,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Run gated deltanet forward.

        Args:
            q: Query tensor [B, H, S, DK].
            k: Key tensor [B, H, S, DK].
            v: Value tensor [B, H, S, DV].
            g: Gate tensor [B, H, S].
            beta: Beta tensor [B, H, S].

        Returns:
            Tuple of (o, S, Aw, Au).
        """
        o, S, Aw, Au = self.kernel(q, k, v, g, beta)
        return o, S, Aw, Au


class GatedDeltaNetPrefillFwdOp(Op):
    """Gated DeltaNet inference prefill operator.

    This is the serving-oriented zero-state prefill interface:
    ``(q, k, v, g, beta) -> (o, final_state)``. It intentionally does not
    expose backward-only training artifacts such as ``Aw`` and ``Au``.
    ``layout="bthd"`` follows the official FLA/Qwen convention
    (``q/k/v/o [B, T, H, D]``, ``g/beta [B, T, H]``). ``layout="bhtd"``
    selects the TileOps head-major convention (``q/k/v/o [B, H, T, D]``,
    ``g/beta [B, H, T]``). ``layout="bhsd"`` is accepted as a backward-compatible
    alias for ``"bhtd"``.
    When ``chunk_size`` is not specified, the op uses a small-stream serving
    default: 128 for ``batch * heads <= 8`` when the sequence length allows it,
    otherwise 64.
    """

    def __init__(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        dim_k: int,
        dim_v: int,
        chunk_size: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
        layout: str = "bhtd",
    ) -> None:
        layout = self._normalize_layout(layout)
        if chunk_size is None:
            streams = batch * heads
            chunk_size = 128 if streams <= 8 and seq_len % 128 == 0 else 64

        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.layout = layout

        if seq_len % chunk_size != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})"
            )

        self.dispatch_kernel(kernel_map)

        kernel_cls = self.kernel_map["GatedDeltaNetPrefillFwdKernel"]
        kernel_dtype = Kernel.dtype_to_str(dtype)
        self.kernel = kernel_cls(
            batch,
            heads,
            seq_len,
            chunk_size,
            dim_k,
            dim_v,
            dtype=kernel_dtype,
            layout=layout,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "GatedDeltaNetPrefillFwdKernel": GatedDeltaNetPrefillFwdKernel,
        }

    @staticmethod
    def _normalize_layout(layout: str) -> str:
        layout = layout.lower()
        if layout == "bhsd":
            return "bhtd"
        if layout in ("bhtd", "bthd"):
            return layout
        raise ValueError(f"Unsupported layout: {layout}")

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        g_shape: tuple[int, ...],
        beta_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        del k_shape, g_shape, beta_shape
        layout = self._normalize_layout(getattr(self, "layout", "bhtd"))
        if layout == "bthd":
            return {
                "o": (q_shape[0], q_shape[1], q_shape[2], v_shape[-1]),
                "final_state": (
                    q_shape[0],
                    q_shape[2],
                    q_shape[-1],
                    v_shape[-1],
                ),
            }
        return {
            "o": tuple(q_shape[:-1]) + (v_shape[-1],),
            "final_state": (
                q_shape[0],
                q_shape[1],
                q_shape[-1],
                v_shape[-1],
            ),
        }

    def _validate_dtypes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> None:
        if self.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
            if tensor.dtype != self.dtype:
                raise ValueError(f"{name}.dtype must be {self.dtype}, got {tensor.dtype}")

    def _validate_shapes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> None:
        if self.layout == "bthd":
            q_shape = (self.batch, self.seq_len, self.heads, self.dim_k)
            v_shape = (self.batch, self.seq_len, self.heads, self.dim_v)
            gate_shape = (self.batch, self.seq_len, self.heads)
        else:
            q_shape = (self.batch, self.heads, self.seq_len, self.dim_k)
            v_shape = (self.batch, self.heads, self.seq_len, self.dim_v)
            gate_shape = (self.batch, self.heads, self.seq_len)
        if tuple(q.shape) != q_shape:
            raise ValueError(f"q must have shape {q_shape}, got {tuple(q.shape)}")
        if tuple(k.shape) != q_shape:
            raise ValueError(f"k must have shape {q_shape}, got {tuple(k.shape)}")
        if tuple(v.shape) != v_shape:
            raise ValueError(f"v must have shape {v_shape}, got {tuple(v.shape)}")
        if tuple(g.shape) != gate_shape:
            raise ValueError(f"g must have shape {gate_shape}, got {tuple(g.shape)}")
        if tuple(beta.shape) != gate_shape:
            raise ValueError(f"beta must have shape {gate_shape}, got {tuple(beta.shape)}")
        if not all(tensor.is_cuda for tensor in (q, k, v, g, beta)):
            raise ValueError("q, k, v, g, and beta must be CUDA tensors")

    def eval_roofline(self) -> tuple[int, int]:
        from tileops.perf.formulas import gated_deltanet_prefill_fwd_roofline

        return gated_deltanet_prefill_fwd_roofline(self)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous()
        beta = beta.contiguous()
        self._validate_dtypes(q, k, v, g, beta)
        self._validate_shapes(q, k, v, g, beta)
        return self.kernel(q, k, v, g, beta)


class GatedDeltaNetBwdOp(Op):
    """Gated DeltaNet backward operator.

    Pipeline: prepare_wy_repr -> fwd (to get Aw, Au) -> bwd kernel -> (dq, dk, dv, dg, dbeta).

    Args:
        batch: Batch size.
        heads: Number of attention heads.
        seq_len: Sequence length (must be divisible by chunk_size).
        dim_k: Key/query dimension.
        dim_v: Value dimension.
        chunk_size: Chunk size for chunked linear attention.
        dtype: Data type for computation.
        kernel_map: Optional kernel overrides.
        tune: Whether to autotune kernels.
    """

    def __init__(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        dim_k: int,
        dim_v: int,
        chunk_size: int = 64,
        dtype: torch.dtype = torch.float32,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.chunk_size = chunk_size
        self.dtype = dtype

        if seq_len % chunk_size != 0:
            raise ValueError(f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})")

        self.dispatch_kernel(kernel_map)

        bwd_kernel_cls = self.kernel_map["GatedDeltaNetBwdKernel"]

        kernel_dtype = Kernel.dtype_to_str(dtype)
        self.kernel = bwd_kernel_cls(
            batch, heads, seq_len, chunk_size, dim_k, dim_v,
            dtype=kernel_dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "GatedDeltaNetBwdKernel": GatedDeltaNetBwdKernel,
        }

    def forward(
        self,
        do: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        S: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run gated deltanet backward.

        Args:
            do: Gradient of output [B, H, S, DV].
            q: Query tensor [B, H, S, DK].
            k: Key tensor [B, H, S, DK].
            v: Value tensor [B, H, S, DV].
            g: Gate tensor [B, H, S].
            beta: Beta tensor [B, H, S].
            S: Per-chunk boundary states from forward [B, H, NC+1, DK, DV].

        Returns:
            Tuple of (dq, dk, dv, dg, dbeta).
        """
        dq, dk, dv, dg, dbeta = self.kernel(do, q, k, v, g, beta, S)
        return dq, dk, dv, dg, dbeta


class _GatedDeltaNetFunction(torch.autograd.Function):
    """Autograd function wrapping TileOPs fwd + bwd kernels."""

    @staticmethod
    def forward(ctx, q, k, v, g, beta, fwd_kernel, bwd_kernel):
        o, S, Aw, Au = fwd_kernel(q, k, v, g, beta)
        ctx.save_for_backward(q, k, v, g, beta, S)
        ctx.bwd_kernel = bwd_kernel
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, g, beta, S = ctx.saved_tensors
        dq, dk, dv, dg, dbeta = ctx.bwd_kernel(do, q, k, v, g, beta, S)
        return dq, dk, dv, dg, dbeta, None, None


class GatedDeltaNetOp(Op):
    """Combined Gated DeltaNet fwd+bwd operator with autograd support.

    Wraps ``GatedDeltaNetFwdKernel`` and ``GatedDeltaNetBwdKernel`` in a
    ``torch.autograd.Function`` so that ``output.backward(do)`` automatically
    invokes the TileOPs backward kernels.

    This makes end-to-end benchmarking against FLA straightforward::

        op = GatedDeltaNetOp(B, H, S, DK, DV, chunk_size, dtype)
        o = op(q, k, v, g, beta)   # forward
        o.backward(do)              # backward via TileOPs kernels

    Layout: BHSD (batch, head, seq_len, dim).

    Args:
        batch: Batch size.
        heads: Number of attention heads.
        seq_len: Sequence length (must be divisible by chunk_size).
        dim_k: Key/query dimension.
        dim_v: Value dimension.
        chunk_size: Chunk size for chunked linear attention.
        dtype: Data type for computation.
        kernel_map: Optional kernel overrides.
        tune: Whether to autotune kernels.
    """

    def __init__(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        dim_k: int,
        dim_v: int,
        chunk_size: int = 64,
        dtype: torch.dtype = torch.float32,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.chunk_size = chunk_size
        self.dtype = dtype

        assert seq_len % chunk_size == 0, (
            f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})"
        )

        self.dispatch_kernel(kernel_map)

        kernel_dtype = Kernel.dtype_to_str(dtype)
        fwd_cls = self.kernel_map["GatedDeltaNetFwdKernel"]
        bwd_cls = self.kernel_map["GatedDeltaNetBwdKernel"]
        self.fwd_kernel = fwd_cls(
            batch, heads, seq_len, chunk_size, dim_k, dim_v,
            dtype=kernel_dtype, tune=tune,
        )
        self.bwd_kernel = bwd_cls(
            batch, heads, seq_len, chunk_size, dim_k, dim_v,
            dtype=kernel_dtype, tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "GatedDeltaNetFwdKernel": GatedDeltaNetFwdKernel,
            "GatedDeltaNetBwdKernel": GatedDeltaNetBwdKernel,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Run gated deltanet forward with autograd backward support.

        Args:
            q: Query tensor [B, H, S, DK].
            k: Key tensor [B, H, S, DK].
            v: Value tensor [B, H, S, DV].
            g: Gate tensor [B, H, S].
            beta: Beta tensor [B, H, S].

        Returns:
            Output tensor o [B, H, S, DV] (supports .backward()).
        """
        return _GatedDeltaNetFunction.apply(
            q, k, v, g, beta, self.fwd_kernel, self.bwd_kernel,
        )


class GatedDeltaNetDecodeOp(Op):
    """Gated DeltaNet decode (single-step recurrence).

    Computes one step of the gated delta rule:
        S_t = S_{t-1} (alpha_t (I - beta_t k_t k_t^T)) + beta_t v_t k_t^T
        o_t = S_t q_t

    Layout: BHD (batch, head, dim).
    Supports float32, float16, and bfloat16 with fp32 accumulation.

    For fp32 dtype, dispatches to a dedicated FP32 kernel that uses
    element-wise matvec instead of T.gemm to avoid TF32 mantissa truncation.
    """

    @staticmethod
    def _raw_cuda_decode_arch_supported() -> bool:
        try:
            sm_version = get_sm_version()
        except Exception:
            return False
        return sm_version in GatedDeltaNetDecodeRawCudaFlaStyleKernel.supported_archs

    @staticmethod
    def _should_use_raw_cuda_decode(
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype,
        tune: bool,
    ) -> bool:
        if tune or dtype != torch.bfloat16 or dim_k != 128 or dim_v != 128:
            return False
        return GatedDeltaNetDecodeOp._raw_cuda_decode_arch_supported()

    def __init__(
        self,
        batch: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype = torch.float32,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)

        # Dispatch:
        #   fp32 -> FP32 kernel (no TF32)
        #   bf16 DK=DV=128 on Hopper -> raw CUDA warp-per-Vtile kernel
        #   other fp16/bf16 shapes -> default TileLang kernel
        if dtype == torch.float32:
            kernel_cls = self.kernel_map["GatedDeltaNetDecodeFP32Kernel"]
        elif self._should_use_raw_cuda_decode(dim_k, dim_v, dtype, tune):
            kernel_cls = self.kernel_map["GatedDeltaNetDecodeRawCudaFlaStyleKernel"]
        else:
            kernel_cls = self.kernel_map["GatedDeltaNetDecodeKernel"]
        kernel_dtype = Kernel.dtype_to_str(dtype)
        self.kernel = kernel_cls(
            batch, heads, dim_k, dim_v,
            dtype=kernel_dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        kernels = {
            "GatedDeltaNetDecodeKernel": GatedDeltaNetDecodeKernel,
            "GatedDeltaNetDecodeFP32Kernel": GatedDeltaNetDecodeFP32Kernel,
        }
        if self._raw_cuda_decode_arch_supported():
            kernels["GatedDeltaNetDecodeRawCudaFlaStyleKernel"] = (
                GatedDeltaNetDecodeRawCudaFlaStyleKernel
            )
        return kernels

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.kernel(q, k, v, g, beta, state)
