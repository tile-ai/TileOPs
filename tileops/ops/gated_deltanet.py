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


def _resolve_gated_bhsd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
    do: Optional[torch.Tensor] = None,
) -> tuple[int, int, int, int, int, torch.dtype]:
    if not all(tensor.is_cuda for tensor in (q, k, v, g, beta)):
        raise ValueError("q, k, v, g, and beta must be CUDA tensors")
    if q.ndim != 4:
        raise ValueError("q must have shape [batch, heads, seq_len, dim_k]")
    batch, heads, seq_len, dim_k = q.shape
    if k.shape != (batch, heads, seq_len, dim_k):
        raise ValueError("k must match q shape")
    if v.ndim != 4 or v.shape[:3] != (batch, heads, seq_len):
        raise ValueError("v must have shape [batch, heads, seq_len, dim_v]")
    dim_v = v.shape[-1]
    if g.shape != (batch, heads, seq_len):
        raise ValueError("g must have shape [batch, heads, seq_len]")
    if beta.shape != (batch, heads, seq_len):
        raise ValueError("beta must have shape [batch, heads, seq_len]")
    if do is not None and do.shape != (batch, heads, seq_len, dim_v):
        raise ValueError("do must have shape [batch, heads, seq_len, dim_v]")
    dtype = q.dtype
    for name, tensor in (("k", k), ("v", v), ("g", g), ("beta", beta)):
        if tensor.dtype != dtype:
            raise ValueError(f"{name}.dtype must be {dtype}, got {tensor.dtype}")
    if do is not None and do.dtype != dtype:
        raise ValueError(f"do.dtype must be {dtype}, got {do.dtype}")
    if seq_len % chunk_size != 0:
        raise ValueError(f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})")
    return batch, heads, seq_len, dim_k, dim_v, dtype


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
        chunk_size: int = 64,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.batch = None
        self.heads = None
        self.seq_len = None
        self.dim_k = None
        self.dim_v = None
        self._requested_chunk_size = chunk_size
        self.chunk_size = chunk_size
        self.dtype = None
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "GatedDeltaNetFwdKernel": GatedDeltaNetFwdKernel,
        }

    def _get_kernel(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (batch, heads, seq_len, self.chunk_size, dim_k, dim_v, dtype, device_index, self.tune)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["GatedDeltaNetFwdKernel"](
                batch,
                heads,
                seq_len,
                self.chunk_size,
                dim_k,
                dim_v,
                dtype=Kernel.dtype_to_str(dtype),
                tune=self.tune,
            )
        return self._kernel_cache[key]

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
        batch, heads, seq_len, dim_k, dim_v, dtype = _resolve_gated_bhsd(
            q, k, v, g, beta, self.chunk_size)
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        self.kernel = self._get_kernel(
            batch, heads, seq_len, dim_k, dim_v, dtype, q.device.index)
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
        chunk_size: Optional[int] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
        layout: str = "bthd",
    ) -> None:
        layout = self._normalize_layout(layout)
        self.batch = None
        self.heads = None
        self.seq_len = None
        self.dim_k = None
        self.dim_v = None
        self.chunk_size = chunk_size
        self.dtype = None
        self.layout = layout
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel = None

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
        layout = self._normalize_layout(getattr(self, "layout", "bthd"))
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
        dtype = q.dtype
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"Unsupported dtype: {dtype}")
        for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
            if tensor.dtype != dtype:
                raise ValueError(f"{name}.dtype must be {dtype}, got {tensor.dtype}")

    def _get_kernel(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        chunk_size: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (
            batch,
            heads,
            seq_len,
            chunk_size,
            dim_k,
            dim_v,
            dtype,
            self.layout,
            device_index,
            self.tune,
        )
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["GatedDeltaNetPrefillFwdKernel"](
                batch,
                heads,
                seq_len,
                chunk_size,
                dim_k,
                dim_v,
                dtype=Kernel.dtype_to_str(dtype),
                layout=self.layout,
                tune=self.tune,
            )
        return self._kernel_cache[key]

    def _validate_shapes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> None:
        if self.layout == "bthd":
            if q.ndim != 4:
                raise ValueError("q must have shape [batch, seq_len, heads, dim_k]")
            batch, seq_len, heads, dim_k = q.shape
            q_shape = (batch, seq_len, heads, dim_k)
            v_shape = (batch, seq_len, heads, v.shape[-1])
            gate_shape = (batch, seq_len, heads)
        else:
            if q.ndim != 4:
                raise ValueError("q must have shape [batch, heads, seq_len, dim_k]")
            batch, heads, seq_len, dim_k = q.shape
            q_shape = (batch, heads, seq_len, dim_k)
            v_shape = (batch, heads, seq_len, v.shape[-1])
            gate_shape = (batch, heads, seq_len)
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
        chunk_size = self._requested_chunk_size
        if chunk_size is None:
            streams = batch * heads
            chunk_size = 128 if streams <= 8 and seq_len % 128 == 0 else 64
        if seq_len % chunk_size != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})"
            )
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim_k = dim_k
        self.dim_v = v.shape[-1]
        self.chunk_size = chunk_size
        self.dtype = q.dtype
        self.kernel = self._get_kernel(
            batch, heads, seq_len, chunk_size, dim_k, self.dim_v, q.dtype, q.device.index)

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
        chunk_size: int = 64,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.batch = None
        self.heads = None
        self.seq_len = None
        self.dim_k = None
        self.dim_v = None
        self.chunk_size = chunk_size
        self.dtype = None
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "GatedDeltaNetBwdKernel": GatedDeltaNetBwdKernel,
        }

    def _get_kernel(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (batch, heads, seq_len, self.chunk_size, dim_k, dim_v, dtype, device_index, self.tune)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["GatedDeltaNetBwdKernel"](
                batch,
                heads,
                seq_len,
                self.chunk_size,
                dim_k,
                dim_v,
                dtype=Kernel.dtype_to_str(dtype),
                tune=self.tune,
            )
        return self._kernel_cache[key]

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
        batch, heads, seq_len, dim_k, dim_v, dtype = _resolve_gated_bhsd(
            q, k, v, g, beta, self.chunk_size, do=do)
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        self.kernel = self._get_kernel(
            batch, heads, seq_len, dim_k, dim_v, dtype, q.device.index)
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

        op = GatedDeltaNetOp(chunk_size=chunk_size)
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
        chunk_size: int = 64,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.batch = None
        self.heads = None
        self.seq_len = None
        self.dim_k = None
        self.dim_v = None
        self.chunk_size = chunk_size
        self.dtype = None
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self._fwd_kernel_cache: Dict[tuple, Kernel] = {}
        self._bwd_kernel_cache: Dict[tuple, Kernel] = {}
        self.fwd_kernel = None
        self.bwd_kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "GatedDeltaNetFwdKernel": GatedDeltaNetFwdKernel,
            "GatedDeltaNetBwdKernel": GatedDeltaNetBwdKernel,
        }

    def _bind_from_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> None:
        batch, heads, seq_len, dim_k, dim_v, dtype = _resolve_gated_bhsd(
            q, k, v, g, beta, self.chunk_size)
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        key = (
            batch,
            heads,
            seq_len,
            self.chunk_size,
            dim_k,
            dim_v,
            dtype,
            q.device.index,
            self.tune,
        )
        if key not in self._fwd_kernel_cache:
            kernel_dtype = Kernel.dtype_to_str(dtype)
            self._fwd_kernel_cache[key] = self.kernel_map["GatedDeltaNetFwdKernel"](
                batch, heads, seq_len, self.chunk_size, dim_k, dim_v,
                dtype=kernel_dtype, tune=self.tune)
            self._bwd_kernel_cache[key] = self.kernel_map["GatedDeltaNetBwdKernel"](
                batch, heads, seq_len, self.chunk_size, dim_k, dim_v,
                dtype=kernel_dtype, tune=self.tune)
        self.fwd_kernel = self._fwd_kernel_cache[key]
        self.bwd_kernel = self._bwd_kernel_cache[key]

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
        self._bind_from_inputs(q, k, v, g, beta)
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
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.batch = None
        self.heads = None
        self.dim_k = None
        self.dim_v = None
        self.dtype = None
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel = None

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

    def _get_kernel(
        self,
        batch: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (batch, heads, dim_k, dim_v, dtype, device_index, self.tune)
        if key not in self._kernel_cache:
            if dtype == torch.float32:
                kernel_cls = self.kernel_map["GatedDeltaNetDecodeFP32Kernel"]
            elif self._should_use_raw_cuda_decode(dim_k, dim_v, dtype, self.tune):
                kernel_cls = self.kernel_map["GatedDeltaNetDecodeRawCudaFlaStyleKernel"]
            else:
                kernel_cls = self.kernel_map["GatedDeltaNetDecodeKernel"]
            self._kernel_cache[key] = kernel_cls(
                batch,
                heads,
                dim_k,
                dim_v,
                dtype=Kernel.dtype_to_str(dtype),
                tune=self.tune,
            )
        return self._kernel_cache[key]

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        g_shape: tuple[int, ...],
        beta_shape: tuple[int, ...],
        state_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        del k_shape, g_shape, beta_shape
        return {
            "o": (q_shape[0], q_shape[1], v_shape[-1]),
            "new_state": state_shape,
        }

    def _validate_dtypes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> None:
        dtype = q.dtype
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"Unsupported dtype: {dtype}")
        for name, tensor in (
            ("q", q),
            ("k", k),
            ("v", v),
            ("g", g),
            ("beta", beta),
            ("state", state),
        ):
            if tensor.dtype != dtype:
                raise ValueError(f"{name}.dtype must be {dtype}, got {tensor.dtype}")

    def _validate_shapes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> None:
        if q.ndim != 3:
            raise ValueError("q must have shape [batch, heads, dim_k]")
        batch, heads, dim_k = q.shape
        if v.ndim != 3 or v.shape[:2] != (batch, heads):
            raise ValueError("v must have shape [batch, heads, dim_v]")
        dim_v = v.shape[2]
        q_shape = (batch, heads, dim_k)
        v_shape = (batch, heads, dim_v)
        gate_shape = (batch, heads)
        state_shape = (batch, heads, dim_k, dim_v)
        expected_shapes = (
            ("q", q, q_shape),
            ("k", k, q_shape),
            ("v", v, v_shape),
            ("g", g, gate_shape),
            ("beta", beta, gate_shape),
            ("state", state, state_shape),
        )
        for name, tensor, expected in expected_shapes:
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {tuple(tensor.shape)}"
        )
        if not all(tensor.is_cuda for tensor in (q, k, v, g, beta, state)):
            raise ValueError("q, k, v, g, beta, and state must be CUDA tensors")
        self.batch = batch
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = q.dtype
        self.kernel = self._get_kernel(batch, heads, dim_k, dim_v, q.dtype, q.device.index)

    def _validate_output_shapes(
        self,
        o: torch.Tensor,
        new_state: torch.Tensor,
    ) -> None:
        o_shape = (self.batch, self.heads, self.dim_v)
        state_shape = (self.batch, self.heads, self.dim_k, self.dim_v)
        if tuple(o.shape) != o_shape:
            raise ValueError(f"o must have shape {o_shape}, got {tuple(o.shape)}")
        if tuple(new_state.shape) != state_shape:
            raise ValueError(
                f"new_state must have shape {state_shape}, got {tuple(new_state.shape)}"
            )

    def eval_roofline(self) -> tuple[int, int]:
        from tileops.perf.formulas import gated_deltanet_decode_roofline

        return gated_deltanet_decode_roofline(self)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate_dtypes(q, k, v, g, beta, state)
        self._validate_shapes(q, k, v, g, beta, state)
        o, new_state = self.kernel(q, k, v, g, beta, state)
        self._validate_output_shapes(o, new_state)
        return o, new_state
