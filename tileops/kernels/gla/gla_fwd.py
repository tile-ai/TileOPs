"""GLA (Gated Linear Attention) Forward Kernel.

Reference:
    https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gla/chunk.py

Algorithm: Chunked GLA forward in 4 stages:
    1. Within-chunk cumulative sum of log-space gates g -> g_cumsum [B, T, H, K]
       (computed in PyTorch inside forward() — sequential scan, not GPU-bound)
    2. Inter-chunk hidden state recurrence -> h [B, NT, H, K, V], ht [B, H, K, V]
       (computed in PyTorch inside forward() — sequential over chunks)
    3. Intra-chunk causal attention matrix -> A [B, T, H, BT]  (TileLang)
    4. Output: o = inter-chunk (q*exp(g_cumsum) @ h) + intra-chunk (A @ v)  (TileLang)

Inputs:
    q  [B, T, H, K]  fp16/bf16  queries
    k  [B, T, H, K]  fp16/bf16  keys
    v  [B, T, H, V]  fp16/bf16  values
    g  [B, T, H, K]  fp16/bf16  log-space forget gates (e.g. F.logsigmoid(...))
    initial_state  [B, H, K, V]  float32  optional initial hidden state

Outputs:
    o  [B, T, H, V]  fp16/bf16
    final_state  [B, H, K, V]  float32  (only when output_final_state=True)
"""

import torch
from typing import Optional, Any, Callable

import tilelang
from tilelang import language as T

from tileops.kernels.kernel import Kernel

LOG2_E = 1.44269504

# ---------------------------------------------------------------------------
# Stage 3: Intra-chunk causal attention matrix  A [B, T, H, BT]
# ---------------------------------------------------------------------------


def _gla_fwd_intra_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    chunk_size: int,
    scale: float,
    dtype: str,
) -> Callable:
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    q_shape = [batch, seq_len, heads, dim_k]
    k_shape = [batch, seq_len, heads, dim_k]
    g_cumsum_shape = [batch, seq_len, heads, dim_k]
    A_shape = [batch, seq_len, heads, chunk_size]

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        })
    def _func(threads: int):

        @T.prim_func
        def _kernel(
                q: T.Tensor(q_shape, dtype),
                k: T.Tensor(k_shape, dtype),
                g_cumsum: T.Tensor(g_cumsum_shape, "float32"),
                A: T.Tensor(A_shape, "float32"),
        ):
            with T.Kernel(batch * heads, num_chunks, threads=threads) as (bx, by):
                i_b = bx // heads
                i_h = bx % heads
                i_c = by
                chunk_start = i_c * chunk_size

                # Shared buffers for inputs
                q_shared = T.alloc_shared([chunk_size, dim_k], dtype)
                k_shared = T.alloc_shared([chunk_size, dim_k], dtype)
                g_shared = T.alloc_shared([chunk_size, dim_k], "float32")

                # Shared buffers for gated q/k (float32 for gemm)
                q_gated = T.alloc_shared([chunk_size, dim_k], "float32")
                k_gated = T.alloc_shared([chunk_size, dim_k], "float32")

                # Fragment accumulator for A [BT, BT]
                acc = T.alloc_fragment([chunk_size, chunk_size], "float32")

                # Load inputs
                T.copy(q[i_b, chunk_start:chunk_start + chunk_size, i_h, :], q_shared)
                T.copy(k[i_b, chunk_start:chunk_start + chunk_size, i_h, :], k_shared)
                T.copy(g_cumsum[i_b, chunk_start:chunk_start + chunk_size, i_h, :], g_shared)

                # q_gated[t, k] = q[t, k] * exp(g_cumsum[t, k])
                # k_gated[t, k] = k[t, k] * exp(-g_cumsum[t, k])
                for i_t, i_k in T.Parallel(chunk_size, dim_k):
                    q_gated[i_t, i_k] = (
                        T.cast(q_shared[i_t, i_k], "float32") * T.exp2(g_shared[i_t, i_k] * LOG2_E))
                    k_gated[i_t, i_k] = (
                        T.cast(k_shared[i_t, i_k], "float32") *
                        T.exp2(-g_shared[i_t, i_k] * LOG2_E))

                # A = q_gated @ k_gated^T  [BT, BT]
                T.fill(acc, 0.0)
                T.gemm(q_gated, k_gated, acc, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # Apply causal mask and scale, write to A
                for i_t, i_j in T.Parallel(chunk_size, chunk_size):
                    A[i_b, chunk_start + i_t, i_h,
                      i_j] = T.if_then_else(i_j <= i_t, acc[i_t, i_j] * scale, 0.0)

        return _kernel

    return _func


@torch.library.custom_op("gla::gla_fwd_intra", mutates_args=())
def _gla_fwd_intra_wrapped(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    chunk_size: int,
    scale: float,
    dtype: str,
    threads: int,
    q: torch.Tensor,
    k: torch.Tensor,
    g_cumsum: torch.Tensor,
) -> torch.Tensor:
    return _gla_fwd_intra_kernel(batch, seq_len, heads, dim_k, chunk_size, scale,
                                 dtype)(threads)(q, k, g_cumsum)


@_gla_fwd_intra_wrapped.register_fake
def _(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    chunk_size: int,
    scale: float,
    dtype: str,
    threads: int,
    *inputs: tuple[Any],
) -> torch.Tensor:
    _ = (dim_k, scale, dtype, threads)
    return torch.empty([batch, seq_len, heads, chunk_size],
                       dtype=torch.float32,
                       device=inputs[0].device)


# ---------------------------------------------------------------------------
# Stage 4: Output computation  o [B, T, H, V]
# ---------------------------------------------------------------------------


def _gla_fwd_o_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    scale: float,
    dtype: str,
) -> Callable:
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    q_shape = [batch, seq_len, heads, dim_k]
    v_shape = [batch, seq_len, heads, dim_v]
    g_cumsum_shape = [batch, seq_len, heads, dim_k]
    A_shape = [batch, seq_len, heads, chunk_size]
    h_shape = [batch, num_chunks, heads, dim_k, dim_v]
    o_shape = [batch, seq_len, heads, dim_v]

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        })
    def _func(threads: int):

        @T.prim_func
        def _kernel(
                q: T.Tensor(q_shape, dtype),
                v: T.Tensor(v_shape, dtype),
                g_cumsum: T.Tensor(g_cumsum_shape, "float32"),
                A: T.Tensor(A_shape, "float32"),
                h: T.Tensor(h_shape, "float32"),
                o: T.Tensor(o_shape, dtype),
        ):
            with T.Kernel(batch * heads, num_chunks, threads=threads) as (bx, by):
                i_b = bx // heads
                i_h = bx % heads
                i_c = by
                chunk_start = i_c * chunk_size

                # Shared buffers for inputs
                q_shared = T.alloc_shared([chunk_size, dim_k], dtype)
                v_shared = T.alloc_shared([chunk_size, dim_v], "float32")
                g_cs_shared = T.alloc_shared([chunk_size, dim_k], "float32")
                A_shared = T.alloc_shared([chunk_size, chunk_size], "float32")
                h_shared = T.alloc_shared([dim_k, dim_v], "float32")

                # Shared buffer for gated q (float32 for gemm)
                q_gated = T.alloc_shared([chunk_size, dim_k], "float32")

                # Fragment accumulator [BT, BV]
                acc = T.alloc_fragment([chunk_size, dim_v], "float32")

                # Load inputs
                T.copy(q[i_b, chunk_start:chunk_start + chunk_size, i_h, :], q_shared)
                T.copy(g_cumsum[i_b, chunk_start:chunk_start + chunk_size, i_h, :], g_cs_shared)
                T.copy(A[i_b, chunk_start:chunk_start + chunk_size, i_h, :], A_shared)
                T.copy(h[i_b, i_c, i_h, :, :], h_shared)

                # Load v as float32
                v_raw = T.alloc_shared([chunk_size, dim_v], dtype)
                T.copy(v[i_b, chunk_start:chunk_start + chunk_size, i_h, :], v_raw)
                for i_t, i_v in T.Parallel(chunk_size, dim_v):
                    v_shared[i_t, i_v] = T.cast(v_raw[i_t, i_v], "float32")

                # q_gated[t, k] = q[t, k] * exp(g_cumsum[t, k])
                for i_t, i_k in T.Parallel(chunk_size, dim_k):
                    q_gated[i_t, i_k] = (
                        T.cast(q_shared[i_t, i_k], "float32") *
                        T.exp2(g_cs_shared[i_t, i_k] * LOG2_E))

                # inter-chunk: acc = scale * q_gated @ h  [BT, BV]
                T.fill(acc, 0.0)
                T.gemm(q_gated, h_shared, acc, policy=T.GemmWarpPolicy.FullRow)
                for i_t, i_v in T.Parallel(chunk_size, dim_v):
                    acc[i_t, i_v] = acc[i_t, i_v] * scale

                # intra-chunk: acc += A @ v  [BT, BV]
                T.gemm(A_shared, v_shared, acc, policy=T.GemmWarpPolicy.FullRow)

                # Write output (cast back to dtype)
                for i_t, i_v in T.Parallel(chunk_size, dim_v):
                    o[i_b, chunk_start + i_t, i_h, i_v] = T.cast(acc[i_t, i_v], dtype)

        return _kernel

    return _func


@torch.library.custom_op("gla::gla_fwd_o", mutates_args=())
def _gla_fwd_o_wrapped(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    scale: float,
    dtype: str,
    threads: int,
    q: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
) -> torch.Tensor:
    return _gla_fwd_o_kernel(batch, seq_len, heads, dim_k, dim_v, chunk_size, scale,
                             dtype)(threads)(q, v, g_cumsum, A, h)


@_gla_fwd_o_wrapped.register_fake
def _(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    scale: float,
    dtype: str,
    threads: int,
    *inputs: tuple[Any],
) -> torch.Tensor:
    _ = (dim_k, chunk_size, scale, dtype, threads)
    return torch.empty([batch, seq_len, heads, dim_v],
                       dtype=inputs[0].dtype,
                       device=inputs[0].device)


# ---------------------------------------------------------------------------
# Kernel class
# ---------------------------------------------------------------------------


class GLAFwdKernel(Kernel):
    """GLA (Gated Linear Attention) forward kernel.

    Implements chunked GLA forward:
        Stage 1 (PyTorch): within-chunk cumulative sum of log-space gates
        Stage 2 (PyTorch): inter-chunk hidden state recurrence
        Stage 3 (TileLang): intra-chunk causal attention matrix A [B, T, H, BT]
        Stage 4 (TileLang): output o = inter-chunk + intra-chunk contributions

    Args:
        batch: Batch size B.
        seq_len: Sequence length T. Must be divisible by chunk_size.
        heads: Number of query heads H.
        dim_k: Key/query head dimension K.
        dim_v: Value head dimension V.
        chunk_size: Chunk size BT (default 64).
        scale: Query scale factor (default 1/sqrt(K)).
        output_final_state: Whether to return the final hidden state.
        dtype: Input tensor dtype (torch.float16 or torch.bfloat16).
        config: Optional kernel config dict (e.g. {"threads": 128}).
        tune: Whether to run autotuning.

    Inputs to forward():
        q  [B, T, H, K]  fp16/bf16
        k  [B, T, H, K]  fp16/bf16
        v  [B, T, H, V]  fp16/bf16
        g  [B, T, H, K]  fp16/bf16  log-space gates
        initial_state  [B, H, K, V]  float32  optional

    Returns:
        (o [B, T, H, V], final_state [B, H, K, V] or None)

    Reference:
        https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gla/chunk.py
    """

    supported_archs: list[int] = [90]

    def __init__(
        self,
        batch: int,
        seq_len: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        chunk_size: int = 64,
        scale: float = -1.0,
        output_final_state: bool = False,
        dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.chunk_size = chunk_size
        self.scale = scale if scale > 0 else dim_k**-0.5
        self.output_final_state = output_final_state
        self.dtype_name = str(dtype).split('.')[-1]
        self.init_config(config, tune)
        # GLAFwdKernel has no single self.kernel to autotune; fall back to default_config
        if not self.config:
            self.config = self.default_config

    @property
    def default_config(self) -> dict:
        return {"threads": 128}

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        dtype_str = self.dtype_name
        threads = self.config["threads"]
        B, T, H, K = self.batch, self.seq_len, self.heads, self.dim_k
        V = self.dim_v
        BT = self.chunk_size
        NT = (T + BT - 1) // BT
        dtype_torch = getattr(torch, dtype_str)

        q = q.to(dtype_torch)
        k = k.to(dtype_torch)
        v = v.to(dtype_torch)
        g = g.to(dtype_torch)

        use_initial_state = initial_state is not None
        if not use_initial_state:
            b_h = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
        else:
            b_h = initial_state.to(torch.float32).clone()

        # Stage 1: within-chunk cumulative sum of gates (PyTorch)
        g_f32 = g.float()
        g_cumsum = torch.empty_like(g_f32)
        for i_c in range(NT):
            cs = i_c * BT
            ce = min(cs + BT, T)
            g_cumsum[:, cs:ce] = torch.cumsum(g_f32[:, cs:ce], dim=1)

        # Stage 2: inter-chunk hidden state recurrence (PyTorch)
        # h_states[b, i_c, h, K, V] = state entering chunk i_c
        h_states = torch.empty(B, NT, H, K, V, dtype=torch.float32, device=q.device)
        for i_c in range(NT):
            cs = i_c * BT
            ce = min(cs + BT, T)
            h_states[:, i_c] = b_h

            # g_last: g_cumsum at last position of this chunk [B, H, K]
            g_last = g_cumsum[:, ce - 1]  # [B, H, K]

            # Decay: b_h[b, h, k, v] *= exp(g_last[b, h, k])
            b_h = b_h * torch.exp(g_last).unsqueeze(-1)

            # Accumulate: b_h += k_adj^T @ v
            # k_adj[b, t, h, k] = k[b, t, h, k] * exp(g_last[b, h, k] - g_cumsum[b, t, h, k])
            k_chunk = k[:, cs:ce].float()  # [B, L, H, K]
            v_chunk = v[:, cs:ce].float()  # [B, L, H, V]
            g_cs_chunk = g_cumsum[:, cs:ce]  # [B, L, H, K]
            k_adj = k_chunk * torch.exp(g_last.unsqueeze(1) - g_cs_chunk)
            b_h = b_h + torch.einsum('blhk,blhv->bhkv', k_adj, v_chunk)

        final_state = b_h if self.output_final_state else None

        # Stage 3: intra-chunk attention matrix (TileLang)
        A = _gla_fwd_intra_wrapped(B, T, H, K, BT, self.scale, dtype_str, threads, q, k, g_cumsum)

        # Stage 4: output (TileLang)
        o = _gla_fwd_o_wrapped(B, T, H, K, V, BT, self.scale, dtype_str, threads, q, v, g_cumsum, A,
                               h_states)

        return o, final_state
