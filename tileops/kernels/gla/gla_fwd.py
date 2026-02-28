import torch
from typing import Optional, Any, Callable

import tilelang
from tilelang import language as T

from tileops.kernels.kernel import Kernel

LOG2_E = 1.44269504


def _gla_fwd_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    scale: float,
    dtype: str,
) -> Callable:
    """GLA (Gated Linear Attention) forward kernel.

    Implements chunked GLA forward in 4 stages within a single @T.prim_func:
        Stage 1: within-chunk cumulative sum of log-space gates -> g_cumsum
        Stage 3: intra-chunk causal attention matrix A = q_gated @ k_gated^T
        Stage 4: output o = scale * q_gated @ h_state + A @ v
        Stage 2: inter-chunk hidden state recurrence -> h_state (carried across chunks)

    Stages run in order 1→3→4→2 so that stage4 reads h_s before stage2 decays it.

    Args:
        q  [B, T, H, K]  fp16/bf16  queries
        k  [B, T, H, K]  fp16/bf16  keys
        v  [B, T, H, V]  fp16/bf16  values
        g  [B, T, H, K]  fp16/bf16  log-space forget gates
        initial_state  [B, H, K, V]  float32  initial hidden state (zeros if unused)
        h_out  [B, NT, H, K, V]  float32  per-chunk hidden states (output)
        o  [B, T, H, V]  fp16/bf16  output

    Reference:
        https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gla/chunk.py
    """
    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        })
    def _gla_fwd_func(threads: int):
        # Shape lists defined inside _gla_fwd_func so the outer closure only
        # captures serializable scalars (int/float/str), satisfying tilelang autotuner.
        accum_dtype = "float32"
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        q_shape = [batch, seq_len, heads, dim_k]
        k_shape = [batch, seq_len, heads, dim_k]
        v_shape = [batch, seq_len, heads, dim_v]
        g_shape = [batch, seq_len, heads, dim_k]
        init_state_shape = [batch, heads, dim_k, dim_v]
        h_out_shape = [batch, num_chunks, heads, dim_k, dim_v]
        o_shape = [batch, seq_len, heads, dim_v]

        @T.macro
        def stage1_cumsum(
            g: T.Tensor(g_shape, dtype),
            i_b: int,
            i_h: int,
            i_c: int,
            g_cumsum_s: T.Buffer([chunk_size, dim_k], accum_dtype),
        ):
            """Within-chunk cumulative sum of log-space gates.
            Reads g directly from global memory to avoid an extra shared buffer.
            """
            chunk_start = i_c * chunk_size
            for i_k in T.Parallel(dim_k):
                g_cumsum_s[0, i_k] = T.cast(g[i_b, chunk_start, i_h, i_k], accum_dtype)
            for i_t in T.Serial(1, chunk_size):
                for i_k in T.Parallel(dim_k):
                    g_cumsum_s[i_t, i_k] = g_cumsum_s[i_t - 1, i_k] + T.cast(
                        g[i_b, chunk_start + i_t, i_h, i_k], accum_dtype)

        @T.macro
        def stage3_intra(
            q: T.Tensor(q_shape, dtype),
            k: T.Tensor(k_shape, dtype),
            g_cumsum_s: T.Buffer([chunk_size, dim_k], accum_dtype),
            A_s: T.Buffer([chunk_size, chunk_size], accum_dtype),
            qf32_s: T.Buffer([chunk_size, dim_k], accum_dtype),
            kf32_s: T.Buffer([chunk_size, dim_k], accum_dtype),
            i_b: int,
            i_h: int,
            i_c: int,
        ):
            """Intra-chunk causal attention matrix A = q_gated @ k_gated^T.
            Reads q and k directly from global memory into qf32_s / kf32_s.
            """
            chunk_start = i_c * chunk_size
            for i_t, i_k in T.Parallel(chunk_size, dim_k):
                qf32_s[i_t, i_k] = T.cast(q[i_b, chunk_start + i_t, i_h, i_k],
                                           accum_dtype) * T.exp2(
                                               g_cumsum_s[i_t, i_k] * LOG2_E)
                kf32_s[i_t, i_k] = T.cast(k[i_b, chunk_start + i_t, i_h, i_k],
                                           accum_dtype) * T.exp2(
                                               -g_cumsum_s[i_t, i_k] * LOG2_E)

            A_frag = T.alloc_fragment([chunk_size, chunk_size], accum_dtype)
            T.fill(A_frag, 0.0)
            T.gemm(qf32_s, kf32_s, A_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

            for i_t, i_j in T.Parallel(chunk_size, chunk_size):
                A_s[i_t, i_j] = T.if_then_else(i_j <= i_t, A_frag[i_t, i_j] * scale, 0.0)

        @T.macro
        def stage4_output(
            q: T.Tensor(q_shape, dtype),
            v: T.Tensor(v_shape, dtype),
            g_cumsum_s: T.Buffer([chunk_size, dim_k], accum_dtype),
            A_s: T.Buffer([chunk_size, chunk_size], accum_dtype),
            h_s: T.Buffer([dim_k, dim_v], accum_dtype),
            o: T.Tensor(o_shape, dtype),
            qf32_s: T.Buffer([chunk_size, dim_k], accum_dtype),
            vf32_s: T.Buffer([chunk_size, dim_v], accum_dtype),
            i_b: int,
            i_h: int,
            i_c: int,
        ):
            """Output: o = scale * q_gated @ h_s + A @ v.
            Called before stage2 so h_s is the pre-decay state entering this chunk.
            Reads q and v directly from global memory.
            """
            chunk_start = i_c * chunk_size
            for i_t, i_k in T.Parallel(chunk_size, dim_k):
                qf32_s[i_t, i_k] = T.cast(q[i_b, chunk_start + i_t, i_h, i_k],
                                           accum_dtype) * T.exp2(
                                               g_cumsum_s[i_t, i_k] * LOG2_E)
            for i_t, i_v in T.Parallel(chunk_size, dim_v):
                vf32_s[i_t, i_v] = T.cast(v[i_b, chunk_start + i_t, i_h, i_v], accum_dtype)

            acc = T.alloc_fragment([chunk_size, dim_v], accum_dtype)
            T.fill(acc, 0.0)
            T.gemm(qf32_s, h_s, acc, policy=T.GemmWarpPolicy.FullRow)
            for i_t, i_v in T.Parallel(chunk_size, dim_v):
                acc[i_t, i_v] = acc[i_t, i_v] * scale
            T.gemm(A_s, vf32_s, acc, policy=T.GemmWarpPolicy.FullRow)

            for i_t, i_v in T.Parallel(chunk_size, dim_v):
                o[i_b, chunk_start + i_t, i_h, i_v] = T.cast(acc[i_t, i_v], dtype)

        @T.macro
        def stage2_recurrence(
            k: T.Tensor(k_shape, dtype),
            v: T.Tensor(v_shape, dtype),
            g_cumsum_s: T.Buffer([chunk_size, dim_k], accum_dtype),
            h_s: T.Buffer([dim_k, dim_v], accum_dtype),
            h_out: T.Tensor(h_out_shape, accum_dtype),
            kf32_s: T.Buffer([chunk_size, dim_k], accum_dtype),
            vf32_s: T.Buffer([chunk_size, dim_v], accum_dtype),
            i_b: int,
            i_h: int,
            i_c: int,
        ):
            """Inter-chunk hidden state recurrence. h_s is carried across chunks.
            Saves pre-decay h_s to h_out, then decays and accumulates.
            Reads k and v directly from global memory.
            """
            chunk_start = i_c * chunk_size
            # Save h entering this chunk to h_out
            T.copy(h_s, h_out[i_b, i_c, i_h, :, :])

            g_last = T.alloc_fragment([dim_k], accum_dtype)
            for i_k in T.Parallel(dim_k):
                g_last[i_k] = g_cumsum_s[chunk_size - 1, i_k]

            # Decay h: h[k, v] *= exp(g_last[k])
            for i_k, i_v in T.Parallel(dim_k, dim_v):
                h_s[i_k, i_v] = h_s[i_k, i_v] * T.exp2(g_last[i_k] * LOG2_E)

            # k_adj[t, k] = k[t, k] * exp(g_last[k] - g_cumsum[t, k])
            for i_t, i_k in T.Parallel(chunk_size, dim_k):
                kf32_s[i_t, i_k] = T.cast(k[i_b, chunk_start + i_t, i_h, i_k],
                                           accum_dtype) * T.exp2(
                                               (g_last[i_k] - g_cumsum_s[i_t, i_k]) * LOG2_E)
            for i_t, i_v in T.Parallel(chunk_size, dim_v):
                vf32_s[i_t, i_v] = T.cast(v[i_b, chunk_start + i_t, i_h, i_v], accum_dtype)

            # h += k_adj^T @ v  [K, V]
            delta_h = T.alloc_fragment([dim_k, dim_v], accum_dtype)
            T.fill(delta_h, 0.0)
            T.gemm(kf32_s, vf32_s, delta_h, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
            for i_k, i_v in T.Parallel(dim_k, dim_v):
                h_s[i_k, i_v] = h_s[i_k, i_v] + delta_h[i_k, i_v]

        @T.prim_func
        def _main(
            q: T.Tensor(q_shape, dtype),
            k: T.Tensor(k_shape, dtype),
            v: T.Tensor(v_shape, dtype),
            g: T.Tensor(g_shape, dtype),
            initial_state: T.Tensor(init_state_shape, accum_dtype),
            h_out: T.Tensor(h_out_shape, accum_dtype),
            o: T.Tensor(o_shape, dtype),
        ):
            with T.Kernel(batch * heads, threads=threads) as bx:
                i_b = bx // heads
                i_h = bx % heads

                # Persistent buffers across the chunk loop.
                # Shared memory budget (dim_k=128, dim_v=128, chunk_size=64):
                #   h_s(65536) + g_cumsum_s(32768) + A_s(16384)
                #   + qf32_s(32768) + kf32_s(32768) + vf32_s(32768) = 212992 < 232448
                h_s = T.alloc_shared([dim_k, dim_v], accum_dtype)
                g_cumsum_s = T.alloc_shared([chunk_size, dim_k], accum_dtype)
                A_s = T.alloc_shared([chunk_size, chunk_size], accum_dtype)
                # Scratch buffers reused across stages each iteration
                qf32_s = T.alloc_shared([chunk_size, dim_k], accum_dtype)
                kf32_s = T.alloc_shared([chunk_size, dim_k], accum_dtype)
                vf32_s = T.alloc_shared([chunk_size, dim_v], accum_dtype)

                T.copy(initial_state[i_b, i_h, :, :], h_s)

                for i_c in T.Serial(num_chunks):
                    stage1_cumsum(g, i_b, i_h, i_c, g_cumsum_s)
                    stage3_intra(q, k, g_cumsum_s, A_s, qf32_s, kf32_s, i_b, i_h, i_c)
                    # stage4 runs before stage2: h_s is still the pre-decay state
                    stage4_output(q, v, g_cumsum_s, A_s, h_s, o, qf32_s, vf32_s, i_b, i_h, i_c)
                    # stage2 decays h_s and accumulates k^T v; saves pre-decay to h_out
                    stage2_recurrence(k, v, g_cumsum_s, h_s, h_out, kf32_s, vf32_s, i_b, i_h,
                                      i_c)

                # Overwrite the last h_out slot with the fully-updated final state
                T.copy(h_s, h_out[i_b, num_chunks - 1, i_h, :, :])

        return _main

    return _gla_fwd_func


@torch.library.custom_op("top::gla_fwd_wrapped_kernel", mutates_args=())
def _gla_fwd_wrapped_kernel(
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
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
    h_out: torch.Tensor,
) -> torch.Tensor:
    return _gla_fwd_kernel(batch, seq_len, heads, dim_k, dim_v, chunk_size, scale,
                           dtype)(threads)(q, k, v, g, initial_state, h_out)


@_gla_fwd_wrapped_kernel.register_fake
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


class GLAFwdKernel(Kernel):
    """GLA (Gated Linear Attention) forward kernel.

    Args:
        q: Query tensor, shape [batch, seq_len, heads, dim_k]
        k: Key tensor, shape [batch, seq_len, heads, dim_k]
        v: Value tensor, shape [batch, seq_len, heads, dim_v]
        g: Log-space forget gates, shape [batch, seq_len, heads, dim_k]
        initial_state: Optional initial hidden state, shape [batch, heads, dim_k, dim_v]

    Computation:
        Chunked GLA forward in 4 TileLang stages per chunk, fused in a single
        T.Serial(NT) loop. Stages run in order 1→3→4→2 so that stage4 reads
        the pre-decay hidden state before stage2 updates it:
          1. Within-chunk cumulative sum of gates -> g_cumsum
          3. Intra-chunk causal attention A = scale * q_gated @ k_gated^T (causal masked)
          4. Output o = scale * q_gated @ h + A @ v  (h is pre-decay)
          2. Inter-chunk hidden state recurrence h += k_adj^T @ v (h carried in shared memory)

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
        self.kernel = _gla_fwd_kernel(batch, seq_len, heads, dim_k, dim_v, chunk_size, self.scale,
                                      self.dtype_name)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"threads": 128}

    @property
    def autotune_configs(self) -> list[dict]:
        return [{"threads": t} for t in [64, 128, 256]]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, H, K = self.batch, self.seq_len, self.heads, self.dim_k
        V = self.dim_v
        BT = self.chunk_size
        NT = (T + BT - 1) // BT
        dtype_torch = getattr(torch, self.dtype_name)

        if initial_state is None:
            init_state = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
        else:
            init_state = initial_state.to(torch.float32)

        h_out = torch.empty(B, NT, H, K, V, dtype=torch.float32, device=q.device)

        o = _gla_fwd_wrapped_kernel(
            B, T, H, K, V, BT, self.scale, self.dtype_name, self.config["threads"],
            q.to(dtype_torch),
            k.to(dtype_torch),
            v.to(dtype_torch),
            g.to(dtype_torch),
            init_state,
            h_out,
        )

        final_state = h_out[:, -1] if self.output_final_state else None
        return o, final_state
