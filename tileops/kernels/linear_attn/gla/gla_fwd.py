from typing import Any, Callable, Optional

import tilelang
import torch
from tilelang import language as T
from tilelang.profiler import do_bench

from tileops.kernels.kernel import Kernel

LOG2_E = 1.44269504


# ---------------------------------------------------------------------------
# Pre-compute: g_cumsum per chunk (parallel, B*H*NC thread blocks)
# ---------------------------------------------------------------------------

def _gla_precompute_g_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    chunk_size: int,
    dtype: str,
) -> Callable:
    """Pre-compute intra-chunk cumulative sum of g.

    Parallel over (batch, heads, chunks): B*H*NC thread blocks.
    Each block computes cumsum for one chunk independently.
    """
    accum_dtype = "float32"
    num_chunks = seq_len // chunk_size

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        })
    def _fn(num_stages, threads=128):
        g_shape = [batch, seq_len, heads, dim_k]
        g_cumsum_shape = [batch, seq_len, heads, dim_k]

        @T.prim_func
        def _main(
            g: T.Tensor(g_shape, dtype),
            g_cumsum: T.Tensor(g_cumsum_shape, accum_dtype),
        ):
            with T.Kernel(batch * heads * num_chunks, threads=threads) as bx:
                i_b = bx // (heads * num_chunks)
                i_h = (bx // num_chunks) % heads
                i_c = bx % num_chunks
                cs = i_c * chunk_size

                g_s = T.alloc_shared([chunk_size, dim_k], dtype)
                g_out_s = T.alloc_shared([chunk_size, dim_k], accum_dtype)

                T.copy(g[i_b, cs:cs + chunk_size, i_h, :], g_s, disable_tma=True)

                for i_k in T.Parallel(dim_k):
                    g_out_s[0, i_k] = T.cast(g_s[0, i_k], accum_dtype)
                for i_t in T.Serial(1, chunk_size):
                    for i_k in T.Parallel(dim_k):
                        g_out_s[i_t, i_k] = (
                            g_out_s[i_t - 1, i_k]
                            + T.cast(g_s[i_t, i_k], accum_dtype)
                        )

                T.copy(g_out_s, g_cumsum[i_b, cs:cs + chunk_size, i_h, :])

        return _main

    return _fn


# ---------------------------------------------------------------------------
# Pass 1: compute h per chunk (sequential, B*H thread blocks)
# Uses pre-computed g_cumsum — no T.Serial cumsum needed.
# ---------------------------------------------------------------------------

def _gla_fwd_h_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: str,
    num_v_partitions: int = 1,
    num_k_partitions: int = 1,
) -> Callable:
    """Compute per-chunk hidden states h in forward order.

    Sequential over chunks (inter-chunk recurrence).
    Uses T.Pipelined + T.copy for async prefetch of k, v, g_cumsum.
    g_cumsum is pre-computed — no T.Serial cumsum in this kernel.

    KV-partition parallelism: splits K and V dimensions across thread blocks
    for higher SM utilization and more square GEMM shapes.
    Grid: B * H * num_k_partitions * num_v_partitions blocks.
    """
    accum_dtype = "float32"
    num_chunks = seq_len // chunk_size
    dim_v_part = dim_v // num_v_partitions
    dim_k_part = dim_k // num_k_partitions
    num_kv = num_k_partitions * num_v_partitions

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        })
    def _h_func(num_stages, threads=128):
        k_shape = [batch, seq_len, heads, dim_k]
        v_shape = [batch, seq_len, heads, dim_v]
        g_cumsum_shape = [batch, seq_len, heads, dim_k]
        init_state_shape = [batch, heads, dim_k, dim_v]
        h_out_shape = [batch, num_chunks + 1, heads, dim_k, dim_v]

        @T.prim_func
        def _main(
            k: T.Tensor(k_shape, dtype),
            v: T.Tensor(v_shape, dtype),
            g_cumsum: T.Tensor(g_cumsum_shape, accum_dtype),
            initial_state: T.Tensor(init_state_shape, accum_dtype),
            h_out: T.Tensor(h_out_shape, accum_dtype),
        ):
            with T.Kernel(batch * heads * num_kv,
                          threads=threads) as bx:
                i_b = bx // (heads * num_kv)
                i_h = (bx // num_kv) % heads
                i_kv = bx % num_kv
                i_kp = i_kv // num_v_partitions
                i_vp = i_kv % num_v_partitions
                k_offset = i_kp * dim_k_part
                v_offset = i_vp * dim_v_part

                h_s = T.alloc_shared([dim_k_part, dim_v_part], accum_dtype)
                k_s = T.alloc_shared([chunk_size, dim_k_part], dtype)
                v_s = T.alloc_shared([chunk_size, dim_v_part], dtype)
                g_cumsum_s = T.alloc_shared([chunk_size, dim_k_part],
                                           accum_dtype)

                # Load initial state KV-slice
                for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                    h_s[i_k, i_v] = initial_state[i_b, i_h,
                                                   k_offset + i_k,
                                                   v_offset + i_v]

                for i_c in T.Pipelined(num_chunks, num_stages=num_stages):
                    T.copy(k[i_b, i_c * chunk_size:(i_c + 1) * chunk_size,
                             i_h, k_offset:k_offset + dim_k_part],
                           k_s, disable_tma=True)
                    T.copy(v[i_b, i_c * chunk_size:(i_c + 1) * chunk_size,
                             i_h, v_offset:v_offset + dim_v_part],
                           v_s, disable_tma=True)
                    T.copy(g_cumsum[i_b,
                                    i_c * chunk_size:(i_c + 1) * chunk_size,
                                    i_h,
                                    k_offset:k_offset + dim_k_part],
                           g_cumsum_s, disable_tma=True)

                    # Save pre-decay h KV-slice
                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        h_out[i_b, i_c, i_h,
                              k_offset + i_k,
                              v_offset + i_v] = h_s[i_k, i_v]

                    # g_last from pre-computed cumsum
                    g_last = T.alloc_fragment([dim_k_part], accum_dtype)
                    for i_k in T.Parallel(dim_k_part):
                        g_last[i_k] = g_cumsum_s[chunk_size - 1, i_k]

                    # Decay h
                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        h_s[i_k, i_v] = (h_s[i_k, i_v]
                                         * T.exp2(g_last[i_k] * LOG2_E))

                    # k_adj in fragment (RS GEMM: A=register, B=shared)
                    k_adj_f = T.alloc_fragment([chunk_size, dim_k_part],
                                              dtype)
                    for i_t, i_k in T.Parallel(chunk_size, dim_k_part):
                        k_adj_f[i_t, i_k] = T.cast(
                            T.cast(k_s[i_t, i_k], accum_dtype)
                            * T.exp2((g_last[i_k] - g_cumsum_s[i_t, i_k])
                                     * LOG2_E),
                            dtype)

                    # h += k_adj^T @ v_slice (RS GEMM)
                    delta_h = T.alloc_fragment([dim_k_part, dim_v_part],
                                              accum_dtype)
                    T.fill(delta_h, 0.0)
                    T.gemm(k_adj_f, v_s, delta_h, transpose_A=True,
                           policy=T.GemmWarpPolicy.FullRow)
                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        h_s[i_k, i_v] = h_s[i_k, i_v] + delta_h[i_k, i_v]

                # Save final state KV-slice
                for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                    h_out[i_b, num_chunks, i_h,
                          k_offset + i_k,
                          v_offset + i_v] = h_s[i_k, i_v]

        return _main

    return _h_func


# ---------------------------------------------------------------------------
# Pass 2: compute output per chunk (parallel, B*H*NC thread blocks)
# Uses pre-computed g_cumsum — no T.Serial cumsum needed.
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
    """Compute output o for each chunk independently.

    Parallel over (batch, heads, chunks): B*H*NC thread blocks.
    Each block reads h[i_c] and g_cumsum from global memory.
    """
    accum_dtype = "float32"
    num_chunks = seq_len // chunk_size

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        })
    def _o_func(num_stages, threads=128):
        q_shape = [batch, seq_len, heads, dim_k]
        k_shape = [batch, seq_len, heads, dim_k]
        v_shape = [batch, seq_len, heads, dim_v]
        g_cumsum_shape = [batch, seq_len, heads, dim_k]
        h_shape = [batch, num_chunks + 1, heads, dim_k, dim_v]
        o_shape = [batch, seq_len, heads, dim_v]

        @T.prim_func
        def _main(
            q: T.Tensor(q_shape, dtype),
            k: T.Tensor(k_shape, dtype),
            v: T.Tensor(v_shape, dtype),
            g_cumsum: T.Tensor(g_cumsum_shape, accum_dtype),
            h: T.Tensor(h_shape, accum_dtype),
            o: T.Tensor(o_shape, dtype),
        ):
            with T.Kernel(batch * heads * num_chunks, threads=threads) as bx:
                i_b = bx // (heads * num_chunks)
                i_h = (bx // num_chunks) % heads
                i_c = bx % num_chunks
                chunk_start = i_c * chunk_size

                # h cast to native dtype for tensor core
                h_cast_s = T.alloc_shared([dim_k, dim_v], dtype)

                # Input buffers
                q_s = T.alloc_shared([chunk_size, dim_k], dtype)
                k_s = T.alloc_shared([chunk_size, dim_k], dtype)
                v_s = T.alloc_shared([chunk_size, dim_v], dtype)
                g_cumsum_s = T.alloc_shared([chunk_size, dim_k], accum_dtype)

                # Compute buffers
                q_gated_s = T.alloc_shared([chunk_size, dim_k], dtype)
                A_s = T.alloc_shared([chunk_size, chunk_size], dtype)

                # Load inputs via T.copy
                T.copy(q[i_b, chunk_start:chunk_start + chunk_size, i_h, :],
                       q_s, disable_tma=True)
                T.copy(k[i_b, chunk_start:chunk_start + chunk_size, i_h, :],
                       k_s, disable_tma=True)
                T.copy(v[i_b, chunk_start:chunk_start + chunk_size, i_h, :],
                       v_s, disable_tma=True)
                T.copy(g_cumsum[i_b, chunk_start:chunk_start + chunk_size, i_h, :],
                       g_cumsum_s, disable_tma=True)

                # Load h[i_c] and cast to native dtype
                for i_k, i_v in T.Parallel(dim_k, dim_v):
                    h_cast_s[i_k, i_v] = T.cast(h[i_b, i_c, i_h, i_k, i_v], dtype)

                # ---- Gated q (inter-chunk term, exp(g_cumsum) <= 1) ----
                for i_t, i_k in T.Parallel(chunk_size, dim_k):
                    q_gated_s[i_t, i_k] = T.cast(
                        T.cast(q_s[i_t, i_k], accum_dtype)
                        * T.exp2(g_cumsum_s[i_t, i_k] * LOG2_E),
                        dtype)

                # ---- A[i,j] = sum_k q[i,k]*k[j,k]*exp(g[i,k]-g[j,k]) ----
                A_frag = T.alloc_fragment([chunk_size, chunk_size], accum_dtype)
                T.fill(A_frag, 0.0)
                for i_k in T.Serial(dim_k):
                    for i_t, i_j in T.Parallel(chunk_size, chunk_size):
                        A_frag[i_t, i_j] = A_frag[i_t, i_j] + (
                            T.cast(q_s[i_t, i_k], accum_dtype)
                            * T.cast(k_s[i_j, i_k], accum_dtype)
                            * T.exp2((g_cumsum_s[i_t, i_k]
                                      - g_cumsum_s[i_j, i_k]) * LOG2_E))
                for i_t, i_j in T.Parallel(chunk_size, chunk_size):
                    A_s[i_t, i_j] = T.cast(
                        T.if_then_else(
                            i_j <= i_t,
                            A_frag[i_t, i_j] * scale,
                            0.0),
                        dtype)

                # ---- o = scale * q_gated @ h + A @ v ----
                acc = T.alloc_fragment([chunk_size, dim_v], accum_dtype)
                T.fill(acc, 0.0)
                T.gemm(q_gated_s, h_cast_s, acc,
                       policy=T.GemmWarpPolicy.FullRow)
                for i_t, i_v in T.Parallel(chunk_size, dim_v):
                    acc[i_t, i_v] = acc[i_t, i_v] * scale
                T.gemm(A_s, v_s, acc, policy=T.GemmWarpPolicy.FullRow)

                for i_t, i_v in T.Parallel(chunk_size, dim_v):
                    o[i_b, chunk_start + i_t, i_h, i_v] = T.cast(
                        acc[i_t, i_v], dtype)

        return _main

    return _o_func


# ---------------------------------------------------------------------------
# Custom op wrappers (kept for torch.compile compatibility)
# ---------------------------------------------------------------------------

@torch.library.custom_op("top::gla_fwd_wrapped_kernel", mutates_args=("h_out",))
def _gla_fwd_wrapped_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    scale: float,
    dtype: str,
    num_stages: int,
    threads: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
    h_out: torch.Tensor,
) -> torch.Tensor:
    # Three-pass: precompute g_cumsum, then h, then o
    g_fn = _gla_precompute_g_kernel(batch, seq_len, heads, dim_k, chunk_size,
                                     dtype)(num_stages, threads)
    h_fn = _gla_fwd_h_kernel(batch, seq_len, heads, dim_k, dim_v, chunk_size,
                              dtype)(num_stages, threads)
    o_fn = _gla_fwd_o_kernel(batch, seq_len, heads, dim_k, dim_v, chunk_size,
                              scale, dtype)(num_stages, threads)
    g_cumsum = g_fn(g)
    h_fn(k, v, g_cumsum, initial_state, h_out)
    return o_fn(q, k, v, g_cumsum, h_out)


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
    num_stages: int,
    threads: int,
    *inputs: tuple[Any],
) -> torch.Tensor:
    _ = (dim_k, chunk_size, scale, dtype, num_stages, threads)
    return torch.empty([batch, seq_len, heads, dim_v],
                       dtype=inputs[0].dtype,
                       device=inputs[0].device)


class GLAFwdKernel(Kernel):
    """GLA (Gated Linear Attention) forward kernel — three-pass architecture.

    Pass 0 (parallel, B*H*NC blocks): Pre-compute g_cumsum per chunk.
    Pass 1 (sequential, B*H blocks): Compute per-chunk hidden states h.
    Pass 2 (parallel, B*H*NC blocks): Compute output o per chunk independently.

    By pre-computing g_cumsum, the sequential h_kernel is free of T.Serial
    cumsum loops, dramatically reducing its latency.

    h_out is saved for the backward pass (no recomputation needed).

    Reference:
        https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gla/chunk.py
    """

    supported_archs: list[int] = [80, 89, 90]

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
        if not tune:
            self._build_kernels(self.config)

    @property
    def default_config(self) -> dict:
        return {
            "num_stages": 3, "threads": 64,
            "num_v_partitions": 4, "num_k_partitions": 2,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        configs = []
        for ns in [1, 2, 3]:
            for t_par in [64, 128, 256]:
                for t_seq in [64, 128, 256]:
                    for nvp in [2, 4]:
                        for nkp in [1, 2]:
                            configs.append({
                                "num_stages": ns,
                                "threads_par": t_par,
                                "threads_seq": t_seq,
                                "num_v_partitions": nvp,
                                "num_k_partitions": nkp,
                            })
        return configs

    def _build_kernels(self, config: dict) -> None:
        """Rebuild all sub-kernels from a config dict."""
        ns = config.get("num_stages", 2)
        thr_seq = config.get("threads_seq", config.get("threads", 256))
        thr_par = config.get("threads_par", config.get("threads", 256))
        num_vp = config.get("num_v_partitions", 4)
        num_kp = config.get("num_k_partitions", 1)
        self._g_fn = _gla_precompute_g_kernel(
            self.batch, self.seq_len, self.heads, self.dim_k,
            self.chunk_size, self.dtype_name,
        )(ns, thr_par)
        self._h_fn = _gla_fwd_h_kernel(
            self.batch, self.seq_len, self.heads, self.dim_k, self.dim_v,
            self.chunk_size, self.dtype_name,
            num_v_partitions=num_vp,
            num_k_partitions=num_kp,
        )(ns, thr_seq)
        self._o_fn = _gla_fwd_o_kernel(
            self.batch, self.seq_len, self.heads, self.dim_k, self.dim_v,
            self.chunk_size, self.scale, self.dtype_name,
        )(ns, thr_par)

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Custom autotuning for multi-kernel forward pass."""
        if self.autotune_configs is None:
            return
        print(f'Start autotuning {self.__class__.__name__} '
              f'({len(self.autotune_configs)} configs)...')

        B, T, H, K, V = (self.batch, self.seq_len, self.heads,
                          self.dim_k, self.dim_v)
        dtype_torch = getattr(torch, self.dtype_name)

        # Generate representative inputs
        q = torch.randn(B, T, H, K, device="cuda", dtype=dtype_torch) * 0.1
        k = torch.randn(B, T, H, K, device="cuda", dtype=dtype_torch) * 0.1
        v = torch.randn(B, T, H, V, device="cuda", dtype=dtype_torch) * 0.1
        g = -torch.rand(B, T, H, K, device="cuda", dtype=dtype_torch).abs()

        best_lat = float('inf')
        best_cfg = None

        for cfg in self.autotune_configs:
            try:
                self._build_kernels(cfg)

                # Warmup run
                self.forward(q, k, v, g)
                torch.cuda.synchronize()

                lat = do_bench(
                    lambda: self.forward(q, k, v, g),
                    warmup=warmup, rep=rep,
                )
                print(f'  config={cfg} -> {lat:.3f}ms')
                if lat < best_lat:
                    best_lat = lat
                    best_cfg = cfg
            except Exception as e:
                print(f'  config={cfg} -> FAILED: {e}')
                continue

        if best_cfg is not None:
            self.config = best_cfg
            self._build_kernels(best_cfg)
            print(f'Best config: {best_cfg} ({best_lat:.3f}ms)')
        else:
            print('Autotuning failed, using default config')
            self.config = self.default_config
            self._build_kernels(self.config)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, H, K, V = self.batch, self.heads, self.dim_k, self.dim_v
        dtype_torch = getattr(torch, self.dtype_name)

        if initial_state is None:
            init_state = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
        else:
            init_state = initial_state.to(torch.float32)

        # Pass 0: pre-compute g_cumsum (parallel, fast)
        g_cumsum = self._g_fn(g.to(dtype_torch))

        # Pass 1: sequential h computation
        h_out = self._h_fn(
            k.to(dtype_torch),
            v.to(dtype_torch),
            g_cumsum,
            init_state,
        )

        # Pass 2: parallel output computation
        o = self._o_fn(
            q.to(dtype_torch),
            k.to(dtype_torch),
            v.to(dtype_torch),
            g_cumsum,
            h_out,
        )

        # Store h_out and g_cumsum for backward access
        self._h_out = h_out
        self._g_cumsum = g_cumsum

        final_state = h_out[:, -1] if self.output_final_state else None
        return o, final_state
