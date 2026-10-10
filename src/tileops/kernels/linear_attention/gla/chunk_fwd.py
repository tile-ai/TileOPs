import functools
from typing import Callable, Optional

import tilelang
import torch
from tilelang import language as T
from tilelang.profiler import do_bench

from tileops.kernels.constants import LOG2E
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.call_spec import (
    GLAChunkCall,
    GLAChunkFwdInterface,
    head_count_refusal,
)
from tileops.kernels.linear_attention.v_tile import GEMM_MIN_N, min_gemm_n
from tileops.utils import get_shared_memory_optin


@functools.lru_cache(maxsize=32)
def gla_precompute_g_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    chunk_size: int,
    dtype: str,
    output_dtype: str = "float32",
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
        },
    )
    def _fn(num_stages, threads=128):
        g_shape = [batch, seq_len, heads, dim_k]
        g_cumsum_shape = [batch, seq_len, heads, dim_k]

        @T.prim_func
        def _main(
            g: T.Tensor(g_shape, dtype),
            g_cumsum: T.Tensor(g_cumsum_shape, output_dtype),
        ):
            with T.Kernel(batch * heads * num_chunks, threads=threads) as bx:
                i_b = bx // (heads * num_chunks)
                i_h = (bx // num_chunks) % heads
                i_c = bx % num_chunks
                cs = i_c * chunk_size

                g_s = T.alloc_shared([chunk_size, dim_k], dtype)
                g_out_s = T.alloc_shared([chunk_size, dim_k], accum_dtype)

                T.copy(g[i_b, cs : cs + chunk_size, i_h, :], g_s, disable_tma=True)

                for i_k in T.Parallel(dim_k):
                    g_out_s[0, i_k] = T.cast(g_s[0, i_k], accum_dtype)
                for i_t in T.Serial(1, chunk_size):
                    for i_k in T.Parallel(dim_k):
                        g_out_s[i_t, i_k] = g_out_s[i_t - 1, i_k] + T.cast(
                            g_s[i_t, i_k], accum_dtype
                        )

                T.copy(g_out_s, g_cumsum[i_b, cs : cs + chunk_size, i_h, :])

        return _main

    return _fn


@functools.lru_cache(maxsize=32)
def gla_fwd_h_kernel(
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

    Sequential over chunks (inter-chunk recurrence); the pipelined loop prefetches k, v and
    g_cumsum.

    KV-partition parallelism: splits K and V dimensions across thread blocks
    for higher SM utilization and more square GEMM shapes.
    Grid: B * H * num_k_partitions * num_v_partitions blocks.
    """
    accum_dtype = "float32"
    num_chunks = seq_len // chunk_size
    if dim_v % num_v_partitions:
        raise ValueError(
            f"dim_v ({dim_v}) is not divisible by num_v_partitions ({num_v_partitions})"
        )
    dim_v_part = dim_v // num_v_partitions
    if dim_v_part < GEMM_MIN_N:
        raise ValueError(
            f"dim_v ({dim_v}) split across num_v_partitions ({num_v_partitions}) gives a "
            f"{dim_v_part}-column T.gemm B operand, below the minimum N extent ({GEMM_MIN_N})"
        )
    dim_k_part = dim_k // num_k_partitions
    num_kv = num_k_partitions * num_v_partitions

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _h_func(num_stages, threads=128):
        if dim_v_part < min_gemm_n(threads):
            raise ValueError(
                f"dim_v ({dim_v}) split across num_v_partitions ({num_v_partitions}) "
                f"gives a {dim_v_part}-column T.gemm B operand, below the minimum N "
                f"extent ({min_gemm_n(threads)}) at {threads} threads"
            )
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
            with T.Kernel(batch * heads * num_kv, threads=threads) as bx:
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
                g_cumsum_s = T.alloc_shared([chunk_size, dim_k_part], accum_dtype)

                for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                    h_s[i_k, i_v] = initial_state[i_b, i_h, k_offset + i_k, v_offset + i_v]

                for i_c in T.Pipelined(num_chunks, num_stages=num_stages):
                    T.copy(
                        k[
                            i_b,
                            i_c * chunk_size : (i_c + 1) * chunk_size,
                            i_h,
                            k_offset : k_offset + dim_k_part,
                        ],
                        k_s,
                        disable_tma=True,
                    )
                    T.copy(
                        v[
                            i_b,
                            i_c * chunk_size : (i_c + 1) * chunk_size,
                            i_h,
                            v_offset : v_offset + dim_v_part,
                        ],
                        v_s,
                        disable_tma=True,
                    )
                    T.copy(
                        g_cumsum[
                            i_b,
                            i_c * chunk_size : (i_c + 1) * chunk_size,
                            i_h,
                            k_offset : k_offset + dim_k_part,
                        ],
                        g_cumsum_s,
                        disable_tma=True,
                    )

                    # h_out[i_c] is the state entering chunk i_c.
                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        h_out[i_b, i_c, i_h, k_offset + i_k, v_offset + i_v] = h_s[i_k, i_v]

                    g_last = T.alloc_fragment([dim_k_part], accum_dtype)
                    for i_k in T.Parallel(dim_k_part):
                        g_last[i_k] = g_cumsum_s[chunk_size - 1, i_k]

                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        h_s[i_k, i_v] = h_s[i_k, i_v] * T.exp2(g_last[i_k] * LOG2E)

                    # k_adj in fragment (RS GEMM: A=register, B=shared)
                    k_adj_f = T.alloc_fragment([chunk_size, dim_k_part], dtype)
                    for i_t, i_k in T.Parallel(chunk_size, dim_k_part):
                        k_adj_f[i_t, i_k] = T.cast(
                            T.cast(k_s[i_t, i_k], accum_dtype)
                            * T.exp2((g_last[i_k] - g_cumsum_s[i_t, i_k]) * LOG2E),
                            dtype,
                        )

                    delta_h = T.alloc_fragment([dim_k_part, dim_v_part], accum_dtype)
                    T.fill(delta_h, 0.0)
                    T.gemm(k_adj_f, v_s, delta_h, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        h_s[i_k, i_v] = h_s[i_k, i_v] + delta_h[i_k, i_v]

                for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                    h_out[i_b, num_chunks, i_h, k_offset + i_k, v_offset + i_v] = h_s[i_k, i_v]

        return _main

    return _h_func


@functools.lru_cache(maxsize=32)
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
        },
    )
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

                q_s = T.alloc_shared([chunk_size, dim_k], dtype)
                k_s = T.alloc_shared([chunk_size, dim_k], dtype)
                v_s = T.alloc_shared([chunk_size, dim_v], dtype)
                g_cumsum_s = T.alloc_shared([chunk_size, dim_k], accum_dtype)

                q_gated_s = T.alloc_shared([chunk_size, dim_k], dtype)
                A_s = T.alloc_shared([chunk_size, chunk_size], dtype)

                T.copy(
                    q[i_b, chunk_start : chunk_start + chunk_size, i_h, :], q_s, disable_tma=True
                )
                T.copy(
                    k[i_b, chunk_start : chunk_start + chunk_size, i_h, :], k_s, disable_tma=True
                )
                T.copy(
                    g_cumsum[i_b, chunk_start : chunk_start + chunk_size, i_h, :],
                    g_cumsum_s,
                    disable_tma=True,
                )

                # ---- Gated q (inter-chunk term, exp(g_cumsum) <= 1) ----
                for i_t, i_k in T.Parallel(chunk_size, dim_k):
                    q_gated_s[i_t, i_k] = T.cast(
                        T.cast(q_s[i_t, i_k], accum_dtype) * T.exp2(g_cumsum_s[i_t, i_k] * LOG2E),
                        dtype,
                    )

                # ---- A[i,j] = sum_k q[i,k]*k[j,k]*exp(g[i,k]-g[j,k]) ----
                A_frag = T.alloc_fragment([chunk_size, chunk_size], accum_dtype)
                T.fill(A_frag, 0.0)
                for i_k in T.Serial(dim_k):
                    for i_t, i_j in T.Parallel(chunk_size, chunk_size):
                        A_frag[i_t, i_j] = A_frag[i_t, i_j] + (
                            T.cast(q_s[i_t, i_k], accum_dtype)
                            * T.cast(k_s[i_j, i_k], accum_dtype)
                            * T.exp2((g_cumsum_s[i_t, i_k] - g_cumsum_s[i_j, i_k]) * LOG2E)
                        )
                for i_t, i_j in T.Parallel(chunk_size, chunk_size):
                    A_s[i_t, i_j] = T.cast(
                        T.if_then_else(i_j <= i_t, A_frag[i_t, i_j] * scale, 0.0), dtype
                    )

                # Loaded once q, k and g are spent, so they do not share the peak with them.
                T.copy(
                    v[i_b, chunk_start : chunk_start + chunk_size, i_h, :], v_s, disable_tma=True
                )
                for i_k, i_v in T.Parallel(dim_k, dim_v):
                    h_cast_s[i_k, i_v] = T.cast(h[i_b, i_c, i_h, i_k, i_v], dtype)

                # ---- o = scale * q_gated @ h + A @ v ----
                acc = T.alloc_fragment([chunk_size, dim_v], accum_dtype)
                T.fill(acc, 0.0)
                T.gemm(q_gated_s, h_cast_s, acc, policy=T.GemmWarpPolicy.FullRow)
                for i_t, i_v in T.Parallel(chunk_size, dim_v):
                    acc[i_t, i_v] = acc[i_t, i_v] * scale
                T.gemm(A_s, v_s, acc, policy=T.GemmWarpPolicy.FullRow)

                for i_t, i_v in T.Parallel(chunk_size, dim_v):
                    o[i_b, chunk_start + i_t, i_h, i_v] = T.cast(acc[i_t, i_v], dtype)

        return _main

    return _o_func


class GLAChunkedFwdKernel(Kernel):
    """GLA (Gated Linear Attention) forward program — three-pass architecture.

    Pass 0 (parallel, B*H*NC blocks): Pre-compute g_cumsum per chunk.
    Pass 1 (sequential, B*H blocks): Compute per-chunk hidden states h.
    Pass 2 (parallel, B*H*NC blocks): Compute output o per chunk independently.

    h_out is kept for the backward pass.

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
        self.dtype_name = str(dtype).split(".")[-1]
        reason = self.region_refusal(dim_k, dim_v, chunk_size)
        if reason:
            raise ValueError(f"{type(self).__name__} does not serve this call: {reason}")
        self.init_config(config, tune)
        if not tune:
            self._build_kernels(self.config)

    @staticmethod
    def region_refusal(dim_k: int, dim_v: int, chunk_size: int) -> Optional[str]:
        """Why the default tiling cannot build these extents, or ``None`` when it can.

        Its three GEMMs run on two warps: the state update over
        ``(dim_k / 2) x (dim_v / 4)`` and the two chunk outputs over
        ``chunk_size x dim_v``. Each warp takes whole 16 x 8 tiles, so the chunk
        either fills both warps' rows (a multiple of 32) or one warp's (16, the
        warps then splitting the columns); the state update likewise needs
        ``dim_k / 2`` a multiple of 32 or exactly 16.
        """
        if not (chunk_size == 16 or chunk_size % 32 == 0):
            return f"chunk_size={chunk_size} must be 16 or a multiple of 32"
        if not (dim_k % 64 == 0 and dim_v % 32 == 0 or dim_k == 32 and dim_v % 64 == 0):
            return (
                f"dim_k={dim_k}, dim_v={dim_v}: dim_k must be a multiple of 64 with dim_v a "
                "multiple of 32, or 32 with dim_v a multiple of 64"
            )
        return None

    @staticmethod
    def _v_partitions(dim_v: int, threads: int, candidates: list[int]) -> list[int]:
        """Return the candidates the recurrence can build at *threads*, widest first."""
        floor = max(GEMM_MIN_N, min_gemm_n(threads))
        return [n for n in candidates if dim_v % n == 0 and dim_v // n >= floor]

    @staticmethod
    def _output_live_bytes(c: int, k: int, v: int, elem: int) -> int:
        """Lower bound on the output pass's shared memory: q, k, the fp32 gate and the gated q
        while A is summed, then the gated q, A, v and the cast state for the two GEMMs."""
        return max(c * k * (3 * elem + 4), (c * k + c * c + c * v + k * v) * elem)

    @staticmethod
    def _state_shared_bytes(c: int, k: int, v: int, elem: int, stages: int) -> int:
        """Shared memory of the state pass over a k x v slice: the fp32 state, and k, v and the
        fp32 gate once per stage."""
        return stages * c * (k * (elem + 4) + v * elem) + k * v * 4

    @property
    def default_config(self) -> dict:
        return self._default_config_for(
            get_shared_memory_optin(),
            self.chunk_size,
            self.dim_k,
            self.dim_v,
            getattr(torch, self.dtype_name).itemsize,
        )

    @classmethod
    def _default_config_for(
        cls, budget: int, chunk_size: int, dim_k: int, dim_v: int, elem: int
    ) -> dict:
        """The config this kernel builds at *budget* bytes of shared memory per block."""
        threads = 64
        num_v_partitions, num_k_partitions = cls._v_partitions(dim_v, threads, [4, 2, 1])[0], 2
        k, v = dim_k // num_k_partitions, dim_v // num_v_partitions
        # Only the state pass pipelines, so its size alone sets the stages.
        stages = next(
            (s for s in (3, 2) if cls._state_shared_bytes(chunk_size, k, v, elem, s) <= budget),
            1,
        )
        return {
            "num_stages": stages,
            "threads": threads,
            "num_v_partitions": num_v_partitions,
            "num_k_partitions": num_k_partitions,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        configs = []
        for ns in [1, 2, 3]:
            for t_par in [64, 128, 256]:
                for t_seq in [64, 128, 256]:
                    for nvp in self._v_partitions(self.dim_v, t_seq, [2, 4]):
                        for nkp in [1, 2]:
                            configs.append(
                                {
                                    "num_stages": ns,
                                    "threads_par": t_par,
                                    "threads_seq": t_seq,
                                    "num_v_partitions": nvp,
                                    "num_k_partitions": nkp,
                                }
                            )
        return configs

    def _build_kernels(self, config: dict) -> None:
        """Rebuild all sub-kernels from a config dict."""
        ns = config.get("num_stages", 2)
        thr_seq = config.get("threads_seq", config.get("threads", 256))
        thr_par = config.get("threads_par", config.get("threads", 256))
        num_vp = config.get("num_v_partitions", 4)
        num_kp = config.get("num_k_partitions", 1)
        self._g_fn = gla_precompute_g_kernel(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            self.chunk_size,
            self.dtype_name,
        )(ns, thr_par)
        self._h_fn = gla_fwd_h_kernel(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            self.dim_v,
            self.chunk_size,
            self.dtype_name,
            num_v_partitions=num_vp,
            num_k_partitions=num_kp,
        )(ns, thr_seq)
        self._o_fn = _gla_fwd_o_kernel(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            self.dim_v,
            self.chunk_size,
            self.scale,
            self.dtype_name,
        )(ns, thr_par)

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Custom autotuning for multi-kernel forward pass."""
        if self.autotune_configs is None:
            return
        print(
            f"Start autotuning {self.__class__.__name__} ({len(self.autotune_configs)} configs)..."
        )

        B, T, H, K, V = (self.batch, self.seq_len, self.heads, self.dim_k, self.dim_v)
        dtype_torch = getattr(torch, self.dtype_name)

        q = torch.randn(B, T, H, K, device="cuda", dtype=dtype_torch) * 0.1
        k = torch.randn(B, T, H, K, device="cuda", dtype=dtype_torch) * 0.1
        v = torch.randn(B, T, H, V, device="cuda", dtype=dtype_torch) * 0.1
        g = -torch.rand(B, T, H, K, device="cuda", dtype=dtype_torch).abs()

        best_lat = float("inf")
        best_cfg = None

        for cfg in self.autotune_configs:
            try:
                self._build_kernels(cfg)

                self.forward(q, k, v, g)
                torch.cuda.synchronize()

                lat = do_bench(
                    lambda: self.forward(q, k, v, g),
                    warmup=warmup,
                    rep=rep,
                )
                print(f"  config={cfg} -> {lat:.3f}ms")
                if lat < best_lat:
                    best_lat = lat
                    best_cfg = cfg
            except Exception as e:
                print(f"  config={cfg} -> FAILED: {e}")
                continue

        if best_cfg is not None:
            self.config = best_cfg
            self._build_kernels(best_cfg)
            print(f"Best config: {best_cfg} ({best_lat:.3f}ms)")
        else:
            print("Autotuning failed, using default config")
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

        g_cumsum = self._g_fn(g.to(dtype_torch))

        h_out = self._h_fn(
            k.to(dtype_torch),
            v.to(dtype_torch),
            g_cumsum,
            init_state,
        )

        o = self._o_fn(
            q.to(dtype_torch),
            k.to(dtype_torch),
            v.to(dtype_torch),
            g_cumsum,
            h_out,
        )

        self._h_out = h_out

        final_state = h_out[:, -1] if self.output_final_state else None
        return o, final_state


class GLAChunkFwdKernel(GLAChunkedFwdKernel, GLAChunkFwdInterface):
    """The chunked forward as the training op calls it, with its own recurrent state."""

    @classmethod
    def refusal(cls, call: GLAChunkCall) -> Optional[str]:
        """Why no program serves this call, or ``None``.

        Refuses a call whose output pass holds more shared memory at once than the device gives,
        a lower bound above the other passes' at their fewest stages. A call within the bound
        that TileLang still cannot place builds and fails at launch.
        """
        reason = head_count_refusal(call.heads) or cls.region_refusal(
            call.dim_k, call.dim_v, call.chunk_size
        )
        if reason is not None or not call.smem_budget:
            return reason
        c, k, v = call.chunk_size, call.dim_k, call.dim_v
        need = cls._output_live_bytes(c, k, v, call.dtype.itemsize)
        if need <= call.smem_budget:
            return None
        return (
            f"needs at least {need} bytes of shared memory per block at chunk {c}, head dims "
            f"{k} / {v} in {call.dtype}; the device gives {call.smem_budget}"
        )

    @classmethod
    def entry_for(cls, call: GLAChunkCall) -> Entry:
        """The final state is always produced, and the initial one is a launch argument.

        The device index is in the identity because the constructor compiles for the
        architecture it is built on.
        """
        arguments = (
            call.batch,
            call.seq_len,
            call.heads,
            call.dim_k,
            call.dim_v,
            call.chunk_size,
            call.scale,
        )
        index = call.device.index if call.device is not None else None
        return (*arguments, call.dtype, index), lambda: cls(
            *arguments, output_final_state=True, dtype=call.dtype
        )
