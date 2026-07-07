import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.trace import trace
from tileops.utils import get_sm_version, str2dtype

__all__ = [
    'GemmFp8BlockScaledKernel',
    'GemmFp8EpilogueKernel',
    'GemmKernel',
    'GemvKernel',
]

class GemmFp8EpilogueKernel(Kernel):
    """Simple TileLang FP8 GEMM for per-tensor scales."""

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        out_dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.kernel = _gemm_fp8_kernel(
            m, n, k, self.dtype_str, self.out_dtype_str, block_scaled=False)
        self.init_config(config, tune)

    @property
    def out_dtype_str(self) -> str:
        return self.dtype_to_str(self.out_dtype)

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 128,
            "block_n": 128,
            "block_k": 128,
            "num_stages": 3,
            "threads": 256,
        }

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.dtype != torch.float8_e4m3fn:
            raise NotImplementedError(
                f"GemmFp8EpilogueKernel only supports torch.float8_e4m3fn, got {self.dtype}")
        compiled = _gemm_fp8_kernel(
            self.m, self.n, self.k, self.dtype_str, self.out_dtype_str,
            block_scaled=False, has_bias=bias is not None)(**self.config)
        if bias is not None:
            return compiled(a, b, scale_a, scale_b, bias)
        return compiled(a, b, scale_a, scale_b)


class GemmFp8BlockScaledKernel(Kernel):
    """Simple TileLang FP8 GEMM for K-block scales."""

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        out_dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.kernel = _gemm_fp8_kernel(
            m, n, k, self.dtype_str, self.out_dtype_str, block_scaled=True)
        self.init_config(config, tune)

    @property
    def out_dtype_str(self) -> str:
        return self.dtype_to_str(self.out_dtype)

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 128,
            "block_n": 128,
            "block_k": 128,
            "num_stages": 3,
            "threads": 256,
        }

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.dtype != torch.float8_e4m3fn:
            raise NotImplementedError(
                f"GemmFp8BlockScaledKernel only supports torch.float8_e4m3fn, got {self.dtype}")
        compiled = _gemm_fp8_kernel(
            self.m, self.n, self.k, self.dtype_str, self.out_dtype_str,
            block_scaled=True, has_bias=bias is not None)(**self.config)
        if bias is not None:
            return compiled(a, b, scale_a, scale_b, bias)
        return compiled(a, b, scale_a, scale_b)


@functools.lru_cache(maxsize=32)
def _gemm_fp8_kernel(
    m: int,
    n: int,
    k: int,
    dtype: str,
    out_dtype: str,
    block_scaled: bool,
    has_bias: bool = False,
) -> Callable:
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gemm_fp8_func(
        block_m: int = 128,
        block_n: int = 128,
        block_k: int = 128,
        num_stages: int = 3,
        threads: int = 256,
    ) -> Callable:
        if block_scaled:
            if block_k > 128:
                raise ValueError(f"block_k must be <= 128 for block128 scaling, got {block_k}")
            if 128 % block_k != 0:
                raise ValueError(f"128 must be divisible by block_k, got {block_k}")
        scale_k = (k + 127) // 128 if block_scaled else 1
        scale_a_shape = (m, scale_k) if block_scaled else (1, 1)
        scale_b_shape = (n, scale_k) if block_scaled else (1, 1)

        @T.prim_func
        def _gemm_fp8_main(
            a: T.Tensor((m, k), dtype),  # type: ignore
            b: T.Tensor((n, k), dtype),  # type: ignore
            scale_a: T.Tensor(scale_a_shape, "float32"),  # type: ignore
            scale_b: T.Tensor(scale_b_shape, "float32"),  # type: ignore
            c: T.Tensor((m, n), out_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                    T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (bx, by):
                a_shared = T.alloc_shared((block_m, block_k), dtype)
                b_shared = T.alloc_shared((block_n, block_k), dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                if block_scaled:
                    partial = T.alloc_fragment((block_m, block_n), accum_dtype)

                T.annotate_layout({
                    a_shared: tilelang.layout.make_swizzled_layout(a_shared),
                    b_shared: tilelang.layout.make_swizzled_layout(b_shared),
                })

                m_start = by * block_m
                n_start = bx * block_n
                T.clear(c_local)

                for kk in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    k_start = kk * block_k
                    for i, j in T.Parallel(block_m, block_k):
                        a_shared[i, j] = T.if_then_else(
                            (m_start + i < m) & (k_start + j < k),
                            a[m_start + i, k_start + j],
                            T.cast(0, dtype),
                        )
                    for i, j in T.Parallel(block_n, block_k):
                        b_shared[i, j] = T.if_then_else(
                            (n_start + i < n) & (k_start + j < k),
                            b[n_start + i, k_start + j],
                            T.cast(0, dtype),
                        )
                    if block_scaled:
                        scale_idx = kk * block_k // 128
                        T.clear(partial)
                        T.gemm(a_shared, b_shared, partial, transpose_B=True,
                               policy=T.GemmWarpPolicy.FullRow)
                        for i, j in T.Parallel(block_m, block_n):
                            if m_start + i < m and n_start + j < n:
                                c_local[i, j] += (
                                    partial[i, j]
                                    * scale_a[m_start + i, scale_idx]
                                    * scale_b[n_start + j, scale_idx]
                                )
                    else:
                        T.gemm(a_shared, b_shared, c_local, transpose_B=True,
                               policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(block_m, block_n):
                    if m_start + i < m and n_start + j < n:
                        if block_scaled:
                            c[m_start + i, n_start + j] = c_local[i, j]
                        else:
                            c[m_start + i, n_start + j] = (
                                c_local[i, j] * scale_a[0, 0] * scale_b[0, 0]
                            )

        @T.prim_func
        def _gemm_fp8_bias_main(
            a: T.Tensor((m, k), dtype),  # type: ignore
            b: T.Tensor((n, k), dtype),  # type: ignore
            scale_a: T.Tensor(scale_a_shape, "float32"),  # type: ignore
            scale_b: T.Tensor(scale_b_shape, "float32"),  # type: ignore
            bias: T.Tensor((n,), out_dtype),  # type: ignore
            c: T.Tensor((m, n), out_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                    T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (bx, by):
                a_shared = T.alloc_shared((block_m, block_k), dtype)
                b_shared = T.alloc_shared((block_n, block_k), dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                if block_scaled:
                    partial = T.alloc_fragment((block_m, block_n), accum_dtype)

                T.annotate_layout({
                    a_shared: tilelang.layout.make_swizzled_layout(a_shared),
                    b_shared: tilelang.layout.make_swizzled_layout(b_shared),
                })

                m_start = by * block_m
                n_start = bx * block_n
                T.clear(c_local)

                for kk in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    k_start = kk * block_k
                    for i, j in T.Parallel(block_m, block_k):
                        a_shared[i, j] = T.if_then_else(
                            (m_start + i < m) & (k_start + j < k),
                            a[m_start + i, k_start + j],
                            T.cast(0, dtype),
                        )
                    for i, j in T.Parallel(block_n, block_k):
                        b_shared[i, j] = T.if_then_else(
                            (n_start + i < n) & (k_start + j < k),
                            b[n_start + i, k_start + j],
                            T.cast(0, dtype),
                        )
                    if block_scaled:
                        scale_idx = kk * block_k // 128
                        T.clear(partial)
                        T.gemm(a_shared, b_shared, partial, transpose_B=True,
                               policy=T.GemmWarpPolicy.FullRow)
                        for i, j in T.Parallel(block_m, block_n):
                            if m_start + i < m and n_start + j < n:
                                c_local[i, j] += (
                                    partial[i, j]
                                    * scale_a[m_start + i, scale_idx]
                                    * scale_b[n_start + j, scale_idx]
                                )
                    else:
                        T.gemm(a_shared, b_shared, c_local, transpose_B=True,
                               policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(block_m, block_n):
                    if m_start + i < m and n_start + j < n:
                        if block_scaled:
                            c[m_start + i, n_start + j] = c_local[i, j] + bias[n_start + j]
                        else:
                            c[m_start + i, n_start + j] = (
                                c_local[i, j] * scale_a[0, 0] * scale_b[0, 0]
                                + bias[n_start + j]
                            )

        return _gemm_fp8_bias_main if has_bias else _gemm_fp8_main

    return _gemm_fp8_func


@functools.lru_cache(maxsize=32)
def _gemm_kernel(m: int,
                 n: int,
                 k: int,
                 trans_a: bool,
                 trans_b: bool,
                 dtype: str = "float16",
                 traced: bool = False) -> Callable:
    """Hand-written warp-specialized GEMM ``C = op(A) @ op(B)`` for Hopper (SM90).

    One producer warpgroup (128 threads) issues TMA loads into a double-buffered
    SMEM ring; one consumer warpgroup (128 threads) runs the WGMMA and accumulates
    over K. All four layouts are covered by ``trans_a`` / ``trans_b`` (forwarded to
    the WGMMA transpose flags): ``A`` is ``[M,K]`` (or ``[K,M]`` transposed), ``B``
    is ``[K,N]`` (or ``[N,K]`` transposed), ``C`` is ``[M,N]``. fp16 / bf16 inputs,
    fp32 accumulation. The auto warp-specialization pass is disabled so it does not
    fire on top of this manual layout.

    TMA constraint: each input's innermost (contiguous) dimension must be a
    multiple of 8 elements (16-byte alignment). That dim is ``K`` for a
    non-transposed ``A`` and a transposed ``B``, ``M`` for a transposed ``A``, and
    ``N`` for a non-transposed ``B``. So NT (``A@Bᵀ``) only requires ``K % 8 == 0``;
    other layouts additionally require the relevant ``M`` / ``N`` to be aligned.

    Args:
        m: Rows of ``op(A)`` / ``C``.
        n: Columns of ``op(B)`` / ``C``.
        k: Contraction dim.
        trans_a: Whether ``A`` is stored transposed (``[K, M]``).
        trans_b: Whether ``B`` is stored transposed (``[N, K]``).
        dtype: Activation / weight dtype string (``"float16"`` or ``"bfloat16"``).
        traced: Build with in-kernel timeline markers materialized (``True``) or
            stripped to zero cost (``False``). **Part of the cache key**: traced
            and untraced builds are distinct cached kernels, so flipping the
            process trace switch never returns a stale variant. Callers pass
            ``trace.enabled`` explicitly rather than letting the build read the
            global switch.

    Returns:
        A ``@tilelang.jit`` factory; calling it with ``(block_m, block_n,
        block_k, num_stages)`` returns the compiled ``prim_func``. When ``traced``
        it materializes the markers and appends a trailing ``slots`` output (so
        ``out_idx`` returns ``(C, slots)``); otherwise ``C`` is the lone output.
    """
    accum_dtype = "float"
    a_shape = (k, m) if trans_a else (m, k)
    b_shape = (n, k) if trans_b else (k, n)

    @tilelang.jit(
        out_idx=trace.out_idx(1, traced),
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gemm_func(block_m: int = 128,
                   block_n: int = 128,
                   block_k: int = 64,
                   num_stages: int = 3) -> Callable:
        # Manual 2-warpgroup WS: 1 producer WG (128 threads) issues TMA, 1
        # consumer WG (128 threads) runs WGMMA. Barrier arrive_counts (128) are
        # bound to this layout, so threads is fixed at 256.
        threads = 256
        k_iters = T.ceildiv(k, block_k)
        # SMEM tile shapes follow the storage layout; the WGMMA transpose flags
        # reconcile them with the logical (M,K) x (K,N) contraction.
        a_tile = (block_k, block_m) if trans_a else (block_m, block_k)
        b_tile = (block_n, block_k) if trans_b else (block_k, block_n)

        @T.prim_func
        def _gemm_main(
                a: T.Tensor(a_shape, dtype),  # type: ignore
                b: T.Tensor(b_shape, dtype),  # type: ignore
                c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                    T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (bx, by):
                # Multi-stage ring of A/B SMEM buffers. Indexed by stage = gi %
                # num_stages; the phase bit flips every num_stages iterations.
                a_smem = T.alloc_shared((num_stages,) + a_tile, dtype)
                b_smem = T.alloc_shared((num_stages,) + b_tile, dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)

                T.annotate_layout({
                    a_smem: tilelang.layout.make_swizzled_layout(a_smem),
                    b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                })

                # Producer→consumer (buffer full) and consumer→producer (buffer
                # empty) barriers, one per ring slot. Each is arrived by exactly
                # one warpgroup (128 threads). Allocated as length-num_stages
                # barrier arrays and indexed by the static slot id.
                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([128] * num_stages)

                # Monotonic per-warpgroup iteration counters; stage = gi %
                # num_stages, phase = (gi // num_stages) % 2.
                gi_prod = T.alloc_var("int32", init=0)
                gi_cons = T.alloc_var("int32", init=0)

                m_start = by * block_m
                n_start = bx * block_n

                tx = T.get_thread_binding()

                if tx < 128:
                    # ── Producer warpgroup: issue TMA loads of A and B tiles. ──
                    # Intern the "producer" group first so it gets gid 0.
                    T.dec_max_nreg(24)
                    with trace.group("producer", lead=0):
                        for ki in T.serial(k_iters):
                            stage = gi_prod % num_stages
                            phase = (gi_prod // num_stages) % 2
                            k_start = ki * block_k
                            # Unroll the ring-slot dispatch at trace time: each
                            # slot gets a static SMEM/barrier index under a
                            # dynamic `stage == s` guard.
                            for s in range(num_stages):
                                if stage == s:
                                    # Wait for this slot to be drained before
                                    # reuse. The consumer leaves the slot in
                                    # empty-phase (phase ^ 1) for the round the
                                    # producer is about to refill; rounds
                                    # 0..num_stages-1 see the init-0 state (phase
                                    # ^ 1 == 1) which is already satisfied by the
                                    # barrier's initial parity.
                                    T.barrier_wait(ab_empty[s], phase ^ 1)
                                    with trace.range("tma", lane="tma"):
                                        if trans_a:
                                            T.tma_copy(
                                                a[k_start:k_start + block_k,
                                                  m_start:m_start + block_m],
                                                a_smem[s, :, :], barrier=ab_full[s])
                                        else:
                                            T.tma_copy(
                                                a[m_start:m_start + block_m,
                                                  k_start:k_start + block_k],
                                                a_smem[s, :, :], barrier=ab_full[s])
                                        if trans_b:
                                            T.tma_copy(
                                                b[n_start:n_start + block_n,
                                                  k_start:k_start + block_k],
                                                b_smem[s, :, :], barrier=ab_full[s])
                                        else:
                                            T.tma_copy(
                                                b[k_start:k_start + block_k,
                                                  n_start:n_start + block_n],
                                                b_smem[s, :, :], barrier=ab_full[s])
                                    with trace.range("arrive", lane="barrier"):
                                        T.barrier_arrive(ab_full[s])
                            gi_prod = gi_prod + 1
                else:
                    # ── Consumer warpgroup: run WGMMA, accumulate over K. ──
                    T.inc_max_nreg(240)
                    T.clear(c_local)
                    with trace.group("consumer", lead=128):
                        for ki in T.serial(k_iters):
                            stage = gi_cons % num_stages
                            phase = (gi_cons // num_stages) % 2
                            for s in range(num_stages):
                                if stage == s:
                                    with trace.range("wait", lane="barrier"):
                                        T.barrier_wait(ab_full[s], phase)
                                    with trace.range("mma", lane="wgmma"):
                                        T.wgmma_gemm(
                                            a_smem[s, :, :],
                                            b_smem[s, :, :],
                                            c_local,
                                            transpose_A=trans_a,
                                            transpose_B=trans_b,
                                            policy=T.GemmWarpPolicy.FullRow,
                                            clear_accum=(ki == 0),
                                        )
                                        T.wait_wgmma(0)
                                    T.warpgroup_fence_operand(c_local, num_regs=64)
                                    T.barrier_arrive(ab_empty[s])
                            gi_cons = gi_cons + 1

                        # Epilogue: guard the M/N tail so partial tiles don't
                        # write out of bounds (m / n need not be multiples of the
                        # block sizes; K tails are zero-filled by TMA).
                        with trace.range("epilogue"):
                            for i, j in T.Parallel(block_m, block_n):
                                if m_start + i < m and n_start + j < n:
                                    c[m_start + i, n_start + j] = c_local[i, j]

                # Build-time flow declaration: producer "arrive" → consumer
                # "wait" (fixed per-iter pairing).
                trace.dag("arrive", "wait")

        # Materialize markers + append ``slots`` when traced; no-op them (identical
        # CUDA to an un-instrumented build) otherwise. Pairs with ``out_idx`` above.
        return trace.finalize(_gemm_main, traced=traced, max_events=1024)

    return _gemm_func


@torch.library.custom_op("top::gemm_wrapped_kernel", mutates_args=())
def _gemm_wrapped_kernel(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Run the warp-specialized GEMM ``C = op(A) @ op(B)`` (torch custom op).

    Kept for ``torch.compile`` compatibility (registered op + ``register_fake``).
    ``GemmKernel.forward`` calls the compiled JIT directly (cf. ``GemvKernel``),
    so this wrapper is not on the eager forward path.
    """
    return _gemm_kernel(m, n, k, trans_a, trans_b, dtype)(
        block_m, block_n, block_k, num_stages)(a, b)


@_gemm_wrapped_kernel.register_fake
def _(m: int, n: int, k: int, trans_a: bool, trans_b: bool, dtype: str,
      block_m: int, block_n: int, block_k: int, num_stages: int,
      *inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.empty((m, n), dtype=inputs[0].dtype, device=inputs[0].device)


class GemmKernel(Kernel):
    """Dense GEMM kernel: a hand-written warp-specialized implementation (SM90).

    Computes ``C = op(A) @ op(B)`` for any ``(trans_a, trans_b)`` layout. One
    producer warpgroup issues TMA loads into a double-buffered SMEM ring; one
    consumer warpgroup runs the WGMMA over K. fp16 / bf16 inputs, fp32
    accumulation. Hopper-only — TMA + WGMMA require SM90.
    """

    supported_archs: list[int] = [90]

    def __init__(self,
                 m: int,
                 n: int,
                 k: int,
                 dtype: torch.dtype,
                 config: Optional[dict] = None,
                 tune: bool = False,
                 trans_a: bool = False,
                 trans_b: bool = False) -> None:
        super().__init__()
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.trans_a = trans_a
        self.trans_b = trans_b

        self.kernel = _gemm_kernel(m, n, k, trans_a, trans_b, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 128,
            "block_n": 128,
            "block_k": 64,
            "num_stages": 3,
        }

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Call the compiled JIT directly (cf. GemvKernel); _gemm_wrapped_kernel is
        # kept only for torch.compile compatibility. trace.run dumps the timeline
        # when tracing is on and otherwise just returns C — so no branch here.
        compiled = _gemm_kernel(
            self.m, self.n, self.k, self.trans_a, self.trans_b, self.dtype_str,
            traced=trace.enabled)(**self.config)
        layout = f"{'T' if self.trans_a else 'N'}{'T' if self.trans_b else 'N'}"
        return trace.run(
            compiled, (a, b),
            stem=f"gemm_{self.m}x{self.n}x{self.k}_{layout}_{self.dtype_str}")


# TODO: add persistent, split-k, steam-k...


@functools.lru_cache(maxsize=32)
def _gemv_kernel(n: int, k: int, dtype: str = "float16") -> Callable:
    accum_dtype = "float"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _gemv_func(
        block_n: int = 8,
        reduce_threads: int = 32,
        num_stages: int = 2,
    ) -> Callable:

        max_transaction_size_in_bits = 128
        tile_k = max_transaction_size_in_bits // (str2dtype[dtype].itemsize * 8)
        block_k = reduce_threads * tile_k

        @T.prim_func
        def _gemv_main(
                a: T.Tensor((k,), dtype),
                b: T.Tensor((n, k), dtype),
                c: T.Tensor((n,), dtype),
        ):
            # threads=(reduce_threads, block_n): tk=threadIdx.x is the fast-varying
            # dimension so consecutive warp threads access consecutive columns of B
            # (same row, stride-1) → coalesced 128-bit loads.
            with T.Kernel(T.ceildiv(n, block_n), threads=(reduce_threads, block_n)) as bn:
                tk = T.get_thread_binding(0)  # threadIdx.x — varies within a warp
                tn = T.get_thread_binding(1)  # threadIdx.y — one row per warp (when reduce_threads=32)
                c_accum = T.alloc_local((1,), accum_dtype)

                T.clear(c_accum)

                # O3: pipeline B loads through shared memory using T.Pipelined.
                # T.copy issues cp.async for the next tile while the current tile is
                # being consumed, hiding HBM3e latency.
                # num_stages=1 → sequential (no overlap), num_stages>=2 → actual pipeline.
                b_shared = T.alloc_shared((block_n, block_k), dtype)
                a_local = T.alloc_local((tile_k,), dtype)

                for bk in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    # disable_tma=True: use cp.async instead of TMA to avoid
                    # mbarrier requirements that TileLang cannot infer for
                    # manually-indexed b_shared in a non-wgmma kernel.
                    T.copy(b[bn * block_n, bk * block_k], b_shared, disable_tma=True)
                    # a is tiny (fits in L1), load directly to registers
                    for _k in T.vectorized(tile_k):
                        a_local[_k] = a[bk * block_k + tk * tile_k + _k]
                    # FMA
                    for _k in T.serial(tile_k):
                        c_accum[0] += a_local[_k].astype(accum_dtype) * b_shared[
                            tn, tk * tile_k + _k].astype(accum_dtype)

                c_reduced = T.alloc_local((1,), accum_dtype)
                with T.attr(
                        T.comm_reducer(lambda x, y: x + y, [T.Cast(accum_dtype, 0)]),
                        "reduce_scope",
                        T.reinterpret(T.uint64(0), dtype="handle"),
                ):
                    T.evaluate(
                        T.tvm_thread_allreduce(
                            T.uint32(1),
                            c_accum[0],
                            True,
                            c_reduced[0],
                            tk,
                            dtype="handle",
                        ))

                c[bn * block_n + tn] = c_reduced[0]

        return _gemv_main

    return _gemv_func


@torch.library.custom_op("top::gemv_wrapped_kernel", mutates_args=())
def _gemv_wrapped_kernel(
    n: int,
    k: int,
    dtype: str,
    block_n: int,
    reduce_threads: int,
    num_stages: int,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return _gemv_kernel(n, k, dtype)(block_n, reduce_threads, num_stages)(a, b)


@_gemv_wrapped_kernel.register_fake
def _(n: int, k: int,
      dtype: str, block_n: int, reduce_threads: int, num_stages: int,
      *inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.empty((n,), dtype=inputs[0].dtype, device=inputs[0].device)


class GemvKernel(Kernel):
    supported_archs: list[int] = [90]

    def __init__(self,
                 n: int,
                 k: int,
                 dtype: torch.dtype,
                 config: Optional[dict] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.n = n
        self.k = k
        self.dtype = dtype

        self.kernel = _gemv_kernel(n, k, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        sm_version = get_sm_version()

        if sm_version in {90}:
            # reduce_threads=32: full warp per row → coalesced B access + warp shuffle reduce
            # block_n=8: 256 threads/block, 448 blocks for n=7168 → ~3.4 blocks/SM on H200
            # num_stages=2: double-buffer B tile to hide HBM3e latency
            return {
                "block_n": 8,
                "reduce_threads": 32,
                "num_stages": 2,
            }

        return {
            "block_n": 32,
            "reduce_threads": 32,
            "num_stages": 1,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        # num_stages=1: sequential shared-memory path (no overlap, baseline for comparison)
        # num_stages>=2: actual pipeline with cp.async prefetch to hide HBM latency
        return [
            {'block_n': bn, 'reduce_threads': rt, 'num_stages': ns}
            for bn in [1, 2, 4, 8, 16]
            for rt in [32]
            for ns in [1, 2, 3]
        ]

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.flatten().contiguous()
        # Call the JIT-compiled kernel directly to avoid Python overhead from
        # closure recreation + JIT cache lookup in _gemv_wrapped_kernel on every
        # forward pass. _gemv_wrapped_kernel is kept for torch.compile compatibility.
        return self.kernel(
            self.config["block_n"],
            self.config["reduce_threads"],
            self.config["num_stages"],
        )(a, b)
