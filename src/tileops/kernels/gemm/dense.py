import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.trace import trace
from tileops.utils import get_sm_count, str2dtype

from .heuristics import (
    SWAP_AB_MPAD,
    best_config,
    gemv_config,
    small_batch_config,
    swap_ab_grid_underfills,
)

_CONSUMER_BAR_WG0 = 8
_CONSUMER_BAR_WG1 = 9


def _tma_misalignment(
    m: int, n: int, k: int, dtype: torch.dtype, trans_a: bool, trans_b: bool
) -> Optional[str]:
    """Why TMA cannot address these operands, or ``None`` when it can.

    Every structure ``GemmKernel`` builds loads its tiles through TMA, whose
    descriptors address the innermost (contiguous) dimension in 16-byte units —
    so that extent must be a multiple of ``16 / itemsize`` elements, 8 for
    fp16 / bf16. Which logical dim is innermost follows the layout: ``K`` for a
    non-transposed ``A`` and a transposed ``B``, ``M`` for a transposed ``A``,
    ``N`` for a non-transposed ``B``. The bandwidth-mode kernels load through
    ``cp.async`` and carry no such requirement.

    Undeclared, an unaligned shape reaches TileLang's descriptor check and dies
    as "Check failed: (result.supported) is false", naming nothing to change.
    """
    step = 16 // dtype.itemsize
    a_dim, a_extent = ("m", m) if trans_a else ("k", k)
    b_dim, b_extent = ("k", k) if trans_b else ("n", n)
    offenders = dict.fromkeys(
        f"{d}={v}" for d, v in ((a_dim, a_extent), (b_dim, b_extent)) if v % step
    )
    if not offenders:
        return None
    layout = f"{'T' if trans_a else 'N'}{'T' if trans_b else 'N'}"
    return (
        f"TMA addresses each operand's innermost dimension in 16-byte units, so it "
        f"must be a multiple of {step} elements for {dtype}; the {layout} layout "
        f"makes that {a_dim} for a and {b_dim} for b, and {', '.join(offenders)}"
    )


__all__ = [
    "GemmFp8BlockScaledKernel",
    "GemmFp8EpilogueKernel",
    "GemmKernel",
    "GemvKernel",
    "SmallBatchGemmKernel",
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
            m, n, k, self.dtype_str, self.out_dtype_str, block_scaled=False
        )
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
                f"GemmFp8EpilogueKernel only supports torch.float8_e4m3fn, got {self.dtype}"
            )
        compiled = _gemm_fp8_kernel(
            self.m,
            self.n,
            self.k,
            self.dtype_str,
            self.out_dtype_str,
            block_scaled=False,
            has_bias=bias is not None,
        )(**self.config)
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
            m, n, k, self.dtype_str, self.out_dtype_str, block_scaled=True
        )
        self.init_config(config, tune)

    @property
    def out_dtype_str(self) -> str:
        return self.dtype_to_str(self.out_dtype)

    @property
    def default_config(self) -> dict:
        # Block scaling keeps both the unscaled WGMMA fragment and the scaled
        # accumulator live. Narrow one output axis on the register-bound prefill
        # shapes where the additional CTAs recover occupancy.
        if (self.m, self.n, self.k) == (4096, 2112, 7168):
            block_m, block_n = 128, 64
        elif (self.m, self.n, self.k) == (4096, 4096, 7168):
            block_m, block_n = 64, 128
        else:
            block_m, block_n = 128, 128
        return {
            "block_m": block_m,
            "block_n": block_n,
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
                f"GemmFp8BlockScaledKernel only supports torch.float8_e4m3fn, got {self.dtype}"
            )
        compiled = _gemm_fp8_kernel(
            self.m,
            self.n,
            self.k,
            self.dtype_str,
            self.out_dtype_str,
            block_scaled=True,
            has_bias=bias is not None,
        )(**self.config)
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
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (
                bx,
                by,
            ):
                a_shared = T.alloc_shared((block_m, block_k), dtype)
                b_shared = T.alloc_shared((block_n, block_k), dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                if block_scaled:
                    partial = T.alloc_fragment((block_m, block_n), accum_dtype)
                    # Reuse each row/column scale across the complete output tile instead
                    # of reloading it for every scaled partial element.
                    scale_a_local = T.alloc_fragment((block_m,), accum_dtype)
                    scale_b_local = T.alloc_fragment((block_n,), accum_dtype)

                T.annotate_layout(
                    {
                        a_shared: tilelang.layout.make_swizzled_layout(a_shared),
                        b_shared: tilelang.layout.make_swizzled_layout(b_shared),
                    }
                )

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
                        for i in T.Parallel(block_m):
                            scale_a_local[i] = T.if_then_else(
                                m_start + i < m,
                                scale_a[m_start + i, scale_idx],
                                0.0,
                            )
                        for j in T.Parallel(block_n):
                            scale_b_local[j] = T.if_then_else(
                                n_start + j < n,
                                scale_b[n_start + j, scale_idx],
                                0.0,
                            )
                        T.clear(partial)
                        T.gemm(
                            a_shared,
                            b_shared,
                            partial,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        for i, j in T.Parallel(block_m, block_n):
                            if m_start + i < m and n_start + j < n:
                                c_local[i, j] += partial[i, j] * scale_a_local[i] * scale_b_local[j]
                    else:
                        T.gemm(
                            a_shared,
                            b_shared,
                            c_local,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )

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
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (
                bx,
                by,
            ):
                a_shared = T.alloc_shared((block_m, block_k), dtype)
                b_shared = T.alloc_shared((block_n, block_k), dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                if block_scaled:
                    partial = T.alloc_fragment((block_m, block_n), accum_dtype)
                    # Reuse each row/column scale across the complete output tile instead
                    # of reloading it for every scaled partial element.
                    scale_a_local = T.alloc_fragment((block_m,), accum_dtype)
                    scale_b_local = T.alloc_fragment((block_n,), accum_dtype)

                T.annotate_layout(
                    {
                        a_shared: tilelang.layout.make_swizzled_layout(a_shared),
                        b_shared: tilelang.layout.make_swizzled_layout(b_shared),
                    }
                )

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
                        for i in T.Parallel(block_m):
                            scale_a_local[i] = T.if_then_else(
                                m_start + i < m,
                                scale_a[m_start + i, scale_idx],
                                0.0,
                            )
                        for j in T.Parallel(block_n):
                            scale_b_local[j] = T.if_then_else(
                                n_start + j < n,
                                scale_b[n_start + j, scale_idx],
                                0.0,
                            )
                        T.clear(partial)
                        T.gemm(
                            a_shared,
                            b_shared,
                            partial,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        for i, j in T.Parallel(block_m, block_n):
                            if m_start + i < m and n_start + j < n:
                                c_local[i, j] += partial[i, j] * scale_a_local[i] * scale_b_local[j]
                    else:
                        T.gemm(
                            a_shared,
                            b_shared,
                            c_local,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )

                for i, j in T.Parallel(block_m, block_n):
                    if m_start + i < m and n_start + j < n:
                        if block_scaled:
                            c[m_start + i, n_start + j] = c_local[i, j] + bias[n_start + j]
                        else:
                            c[m_start + i, n_start + j] = (
                                c_local[i, j] * scale_a[0, 0] * scale_b[0, 0] + bias[n_start + j]
                            )

        return _gemm_fp8_bias_main if has_bias else _gemm_fp8_main

    return _gemm_fp8_func


@functools.lru_cache(maxsize=32)
def _gemm_kernel(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: str = "float16",
    traced: bool = False,
    *,
    sm_count: int,
) -> Callable:
    """Hand-written warp-specialized GEMM ``C = op(A) @ op(B)`` for Hopper (SM90).

    One producer warpgroup (128 threads) issues TMA loads into a double-buffered
    SMEM ring; one consumer warpgroup (128 threads) runs the WGMMA and accumulates
    over K. All four layouts are covered by ``trans_a`` / ``trans_b`` (forwarded to
    the WGMMA transpose flags): ``A`` is $[M \\times K]$ (or $[K \\times M]$ transposed), ``B``
    is $[K \\times N]$ (or $[N \\times K]$ transposed), ``C`` is $[M \\times N]$. fp16 / bf16 inputs,
    fp32 accumulation. The auto warp-specialization pass is disabled so it does not
    fire on top of this manual layout.

    Operands must satisfy TMA's innermost-dimension alignment, which
    ``_tma_misalignment`` states and ``GemmKernel`` refuses on.

    Args:
        m: Rows of ``op(A)`` / ``C``.
        n: Columns of ``op(B)`` / ``C``.
        k: Contraction dim.
        trans_a: Whether ``A`` is stored transposed ($[K \\times M]$).
        trans_b: Whether ``B`` is stored transposed ($[N \\times K]$).
        dtype: Activation / weight dtype string (``"float16"`` or ``"bfloat16"``).
        traced: Build with in-kernel timeline markers materialized (``True``) or
            stripped to zero cost (``False``). **Part of the cache key**: traced
            and untraced builds are distinct cached kernels, so flipping the
            process trace switch never returns a stale variant. Callers pass
            ``trace.enabled`` explicitly rather than letting the build read the
            global switch.
        sm_count: Device SM count, deciding the multi-wave TMA-store epilogue
            gate. Part of the cache key so a kernel built for one GPU is never
            reused on another.

    Returns:
        A ``@tilelang.jit`` factory; calling it with ``(block_m, block_n,
        block_k, num_stages, panel_size)`` returns the compiled ``prim_func``.
        When ``traced``
        it materializes the markers and appends a trailing ``slots`` output (so
        ``out_idx`` returns ``(C, slots)``); otherwise ``C`` is the lone output.
    """
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=trace.out_idx(1, traced),
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gemm_func(
        block_m: int = 128,
        block_n: int = 128,
        block_k: int = 64,
        num_stages: int = 3,
        panel_size: int = 10,
    ) -> Callable:
        # Manual 2-warpgroup WS: 1 producer WG (128 threads) issues TMA, 1
        # consumer WG (128 threads) runs WGMMA. Barrier arrive_counts (128) are
        # bound to this layout, so threads is fixed at 256.
        threads = 256
        k_iters = T.ceildiv(k, block_k)
        # SMEM tile shapes follow the storage layout; the WGMMA transpose flags
        # reconcile them with the logical (M,K) x (K,N) contraction.
        a_tile = (block_k, block_m) if trans_a else (block_m, block_k)
        b_tile = (block_n, block_k) if trans_b else (block_k, block_n)
        grid_size = -(-n // block_n) * -(-m // block_m)
        tma_epilogue = (n * 2) % 16 == 0 and grid_size > sm_count

        @T.prim_func
        def _gemm_main(
            a: T.Tensor((k, m) if trans_a else (m, k), dtype),  # type: ignore
            b: T.Tensor((n, k) if trans_b else (k, n), dtype),  # type: ignore
            c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (
                bx,
                by,
            ):
                T.use_swizzle(panel_size, enable=panel_size > 0)
                # Multi-stage ring of A/B SMEM buffers. Indexed by stage = gi %
                # num_stages; the phase bit flips every num_stages iterations.
                a_smem = T.alloc_shared((num_stages,) + a_tile, dtype)
                b_smem = T.alloc_shared((num_stages,) + b_tile, dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)

                if tma_epilogue:
                    c_smem = T.alloc_shared((block_m, block_n), dtype)
                    T.annotate_layout(
                        {
                            a_smem: tilelang.layout.make_swizzled_layout(a_smem),
                            b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                            c_smem: tilelang.layout.make_swizzled_layout(c_smem),
                        }
                    )
                else:
                    T.annotate_layout(
                        {
                            a_smem: tilelang.layout.make_swizzled_layout(a_smem),
                            b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                        }
                    )

                # Producer→consumer (buffer full) and consumer→producer (buffer
                # empty) barriers, one per ring slot. Each is arrived by exactly
                # one warpgroup (128 threads). Allocated as length-num_stages
                # barrier arrays and indexed by the static slot id.
                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([128] * num_stages)

                # Monotonic per-warpgroup iteration counters; stage = gi %
                # num_stages, phase = (gi // num_stages) % 2.

                m_start = by * block_m
                n_start = bx * block_n

                ps = T.alloc_local((1,), "int32")

                tx = T.get_thread_binding()

                if tx < 128:
                    # ── Producer warpgroup: issue TMA loads of A and B tiles. ──
                    # Intern the "producer" group first so it gets gid 0.
                    T.dec_max_nreg(24)
                    with trace.group("producer", lead=0):
                        for ki in T.serial(k_iters):
                            slot = ki % num_stages
                            phase = (ki // num_stages) % 2
                            k_start = ki * block_k
                            T.barrier_wait(ab_empty[slot], phase ^ 1)
                            with trace.range("tma", lane="tma"):
                                if trans_a:
                                    T.tma_copy(
                                        a[k_start : k_start + block_k, m_start : m_start + block_m],
                                        a_smem[slot, :, :],
                                        barrier=ab_full[slot],
                                    )
                                else:
                                    T.tma_copy(
                                        a[m_start : m_start + block_m, k_start : k_start + block_k],
                                        a_smem[slot, :, :],
                                        barrier=ab_full[slot],
                                    )
                                if trans_b:
                                    T.tma_copy(
                                        b[n_start : n_start + block_n, k_start : k_start + block_k],
                                        b_smem[slot, :, :],
                                        barrier=ab_full[slot],
                                    )
                                else:
                                    T.tma_copy(
                                        b[k_start : k_start + block_k, n_start : n_start + block_n],
                                        b_smem[slot, :, :],
                                        barrier=ab_full[slot],
                                    )
                            with trace.range("arrive", lane="barrier"):
                                T.barrier_arrive(ab_full[slot])
                else:
                    # ── Consumer warpgroup: run WGMMA, accumulate over K. ──
                    T.inc_max_nreg(240)
                    num_accum_regs = (block_m * block_n) // 128
                    with trace.group("consumer", lead=128):
                        for ki in T.serial(k_iters):
                            slot = ki % num_stages
                            phase = (ki // num_stages) % 2
                            with trace.range("wait", lane="barrier"):
                                T.barrier_wait(ab_full[slot], phase)
                            with trace.range("mma", lane="wgmma"):
                                T.wgmma_gemm(
                                    a_smem[slot, :, :],
                                    b_smem[slot, :, :],
                                    c_local,
                                    transpose_A=trans_a,
                                    transpose_B=trans_b,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    clear_accum=(ki == 0),
                                )
                            if ki > 0:
                                T.wait_wgmma(1)
                                T.warpgroup_fence_operand(c_local, num_regs=num_accum_regs)
                                T.barrier_arrive(ab_empty[ps[0]])
                            ps[0] = slot

                        T.wait_wgmma(0)
                        T.warpgroup_fence_operand(c_local, num_regs=num_accum_regs)
                        T.barrier_arrive(ab_empty[ps[0]])
                        if tma_epilogue:
                            with trace.range("epilogue"):
                                T.copy(c_local, c_smem)
                                T.fence_proxy_async()
                                T.sync_threads(barrier_id=4, arrive_count=128)
                                T.copy(c_smem, c[m_start, n_start])
                        else:
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


@functools.lru_cache(maxsize=32)
def _gemm_splitk_kernel(
    m: int, n: int, k: int, trans_a: bool, trans_b: bool, dtype: str = "float16"
) -> Callable:
    """Split-K variant of the warp-specialized GEMM mainloop.

    The K contraction is sliced across ``split_k`` CTAs (grid z). Each CTA
    runs the same producer/consumer pipeline as ``_gemm_kernel`` over its
    K slice and writes an fp32 partial tile to the workspace
    ``w[split_k, m, n]``; ``_splitk_reduce_kernel`` then sums the slices and
    casts to the storage dtype. Splitting only pays off when the natural
    (M, N) grid underfills the GPU — see ``GemmKernel.forward`` for the
    dispatch. ``split_k`` must divide the K-tile count evenly.

    Args:
        m: Rows of ``op(A)`` / ``C``.
        n: Columns of ``op(B)`` / ``C``.
        k: Contraction dim.
        trans_a: Whether ``A`` is stored transposed (``[K, M]``).
        trans_b: Whether ``B`` is stored transposed (``[N, K]``).
        dtype: Activation / weight dtype string (``"float16"`` or ``"bfloat16"``).

    Returns:
        A ``@tilelang.jit`` factory; calling it with ``(block_m, block_n,
        block_k, num_stages, panel_size, split_k)`` returns the compiled
        ``prim_func`` producing the fp32 workspace.
    """
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gemm_splitk_func(
        block_m: int = 128,
        block_n: int = 128,
        block_k: int = 64,
        num_stages: int = 4,
        panel_size: int = 16,
        split_k: int = 2,
    ) -> Callable:
        threads = 256
        k_iters = T.ceildiv(k, block_k)
        if k_iters % split_k != 0:
            raise ValueError(
                f"split_k={split_k} must divide the K-tile count evenly "
                f"(k={k}, block_k={block_k} -> {k_iters} tiles)"
            )
        k_slice = k_iters // split_k
        a_tile = (block_k, block_m) if trans_a else (block_m, block_k)
        b_tile = (block_n, block_k) if trans_b else (block_k, block_n)

        @T.prim_func
        def _gemm_splitk_main(
            a: T.Tensor((k, m) if trans_a else (m, k), dtype),  # type: ignore
            b: T.Tensor((n, k) if trans_b else (k, n), dtype),  # type: ignore
            w: T.Tensor((split_k, m, n), accum_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                T.ceildiv(n, block_n), T.ceildiv(m, block_m), split_k, threads=threads
            ) as (bx, by, bz):
                T.use_swizzle(panel_size, enable=panel_size > 0)
                a_smem = T.alloc_shared((num_stages,) + a_tile, dtype)
                b_smem = T.alloc_shared((num_stages,) + b_tile, dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)

                T.annotate_layout(
                    {
                        a_smem: tilelang.layout.make_swizzled_layout(a_smem),
                        b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                    }
                )

                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([128] * num_stages)
                ps = T.alloc_local((1,), "int32")

                m_start = by * block_m
                n_start = bx * block_n
                ki_base = bz * k_slice

                tx = T.get_thread_binding()

                if tx < 128:
                    T.dec_max_nreg(24)
                    for ki in T.serial(k_slice):
                        slot = ki % num_stages
                        phase = (ki // num_stages) % 2
                        k_start = (ki_base + ki) * block_k
                        T.barrier_wait(ab_empty[slot], phase ^ 1)
                        if trans_a:
                            T.tma_copy(
                                a[k_start : k_start + block_k, m_start : m_start + block_m],
                                a_smem[slot, :, :],
                                barrier=ab_full[slot],
                            )
                        else:
                            T.tma_copy(
                                a[m_start : m_start + block_m, k_start : k_start + block_k],
                                a_smem[slot, :, :],
                                barrier=ab_full[slot],
                            )
                        if trans_b:
                            T.tma_copy(
                                b[n_start : n_start + block_n, k_start : k_start + block_k],
                                b_smem[slot, :, :],
                                barrier=ab_full[slot],
                            )
                        else:
                            T.tma_copy(
                                b[k_start : k_start + block_k, n_start : n_start + block_n],
                                b_smem[slot, :, :],
                                barrier=ab_full[slot],
                            )
                        T.barrier_arrive(ab_full[slot])
                else:
                    T.inc_max_nreg(240)
                    num_accum_regs = (block_m * block_n) // 128
                    for ki in T.serial(k_slice):
                        slot = ki % num_stages
                        phase = (ki // num_stages) % 2
                        T.barrier_wait(ab_full[slot], phase)
                        T.wgmma_gemm(
                            a_smem[slot, :, :],
                            b_smem[slot, :, :],
                            c_local,
                            transpose_A=trans_a,
                            transpose_B=trans_b,
                            policy=T.GemmWarpPolicy.FullRow,
                            clear_accum=(ki == 0),
                        )
                        if ki > 0:
                            T.wait_wgmma(1)
                            T.warpgroup_fence_operand(c_local, num_regs=num_accum_regs)
                            T.barrier_arrive(ab_empty[ps[0]])
                        ps[0] = slot

                    T.wait_wgmma(0)
                    T.warpgroup_fence_operand(c_local, num_regs=num_accum_regs)
                    T.barrier_arrive(ab_empty[ps[0]])
                    for i, j in T.Parallel(block_m, block_n):
                        if m_start + i < m and n_start + j < n:
                            w[bz, m_start + i, n_start + j] = c_local[i, j]

        return _gemm_splitk_main

    return _gemm_splitk_func


@functools.lru_cache(maxsize=32)
def _splitk_reduce_kernel(split_k: int, m: int, n: int, dtype: str = "float16") -> Callable:
    """Reduce the split-K fp32 workspace into the final output.

    Sums ``w[split_k, m, n]`` over the slice axis in fp32 and casts to the
    storage dtype at the boundary. Bandwidth-trivial elementwise kernel; the
    workspace of the shapes worth splitting fits in L2.

    ``C`` is an explicit parameter rather than a JIT-allocated output
    (``out_idx``): the caller allocates it *before* launching the mainloop, so
    the allocation no longer sits between the two launches. On short-mainloop
    shapes the mainloop drains in ~14 us while the host is still allocating,
    and the span metric charges that idle to us (see ``_splitk_pair``).
    """
    accum_dtype = "float"

    @tilelang.jit(compile_flags=["-O3", "-DENABLE_BF16"])
    def _splitk_reduce_func(elems_per_cta: int = 1024) -> Callable:
        def _slice_sum(w, gi, gj):
            expr = w[0, gi, gj]
            for s in range(1, split_k):
                expr = expr + w[s, gi, gj]
            return expr

        @T.prim_func
        def _splitk_reduce_main(
            w: T.Tensor((split_k, m, n), accum_dtype),  # type: ignore
            c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            total = m * n
            with T.Kernel(T.ceildiv(total, elems_per_cta), threads=256) as bx:
                base = bx * elems_per_cta
                for t in T.Parallel(elems_per_cta):
                    idx = base + t
                    if idx < total:
                        gi = idx // n
                        gj = idx % n
                        c[gi, gj] = T.cast(_slice_sum(w, gi, gj), dtype)

        return _splitk_reduce_main

    return _splitk_reduce_func


@functools.lru_cache(maxsize=32)
def _gemm_coop2_kernel(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: str = "float16",
    *,
    sm_count: int,
) -> Callable:
    """Persistent 2-consumer (cooperative) warp-specialized GEMM for Hopper.

    Matches the cuBLAS Hopper cooperative (``coopA``) layout: one producer
    warpgroup (128 threads) plus **two** consumer warpgroups (256 threads,
    384 total). A ``block_m x block_n`` output tile is split along M — each
    consumer owns ``block_m // 2`` rows and runs its own WGMMA; the ``B`` tile
    is loaded once and shared (split-A / shared-B). Two math warpgroups double
    the WGMMA issue rate over the single-consumer ``_gemm_kernel``, which is the
    edge on compute-bound prefill shapes (large M, GPU-filling grid).

    A static-wave persistent loop (grid = ``sm_count``; each CTA sweeps tile ids
    ``flat_id = sm_count * w + pid``) overlaps a tile's TMA-store epilogue with
    the next tile's mainloop prologue: the ring counters carry across waves so
    the producer keeps prefetching. A Triton-style grouped tile order
    (``group_size_m``) keeps concurrently-resident CTAs on a shared ``B`` column
    stripe for L2 reuse.

    NT only (``A[m,k] @ B[n,k]ᵀ``): the split-A layout and shared ``B`` ring are
    specific to a non-transposed ``A`` and transposed ``B``. Other layouts fall
    back to ``_gemm_kernel``. M / N tails are handled by a predicated scalar
    epilogue (full tiles use the TMA store); K tails are TMA zero-filled.

    Args:
        m: Rows of ``A`` / ``C``.
        n: Columns of ``op(B)`` / ``C``.
        k: Contraction dim.
        trans_a: Must be ``False`` (NT layout).
        trans_b: Must be ``True`` (NT layout).
        dtype: Activation / weight dtype string (``"float16"`` / ``"bfloat16"``).
        sm_count: Persistent grid width — the device SM count. Part of the cache
            key so a kernel built for one GPU is never reused on another.

    Returns:
        A ``@tilelang.jit`` factory; calling it with ``(block_n, block_k,
        num_stages, group_size_m, stage_n)`` returns the compiled ``prim_func``.
    """
    if trans_a or not trans_b:
        raise ValueError("_gemm_coop2_kernel is NT-only (trans_a=False, trans_b=True)")
    accum_dtype = "float"
    block_m = 128

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gemm_coop2_func(
        block_n: int = 256,
        block_k: int = 64,
        num_stages: int = 3,
        group_size_m: int = 16,
        stage_n: int = 0,
    ) -> Callable:
        half_m = block_m // 2
        nr = (half_m * block_n) // 128
        sn = block_n if stage_n <= 0 else stage_n
        n_chunks = block_n // sn
        num_pid_m = -(-m // block_m)
        num_pid_n = -(-n // block_n)
        total_tiles = num_pid_m * num_pid_n
        max_waves = -(-total_tiles // sm_count) + 1
        k_iters = T.ceildiv(k, block_k)

        @T.macro
        def decode(flat_id, mt, nt):
            gin = T.int32(group_size_m * num_pid_n)
            gid = flat_id // gin
            first_m = gid * T.int32(group_size_m)
            gsize = T.min(T.int32(group_size_m), T.int32(num_pid_m) - first_m)
            mt[0] = first_m + (flat_id % gin) % gsize
            nt[0] = (flat_id % gin) // gsize

        @T.prim_func
        def _gemm_coop2_main(
            a: T.Tensor((m, k), dtype),  # type: ignore
            b: T.Tensor((n, k), dtype),  # type: ignore
            c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(sm_count, threads=384) as (pid,):
                a_smem_top = T.alloc_shared((num_stages, half_m, block_k), dtype)
                a_smem_bot = T.alloc_shared((num_stages, half_m, block_k), dtype)
                b_smem = T.alloc_shared((num_stages, block_n, block_k), dtype)
                c_local_0 = T.alloc_fragment((half_m, block_n), accum_dtype)
                c_local_1 = T.alloc_fragment((half_m, block_n), accum_dtype)
                c_cast_0 = T.alloc_fragment((half_m, block_n), dtype)
                c_cast_1 = T.alloc_fragment((half_m, block_n), dtype)
                c_smem_0 = T.alloc_shared((half_m, sn), dtype)
                c_smem_1 = T.alloc_shared((half_m, sn), dtype)

                T.annotate_layout(
                    {
                        a_smem_top: tilelang.layout.make_swizzled_layout(a_smem_top),
                        a_smem_bot: tilelang.layout.make_swizzled_layout(a_smem_bot),
                        b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                        c_smem_0: tilelang.layout.make_swizzled_layout(c_smem_0),
                        c_smem_1: tilelang.layout.make_swizzled_layout(c_smem_1),
                    }
                )

                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([256] * num_stages)

                gi_prod = T.alloc_var("int32", init=0)
                gi_cons_0 = T.alloc_var("int32", init=0)
                gi_cons_1 = T.alloc_var("int32", init=0)
                ps0 = T.alloc_local((1,), "int32")
                ps1 = T.alloc_local((1,), "int32")
                mt = T.alloc_local((1,), "int32")
                nt = T.alloc_local((1,), "int32")

                tx = T.get_thread_binding()

                if tx < 128:
                    T.dec_max_nreg(24)
                    for w in T.serial(max_waves):
                        flat_id = T.int32(sm_count) * w + pid
                        if flat_id < total_tiles:
                            decode(flat_id, mt, nt)
                            m_start = mt[0] * block_m
                            n_start = nt[0] * block_n
                            for ki in T.Pipelined(k_iters, num_stages=0):
                                slot = gi_prod % num_stages
                                ks = ki * block_k
                                T.barrier_wait(ab_empty[slot], ((gi_prod // num_stages) & 1) ^ 1)
                                T.tma_copy(
                                    a[m_start : m_start + half_m, ks : ks + block_k],
                                    a_smem_top[slot, :, :],
                                    barrier=ab_full[slot],
                                )
                                T.tma_copy(
                                    a[m_start + half_m : m_start + block_m, ks : ks + block_k],
                                    a_smem_bot[slot, :, :],
                                    barrier=ab_full[slot],
                                )
                                T.tma_copy(
                                    b[n_start : n_start + block_n, ks : ks + block_k],
                                    b_smem[slot, :, :],
                                    barrier=ab_full[slot],
                                )
                                T.barrier_arrive(ab_full[slot])
                                gi_prod = gi_prod + 1

                elif tx < 256:
                    T.inc_max_nreg(240)
                    for w in T.serial(max_waves):
                        flat_id = T.int32(sm_count) * w + pid
                        if flat_id < total_tiles:
                            decode(flat_id, mt, nt)
                            m_start = mt[0] * block_m
                            n_start = nt[0] * block_n
                            arows = T.min(T.int32(half_m), T.int32(m) - m_start)
                            acols = T.min(T.int32(block_n), T.int32(n) - n_start)
                            for ki in T.Pipelined(k_iters, num_stages=0):
                                slot = gi_cons_0 % num_stages
                                T.barrier_wait(ab_full[slot], (gi_cons_0 // num_stages) & 1)
                                T.wgmma_gemm(
                                    a_smem_top[slot, :, :],
                                    b_smem[slot, :, :],
                                    c_local_0,
                                    transpose_B=True,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    clear_accum=(ki == 0),
                                )
                                if ki > 0:
                                    T.wait_wgmma(1)
                                    T.barrier_arrive(ab_empty[ps0[0]])
                                ps0[0] = slot
                                gi_cons_0 = gi_cons_0 + 1
                            T.wait_wgmma(0)
                            T.barrier_arrive(ab_empty[ps0[0]])
                            T.warpgroup_fence_operand(c_local_0, num_regs=nr)
                            T.copy(c_local_0, c_cast_0)
                            if arows == T.int32(half_m) and acols == T.int32(block_n):
                                for ch in range(n_chunks):
                                    c0 = ch * sn
                                    T.sync_threads(barrier_id=_CONSUMER_BAR_WG0, arrive_count=128)
                                    T.copy(c_cast_0[:, c0 : c0 + sn], c_smem_0)
                                    T.fence_proxy_async()
                                    T.sync_threads(barrier_id=_CONSUMER_BAR_WG0, arrive_count=128)
                                    T.copy(c_smem_0, c[m_start, n_start + c0])
                            else:
                                for i, j in T.Parallel(half_m, block_n):
                                    if i < arows and j < acols:
                                        c[m_start + i, n_start + j] = c_cast_0[i, j]

                else:
                    T.inc_max_nreg(240)
                    for w in T.serial(max_waves):
                        flat_id = T.int32(sm_count) * w + pid
                        if flat_id < total_tiles:
                            decode(flat_id, mt, nt)
                            m_start = mt[0] * block_m
                            n_start = nt[0] * block_n
                            arows = T.max(
                                T.int32(0),
                                T.min(T.int32(half_m), T.int32(m) - m_start - T.int32(half_m)),
                            )
                            acols = T.min(T.int32(block_n), T.int32(n) - n_start)
                            for ki in T.Pipelined(k_iters, num_stages=0):
                                slot = gi_cons_1 % num_stages
                                T.barrier_wait(ab_full[slot], (gi_cons_1 // num_stages) & 1)
                                T.wgmma_gemm(
                                    a_smem_bot[slot, :, :],
                                    b_smem[slot, :, :],
                                    c_local_1,
                                    transpose_B=True,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    clear_accum=(ki == 0),
                                )
                                if ki > 0:
                                    T.wait_wgmma(1)
                                    T.barrier_arrive(ab_empty[ps1[0]])
                                ps1[0] = slot
                                gi_cons_1 = gi_cons_1 + 1
                            T.wait_wgmma(0)
                            T.barrier_arrive(ab_empty[ps1[0]])
                            T.warpgroup_fence_operand(c_local_1, num_regs=nr)
                            T.copy(c_local_1, c_cast_1)
                            if arows == T.int32(half_m) and acols == T.int32(block_n):
                                for ch in range(n_chunks):
                                    c0 = ch * sn
                                    T.sync_threads(barrier_id=_CONSUMER_BAR_WG1, arrive_count=128)
                                    T.copy(c_cast_1[:, c0 : c0 + sn], c_smem_1)
                                    T.fence_proxy_async()
                                    T.sync_threads(barrier_id=_CONSUMER_BAR_WG1, arrive_count=128)
                                    T.copy(c_smem_1, c[m_start + half_m, n_start + c0])
                            elif arows > T.int32(0):
                                for i, j in T.Parallel(half_m, block_n):
                                    if i < arows and j < acols:
                                        c[m_start + half_m + i, n_start + j] = c_cast_1[i, j]

        return _gemm_coop2_main

    return _gemm_coop2_func


@functools.lru_cache(maxsize=32)
def _gemm_coop2_splitk_kernel(
    m: int, n: int, k: int, trans_a: bool, trans_b: bool, dtype: str = "float16"
) -> Callable:
    """Split-K variant of the 2-consumer (cooperative) GEMM mainloop (NT).

    For small-M shapes whose natural (M, N) grid underfills the GPU with a
    single K-slice, this slices K across grid-z CTAs. Each CTA runs the coop2
    mainloop (1 producer + 2 math WGs, split-A / shared-B, ``block_m`` fixed at
    128 = two 64-row consumers) over its K-slice and writes an fp32 partial tile
    into ``w[split_k, m, n]``; ``_splitk_reduce_kernel`` then sums the slices and
    casts to the storage dtype.

    The 2-consumer mainloop is the more WGMMA-efficient of the two, so it takes
    the large-K decode shapes, whose slices amortize the reduce round-trip;
    dispatch keeps short-K shapes on ``_gemm_splitk_kernel``.

    NT only. ``split_k`` must divide the K-tile count evenly.

    Args:
        m: Rows of ``A`` / ``C``.
        n: Columns of ``op(B)`` / ``C``.
        k: Contraction dim.
        trans_a: Must be ``False`` (NT layout).
        trans_b: Must be ``True`` (NT layout).
        dtype: Activation / weight dtype string (``"float16"`` / ``"bfloat16"``).

    Returns:
        A ``@tilelang.jit`` factory; calling it with ``(block_n, block_k,
        num_stages, split_k)`` returns the compiled ``prim_func`` producing the
        fp32 workspace ``w[split_k, m, n]``.
    """
    if trans_a or not trans_b:
        raise ValueError("_gemm_coop2_splitk_kernel is NT-only (trans_a=False, trans_b=True)")
    accum_dtype = "float"
    block_m = 128

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gemm_coop2_splitk_func(
        block_n: int = 64, block_k: int = 128, num_stages: int = 4, split_k: int = 4
    ) -> Callable:
        half_m = block_m // 2
        nr = (half_m * block_n) // 128
        k_iters_total = T.ceildiv(k, block_k)
        if k_iters_total % split_k != 0:
            raise ValueError(
                f"split_k={split_k} must divide the K-tile count evenly "
                f"(k={k}, block_k={block_k} -> {k_iters_total} tiles)"
            )
        k_slice = k_iters_total // split_k

        @T.prim_func
        def _gemm_coop2_splitk_main(
            a: T.Tensor((m, k), dtype),  # type: ignore
            b: T.Tensor((n, k), dtype),  # type: ignore
            w: T.Tensor((split_k, m, n), accum_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), split_k, threads=384) as (
                bx,
                by,
                bz,
            ):
                a_smem_top = T.alloc_shared((num_stages, half_m, block_k), dtype)
                a_smem_bot = T.alloc_shared((num_stages, half_m, block_k), dtype)
                b_smem = T.alloc_shared((num_stages, block_n, block_k), dtype)
                c_local_0 = T.alloc_fragment((half_m, block_n), accum_dtype)
                c_local_1 = T.alloc_fragment((half_m, block_n), accum_dtype)
                T.annotate_layout(
                    {
                        a_smem_top: tilelang.layout.make_swizzled_layout(a_smem_top),
                        a_smem_bot: tilelang.layout.make_swizzled_layout(a_smem_bot),
                        b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                    }
                )
                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([256] * num_stages)

                ps0 = T.alloc_local((1,), "int32")
                ps1 = T.alloc_local((1,), "int32")

                m_start = by * block_m
                n_start = bx * block_n
                ki_base = bz * k_slice
                tx = T.get_thread_binding()

                if tx < 128:
                    T.dec_max_nreg(24)
                    for ki in T.Pipelined(k_slice, num_stages=0):
                        slot = ki % num_stages
                        ks = (ki_base + ki) * block_k
                        T.barrier_wait(ab_empty[slot], ((ki // num_stages) % 2) ^ 1)
                        T.tma_copy(
                            a[m_start : m_start + half_m, ks : ks + block_k],
                            a_smem_top[slot, :, :],
                            barrier=ab_full[slot],
                        )
                        T.tma_copy(
                            a[m_start + half_m : m_start + block_m, ks : ks + block_k],
                            a_smem_bot[slot, :, :],
                            barrier=ab_full[slot],
                        )
                        T.tma_copy(
                            b[n_start : n_start + block_n, ks : ks + block_k],
                            b_smem[slot, :, :],
                            barrier=ab_full[slot],
                        )
                        T.barrier_arrive(ab_full[slot])
                elif tx < 256:
                    T.inc_max_nreg(240)
                    for ki in T.Pipelined(k_slice, num_stages=0):
                        slot = ki % num_stages
                        T.barrier_wait(ab_full[slot], (ki // num_stages) % 2)
                        T.wgmma_gemm(
                            a_smem_top[slot, :, :],
                            b_smem[slot, :, :],
                            c_local_0,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            clear_accum=(ki == 0),
                        )
                        if ki > 0:
                            T.wait_wgmma(1)
                            T.barrier_arrive(ab_empty[ps0[0]])
                        ps0[0] = slot
                    T.wait_wgmma(0)
                    T.barrier_arrive(ab_empty[ps0[0]])
                    T.warpgroup_fence_operand(c_local_0, num_regs=nr)
                    for i, j in T.Parallel(half_m, block_n):
                        if m_start + i < m and n_start + j < n:
                            w[bz, m_start + i, n_start + j] = c_local_0[i, j]
                else:
                    T.inc_max_nreg(240)
                    for ki in T.Pipelined(k_slice, num_stages=0):
                        slot = ki % num_stages
                        T.barrier_wait(ab_full[slot], (ki // num_stages) % 2)
                        T.wgmma_gemm(
                            a_smem_bot[slot, :, :],
                            b_smem[slot, :, :],
                            c_local_1,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            clear_accum=(ki == 0),
                        )
                        if ki > 0:
                            T.wait_wgmma(1)
                            T.barrier_arrive(ab_empty[ps1[0]])
                        ps1[0] = slot
                    T.wait_wgmma(0)
                    T.barrier_arrive(ab_empty[ps1[0]])
                    T.warpgroup_fence_operand(c_local_1, num_regs=nr)
                    for i, j in T.Parallel(half_m, block_n):
                        if m_start + half_m + i < m and n_start + j < n:
                            w[bz, m_start + half_m + i, n_start + j] = c_local_1[i, j]

        return _gemm_coop2_splitk_main

    return _gemm_coop2_splitk_func


@functools.lru_cache(maxsize=32)
def _splitk_pair(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: str,
    coop2: bool,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    panel_size: int,
    split_k: int,
) -> tuple[Callable, Callable]:
    """Resolve the (mainloop, reduce) compiled pair for a split-K config.

    Both split-K paths run two kernels back to back, so every microsecond the
    host spends between the two launches is GPU idle the span metric charges
    to us: on the short-mainloop shapes the mainloop drains before the reduce
    is even enqueued. Folding the builder lookup and the ``@tilelang.jit``
    factory call of *both* kernels into one cached resolution keeps that
    window to the two launches themselves (measured: inter-kernel gap
    4.6-5.7 us -> 1.7-2.6 us on the m<=128 x 2112 x 7168 family).

    The other host step that used to land in that window is allocating ``C``.
    ``_splitk_reduce_kernel`` therefore takes it as an explicit parameter, and
    both callers allocate it *before* launching the mainloop, which closes the
    remaining gap to the 1.1 us floor torch measures on the same rows.
    """
    if coop2:
        mainloop = _gemm_coop2_splitk_kernel(m, n, k, trans_a, trans_b, dtype)(
            block_n, block_k, num_stages, split_k
        )
    else:
        mainloop = _gemm_splitk_kernel(m, n, k, trans_a, trans_b, dtype)(
            block_m, block_n, block_k, num_stages, panel_size, split_k
        )
    return mainloop, _splitk_reduce_kernel(split_k, m, n, dtype)()


@functools.lru_cache(maxsize=32)
def _gemm_simple_kernel(
    m: int, n: int, k: int, trans_a: bool, trans_b: bool, dtype: str = "float16"
) -> Callable:
    """Non-warp-specialized pipelined GEMM for short-mainloop shapes (SM90).

    A stock ``T.Pipelined`` + ``T.gemm`` kernel: every thread cooperates in
    both copy and math, and the compiler schedules the cp.async/TMA overlap.
    On short-K skinny-M NT shapes (the decode-down family: ~16 K iterations,
    about one CTA wave) this beats the warp-specialized kernel by ~4% — with
    so short a mainloop the WS producer warpgroup's fixed costs (barrier
    protocol per iteration, idle tail, 128 threads not doing math) outweigh
    the benefit of its hand-managed deeper ring.

    Selected via config only (``simple: True``, pinned per-shape in
    ``GemmKernel._TUNED_CONFIGS``). Requires tiles that divide the problem
    exactly; the builder raises ``ValueError`` otherwise.
    """
    if trans_a:
        raise ValueError("_gemm_simple_kernel supports trans_a=False only")
    accum_dtype = "float"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _gemm_simple_func(
        block_m: int = 64,
        block_n: int = 128,
        block_k: int = 128,
        num_stages: int = 4,
        threads: int = 128,
        panel_size: int = 8,
        cluster_m: int = 1,
    ) -> Callable:
        if m % block_m or n % block_n or k % block_k:
            raise ValueError(
                f"_gemm_simple_kernel requires exact tiling: got "
                f"m={m} % {block_m}, n={n} % {block_n}, k={k} % {block_k}"
            )
        if cluster_m > 1:
            if (m // block_m) % cluster_m:
                raise ValueError(
                    f"cluster_m={cluster_m} must divide the M grid ({m // block_m} tiles)"
                )
            if panel_size > 0:
                raise ValueError("cluster_m > 1 requires panel_size == 0")
        b_tile = (block_n, block_k) if trans_b else (block_k, block_n)

        def _launch():
            if cluster_m > 1:
                return T.ClusterKernel(
                    n // block_n, m // block_m, cluster_dims=(1, cluster_m, 1), threads=threads
                )
            return T.Kernel(n // block_n, m // block_m, threads=threads)

        @T.prim_func
        def _gemm_simple_main(
            a: T.Tensor((m, k), dtype),  # type: ignore
            b: T.Tensor((n, k) if trans_b else (k, n), dtype),  # type: ignore
            c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with _launch() as (bx, by):
                T.use_swizzle(panel_size, enable=panel_size > 0)
                a_smem = T.alloc_shared((block_m, block_k), dtype)
                b_smem = T.alloc_shared(b_tile, dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                T.clear(c_local)
                for ki in T.Pipelined(k // block_k, num_stages=num_stages):
                    T.copy(a[by * block_m, ki * block_k], a_smem)
                    if trans_b:
                        T.copy(b[bx * block_n, ki * block_k], b_smem)
                    else:
                        T.copy(b[ki * block_k, bx * block_n], b_smem)
                    T.gemm(a_smem, b_smem, c_local, transpose_B=trans_b)
                T.copy(c_local, c[by * block_m, bx * block_n])

        return _gemm_simple_main

    return _gemm_simple_func


@functools.lru_cache(maxsize=32)
def _gemm_swap_ab_kernel(
    m: int, n: int, k: int, trans_a: bool, trans_b: bool, dtype: str = "float16"
) -> Callable:
    """Operand-swapped tiny-m NT GEMM: ``C[m,n] = A[m,k] @ B[n,k]ᵀ``, m <= 8.

    Tiling the output the usual way wastes the M dimension at ``m <= 8``: WGMMA
    needs 64 rows, so ``A`` is padded 8-32x and the grid is only
    ``ceil(n / block_n)`` CTAs, too few to fill the device.

    Computing the transpose instead, ``Cᵀ[n,m] = B[n,k] @ A[m,k]ᵀ``, keeps the
    same NT operand form but puts ``n`` on the 64-row WGMMA axis and ``m`` on
    the 8-wide one: no padding waste, and the grid becomes
    ``ceil(n / block_nn)``. The epilogue stages the ``(block_nn, 8)`` tile
    through SMEM and writes ``c[mi, n0 + j]``, contiguous along ``n``.

    Only worth it when that grid fills enough of the device — see
    ``heuristics._swap_ab_stages``, which also sets ``num_stages``: with
    fewer CTAs resident the ring has to be deeper to hide the same latency.

    Args:
        m: Batch rows (2..8).
        n: Output columns (weight rows).
        k: Contraction dim; innermost for both ``a`` and ``b``.
        trans_a: Must be ``False`` (NT layout).
        trans_b: Must be ``True`` (NT layout).
        dtype: Activation / weight dtype string (``"float16"`` / ``"bfloat16"``).

    Returns:
        A ``@tilelang.jit`` factory; calling it with ``(block_nn, block_k,
        num_stages)`` returns the compiled ``prim_func``.
    """
    if trans_a or not trans_b:
        raise ValueError("_gemm_swap_ab_kernel is NT-only (trans_a=False, trans_b=True)")
    if m > SWAP_AB_MPAD:
        raise ValueError(f"_gemm_swap_ab_kernel serves m <= {SWAP_AB_MPAD}, got m={m}")
    accum_dtype = "float"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _gemm_swap_ab_func(block_nn: int = 64, block_k: int = 128, num_stages: int = 4) -> Callable:
        mpad = SWAP_AB_MPAD

        @T.prim_func
        def _gemm_swap_ab_main(
            a: T.Tensor((m, k), dtype),  # type: ignore
            b: T.Tensor((n, k), dtype),  # type: ignore
            c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(n, block_nn), threads=128) as bx:
                b_smem = T.alloc_shared((block_nn, block_k), dtype)
                a_smem = T.alloc_shared((mpad, block_k), dtype)
                ct_local = T.alloc_fragment((block_nn, mpad), accum_dtype)
                ct_cast = T.alloc_fragment((block_nn, mpad), dtype)
                ct_smem = T.alloc_shared((block_nn, mpad), dtype)
                T.clear(ct_local)
                for ki in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    T.copy(b[bx * block_nn, ki * block_k], b_smem)
                    T.copy(a[0, ki * block_k], a_smem)
                    T.gemm(b_smem, a_smem, ct_local, transpose_B=True)
                T.copy(ct_local, ct_cast)
                T.copy(ct_cast, ct_smem)
                for mi, j in T.Parallel(mpad, block_nn):
                    if mi < m and bx * block_nn + j < n:
                        c[mi, bx * block_nn + j] = ct_smem[j, mi]

        return _gemm_swap_ab_main

    return _gemm_swap_ab_func


@functools.lru_cache(maxsize=32)
def _gemm_coop2s_kernel(
    m: int, n: int, k: int, trans_a: bool, trans_b: bool, dtype: str = "float16"
) -> Callable:
    """Single-tile 2-consumer (cooperative) GEMM for small NN shapes (SM90).

    ``_gemm_coop2_kernel`` stripped of its persistent loop: the grid *is* the
    tile grid (``n / block_n`` by ``m / block_m``), so a CTA computes exactly
    one output tile and needs no cross-wave ring carry, no tile decode and no
    grouped swizzle. Small square shapes cannot amortize that machinery: the
    mainloop is only ``k / block_k`` iterations. One producer warpgroup issues
    TMA into a ``num_stages`` ring; two consumer warpgroups each own
    ``block_m // 2 = 64`` rows and share the ``B`` tile (split-A / shared-B).

    NN only (``A[m,k] @ B[k,n]``): ``B`` tiles load as ``(block_k, block_n)``
    and feed WGMMA with ``transpose_B=False``. Requires tiles that divide the
    problem exactly; the builder raises ``ValueError`` otherwise. Selected via
    config only (``coop2s: True``, pinned per-shape in
    ``GemmKernel._TUNED_CONFIGS``).

    Args:
        m: Rows of ``A`` / ``C``.
        n: Columns of ``B`` / ``C``.
        k: Contraction dim.
        trans_a: Must be ``False`` (NN layout).
        trans_b: Must be ``False`` (NN layout).
        dtype: Activation / weight dtype string (``"float16"`` / ``"bfloat16"``).

    Returns:
        A ``@tilelang.jit`` factory; calling it with ``(block_n, block_k,
        num_stages)`` returns the compiled ``prim_func``.
    """
    if trans_a or trans_b:
        raise ValueError("_gemm_coop2s_kernel is NN-only (trans_a=False, trans_b=False)")
    accum_dtype = "float"
    block_m = 128

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gemm_coop2s_func(block_n: int = 64, block_k: int = 128, num_stages: int = 4) -> Callable:
        if m % block_m or n % block_n or k % block_k:
            raise ValueError(
                f"_gemm_coop2s_kernel requires exact tiling: got "
                f"m={m} % {block_m}, n={n} % {block_n}, k={k} % {block_k}"
            )
        half_m = block_m // 2
        nr = (half_m * block_n) // 128
        k_iters = k // block_k

        @T.prim_func
        def _gemm_coop2s_main(
            a: T.Tensor((m, k), dtype),  # type: ignore
            b: T.Tensor((k, n), dtype),  # type: ignore
            c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(n // block_n, m // block_m, threads=384) as (bx, by):
                a_smem_top = T.alloc_shared((num_stages, half_m, block_k), dtype)
                a_smem_bot = T.alloc_shared((num_stages, half_m, block_k), dtype)
                b_smem = T.alloc_shared((num_stages, block_k, block_n), dtype)
                c_local_0 = T.alloc_fragment((half_m, block_n), accum_dtype)
                c_local_1 = T.alloc_fragment((half_m, block_n), accum_dtype)
                c_cast_0 = T.alloc_fragment((half_m, block_n), dtype)
                c_cast_1 = T.alloc_fragment((half_m, block_n), dtype)
                c_smem_0 = T.alloc_shared((half_m, block_n), dtype)
                c_smem_1 = T.alloc_shared((half_m, block_n), dtype)

                T.annotate_layout(
                    {
                        a_smem_top: tilelang.layout.make_swizzled_layout(a_smem_top),
                        a_smem_bot: tilelang.layout.make_swizzled_layout(a_smem_bot),
                        b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                        c_smem_0: tilelang.layout.make_swizzled_layout(c_smem_0),
                        c_smem_1: tilelang.layout.make_swizzled_layout(c_smem_1),
                    }
                )

                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([256] * num_stages)
                m_start = by * block_m
                n_start = bx * block_n
                ps0 = T.alloc_local((1,), "int32")
                ps1 = T.alloc_local((1,), "int32")

                tx = T.get_thread_binding()

                if tx < 128:
                    T.dec_max_nreg(24)
                    for ki in T.serial(k_iters):
                        slot = ki % num_stages
                        phase = (ki // num_stages) % 2
                        ks = ki * block_k
                        T.barrier_wait(ab_empty[slot], phase ^ 1)
                        T.tma_copy(
                            a[m_start : m_start + half_m, ks : ks + block_k],
                            a_smem_top[slot, :, :],
                            barrier=ab_full[slot],
                        )
                        T.tma_copy(
                            a[m_start + half_m : m_start + block_m, ks : ks + block_k],
                            a_smem_bot[slot, :, :],
                            barrier=ab_full[slot],
                        )
                        T.tma_copy(
                            b[ks : ks + block_k, n_start : n_start + block_n],
                            b_smem[slot, :, :],
                            barrier=ab_full[slot],
                        )
                        T.barrier_arrive(ab_full[slot])
                elif tx < 256:
                    T.inc_max_nreg(240)
                    for ki in T.serial(k_iters):
                        slot = ki % num_stages
                        phase = (ki // num_stages) % 2
                        T.barrier_wait(ab_full[slot], phase)
                        T.wgmma_gemm(
                            a_smem_top[slot, :, :],
                            b_smem[slot, :, :],
                            c_local_0,
                            transpose_B=False,
                            policy=T.GemmWarpPolicy.FullRow,
                            clear_accum=(ki == 0),
                        )
                        if ki > 0:
                            T.wait_wgmma(1)
                            T.barrier_arrive(ab_empty[ps0[0]])
                            T.warpgroup_fence_operand(c_local_0, num_regs=nr)
                        ps0[0] = slot
                    T.wait_wgmma(0)
                    T.warpgroup_fence_operand(c_local_0, num_regs=nr)
                    T.barrier_arrive(ab_empty[ps0[0]])
                    T.copy(c_local_0, c_cast_0)
                    T.sync_threads(barrier_id=_CONSUMER_BAR_WG0, arrive_count=128)
                    T.copy(c_cast_0, c_smem_0)
                    T.fence_proxy_async()
                    T.sync_threads(barrier_id=_CONSUMER_BAR_WG0, arrive_count=128)
                    T.copy(c_smem_0, c[m_start, n_start])
                else:
                    T.inc_max_nreg(240)
                    for ki in T.serial(k_iters):
                        slot = ki % num_stages
                        phase = (ki // num_stages) % 2
                        T.barrier_wait(ab_full[slot], phase)
                        T.wgmma_gemm(
                            a_smem_bot[slot, :, :],
                            b_smem[slot, :, :],
                            c_local_1,
                            transpose_B=False,
                            policy=T.GemmWarpPolicy.FullRow,
                            clear_accum=(ki == 0),
                        )
                        if ki > 0:
                            T.wait_wgmma(1)
                            T.barrier_arrive(ab_empty[ps1[0]])
                            T.warpgroup_fence_operand(c_local_1, num_regs=nr)
                        ps1[0] = slot
                    T.wait_wgmma(0)
                    T.warpgroup_fence_operand(c_local_1, num_regs=nr)
                    T.barrier_arrive(ab_empty[ps1[0]])
                    T.copy(c_local_1, c_cast_1)
                    T.sync_threads(barrier_id=_CONSUMER_BAR_WG1, arrive_count=128)
                    T.copy(c_cast_1, c_smem_1)
                    T.fence_proxy_async()
                    T.sync_threads(barrier_id=_CONSUMER_BAR_WG1, arrive_count=128)
                    T.copy(c_smem_1, c[m_start + half_m, n_start])

        return _gemm_coop2s_main

    return _gemm_coop2s_func


@torch.library.custom_op("tileops::gemm_wrapped_kernel", mutates_args=())
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
    panel_size: int,
    split_k: int,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Run the warp-specialized GEMM ``C = op(A) @ op(B)`` (torch custom op).

    Kept for ``torch.compile`` compatibility (registered op + ``register_fake``).
    ``GemmKernel.forward`` calls the compiled JIT directly (cf. ``GemvKernel``),
    so this wrapper is not on the eager forward path.
    """
    if split_k > 1:
        mainloop, reduce_ = _splitk_pair(
            m,
            n,
            k,
            trans_a,
            trans_b,
            dtype,
            False,
            block_m,
            block_n,
            block_k,
            num_stages,
            panel_size,
            split_k,
        )
        c = torch.empty((m, n), dtype=a.dtype, device=a.device)
        reduce_(mainloop(a, b), c)
        return c
    return _gemm_kernel(m, n, k, trans_a, trans_b, dtype, sm_count=get_sm_count())(
        block_m, block_n, block_k, num_stages, panel_size
    )(a, b)


@_gemm_wrapped_kernel.register_fake
def _(
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
    panel_size: int,
    split_k: int,
    *inputs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    return torch.empty((m, n), dtype=inputs[0].dtype, device=inputs[0].device)


class GemmKernel(Kernel):
    """Dense GEMM kernel family: hand-written Hopper (SM90) implementations.

    Computes ``C = op(A) @ op(B)`` for any ``(trans_a, trans_b)`` layout. The
    default structure is warp-specialized: one producer warpgroup issues TMA
    loads into a multi-stage SMEM ring, one consumer warpgroup runs the WGMMA
    over K. Structure flags in ``config`` select the coop2 / coop2s /
    coop2_splitk / simple / split-K variants instead (see ``forward``).
    fp16 / bf16 inputs, fp32 accumulation. Hopper-only — TMA + WGMMA
    require SM90.
    """

    supported_archs: list[int] = [90]
    general = True

    _STRUCTURE_FLAGS = ("coop2", "coop2s", "coop2_splitk", "simple", "swap_ab")

    def init_config(self, config: Optional[dict] = None, tune: bool = False) -> None:
        """Take a structure-flagged explicit config verbatim.

        The base merge walks ``default_config``'s keys — right for one schema per
        class, wrong here: it drops the caller's flag and grafts their tile values
        onto whichever structure the selector picked. Asking for ``coop2s`` on a
        shape served by ``coop2`` produced a ``coop2`` config at ``coop2s``' tile
        width, a combination ``_enumerate`` deliberately excludes.

        ``default_config`` applies the same rule to ``_TUNED_CONFIGS`` hits over a
        narrower flag set; widening it there would stop merging the modal keys
        into the shipped ``simple`` / ``coop2_splitk`` pins, so the two stay
        separate deliberately.
        """
        if config is not None and any(config.get(f) for f in self._STRUCTURE_FLAGS):
            self.config = dict(config)
            print(f"{type(self).__name__} initialized with config: {self.config}")
            return
        super().init_config(config, tune)

    @classmethod
    def applies(cls, call) -> bool:
        return (
            _tma_misalignment(call.m, call.n, call.k, call.dtype, call.trans_a, call.trans_b)
            is None
        )

    @classmethod
    def refusal(cls, call) -> Optional[str]:
        archs = cls.supported_archs
        if archs is not None and call.arch not in archs:
            return super().refusal(call)
        return _tma_misalignment(call.m, call.n, call.k, call.dtype, call.trans_a, call.trans_b)

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
        trans_a: bool = False,
        trans_b: bool = False,
    ) -> None:
        super().__init__()
        misaligned = _tma_misalignment(m, n, k, dtype, trans_a, trans_b)
        if misaligned is not None:
            raise ValueError(f"{type(self).__name__} cannot serve {m}x{n}x{k}: {misaligned}")
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.sm_count = get_sm_count()
        self.device_name = torch.cuda.get_device_name()

        self.kernel = _gemm_kernel(
            m, n, k, trans_a, trans_b, self.dtype_str, sm_count=self.sm_count
        )

        self.init_config(config, tune)

    _TUNED_CONFIGS: dict = {
        (128, 2112, 7168, False, True, "bfloat16"): {
            "coop2_splitk": True,
            "block_n": 64,
            "block_k": 128,
            "num_stages": 4,
            "split_k": 4,
        },
        (1024, 1024, 1024, False, False, "float16"): {
            "coop2s": True,
            "block_n": 64,
            "block_k": 128,
            "num_stages": 4,
        },
        (1024, 1024, 1024, False, False, "bfloat16"): {
            "coop2s": True,
            "block_n": 64,
            "block_k": 128,
            "num_stages": 4,
        },
        (128, 7168, 2048, False, True, "bfloat16"): {
            "simple": True,
            "block_m": 64,
            "block_n": 128,
            "block_k": 128,
            "num_stages": 4,
            "threads": 128,
            "panel_size": 0,
            "cluster_m": 2,
        },
        (64, 7168, 2048, False, True, "bfloat16"): {
            "simple": True,
            "block_m": 64,
            "block_n": 64,
            "block_k": 128,
            "num_stages": 4,
            "threads": 128,
            "panel_size": 8,
        },
        (4096, 2112, 7168, False, True, "bfloat16"): {
            "coop2": True,
            "block_n": 192,
            "block_k": 64,
            "num_stages": 5,
            "group_size_m": 16,
            "stage_n": 96,
        },
        (4096, 4096, 7168, False, True, "float16"): {
            "coop2": True,
            "block_n": 256,
            "block_k": 64,
            "num_stages": 3,
            "group_size_m": 16,
            "stage_n": 0,
        },
        (4096, 4096, 7168, False, True, "bfloat16"): {
            "coop2": True,
            "block_n": 256,
            "block_k": 64,
            "num_stages": 3,
            "group_size_m": 16,
            "stage_n": 0,
        },
        (4096, 7168, 2048, False, True, "bfloat16"): {
            "coop2": True,
            "block_n": 256,
            "block_k": 64,
            "num_stages": 4,
            "group_size_m": 16,
            "stage_n": 128,
        },
        (4096, 7168, 16384, False, True, "bfloat16"): {
            "coop2": True,
            "block_n": 256,
            "block_k": 64,
            "num_stages": 3,
            "group_size_m": 16,
            "stage_n": 0,
        },
        (4096, 24576, 1536, False, True, "bfloat16"): {
            "coop2": True,
            "block_n": 256,
            "block_k": 64,
            "num_stages": 3,
            "group_size_m": 16,
            "stage_n": 0,
        },
    }

    @property
    def default_config(self) -> dict:
        modal = {
            "block_m": 128,
            "block_n": 128,
            "block_k": 64,
            "num_stages": 4,
            "panel_size": 16,
            "split_k": 1,
        }
        override = self._TUNED_CONFIGS.get(
            (self.m, self.n, self.k, self.trans_a, self.trans_b, self.dtype_str)
        )
        if override is not None:
            self_contained = override.get("coop2") or override.get("coop2s")
            return dict(override) if self_contained else {**modal, **override}
        scored = best_config(
            self.m, self.n, self.k, self.trans_a, self.trans_b, self.sm_count, self.device_name
        )
        return scored if scored is not None else modal

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.config.get("simple"):
            cfg = self.config
            compiled = _gemm_simple_kernel(
                self.m, self.n, self.k, self.trans_a, self.trans_b, self.dtype_str
            )(
                cfg["block_m"],
                cfg["block_n"],
                cfg["block_k"],
                cfg["num_stages"],
                cfg.get("threads", 128),
                cfg.get("panel_size", 8),
                cfg.get("cluster_m", 1),
            )
            return compiled(a, b)

        if self.config.get("swap_ab"):
            cfg = self.config
            compiled = _gemm_swap_ab_kernel(
                self.m, self.n, self.k, self.trans_a, self.trans_b, self.dtype_str
            )(cfg["block_nn"], cfg["block_k"], cfg["num_stages"])
            return compiled(a, b)

        if self.config.get("coop2s"):
            cfg = self.config
            compiled = _gemm_coop2s_kernel(
                self.m, self.n, self.k, self.trans_a, self.trans_b, self.dtype_str
            )(cfg["block_n"], cfg["block_k"], cfg["num_stages"])
            return compiled(a, b)

        if self.config.get("coop2"):
            cfg = self.config
            compiled = _gemm_coop2_kernel(
                self.m,
                self.n,
                self.k,
                self.trans_a,
                self.trans_b,
                self.dtype_str,
                sm_count=self.sm_count,
            )(
                cfg["block_n"],
                cfg["block_k"],
                cfg["num_stages"],
                cfg["group_size_m"],
                cfg.get("stage_n", 0),
            )
            return compiled(a, b)

        if self.config.get("coop2_splitk"):
            cfg = self.config
            mainloop, reduce_ = _splitk_pair(
                self.m,
                self.n,
                self.k,
                self.trans_a,
                self.trans_b,
                self.dtype_str,
                True,
                0,
                cfg["block_n"],
                cfg["block_k"],
                cfg["num_stages"],
                0,
                cfg["split_k"],
            )
            c = torch.empty((self.m, self.n), dtype=a.dtype, device=a.device)
            reduce_(mainloop(a, b), c)
            return c

        split_k = self.config.get("split_k", 1)
        if split_k > 1:
            cfg = self.config
            mainloop, reduce_ = _splitk_pair(
                self.m,
                self.n,
                self.k,
                self.trans_a,
                self.trans_b,
                self.dtype_str,
                False,
                cfg["block_m"],
                cfg["block_n"],
                cfg["block_k"],
                cfg["num_stages"],
                cfg["panel_size"],
                split_k,
            )
            c = torch.empty((self.m, self.n), dtype=a.dtype, device=a.device)
            reduce_(mainloop(a, b), c)
            return c

        main_cfg = {k2: v for k2, v in self.config.items() if k2 != "split_k"}
        compiled = _gemm_kernel(
            self.m,
            self.n,
            self.k,
            self.trans_a,
            self.trans_b,
            self.dtype_str,
            traced=trace.enabled,
            sm_count=self.sm_count,
        )(**main_cfg)
        layout = f"{'T' if self.trans_a else 'N'}{'T' if self.trans_b else 'N'}"
        return trace.run(
            compiled, (a, b), stem=f"gemm_{self.m}x{self.n}x{self.k}_{layout}_{self.dtype_str}"
        )


@functools.lru_cache(maxsize=32)
def _gemm_small_batch_kernel(m: int, n: int, k: int, dtype: str = "float16") -> Callable:
    """Bandwidth-bound NT GEMM ``C[m,n] = A[m,k] @ B[n,k]ᵀ`` for small ``m`` (SM90).

    The weight matrix ``B`` is streamed once through a cp.async SMEM ring and
    each B-tile is reused across all ``m`` rows — the regime where arithmetic
    intensity is ``~m`` and saturating HBM, not tensor-core rate, is the goal.
    One ``tvm_thread_allreduce`` over the ``tk`` reduce lanes runs per output
    row.

    Serves both bandwidth-mode kernels of this family: ``SmallBatchGemmKernel``
    at its dispatched ``m``, and ``GemvKernel`` at ``m = 1`` (the matrix-vector
    case is this kernel with a one-row ``A``, not a separate implementation —
    ``for mi in T.serial(1)`` folds away).

    The epilogue write is N-tail guarded, which is load-bearing rather than
    defensive: ``gemv_config`` selects ``block_n = 2`` for ``k >= 12288``, and
    without the guard an odd ``n`` writes one element past ``c[.., n-1]``. The K
    tail needs no mask written here — both tail loads are predicated by the
    buffer extents in the emitted code (see the mainloop comment).

    Args:
        m: Batch rows (``1`` for the GEMV case, else the dispatched small ``m``).
        n: Output columns (weight rows).
        k: Contraction dim; the innermost dim of both ``a`` and ``b``.
        dtype: TileLang dtype string (``"float16"`` / ``"bfloat16"``).

    Returns:
        A JIT factory ``(block_n, reduce_threads, num_stages) -> compiled`` whose
        compiled kernel maps ``(a[m,k], b[n,k]) -> c[m,n]``.
    """
    accum_dtype = "float"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _gemm_small_batch_func(
        block_n: int = 1,
        reduce_threads: int = 128,
        num_stages: int = 4,
    ) -> Callable:
        tile_k = 128 // (str2dtype[dtype].itemsize * 8)
        block_k = reduce_threads * tile_k

        @T.prim_func
        def _gemm_small_batch_main(
            a: T.Tensor((m, k), dtype),
            b: T.Tensor((n, k), dtype),
            c: T.Tensor((m, n), dtype),
        ):
            with T.Kernel(T.ceildiv(n, block_n), threads=(reduce_threads, block_n)) as bn:
                tk = T.get_thread_binding(0)
                tn = T.get_thread_binding(1)
                c_accum = T.alloc_local((m,), accum_dtype)
                T.clear(c_accum)
                b_shared = T.alloc_shared((block_n, block_k), dtype)
                a_local = T.alloc_local((m, tile_k), dtype)

                for bk in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    T.copy(b[bn * block_n, bk * block_k], b_shared, disable_tma=True)
                    for mi in T.serial(m):
                        for _k in T.vectorized(tile_k):
                            a_local[mi, _k] = a[mi, bk * block_k + tk * tile_k + _k]
                    for mi in T.serial(m):
                        for _k in T.serial(tile_k):
                            c_accum[mi] += a_local[mi, _k].astype(accum_dtype) * b_shared[
                                tn, tk * tile_k + _k
                            ].astype(accum_dtype)

                c_reduced = T.alloc_local((1,), accum_dtype)
                for mi in T.serial(m):
                    with T.attr(
                        T.comm_reducer(lambda x, y: x + y, [T.Cast(accum_dtype, 0)]),
                        "reduce_scope",
                        T.reinterpret(T.uint64(0), dtype="handle"),
                    ):
                        T.evaluate(
                            T.tvm_thread_allreduce(
                                T.uint32(1), c_accum[mi], True, c_reduced[0], tk, dtype="handle"
                            )
                        )
                    if bn * block_n + tn < n:
                        c[mi, bn * block_n + tn] = c_reduced[0]

        return _gemm_small_batch_main

    return _gemm_small_batch_func


@torch.library.custom_op("tileops::gemv_wrapped_kernel", mutates_args=())
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
    """The GEMV path as a registered op; off the eager path, as ``_gemm_wrapped_kernel``.

    Adapts ranks around the shared ``[m, k] -> [m, n]`` body: this op's registered
    ``a[k] -> c[n]`` contract predates that body and callers depend on it.
    """
    c = _gemm_small_batch_kernel(1, n, k, dtype)(block_n, reduce_threads, num_stages)(
        a.reshape(1, -1), b
    )
    return c.reshape(n)


@_gemv_wrapped_kernel.register_fake
def _(
    n: int,
    k: int,
    dtype: str,
    block_n: int,
    reduce_threads: int,
    num_stages: int,
    *inputs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    return torch.empty((n,), dtype=inputs[0].dtype, device=inputs[0].device)


_TILE_K = 8
_SMEM_CAP = 224 * 1024


def _bandwidth_autotune_grid(rts: tuple, bns: tuple, nss: tuple) -> list[dict]:
    """Config grid for the bandwidth-mode kernels, guarded by thread and SMEM caps."""
    return [
        {"block_n": bn, "reduce_threads": rt, "num_stages": ns}
        for rt in rts
        for bn in bns
        if rt * bn <= 1024
        for ns in nss
        if bn * (rt * _TILE_K) * 2 * ns <= _SMEM_CAP
    ]


class GemvKernel(Kernel):
    """Matrix-vector product; serves the layouts a vector operand can take."""

    supported_archs: list[int] = [90]

    @classmethod
    def applies(cls, call) -> bool:
        return call.gemv_mode is not None

    def __init__(
        self, n: int, k: int, dtype: torch.dtype, config: Optional[dict] = None, tune: bool = False
    ) -> None:
        super().__init__()
        self.n = n
        self.k = k
        self.dtype = dtype

        self.kernel = _gemm_small_batch_kernel(1, n, k, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return gemv_config(self.k)

    @property
    def autotune_configs(self) -> list[dict]:
        return _bandwidth_autotune_grid((32, 64, 128, 256), (1, 2, 4, 8, 16), (1, 2, 3, 4, 5, 6))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.reshape(1, -1).contiguous()
        return self.kernel(
            self.config["block_n"],
            self.config["reduce_threads"],
            self.config["num_stages"],
        )(a, b)


class SmallBatchGemmKernel(Kernel):
    """Small-batch (small-m, NT) kernel-mode of ``GemmFwdOp`` — a batched GEMV.

    Builds :func:`_gemm_small_batch_kernel`, the same body :class:`GemvKernel`
    builds at ``m = 1``; the two classes differ only in the region they serve and
    the config band they pick. Its inner loop pays ``m`` FMAs and ``m`` converts
    per weight element on CUDA cores, so its lead over the tensor-core
    ``GemmKernel`` shrinks as ``m`` grows; :meth:`applies` states the band.

    Scope: SM90, NT only — ``B`` is ``[N,K]``, so K is contiguous and the
    reduction over it coalesces; no other layout has that property. The kernel is
    correct for any ``m``; the band above is what it claims.

    Args:
        m: Batch rows.
        n: Output columns.
        k: Contraction dim.
        dtype: Input/output torch dtype (fp16 / bf16); fp32 accumulation.
        config: Optional explicit config; defaults to :attr:`default_config`.
        tune: Whether to autotune over :attr:`autotune_configs`.
    """

    supported_archs: list[int] = [90]

    @classmethod
    def applies(cls, call) -> bool:
        """``m == 2`` NT, while a 64-wide n-tiling still underfills a wave.

        Above that fill a generic config streams the same weights with no padded
        ``A`` re-read and wins; ``m == 1`` belongs to :class:`GemvKernel`.
        """
        if call.trans_a or not call.trans_b or call.m != 2:
            return False
        return swap_ab_grid_underfills(call.n, call.sm_count)

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.kernel = _gemm_small_batch_kernel(m, n, k, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return small_batch_config(self.n, self.k, get_sm_count())

    @property
    def autotune_configs(self) -> list[dict]:
        return _bandwidth_autotune_grid((32, 64, 128), (1, 2, 4), (2, 3, 4, 5))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.kernel(
            self.config["block_n"],
            self.config["reduce_threads"],
            self.config["num_stages"],
        )(a, b)
