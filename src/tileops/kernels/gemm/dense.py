import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.trace import trace
from tileops.utils import get_sm_count, str2dtype

from .call_spec import gemv_region
from .call_spec import gemv_region, small_batch_region
from .heuristics import SWAP_AB_MPAD as _SWAP_AB_MPAD
from .heuristics import SWAP_AB_MPAD, best_config, gemv_config, small_batch_config
from .heuristics import best_config as _heuristic_best_config
from .heuristics import gemv_config, small_batch_config

# ── SMEM ring protocol, shared by every warp-specialized kernel below ──
#
# ``_gemm_kernel``, ``_gemm_splitk_kernel``, ``_gemm_coop2_kernel``,
# ``_gemm_coop2_splitk_kernel``, ``_gemm_coop2s_kernel``. Producer arrives on
# ``ab_full[s]`` once slot ``s`` holds its TMA payload; every consumer arrives
# on ``ab_empty[s]`` once it is done reading ``s`` (``arrive_count`` 128 on
# full, 128 or 256 on empty depending on consumer count).
#
# ``slot = ki % num_stages``, ``phase = (ki // num_stages) % 2``. The producer
# waits on ``phase ^ 1``: the consumer leaves a slot in the inverted phase for
# the round about to be refilled, and rounds ``0..num_stages-1`` see the init-0
# state the barrier's initial parity already satisfies.
#
# The consumer releases a slot one iteration late — it drains the previous
# WGMMA (``wait_wgmma(1)``) and frees that slot while the WGMMA it just issued
# stays in flight, overlapping the next ``barrier_wait``.
#
# Three details are load-bearing and were each paid for in measurement; changing
# any of them back needs new numbers:
#
# - ``ps*[0]`` carries the previous slot in a register. Recomputing it as
#   ``(ki - 1) % num_stages`` adds a second modulo to the hot loop and cost
#   4-13% on the coop2 rows (worst at ``num_stages=5``: prefill-gate-up
#   0.904 -> 0.803; k-dominant 1.021 -> 0.886).
# - Slot indices are dynamic. Unrolling the dispatch to hand each op a
#   compile-time index (``for s in range(num_stages): if slot == s:``) was
#   believed to be required by the timeline tracer; it is not — dynamic
#   indexing decodes to the same traced timeline event-for-event and runs
#   2-3% faster.
# - ``_gemm_coop2_kernel`` is the only kernel that needs its own ``gi_*``
#   counters: its persistent wave loop carries the ring across tiles, so the
#   ring index outlives one tile's ``ki``. Elsewhere such a counter is
#   identically equal to ``ki``.
#
# ``warpgroup_fence_operand`` placement is *not* settled and the kernels below
# differ on it: the single-consumer pair and coop2s fence inside the
# deferred-release branch, the two coop2 kernels only after the final drain.
# Neither form has ever changed a result here, and the one attempt to unify them
# changed ``ps*[0]`` in the same run, so it separated nothing. Settling it needs
# a single-axis measurement at coop2's accumulator width (``nr = 128`` at
# ``block_n = 256``), where per-iteration cost is largest.
#
# Named-barrier ids for the per-warpgroup epilogues of the 2-consumer kernels.
# TileLang allocates its own ids for the syncs it inserts around a
# fragment-to-shared copy, and it hands them out from the bottom: with 4 and 5
# here it took 3 for one consumer and 4 for the other, so that consumer's
# implicit barrier aliased the other's explicit one and could release it early
# -- a store could then read a half-written ``c_smem``. Five of the six shipped
# coop2 configs aliased that way (``compute-sanitizer --tool synccheck`` flags
# it; results stayed right only because the two warpgroups run near lockstep).
# Keep these clear of the range TileLang allocates from, and re-run synccheck
# over the generated source after touching either epilogue.
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
        # TMA-store epilogue needs C's row stride 16-byte aligned (descriptor
        # constraint). It also only pays off on multi-wave grids: T.copy's
        # store drains via tma_store_wait, and that drain is hidden by the
        # next wave's compute. On a single wave (H100/H200: 132 SMs) it sits
        # exposed on the critical path and loses to fire-and-forget scalar
        # stores (measured: decode shapes -13..-22%). Resolved at trace time;
        # the fallback keeps the supported-shape envelope unchanged.
        grid_size = -(-n // block_n) * -(-m // block_m)
        tma_epilogue = (n * 2) % 16 == 0 and grid_size > 132

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
                # L2 tile rasterization: serpentine panels of ``panel_size``
                # M-rows so concurrently resident CTAs cover a compact M x N
                # patch and re-read A/B panels from L2 instead of DRAM.
                # ``panel_size <= 0`` disables the remap (identity order).
                T.use_swizzle(panel_size, enable=panel_size > 0)
                # Multi-stage ring of A/B SMEM buffers. Indexed by stage = gi %
                # num_stages; the phase bit flips every num_stages iterations.
                a_smem = T.alloc_shared((num_stages,) + a_tile, dtype)
                b_smem = T.alloc_shared((num_stages,) + b_tile, dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)

                if tma_epilogue:
                    # Epilogue staging tile: the accumulator fragment is cast
                    # to the storage dtype here, then written back with one
                    # TMA store.
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

                # Previous ring slot, kept in a register: recomputing it
                # costs a second ``% num_stages`` in the hot loop.
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
                        # Epilogue (TMA path): cast the fp32 accumulator into
                        # the SMEM staging tile, then issue one TMA store. M/N
                        # tails need no predicate — they sit on the physical
                        # bounds of ``c``, and the TMA descriptor drops
                        # out-of-bounds coordinates in hardware (cf. K tails
                        # zero-filled by TMA on the load side).
                        if tma_epilogue:
                            with trace.range("epilogue"):
                                T.copy(c_local, c_smem)
                                # Order the generic-proxy SMEM writes above
                                # before the async-proxy TMA read below, and
                                # align all 128 consumer threads so the store
                                # never reads a half-written c_smem. WG-scoped
                                # named barrier, not T.sync_threads() — this
                                # branch is warpgroup-divergent (the producer
                                # never reaches it).
                                T.fence_proxy_async()
                                T.sync_threads(barrier_id=4, arrive_count=128)
                                T.copy(c_smem, c[m_start, n_start])
                        else:
                            # Fallback (misaligned N): predicated scalar store,
                            # guarding the M/N tail against out-of-bounds
                            # writes.
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
                # Rasterization remaps (bx, by) only; the z (K-slice) index
                # passes through untouched.
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
                # First K tile owned by this CTA's slice. The global K tail
                # (k % block_k != 0) lands in the last slice and is
                # zero-filled by TMA, as in the non-split kernel.
                ki_base = bz * k_slice

                tx = T.get_thread_binding()

                if tx < 128:
                    # ── Producer warpgroup: TMA loads for this K slice. ──
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
                    # ── Consumer warpgroup: WGMMA over this K slice. ──
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
                    # Epilogue: predicated scalar store of the fp32 partial.
                    # Split-K targets underfilled grids where the TMA-store
                    # drain would sit exposed (cf. the grid > 132 gate in
                    # ``_gemm_kernel``), so the scalar store is deliberate.
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
            # Plain-Python expression builder: runs natively at trace time
            # (only the prim_func body is rewritten by the tracer), unrolling
            # the slice sum into one fp32 Add tree. A traced ``for`` would
            # rebind an immutable var across loop frames, which the eager
            # builder rejects.
            expr = w[0, gi, gj]
            for s in range(1, split_k):
                expr = expr + w[s, gi, gj]
            return expr

        @T.prim_func
        def _splitk_reduce_main(
            w: T.Tensor((split_k, m, n), accum_dtype),  # type: ignore
            c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            # Flat 1-D tiling over all m*n outputs. The 2-D (block_m, block_n)
            # tiling launched only ceil(m/bm)*ceil(n/bn) CTAs — e.g. 34 for a
            # 128x2112 output, ~26% of an H200's 132 SMs, so the reduce ran
            # memory-starved. A flat grid sizes CTAs to the element count, so
            # the reduce runs at full occupancy; adjacent threads touch
            # adjacent (gi, gj), keeping the workspace reads and C write
            # coalesced. ``elems_per_cta=1024`` (264 CTAs for 128x2112)
            # measured fastest across {256..4096}.
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
    sm_count: int = 132,
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

    Wave quantization is the dominant residual against cuBLAS here. Sweeping
    ``n`` at a fixed config, the ratio to cuBLAS moves with
    ``tiles / (ceil(tiles / sm_count) * sm_count)``: over n in
    {1920, 2112, 2304, 2496, 3168} the two agree within 4% (ratio/fill spans
    0.96-1.03). That is a correlation across five different problems, not an
    attribution -- cuBLAS's own tiling quantizes too, and how much is not
    observable from here. Two ways of removing it were measured and rejected:

    - every wave-filling tile shape costs more than the fill is worth. On
      prefill-gate-up (4096x2112x7168) ``block_n=64`` gives 33 n-tiles, exactly
      8 waves on 132 SMs, and measures 385-452 TF depending on structure
      against 715 TF for the shipped ``block_n=192`` at 89% fill.
    - stream-K tail balancing -- whole tiles for the even rounds, the leftover
      tiles split along K across the full grid, fp32 partials combined by TMA
      region reduce under a counter election -- is numerically correct but the
      workspace path costs more than it recovers. Holding geometry fixed and
      toggling only the path, a control doing the *same* whole-tile work
      measures 181 us without it and 229 us with it: +27% for a prize worth
      11%. The penalty is proportional rather than fixed, so a long mainloop
      does not amortize it; the same effect closed the fused split-K route on
      the much shorter decode shapes.

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
    block_m = 128  # cooperative: two consumers each own block_m // 2 = 64 rows

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
        nr = (half_m * block_n) // 128  # fp32 accum regs per consumer thread
        # Epilogue SMEM-staging chunk width. ``stage_n < block_n`` splits the
        # TMA store into ``block_n / stage_n`` chunks, shrinking the C_shared
        # staging so a deeper A/B ring (num_stages=4) fits the 227 KB SMEM cap.
        sn = block_n if stage_n <= 0 else stage_n
        n_chunks = block_n // sn
        num_pid_m = -(-m // block_m)
        num_pid_n = -(-n // block_n)
        total_tiles = num_pid_m * num_pid_n
        max_waves = -(-total_tiles // sm_count) + 1
        k_iters = T.ceildiv(k, block_k)

        @T.macro
        def decode(flat_id, mt, nt):
            # Grouped tile order: group_size_m consecutive m-tiles share an
            # n-tile stripe so concurrent CTAs reuse B columns in L2.
            # group_size_m=1 recovers row-major.
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

                # Producer (arrive_count 128) fills a slot; both consumers
                # (arrive_count 256) must drain it before the producer refills.
                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([256] * num_stages)

                gi_prod = T.alloc_var("int32", init=0)
                gi_cons_0 = T.alloc_var("int32", init=0)
                gi_cons_1 = T.alloc_var("int32", init=0)
                # Previous ring slot, carried in a register rather than
                # recomputed. See the deferred-release note in the module header:
                # a second ``% num_stages`` in this loop costs 4-13%.
                ps0 = T.alloc_local((1,), "int32")
                ps1 = T.alloc_local((1,), "int32")
                mt = T.alloc_local((1,), "int32")
                nt = T.alloc_local((1,), "int32")

                tx = T.get_thread_binding()

                if tx < 128:
                    # ── Producer WG: 3 TMAs (A_top, A_bot, B) per K step. ──
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
                    # ── Consumer WG0: top half rows [0, half_m). ──
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
                    # ── Consumer WG1: bottom half rows [half_m, block_m). ──
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

    The 2-consumer mainloop is more WGMMA-efficient than the single-consumer
    ``_gemm_splitk_kernel``, so it wins on **large-K** decode shapes (e.g.
    decode-gate-up 128x2112x7168: the K=7168 slice is long enough to amortize
    the reduce, 33 n-tiles x split_k=4 fill the 132-SM wave — 0.98x -> 1.04x,
    beating cuBLAS). It loses on small-K decode shapes where the K-slice is too
    short to amortize the reduce round-trip — dispatch keeps those on the
    single-consumer path.

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

                # Previous ring slot, carried in a register rather than
                # recomputed: a second ``% num_stages`` in this loop is the
                # cost documented in the module header.
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
                # TileLang only lowers the swizzle annotation for clusters on
                # the X grid dim; our cluster pairs M tiles on Y.
                raise ValueError("cluster_m > 1 requires panel_size == 0")
        b_tile = (block_n, block_k) if trans_b else (block_k, block_n)

        # ``cluster_m > 1`` pairs the M tiles sharing one N column into a
        # cluster: their B reads are co-scheduled, so the second tile's stream
        # resolves in L2 instead of DRAM (bare geometry only — no multicast, no
        # cluster-sync primitives). The body is the same either way; only the
        # launch differs. ``use_swizzle`` stays in both because the guard above
        # forces ``panel_size == 0`` under a cluster, which disables it.
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
    ``ceil(n / block_n)`` CTAs — 56 of an H200's 132 on the decode-down shape,
    which streams the weights at ~2.2 TB/s where the ``m = 1`` GEMV reaches
    2.7 TB/s.

    Computing the transpose instead, ``Cᵀ[n,m] = B[n,k] @ A[m,k]ᵀ``, keeps the
    same NT operand form but puts ``n`` on the 64-row WGMMA axis and ``m`` on
    the 8-wide one: no padding waste, and the grid becomes
    ``ceil(n / block_nn)``. The epilogue stages the ``(block_nn, 8)`` tile
    through SMEM and writes ``c[mi, n0 + j]``, contiguous along ``n``.

    Only worth it when that grid fills enough of the device — see
    ``heuristics.swap_ab_stages``, which also sets ``num_stages``: with
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
                    # Partial tiles zero-fill, so the K tail and the m < mpad
                    # rows contribute nothing to the accumulator.
                    T.copy(b[bx * block_nn, ki * block_k], b_smem)
                    T.copy(a[0, ki * block_k], a_smem)
                    T.gemm(b_smem, a_smem, ct_local, transpose_B=True)
                T.copy(ct_local, ct_cast)
                T.copy(ct_cast, ct_smem)
                # Transpose out of SMEM: consecutive threads take consecutive
                # ``j`` so each row of C is written coalesced.
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
    grouped swizzle. Small square shapes cannot amortize that machinery — the
    mainloop is only ``k / block_k`` iterations — and the single-consumer
    structures cap at 0.87x cuBLAS there across a 13-variant tile sweep.

    The shape of the kernel is cuBLAS's own winner on square-1k
    (``nvjet_tst_128x64_64x8_1x2_h_bz_NNT``: 384 threads / 3 warpgroups over a
    128x64 tile): one producer warpgroup issues TMA into a ``num_stages`` ring;
    two consumer warpgroups each own ``block_m // 2 = 64`` rows and share the
    ``B`` tile (split-A / shared-B).

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
    block_m = 128  # cooperative: two consumers each own block_m // 2 = 64 rows

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
        nr = (half_m * block_n) // 128  # fp32 accum regs per consumer thread
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

                # Producer (arrive_count 128) fills a slot; both consumers
                # (arrive_count 256) must drain it before the producer refills.
                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([256] * num_stages)
                m_start = by * block_m
                n_start = bx * block_n
                # Previous ring slot, kept in a register: recomputing it
                # costs a second ``% num_stages`` in the hot loop.
                ps0 = T.alloc_local((1,), "int32")
                ps1 = T.alloc_local((1,), "int32")

                tx = T.get_thread_binding()

                if tx < 128:
                    # ── Producer WG: TMA the A halves and the shared B tile. ──
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
                    # ── Consumer WG0: top half rows [0, half_m). ──
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
                    # ── Consumer WG1: bottom half rows [half_m, block_m). ──
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
        c = torch.empty((m, n), dtype=a.dtype, device=a.device)
        w = _gemm_splitk_kernel(m, n, k, trans_a, trans_b, dtype)(
            block_m, block_n, block_k, num_stages, panel_size, split_k
        )(a, b)
        _splitk_reduce_kernel(split_k, m, n, dtype)()(w, c)
        return c
    return _gemm_kernel(m, n, k, trans_a, trans_b, dtype)(
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
    fp16 / bf16 inputs,
    fp32 accumulation. Hopper-only — TMA + WGMMA require SM90.
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
        # The base relays only a yes/no from ``applies``; name the dimension
        # instead. It still words the architecture message.
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
        # Selection already asks ``refusal`` this, but a caller can construct
        # the kernel directly (tests, benchmarks, probes); refuse there too
        # rather than let it reach TileLang's descriptor check.
        misaligned = _tma_misalignment(m, n, k, dtype, trans_a, trans_b)
        if misaligned is not None:
            raise ValueError(f"{type(self).__name__} cannot serve {m}x{n}x{k}: {misaligned}")
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.trans_a = trans_a
        self.trans_b = trans_b
        # Persistent-grid width for the coop2 (2-consumer) kernel: the device SM
        # count. Fixed per device, so it is a build-time constant of that kernel.
        self.sm_count = get_sm_count()

        self.kernel = _gemm_kernel(m, n, k, trans_a, trans_b, self.dtype_str)

        self.init_config(config, tune)

    # Per-shape tuned overrides (H200), keyed by
    # ``(m, n, k, trans_a, trans_b, dtype_str)``. Exact hits are authoritative;
    # every other shape falls to the analytic selector (see ``default_config``).
    # Entries without a structure flag merge over the modal base config.
    # Entries are locked in only when a per-shape config beats the selector's
    # pick reproducibly (small-M kernels are event-timing-noisy; a marginal
    # win that flips sign across runs is left unpinned).
    #
    # Gains below are CUPTI kernel-only + L2-flush (the acceptance protocol);
    # small-M weight matrices fit in L2, so an unflushed event loop
    # over-reports them badly.
    #   square-1k: 64x128x128 doubles the M-tile count over the modal 128-row
    #     tile and deepens block_k reuse.
    #
    # Large-M NT prefill shapes route to the 2-consumer persistent kernel
    # (``coop2``: 1 producer + 2 math warpgroups, split-A / shared-B, static-wave
    # persistent loop + grouped tile swizzle), matching cuBLAS's Hopper
    # cooperative layout (see ``_gemm_coop2_kernel``). Fields: block_n, block_k,
    # num_stages, group_size_m, stage_n (epilogue SMEM chunk width; 0 = full
    # block_n). block_m is fixed at 128 (two 64-row consumers). The entries
    # below only pin per-shape tuning:
    #   prefill-attn / k-dominant / wide-n: ns=3 g=16 (deepest full-epilogue
    #     ring bn=256/bk=64 allows in 227 KB SMEM).
    #   prefill-down (shallow K=2048): block_k=64 with a half-width (stage_n=128)
    #     epilogue. Chunking the store halves the C staging, which buys the
    #     fourth ring stage inside 227 KB; that beats spending the same SMEM on
    #     a bk=32 ring deep enough to reach ns=6 (0.921-0.927x vs 0.910-0.916x,
    #     three independent rounds). stage_n=32 measures the same as 128 — what
    #     pays is freeing the SMEM, not the chunk width.
    #     The same swap on wide-n is a mirage: it read +1.8pp in one round and
    #     -1.5pp in two more, on a row where cuBLASLt's algorithm choice swings
    #     the baseline 394-435 us between rounds. wide-n keeps ns=3/stage_n=0.
    #   gate-up (N=2112): block_n=192 (2112=192*11, no tail waste — bn=256
    #     wastes the 8.25th tile and drops to 0.68x); ns=5, stage_n=96.
    _TUNED_CONFIGS: dict = {
        # decode-gate-up (M=128, large K=7168): the 2-consumer split-K mainloop
        # (bn=64 -> 33 n-tiles x split_k=4 = 132 CTAs, one full wave) beats the
        # single-consumer split-K. 0.98x -> 1.04x cuBLAS (K is long enough to
        # amortize the reduce). Small-K decode shapes (decode-down) stay on the
        # single-consumer path — there the K-slice is too short to amortize.
        (128, 2112, 7168, False, True, "bfloat16"): {
            "coop2_splitk": True,
            "block_n": 64,
            "block_k": 128,
            "num_stages": 4,
            "split_k": 4,
        },
        # square-1k (NN): the single-tile 2-consumer kernel (``coop2s``) is the
        # only structure that clears the 0.87x ceiling the single-consumer tile
        # menu tops out at — it reproduces cuBLAS's own 384-thread / 128x64
        # layout on this shape. 0.82x -> 0.94x. See ``_gemm_coop2s_kernel``.
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
        # decode-down family (skinny-M NT, n=7168, k=2048): the non-warp-
        # specialized pipelined kernel (``simple``) wins the short-mainloop
        # regime (16 K-iters, ~1 CTA wave) by ~4% over the WS kernel — the
        # producer warpgroup's fixed costs outweigh its deeper ring there.
        # Verified per-rep interleaved, two independent fresh-build rounds.
        # m=128: pairing the two M tiles per N column into a (1, 2) cluster
        # co-schedules their B streams (second read resolves in L2), worth a
        # further ~2.5% (15.3 -> 14.9 us; cuBLAS nvjet gets its extra margin
        # from TMA multicast, which TileLang cannot express). Swizzle is off
        # because TileLang rejects the annotation for clusters on Y.
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
        # Two-tier selection:
        #   1. ``_TUNED_CONFIGS`` exact hit — human-pinned shapes stay
        #      authoritative and are never overridden by the model;
        #   2. otherwise the analytic selector picks structure + tiles
        #      (``heuristics.best_config``). It replaces both the old
        #      modal 128x128x64 fallback and the hand-written "coop2 when the
        #      grid fills the GPU" gate: the selector reproduces those choices
        #      where they were measured best and picks narrower tiles /
        #      split-K for the shapes the modal default starved.
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
            # coop2 / coop2s configs are self-contained (own schema); do not
            # merge the single-consumer modal keys into them.
            self_contained = override.get("coop2") or override.get("coop2s")
            return dict(override) if self_contained else {**modal, **override}
        return best_config(self.m, self.n, self.k, self.trans_a, self.trans_b, self.sm_count)

    # No ``autotune_configs``: the in-tree tuner wraps only ``self.kernel``
    # (the basic mainloop builder), so a sweep cannot reach the
    # structure-flagged paths (coop2 / coop2_splitk / simple / split-K run
    # through other builders plus a reduce pass) and would silently lose to
    # ``default_config`` on the shapes those paths win. ``tune=True``
    # therefore falls back to ``default_config``; per-shape tuning runs the
    # CUPTI kernel-only protocol offline and pins winners in
    # ``_TUNED_CONFIGS`` (measurement note there). TileLang's event backend
    # does flush L2 before each timed rep; the residual event-vs-CUPTI delta
    # is launch-gap wall time and mean aggregation (µs-scale).

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Simple (non-warp-specialized) pipelined path for short-mainloop
        # shapes pinned in ``_TUNED_CONFIGS`` (``simple: True``).
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

        # swap_ab path: operand-swapped tiny-m NT kernel. Selected by the
        # analytic band (``heuristics._tiny_m_config``) whenever its
        # ``ceil(n / block_nn)`` grid fills enough of the device.
        if self.config.get("swap_ab"):
            cfg = self.config
            compiled = _gemm_swap_ab_kernel(
                self.m, self.n, self.k, self.trans_a, self.trans_b, self.dtype_str
            )(cfg["block_nn"], cfg["block_k"], cfg["num_stages"])
            return compiled(a, b)

        # coop2s path: single-tile 2-consumer kernel for small NN shapes whose
        # mainloop is too short to amortize the persistent loop. Selected via
        # config (``coop2s``) from ``_TUNED_CONFIGS``.
        if self.config.get("coop2s"):
            cfg = self.config
            compiled = _gemm_coop2s_kernel(
                self.m, self.n, self.k, self.trans_a, self.trans_b, self.dtype_str
            )(cfg["block_n"], cfg["block_k"], cfg["num_stages"])
            return compiled(a, b)

        # coop2 path: persistent 2-consumer (cooperative) kernel for large-M NT
        # shapes whose grid fills the GPU. Selected via config (``coop2``) from
        # ``default_config`` / ``_TUNED_CONFIGS``. Called directly (like split-K)
        # — it carries no trace instrumentation.
        if self.config.get("coop2"):
            cfg = self.config
            compiled = _gemm_coop2_kernel(
                self.m, self.n, self.k, self.trans_a, self.trans_b, self.dtype_str, self.sm_count
            )(
                cfg["block_n"],
                cfg["block_k"],
                cfg["num_stages"],
                cfg["group_size_m"],
                cfg.get("stage_n", 0),
            )
            return compiled(a, b)

        # coop2 split-K path: 2-consumer mainloop sliced over K into an fp32
        # workspace, then reduced. For large-K small-M shapes that underfill the
        # (M, N) grid (see ``_gemm_coop2_splitk_kernel``).
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
            # Allocate C before the first launch: see ``_splitk_pair``.
            c = torch.empty((self.m, self.n), dtype=a.dtype, device=a.device)
            reduce_(mainloop(a, b), c)
            return c

        # Split-K path: slice K across grid-z CTAs into an fp32 workspace,
        # then reduce. Selected via config only (``split_k > 1``); the in-tree
        # tuner cannot rank it (see the class-level ``autotune_configs`` note).
        # Worth trying when the natural grid underfills the GPU (< 132 CTAs
        # on H100/H200).
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
            # Allocate C before the first launch: see ``_splitk_pair``.
            c = torch.empty((self.m, self.n), dtype=a.dtype, device=a.device)
            reduce_(mainloop(a, b), c)
            return c

        # Call the compiled JIT directly (cf. GemvKernel); _gemm_wrapped_kernel is
        # kept only for torch.compile compatibility. trace.run dumps the timeline
        # when tracing is on and otherwise just returns C — so no branch here.
        main_cfg = {k2: v for k2, v in self.config.items() if k2 != "split_k"}
        compiled = _gemm_kernel(
            self.m, self.n, self.k, self.trans_a, self.trans_b, self.dtype_str, traced=trace.enabled
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
    tail needs no explicit mask — the partial-tile ``T.copy`` zero-fills the
    out-of-bounds ``b_shared`` and the ``x * 0`` product masks the
    (unpredicated) out-of-bounds ``a`` read.

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
            # threads=(reduce_threads, block_n): tk=threadIdx.x is the fast reduce
            # lane over K (consecutive threads read consecutive B columns →
            # coalesced 128-bit loads); tn=threadIdx.y selects the output column.
            with T.Kernel(T.ceildiv(n, block_n), threads=(reduce_threads, block_n)) as bn:
                tk = T.get_thread_binding(0)
                tn = T.get_thread_binding(1)
                c_accum = T.alloc_local((m,), accum_dtype)
                T.clear(c_accum)
                b_shared = T.alloc_shared((block_n, block_k), dtype)
                a_local = T.alloc_local((m, tile_k), dtype)

                for bk in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    T.copy(b[bn * block_n, bk * block_k], b_shared, disable_tma=True)
                    # Both loads reach past ``k`` on the last iteration whenever
                    # ``block_k`` does not divide it, and both are guarded -- by
                    # the buffer extents, not by anything written here. The
                    # emitted code peels that iteration into a
                    # ``cp_async_gs_conditional`` for b and a
                    # ``threadIdx.x < (k - bk * block_k) / tile_k`` branch for a.
                    # Verified at k=3000 with block_k=1024: an ``a`` whose tail
                    # is filled with NaN leaves the result unchanged.
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


# Autotune-grid geometry for the cp.async B-ring kernel
# (``_gemm_small_batch_kernel``, which both bandwidth-mode Kernel classes build):
# 128-bit fp16/bf16 loads give
# 8 elements per thread per tile; the ring guard budgets 224 KB — headroom
# under the 227 KB SM90 opt-in ceiling (``heuristics._SMEM_BUDGET``)
# because it models only the B ring, not the reduction scratch.
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
        return gemv_region(call)

    def __init__(
        self, n: int, k: int, dtype: torch.dtype, config: Optional[dict] = None, tune: bool = False
    ) -> None:
        super().__init__()
        self.n = n
        self.k = k
        self.dtype = dtype

        # The matrix-vector case is ``_gemm_small_batch_kernel`` at m = 1;
        # there is no separate GEMV body.
        self.kernel = _gemm_small_batch_kernel(1, n, k, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        # Measured SM90 band rules live with the rest of the family's shape
        # policy in ``heuristics.gemv_config``; ``tune=True`` replaces them
        # with the ``autotune_configs`` sweep winner.
        return gemv_config(self.k)

    @property
    def autotune_configs(self) -> list[dict]:
        # reduce_threads>32 opens a cross-warp SMEM tree reduction: more threads
        # per output row raise memory-level parallelism (the lever on bandwidth-
        # bound GEMV). num_stages>=2 pipelines the B-tile cp.async prefetch.
        return _bandwidth_autotune_grid((32, 64, 128, 256), (1, 2, 4, 8, 16), (1, 2, 3, 4, 5, 6))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # One-row ``A`` for the shared small-batch body; the caller reshapes the
        # ``(1, n)`` result. Called directly rather than through
        # ``_gemv_wrapped_kernel`` (kept for torch.compile) to avoid per-forward
        # closure recreation and a JIT cache lookup.
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
    the measured config band they pick. Its inner loop pays ``m`` FMAs and ``m``
    converts per weight element on CUDA cores, so the lead over the tensor-core
    ``GemmKernel`` shrinks as ``m`` grows — the measured crossover and the
    dispatch band live in :func:`~tileops.kernels.gemm.call_spec.small_batch_region`.

    Scope: SM90, NT only — ``B`` is ``[N,K]``, so K is contiguous and the
    reduction over it coalesces; no other layout has that property. The kernel
    is correct for any ``m``; the region it claims is ``m == 2`` on an n too
    narrow to fill the device (:func:`~tileops.kernels.gemm.call_spec.small_batch_region`).

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
        return small_batch_region(call)

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
        # Measured band rule lives with the rest of the family's shape policy
        # in ``heuristics.small_batch_config``; ``tune=True`` replaces it
        # with the ``autotune_configs`` sweep winner.
        return small_batch_config(self.n, self.k, get_sm_count())

    @property
    def autotune_configs(self) -> list[dict]:
        # Narrower than the GEMV grid above: block_n > 4 starves the per-row
        # reduction once m rows share each B tile.
        return _bandwidth_autotune_grid((32, 64, 128), (1, 2, 4), (2, 3, 4, 5))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.kernel(
            self.config["block_n"],
            self.config["reduce_threads"],
            self.config["num_stages"],
        )(a, b)
