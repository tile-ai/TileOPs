"""V2 persistent grouped-GEMM kernel: warp-specialized + static wave + TMA store.

Mirrors the structure of the TileLang SM100 example
``gemm_tcgen5mma_ws_persistent.py``, with all SM100-only primitives
(TMEM, tcgen05_gemm, CLC, 2-CTA MMA) replaced by SM90 equivalents.

Design: see tileops_md/2026-05-15_grouped_gemm_persistent_v2_design.md.

Two compile-time templates routed by ``block_m``:
  * ``block_m ≤ 64``  → Pingpong: 2 math WGs process independent tiles
  * ``block_m ≥ 128`` → Cooperative: 2 math WGs share one tile's M

K-aligned only.  K-unaligned use ``GroupedGemmPersistentKernel``.

Uses a *static-wave scheduler* (no atomic counter): each CTA enumerates
its tile IDs as ``flat_id_0 = 2*(sm_count*w + pid)`` and
``flat_id_1 = flat_id_0 + 1`` for wave ``w ∈ [0, _max_waves)``.  Tiles past
``s_total`` are skipped via a valid mask.  This removes scheduler latency
and produces a deterministic tile-to-CTA map (group-size=8 threadblock
swizzle is deferred: the M axis is partitioned by expert via prefix-sum/
binary-search, so naive swizzle across expert boundaries is unsafe).

A per-WG ring buffer with ``num_stages`` slots overlaps TMA loads with
WGMMA.  Default ``num_stages=4`` (autotuned over {2..6} with H100 SMEM
limit pruning).
"""
import functools
import math
import os

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

from tileops.kernels.kernel_base import Kernel

_ANCHOR_HELPER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "_anchor_helper.h")
)

__all__ = ["GroupedGemmPersistent3WGKernel"]

_DEFAULT_CONFIG = {
    "block_m": 128,        # cooperative template (split-A, shared-B)
    "block_n": 256,
    "block_k": 64,
    "num_stages": 3,
    "threads": 384,
    "group_size_m": 1,
}


class GroupedGemmPersistent3WGKernel(Kernel):
    """V2 persistent grouped-GEMM kernel (K-aligned only)."""

    supported_archs: list[int] = [90]

    def __init__(self, numel, num_experts, N, K,
                 dtype=torch.bfloat16, sm_count=None,
                 config=None, tune=False):
        super().__init__()
        self.numel = numel
        self.num_experts = num_experts
        self.N = N
        self.K = K
        self.dtype = dtype
        if sm_count is None:
            sm_count = torch.cuda.get_device_properties(
                torch.cuda.current_device()).multi_processor_count
        self.sm_count = sm_count
        self.kernel = lambda: _persistent_grouped_gemm_v2_kernel(
            self.numel, self.num_experts, self.N, self.K,
            self.dtype_str, self.sm_count, self.config["block_k"])
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return dict(_DEFAULT_CONFIG)

    @property
    def autotune_configs(self) -> list[dict]:
        SMEM_LIMIT = 228 * 1024
        bytes_per_elem = 2  # bf16/fp16
        configs = []
        # ── Pingpong template (bm <= 64): 4 SMEM streams (A_wg0/1, B_wg0/1) ──
        for block_m in (64,):
            for block_n in (128, 256):
                for block_k in (64,):
                    for num_stages in (2, 3, 4, 5, 6):
                        # 2 × A_smem_wgX (ns, bm, bk) + 2 × B_smem_wgX (ns, bn, bk).
                        smem_main = (
                            2 * num_stages * block_m * block_k * bytes_per_elem
                            + 2 * num_stages * block_n * block_k * bytes_per_elem
                        )
                        # Per-WG C_shared (bm, bn) for TMA-store epilogue.
                        smem_c = 2 * block_m * block_n * bytes_per_elem
                        if smem_main + smem_c > SMEM_LIMIT:
                            continue
                        configs.append({
                            "block_m": block_m, "block_n": block_n,
                            "block_k": block_k, "num_stages": num_stages,
                            "threads": 384, "group_size_m": 1,
                        })
        # ── Cooperative template (bm >= 128): split-A, shared B ──
        # SMEM: 2 × A_smem (ns, bm/2, bk) + 1 × B_smem (ns, bn, bk).
        # Same total A footprint as pingpong, but the duplicated B ring is
        # eliminated, freeing budget for bn=256 with ns=3 at bm=128.
        for block_m in (128,):
            half_m = block_m // 2
            for block_n in (128, 256):
                for block_k in (64,):
                    for num_stages in (2, 3, 4, 5, 6):
                        # 2 × A_smem_{top,bot} (ns, half_m, bk) + 1 × B_smem.
                        smem_main = (
                            2 * num_stages * half_m * block_k * bytes_per_elem
                            + num_stages * block_n * block_k * bytes_per_elem
                        )
                        # Per-WG C_shared (half_m, bn) for TMA-store epilogue.
                        smem_c = 2 * half_m * block_n * bytes_per_elem
                        if smem_main + smem_c > SMEM_LIMIT:
                            continue
                        configs.append({
                            "block_m": block_m, "block_n": block_n,
                            "block_k": block_k, "num_stages": num_stages,
                            "threads": 384, "group_size_m": 1,
                        })
        return configs

    def forward(self, A, B, true_sizes, true_offsets):
        C = torch.zeros(self.numel, self.N, dtype=self.dtype, device=A.device)
        block_m = self.config["block_m"]
        block_n = self.config["block_n"]
        block_k = self.config["block_k"]
        if self.K % block_k != 0:
            raise ValueError(f"K-aligned only: K={self.K}, block_k={block_k}")
        if self.N % block_n != 0:
            raise ValueError(f"N-aligned only: N={self.N}, block_n={block_n}")
        if A.shape[0] < self.numel + block_m:
            A = F.pad(A, (0, 0, 0, block_m))
        gemm_fn = _persistent_grouped_gemm_v2_kernel(
            self.numel, self.num_experts, self.N, self.K,
            self.dtype_str, self.sm_count, block_k,
        )(
            block_m, block_n, block_k,
            self.config["num_stages"],
            self.config["threads"],
            self.config.get("group_size_m", 1),
        )
        gemm_fn(A, B, true_sizes, true_offsets, C)
        return C


def _make_pingpong_kernel(numel, num_experts, N, K, dtype, sm_count,
                          block_m, block_n, block_k, num_stages, threads,
                          group_size_m):
    """Build a @T.prim_func for the pingpong template (block_m <= 64).

    Two math WGs each work an independent tile per wave; each WG holds
    its own SMEM ring (A_smem_wgX, B_smem_wgX of shape (num_stages, ...)).
    Pingpong (1 producer + 2 consumer WGs) + static-wave scheduler +
    per-WG ring buffer (num_stages slots).
    """
    accum_dtype = "float"
    log2_up = max(1, math.ceil(math.log2(num_experts + 1)))

    # V2 pingpong layout: 1 producer + 2 consumers, each WG = 128 threads.
    assert threads == 384, (
        f"V2 pingpong persistent grouped GEMM requires threads=384 "
        f"(1 producer + 2 consumer WGs); got threads={threads}"
    )
    _num_pid_n = math.ceil(N / block_n)
    _max_tiles = numel // block_m + num_experts  # over-estimate of M tiles
    _total_ctas_ub = _max_tiles * _num_pid_n
    # Pingpong: each CTA processes 2 tiles per wave (one per math WG).
    # Slack of 1 covers the case where _total_ctas_ub is not a multiple
    # of 2 * sm_count.
    _max_waves = (_total_ctas_ub + 2 * sm_count - 1) // (2 * sm_count) + 1
    _k_iters = K // block_k
    A_shape = (numel + block_m, K)

    @T.prim_func
    def _gemm_main_v2(
        A: T.Tensor(A_shape, dtype),                        # type: ignore  # noqa: F821
        B: T.Tensor((num_experts, N, K), dtype),            # type: ignore  # noqa: F821
        true_sizes: T.Tensor((num_experts,), "int32"),      # noqa: F821
        true_offsets: T.Tensor((num_experts,), "int32"),    # noqa: F821
        C: T.Tensor((numel, N), dtype),                     # type: ignore  # noqa: F821
    ):
        with T.Kernel(sm_count, threads=threads) as (pid,):
            # ── Per-WG ring-buffered SMEM (num_stages slots) ──
            A_smem_wg0 = T.alloc_shared((num_stages, block_m, block_k), dtype)
            B_smem_wg0 = T.alloc_shared((num_stages, block_n, block_k), dtype)
            A_smem_wg1 = T.alloc_shared((num_stages, block_m, block_k), dtype)
            B_smem_wg1 = T.alloc_shared((num_stages, block_n, block_k), dtype)
            C_local_wg0 = T.alloc_fragment((block_m, block_n), accum_dtype)
            C_local_wg1 = T.alloc_fragment((block_m, block_n), accum_dtype)

            # ── Phase-4 epilogue: per-WG cast fragment + TMA-store SMEM ──
            # Fast path (full m-tile): cast→C_shared→TMA store.
            # Slow path (partial m-tile, arows < block_m): cast→predicated STG.
            C_local_cast_wg0 = T.alloc_fragment((block_m, block_n), dtype)
            C_local_cast_wg1 = T.alloc_fragment((block_m, block_n), dtype)
            C_shared_wg0 = T.alloc_shared((block_m, block_n), dtype)
            C_shared_wg1 = T.alloc_shared((block_m, block_n), dtype)

            # ── Scheduler SMEM (no atomic counter / tile pair anymore) ──
            s_cum = T.alloc_shared((num_experts + 1,), "int32")
            s_total = T.alloc_shared((1,), "int32")
            lo = T.alloc_local((1,), "int32")
            hi = T.alloc_local((1,), "int32")
            # Per-tile metadata hoisted to alloc_local so the interleaved
            # K-loop body can read m_start/n_start/expert_id across
            # IfFrame boundaries (TileLang ir_builder forbids cross-frame
            # reads of immutable Vars; Buffer loads are exempt).
            ms0 = T.alloc_local((1,), "int32")
            ns0 = T.alloc_local((1,), "int32")
            ex0 = T.alloc_local((1,), "int32")
            v0 = T.alloc_local((1,), "int32")
            ms1 = T.alloc_local((1,), "int32")
            ns1 = T.alloc_local((1,), "int32")
            ex1 = T.alloc_local((1,), "int32")
            v1 = T.alloc_local((1,), "int32")

            T.annotate_layout({
                A_smem_wg0: tilelang.layout.make_swizzled_layout(A_smem_wg0),
                B_smem_wg0: tilelang.layout.make_swizzled_layout(B_smem_wg0),
                A_smem_wg1: tilelang.layout.make_swizzled_layout(A_smem_wg1),
                B_smem_wg1: tilelang.layout.make_swizzled_layout(B_smem_wg1),
                C_shared_wg0: tilelang.layout.make_swizzled_layout(C_shared_wg0),
                C_shared_wg1: tilelang.layout.make_swizzled_layout(C_shared_wg1),
            })

            # ── Per-WG producer/consumer barriers (arrive_count=128) ──
            ab_full_wg0 = T.alloc_barrier([128] * num_stages)
            ab_empty_wg0 = T.alloc_barrier([128] * num_stages)
            ab_full_wg1 = T.alloc_barrier([128] * num_stages)
            ab_empty_wg1 = T.alloc_barrier([128] * num_stages)

            # ── Phase counters: monotonic, never reset ──
            gi_prod_0 = T.alloc_var("int32", init=0)
            gi_prod_1 = T.alloc_var("int32", init=0)
            gi_cons_0 = T.alloc_var("int32", init=0)
            gi_cons_1 = T.alloc_var("int32", init=0)

            tx = T.get_thread_binding()

            # ═════════════════════════════════════════════════════════
            # Producer WG: tx < 128
            # ═════════════════════════════════════════════════════════
            # Static-wave scheduler: each CTA enumerates tile IDs
            # ``flat_id_0 = 2*(sm_count*w + pid)`` and
            # ``flat_id_1 = flat_id_0 + 1`` for wave w.  No atomic.
            if tx < 128:
                T.dec_max_nreg(24)

                if tx == 0:
                    s_cum[0] = T.int32(0)
                    for e in T.serial(num_experts):
                        s_cum[e + 1] = s_cum[e] + (true_sizes[e] + (block_m - 1)) // block_m
                    s_total[0] = s_cum[num_experts] * T.int32(_num_pid_n)
                # CTA-wide sync: publishes s_cum/s_total to consumers
                # (this pairs with the corresponding sync_threads in
                # the elif/else branches below).
                T.sync_threads()

                for w in T.serial(_max_waves):
                    total = s_total[0]
                    base = T.int32(2) * (T.int32(sm_count) * w + pid)

                    # ── Resolve WG0 tile metadata into alloc_local ──
                    flat_id_0 = base
                    if flat_id_0 < total:
                        m_tile_0 = flat_id_0 // T.int32(_num_pid_n)
                        n_tile_0 = flat_id_0 % T.int32(_num_pid_n)

                        lo[0] = T.int32(0)
                        hi[0] = T.int32(num_experts - 1)
                        for _bs in T.serial(log2_up):
                            mid = (lo[0] + hi[0]) >> T.int32(1)
                            if s_cum[mid + 1] <= m_tile_0:
                                lo[0] = mid + T.int32(1)
                            else:
                                hi[0] = mid
                        ex0[0] = lo[0]
                        ms0[0] = (true_offsets[ex0[0]]
                                  + (m_tile_0 - s_cum[ex0[0]]) * T.int32(block_m))
                        ns0[0] = n_tile_0 * T.int32(block_n)
                        v0[0] = T.int32(1)
                    else:
                        v0[0] = T.int32(0)

                    # ── Resolve WG1 tile metadata into alloc_local ──
                    flat_id_1 = base + T.int32(1)
                    if flat_id_1 < total:
                        m_tile_1 = flat_id_1 // T.int32(_num_pid_n)
                        n_tile_1 = flat_id_1 % T.int32(_num_pid_n)

                        lo[0] = T.int32(0)
                        hi[0] = T.int32(num_experts - 1)
                        for _bs in T.serial(log2_up):
                            mid = (lo[0] + hi[0]) >> T.int32(1)
                            if s_cum[mid + 1] <= m_tile_1:
                                lo[0] = mid + T.int32(1)
                            else:
                                hi[0] = mid
                        ex1[0] = lo[0]
                        ms1[0] = (true_offsets[ex1[0]]
                                  + (m_tile_1 - s_cum[ex1[0]]) * T.int32(block_m))
                        ns1[0] = n_tile_1 * T.int32(block_n)
                        v1[0] = T.int32(1)
                    else:
                        v1[0] = T.int32(0)

                    # ── Interleaved K-loop: WG0 then WG1 each k ──
                    for k in T.Pipelined(_k_iters, num_stages=0):
                        k_start = k * block_k

                        # WG0 stream (ring slot = gi_prod_0 % num_stages)
                        if v0[0] != 0:
                            slot0 = gi_prod_0 % num_stages
                            T.barrier_wait(
                                ab_empty_wg0[slot0],
                                ((gi_prod_0 // num_stages) & 1) ^ 1)
                            T.tma_copy(
                                A[ms0[0]:ms0[0] + block_m,
                                  k_start:k_start + block_k],
                                A_smem_wg0[slot0, :, :],
                                barrier=ab_full_wg0[slot0],
                            )
                            T.tma_copy(
                                B[ex0[0],
                                  ns0[0]:ns0[0] + block_n,
                                  k_start:k_start + block_k],
                                B_smem_wg0[slot0, :, :],
                                barrier=ab_full_wg0[slot0],
                            )
                            T.barrier_arrive(ab_full_wg0[slot0])
                            gi_prod_0 = gi_prod_0 + 1

                        # WG1 stream (ring slot = gi_prod_1 % num_stages)
                        if v1[0] != 0:
                            slot1 = gi_prod_1 % num_stages
                            T.barrier_wait(
                                ab_empty_wg1[slot1],
                                ((gi_prod_1 // num_stages) & 1) ^ 1)
                            T.tma_copy(
                                A[ms1[0]:ms1[0] + block_m,
                                  k_start:k_start + block_k],
                                A_smem_wg1[slot1, :, :],
                                barrier=ab_full_wg1[slot1],
                            )
                            T.tma_copy(
                                B[ex1[0],
                                  ns1[0]:ns1[0] + block_n,
                                  k_start:k_start + block_k],
                                B_smem_wg1[slot1, :, :],
                                barrier=ab_full_wg1[slot1],
                            )
                            T.barrier_arrive(ab_full_wg1[slot1])
                            gi_prod_1 = gi_prod_1 + 1

            # ═════════════════════════════════════════════════════════
            # Consumer WG0: 128 ≤ tx < 256 — processes flat_id_0 per wave
            # ═════════════════════════════════════════════════════════
            elif tx < 256:
                T.inc_max_nreg(240)
                # CTA-wide sync (pairs with producer's post-init sync).
                T.sync_threads()

                for w in T.serial(_max_waves):
                    total = s_total[0]
                    flat_id_0 = T.int32(2) * (T.int32(sm_count) * w + pid)
                    valid_0 = flat_id_0 < total

                    if valid_0:
                        m_tile_0 = flat_id_0 // T.int32(_num_pid_n)
                        n_tile_0 = flat_id_0 % T.int32(_num_pid_n)

                        lo[0] = T.int32(0)
                        hi[0] = T.int32(num_experts - 1)
                        for _bs in T.serial(log2_up):
                            mid = (lo[0] + hi[0]) >> T.int32(1)
                            if s_cum[mid + 1] <= m_tile_0:
                                lo[0] = mid + T.int32(1)
                            else:
                                hi[0] = mid
                        expert_id_0 = lo[0]
                        row_0 = (m_tile_0 - s_cum[expert_id_0]) * T.int32(block_m)
                        m_start_0 = true_offsets[expert_id_0] + row_0
                        n_start_0 = n_tile_0 * T.int32(block_n)
                        arows_0 = T.min(T.int32(block_m),
                                        true_sizes[expert_id_0] - row_0)
                        acols_0 = T.min(T.int32(block_n),
                                        T.int32(N) - n_start_0)


                        for k in T.Pipelined(_k_iters, num_stages=0):
                            slot = gi_cons_0 % num_stages
                            T.barrier_wait(ab_full_wg0[slot],
                                           (gi_cons_0 // num_stages) & 1)
                            T.wgmma_gemm(
                                A_smem_wg0[slot, :, :],
                                B_smem_wg0[slot, :, :],
                                C_local_wg0,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow,
                                clear_accum=(k == 0),
                            )
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(C_local_wg0, num_regs=64)
                            T.barrier_arrive(ab_empty_wg0[slot])
                            gi_cons_0 = gi_cons_0 + 1

                        # ── Phase-4 epilogue: TMA-store fast path / STG fallback ──
                        # Cast fp32 accumulator → dtype fragment once, then
                        # branch on full vs partial m-tile (acols is always
                        # block_n since N % block_n == 0 is enforced).
                        T.copy(C_local_wg0, C_local_cast_wg0)
                        if arows_0 == T.int32(block_m):
                            # Fast path: full tile → SMEM staging → TMA store.
                            T.copy(C_local_cast_wg0, C_shared_wg0)
                            T.copy(C_shared_wg0,
                                   C[m_start_0, n_start_0])
                        else:
                            # Slow path: partial tile → predicated direct STG.
                            for i, j in T.Parallel(block_m, block_n):
                                if i < arows_0 and j < acols_0:
                                    C[m_start_0 + i, n_start_0 + j] = C_local_cast_wg0[i, j]

            # ═════════════════════════════════════════════════════════
            # Consumer WG1: tx ≥ 256 — processes flat_id_1 per wave
            # ═════════════════════════════════════════════════════════
            else:
                T.inc_max_nreg(240)
                # CTA-wide sync (pairs with producer's post-init sync).
                T.sync_threads()

                for w in T.serial(_max_waves):
                    total = s_total[0]
                    flat_id_1 = T.int32(2) * (T.int32(sm_count) * w + pid) + T.int32(1)
                    valid_1 = flat_id_1 < total

                    if valid_1:
                        m_tile_1 = flat_id_1 // T.int32(_num_pid_n)
                        n_tile_1 = flat_id_1 % T.int32(_num_pid_n)

                        lo[0] = T.int32(0)
                        hi[0] = T.int32(num_experts - 1)
                        for _bs in T.serial(log2_up):
                            mid = (lo[0] + hi[0]) >> T.int32(1)
                            if s_cum[mid + 1] <= m_tile_1:
                                lo[0] = mid + T.int32(1)
                            else:
                                hi[0] = mid
                        expert_id_1 = lo[0]
                        row_1 = (m_tile_1 - s_cum[expert_id_1]) * T.int32(block_m)
                        m_start_1 = true_offsets[expert_id_1] + row_1
                        n_start_1 = n_tile_1 * T.int32(block_n)
                        arows_1 = T.min(T.int32(block_m),
                                        true_sizes[expert_id_1] - row_1)
                        acols_1 = T.min(T.int32(block_n),
                                        T.int32(N) - n_start_1)


                        for k in T.Pipelined(_k_iters, num_stages=0):
                            slot = gi_cons_1 % num_stages
                            T.barrier_wait(ab_full_wg1[slot],
                                           (gi_cons_1 // num_stages) & 1)
                            T.wgmma_gemm(
                                A_smem_wg1[slot, :, :],
                                B_smem_wg1[slot, :, :],
                                C_local_wg1,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow,
                                clear_accum=(k == 0),
                            )
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(C_local_wg1, num_regs=64)
                            T.barrier_arrive(ab_empty_wg1[slot])
                            gi_cons_1 = gi_cons_1 + 1

                        # ── Phase-4 epilogue: TMA-store fast path / STG fallback ──
                        # Cast fp32 accumulator → dtype fragment once, then
                        # branch on full vs partial m-tile (acols is always
                        # block_n since N % block_n == 0 is enforced).
                        T.copy(C_local_wg1, C_local_cast_wg1)
                        if arows_1 == T.int32(block_m):
                            # Fast path: full tile → SMEM staging → TMA store.
                            T.copy(C_local_cast_wg1, C_shared_wg1)
                            T.copy(C_shared_wg1,
                                   C[m_start_1, n_start_1])
                        else:
                            # Slow path: partial tile → predicated direct STG.
                            for i, j in T.Parallel(block_m, block_n):
                                if i < arows_1 and j < acols_1:
                                    C[m_start_1 + i, n_start_1 + j] = C_local_cast_wg1[i, j]

    return _gemm_main_v2


def _make_cooperative_kernel(numel, num_experts, N, K, dtype, sm_count,
                             block_m, block_n, block_k, num_stages, threads,
                             group_size_m):
    """Build a @T.prim_func for the cooperative template (block_m >= 128).

    Both math WGs split the M dimension of one shared tile in half.
    Producer WG (tx<128) issues 3 TMAs per K-step: A_top, A_bot, and B,
    then arrives ab_full.  Math WG0 (128≤tx<256) reads A_smem_top and
    consumes the full B; Math WG1 (tx≥256) reads A_smem_bot and the
    same B.  Both arrive ab_empty (count=256) before the next K step.

    Why ``A`` is split into two physical SMEM buffers (A_smem_top,
    A_smem_bot) rather than one shared ``A_shared`` and row-slicing:
    TileLang 0.1.9's ``T.wgmma_gemm`` asserts ``A_offset[-2] == 0`` —
    i.e. A's first-dim offset must be zero, so ``A_shared[stage, half:bm]``
    is rejected.  Splitting A keeps every WGMMA call zero-offset.

    SMEM accounting (vs pingpong):
      * Pingpong: 2 × (A_smem + B_smem) rings  → 2·ns·(bm + bn)·bk·2 bytes
      * Cooperative: 2 × A_smem half-rings + 1 × B_smem ring
                                              → ns·(bm + bn)·bk·2 bytes
    Cooperative saves the duplicated B ring, which lets ``bn=256`` with
    ``ns≥3`` fit within H100's 228 KB cap when bm=128.
    """
    accum_dtype = "float"
    log2_up = max(1, math.ceil(math.log2(num_experts + 1)))

    assert threads == 384, (
        f"V2 cooperative persistent grouped GEMM requires threads=384 "
        f"(1 producer + 2 consumer WGs); got threads={threads}"
    )
    assert block_m % 2 == 0, (
        f"cooperative template requires even block_m (each math WG owns "
        f"block_m/2 rows); got block_m={block_m}"
    )
    half_m = block_m // 2
    assert half_m >= 64, (
        f"cooperative template requires block_m >= 128 (half_m={half_m} < "
        f"WGMMA minimum M=64)"
    )

    _num_pid_n = math.ceil(N / block_n)
    _max_tiles = numel // block_m + num_experts  # over-estimate of M tiles
    _total_ctas_ub = _max_tiles * _num_pid_n
    # Cooperative: each CTA processes 1 tile per wave.  Slack of 1 covers
    # the case where _total_ctas_ub is not a multiple of sm_count.
    _max_waves = (_total_ctas_ub + sm_count - 1) // sm_count + 1
    _k_iters = K // block_k
    A_shape = (numel + block_m, K)

    @T.prim_func
    def _gemm_main_v2_coop(
        A: T.Tensor(A_shape, dtype),                        # type: ignore  # noqa: F821
        B: T.Tensor((num_experts, N, K), dtype),            # type: ignore  # noqa: F821
        true_sizes: T.Tensor((num_experts,), "int32"),      # noqa: F821
        true_offsets: T.Tensor((num_experts,), "int32"),    # noqa: F821
        C: T.Tensor((numel, N), dtype),                     # type: ignore  # noqa: F821
    ):
        with T.Kernel(sm_count, threads=threads) as (pid,):
            # ── Split-A SMEM rings (zero-offset WGMMA) + shared B ring ──
            A_smem_top = T.alloc_shared((num_stages, half_m, block_k), dtype)
            A_smem_bot = T.alloc_shared((num_stages, half_m, block_k), dtype)
            B_smem = T.alloc_shared((num_stages, block_n, block_k), dtype)
            # Per-WG half-tile fragments (block_m/2 × block_n).
            C_local_wg0 = T.alloc_fragment((half_m, block_n), accum_dtype)
            C_local_wg1 = T.alloc_fragment((half_m, block_n), accum_dtype)

            # ── TMA-store epilogue staging (per-WG half-tile) ──
            C_local_cast_wg0 = T.alloc_fragment((half_m, block_n), dtype)
            C_local_cast_wg1 = T.alloc_fragment((half_m, block_n), dtype)
            C_shared_wg0 = T.alloc_shared((half_m, block_n), dtype)
            C_shared_wg1 = T.alloc_shared((half_m, block_n), dtype)

            # ── Scheduler SMEM (single tile per wave; one binary search) ──
            s_cum = T.alloc_shared((num_experts + 1,), "int32")
            s_total = T.alloc_shared((1,), "int32")
            lo = T.alloc_local((1,), "int32")
            hi = T.alloc_local((1,), "int32")
            # Per-tile metadata in alloc_local (same pattern as pingpong).
            ms = T.alloc_local((1,), "int32")
            ns_ = T.alloc_local((1,), "int32")
            ex = T.alloc_local((1,), "int32")
            v = T.alloc_local((1,), "int32")

            T.annotate_layout({
                A_smem_top: tilelang.layout.make_swizzled_layout(A_smem_top),
                A_smem_bot: tilelang.layout.make_swizzled_layout(A_smem_bot),
                B_smem: tilelang.layout.make_swizzled_layout(B_smem),
                C_shared_wg0: tilelang.layout.make_swizzled_layout(C_shared_wg0),
                C_shared_wg1: tilelang.layout.make_swizzled_layout(C_shared_wg1),
            })

            # ── Single shared barrier set: producer WG (128 threads)
            #     fills, both math WGs (256 threads total) drain ──
            ab_full = T.alloc_barrier([128] * num_stages)
            ab_empty = T.alloc_barrier([256] * num_stages)

            gi_prod = T.alloc_var("int32", init=0)
            gi_cons_0 = T.alloc_var("int32", init=0)
            gi_cons_1 = T.alloc_var("int32", init=0)

            tx = T.get_thread_binding()

            # ═════════════════════════════════════════════════════════
            # Producer WG: tx < 128
            # Static-wave scheduler: flat_id = sm_count * w + pid (one
            # tile per CTA per wave; both math WGs co-process it).
            # ═════════════════════════════════════════════════════════
            if tx < 128:
                T.dec_max_nreg(24)

                if tx == 0:
                    s_cum[0] = T.int32(0)
                    for e in T.serial(num_experts):
                        s_cum[e + 1] = s_cum[e] + (true_sizes[e] + (block_m - 1)) // block_m
                    s_total[0] = s_cum[num_experts] * T.int32(_num_pid_n)
                T.sync_threads()

                for w in T.serial(_max_waves):
                    total = s_total[0]
                    flat_id = T.int32(sm_count) * w + pid

                    if flat_id < total:
                        m_tile = flat_id // T.int32(_num_pid_n)
                        n_tile = flat_id % T.int32(_num_pid_n)

                        lo[0] = T.int32(0)
                        hi[0] = T.int32(num_experts - 1)
                        for _bs in T.serial(log2_up):
                            mid = (lo[0] + hi[0]) >> T.int32(1)
                            if s_cum[mid + 1] <= m_tile:
                                lo[0] = mid + T.int32(1)
                            else:
                                hi[0] = mid
                        ex[0] = lo[0]
                        ms[0] = (true_offsets[ex[0]]
                                 + (m_tile - s_cum[ex[0]]) * T.int32(block_m))
                        ns_[0] = n_tile * T.int32(block_n)
                        v[0] = T.int32(1)
                    else:
                        v[0] = T.int32(0)

                    for k in T.Pipelined(_k_iters, num_stages=0):
                        k_start = k * block_k

                        if v[0] != 0:
                            slot = gi_prod % num_stages
                            T.barrier_wait(
                                ab_empty[slot],
                                ((gi_prod // num_stages) & 1) ^ 1)
                            # Top half of A (rows ms..ms+half_m).
                            T.tma_copy(
                                A[ms[0]:ms[0] + half_m,
                                  k_start:k_start + block_k],
                                A_smem_top[slot, :, :],
                                barrier=ab_full[slot],
                            )
                            # Bottom half of A (rows ms+half_m..ms+block_m).
                            T.tma_copy(
                                A[ms[0] + half_m:ms[0] + block_m,
                                  k_start:k_start + block_k],
                                A_smem_bot[slot, :, :],
                                barrier=ab_full[slot],
                            )
                            # Full B tile (shared between the two math WGs).
                            T.tma_copy(
                                B[ex[0],
                                  ns_[0]:ns_[0] + block_n,
                                  k_start:k_start + block_k],
                                B_smem[slot, :, :],
                                barrier=ab_full[slot],
                            )
                            T.barrier_arrive(ab_full[slot])
                            gi_prod = gi_prod + 1

            # ═════════════════════════════════════════════════════════
            # Consumer WG0: 128 ≤ tx < 256 — top half (rows 0..half_m)
            # ═════════════════════════════════════════════════════════
            elif tx < 256:
                T.inc_max_nreg(240)
                T.sync_threads()

                for w in T.serial(_max_waves):
                    total = s_total[0]
                    flat_id = T.int32(sm_count) * w + pid

                    if flat_id < total:
                        m_tile = flat_id // T.int32(_num_pid_n)
                        n_tile = flat_id % T.int32(_num_pid_n)

                        lo[0] = T.int32(0)
                        hi[0] = T.int32(num_experts - 1)
                        for _bs in T.serial(log2_up):
                            mid = (lo[0] + hi[0]) >> T.int32(1)
                            if s_cum[mid + 1] <= m_tile:
                                lo[0] = mid + T.int32(1)
                            else:
                                hi[0] = mid
                        expert_id = lo[0]
                        row = (m_tile - s_cum[expert_id]) * T.int32(block_m)
                        m_start = true_offsets[expert_id] + row
                        n_start = n_tile * T.int32(block_n)
                        # Top-half row count: clamp(true_arows, 0, half_m).
                        true_arows = true_sizes[expert_id] - row
                        arows0 = T.max(T.int32(0),
                                       T.min(T.int32(half_m), true_arows))


                        for k in T.Pipelined(_k_iters, num_stages=0):
                            slot = gi_cons_0 % num_stages
                            T.barrier_wait(ab_full[slot],
                                           (gi_cons_0 // num_stages) & 1)
                            T.wgmma_gemm(
                                A_smem_top[slot, :, :],
                                B_smem[slot, :, :],
                                C_local_wg0,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow,
                                clear_accum=(k == 0),
                            )
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(C_local_wg0, num_regs=64)
                            T.barrier_arrive(ab_empty[slot])
                            gi_cons_0 = gi_cons_0 + 1

                        # ── Epilogue (top half) ──
                        T.copy(C_local_wg0, C_local_cast_wg0)
                        if arows0 == T.int32(half_m):
                            # Fast path: full top-half tile via TMA store.
                            T.copy(C_local_cast_wg0, C_shared_wg0)
                            T.copy(C_shared_wg0,
                                   C[m_start, n_start])
                        else:
                            # Slow path: predicated direct STG (rows 0..arows0).
                            for i, j in T.Parallel(half_m, block_n):
                                if i < arows0:
                                    C[m_start + i, n_start + j] = C_local_cast_wg0[i, j]

            # ═════════════════════════════════════════════════════════
            # Consumer WG1: tx ≥ 256 — bottom half (rows half_m..block_m)
            # ═════════════════════════════════════════════════════════
            else:
                T.inc_max_nreg(240)
                T.sync_threads()

                for w in T.serial(_max_waves):
                    total = s_total[0]
                    flat_id = T.int32(sm_count) * w + pid

                    if flat_id < total:
                        m_tile = flat_id // T.int32(_num_pid_n)
                        n_tile = flat_id % T.int32(_num_pid_n)

                        lo[0] = T.int32(0)
                        hi[0] = T.int32(num_experts - 1)
                        for _bs in T.serial(log2_up):
                            mid = (lo[0] + hi[0]) >> T.int32(1)
                            if s_cum[mid + 1] <= m_tile:
                                lo[0] = mid + T.int32(1)
                            else:
                                hi[0] = mid
                        expert_id = lo[0]
                        row = (m_tile - s_cum[expert_id]) * T.int32(block_m)
                        m_start = true_offsets[expert_id] + row
                        n_start = n_tile * T.int32(block_n)
                        # Bottom-half row count: clamp(true_arows-half_m, 0, half_m).
                        true_arows = true_sizes[expert_id] - row
                        arows1 = T.max(T.int32(0),
                                       T.min(T.int32(half_m),
                                             true_arows - T.int32(half_m)))


                        for k in T.Pipelined(_k_iters, num_stages=0):
                            slot = gi_cons_1 % num_stages
                            T.barrier_wait(ab_full[slot],
                                           (gi_cons_1 // num_stages) & 1)
                            T.wgmma_gemm(
                                A_smem_bot[slot, :, :],
                                B_smem[slot, :, :],
                                C_local_wg1,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow,
                                clear_accum=(k == 0),
                            )
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(C_local_wg1, num_regs=64)
                            T.barrier_arrive(ab_empty[slot])
                            gi_cons_1 = gi_cons_1 + 1

                        # ── Epilogue (bottom half) ──
                        T.copy(C_local_wg1, C_local_cast_wg1)
                        if arows1 == T.int32(half_m):
                            # Fast path: full bottom-half tile via TMA store.
                            T.copy(C_local_cast_wg1, C_shared_wg1)
                            T.copy(C_shared_wg1,
                                   C[m_start + half_m, n_start])
                        elif arows1 > T.int32(0):
                            # Slow path: predicated direct STG.
                            for i, j in T.Parallel(half_m, block_n):
                                if i < arows1:
                                    C[m_start + half_m + i, n_start + j] = C_local_cast_wg1[i, j]
                        # else: bottom half empty (true_arows ≤ half_m), skip writes

    return _gemm_main_v2_coop


@functools.lru_cache(maxsize=64)
def _persistent_grouped_gemm_v2_kernel(numel, num_experts, N, K, dtype,
                                       sm_count, block_k):
    """V2 persistent grouped-GEMM JIT factory.

    Dispatches between two prim_func templates by ``block_m``:
      * ``block_m <= 64``  → pingpong (2 math WGs work independent tiles)
      * ``block_m >= 128`` → cooperative (2 math WGs split one tile's M
        via split-A; B is shared)
    """
    if K % block_k != 0:
        raise ValueError(
            f"GroupedGemmPersistent3WGKernel requires K-alignment "
            f"(K={K} must be divisible by block_k={block_k}). "
            f"Use GroupedGemmPersistentKernel for K-unaligned shapes."
        )

    @tilelang.jit(
        out_idx=[],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16", "-include", _ANCHOR_HELPER_PATH],
    )
    def _func(block_m, block_n, block_k, num_stages, threads, group_size_m):
        if block_m <= 64:
            return _make_pingpong_kernel(
                numel, num_experts, N, K, dtype, sm_count,
                block_m, block_n, block_k, num_stages, threads, group_size_m,
            )
        else:
            return _make_cooperative_kernel(
                numel, num_experts, N, K, dtype, sm_count,
                block_m, block_n, block_k, num_stages, threads, group_size_m,
            )

    return _func
