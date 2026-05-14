"""V2 persistent grouped-GEMM kernel: warp-specialized + static wave + TMA store.

Mirrors the structure of the TileLang SM100 example
``gemm_tcgen5mma_ws_persistent.py``, with all SM100-only primitives
(TMEM, tcgen05_gemm, CLC, 2-CTA MMA) replaced by SM90 equivalents.

Design: see tileops_md/2026-05-15_grouped_gemm_persistent_v2_design.md.

Two compile-time templates routed by ``block_m``:
  * ``block_m ≤ 64``  → Pingpong: 2 math WGs process independent tiles
  * ``block_m ≥ 128`` → Cooperative: 2 math WGs share one tile's M

K-aligned only.  K-unaligned use ``GroupedGemmPersistentKernel``.

V2 phase 1: pingpong (atomic_add scheduler, parity drop-in).
Literal port of GroupedGemmPersistent3WGKernel; only names differ.  Future
phases will swap the scheduler, replace the double-buffer with a
num_stages ring, add TMA-store epilogue, and add a cooperative template.

Defaults differ from 3WG: ``num_stages`` is 2 (phase-1 baseline for the
upcoming ring-buffer phase) and ``autotune_configs`` returns a single
config; both will expand in later phases.
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

__all__ = ["GroupedGemmPersistentV2Kernel"]

_DEFAULT_CONFIG = {
    "block_m": 64,         # phase 1: bm=64 (pingpong) only
    "block_n": 256,
    "block_k": 64,
    "num_stages": 2,
    "threads": 384,
    "group_size_m": 1,
}


class GroupedGemmPersistentV2Kernel(Kernel):
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
        self._tile_counter: torch.Tensor | None = None

    @property
    def default_config(self) -> dict:
        return dict(_DEFAULT_CONFIG)

    @property
    def autotune_configs(self) -> list[dict]:
        return [dict(_DEFAULT_CONFIG)]  # phase 1: single config

    def forward(self, A, B, true_sizes, true_offsets):
        if self._tile_counter is None or self._tile_counter.device != A.device:
            self._tile_counter = torch.zeros(1, dtype=torch.int32, device=A.device)
        else:
            self._tile_counter.zero_()
        C = torch.zeros(self.numel, self.N, dtype=self.dtype, device=A.device)
        block_m = self.config["block_m"]
        block_n = self.config["block_n"]
        block_k = self.config["block_k"]
        if self.K % block_k != 0:
            raise ValueError(f"K-aligned only: K={self.K}, block_k={block_k}")
        if self.N % block_n != 0:
            raise ValueError(f"N-aligned only: N={self.N}, block_n={block_n}")
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
        gemm_fn(A, B, true_sizes, true_offsets, C, self._tile_counter)
        return C


@functools.lru_cache(maxsize=64)
def _persistent_grouped_gemm_v2_kernel(numel, num_experts, N, K, dtype,
                                       sm_count, block_k):
    """Build a V2 persistent grouped-GEMM JIT factory.

    V2 phase 1: pingpong (atomic_add scheduler, parity drop-in).
    Literal port of _persistent_grouped_gemm_3wg_kernel; only the outer
    factory name and internal prim_func name differ.
    """
    accum_dtype = "float"
    log2_up = max(1, math.ceil(math.log2(num_experts + 1)))

    if K % block_k != 0:
        raise ValueError(
            f"GroupedGemmPersistentV2Kernel requires K-alignment "
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
        # V2 pingpong layout: 1 producer + 2 consumers, each WG = 128 threads.
        assert threads == 384, (
            f"V2 pingpong persistent grouped GEMM requires threads=384 "
            f"(1 producer + 2 consumer WGs); got threads={threads}"
        )
        _num_pid_n = math.ceil(N / block_n)
        _max_tiles = numel // block_m + num_experts
        _total_ctas_ub = _max_tiles * _num_pid_n
        # Each iteration claims 2 tiles per CTA (atomic_add(+2)); slack of 2
        # absorbs the case where total_ctas_ub is not a multiple of 2*sm_count.
        # Each iteration claims 2 tiles per CTA (atomic_add(+2)); slack of 2
        # absorbs the case where total_ctas_ub is not a multiple of 2*sm_count.
        _max_iters = (_total_ctas_ub + 2 * sm_count - 1) // (2 * sm_count) + 2
        _k_iters = K // block_k
        A_shape = (numel + block_m, K)

        @T.prim_func
        def _gemm_main_v2(
            A: T.Tensor(A_shape, dtype),                        # type: ignore  # noqa: F821
            B: T.Tensor((num_experts, N, K), dtype),            # type: ignore  # noqa: F821
            true_sizes: T.Tensor((num_experts,), "int32"),      # noqa: F821
            true_offsets: T.Tensor((num_experts,), "int32"),    # noqa: F821
            C: T.Tensor((numel, N), dtype),                     # type: ignore  # noqa: F821
            tile_counter: T.Tensor((1,), "int32"),              # noqa: F821
        ):
            with T.Kernel(sm_count, threads=threads) as (pid,):
                # ── Per-WG double-buffered SMEM (independent tiles) ──
                A_smem_wg0_0 = T.alloc_shared((block_m, block_k), dtype)
                A_smem_wg0_1 = T.alloc_shared((block_m, block_k), dtype)
                B_smem_wg0_0 = T.alloc_shared((block_n, block_k), dtype)
                B_smem_wg0_1 = T.alloc_shared((block_n, block_k), dtype)
                A_smem_wg1_0 = T.alloc_shared((block_m, block_k), dtype)
                A_smem_wg1_1 = T.alloc_shared((block_m, block_k), dtype)
                B_smem_wg1_0 = T.alloc_shared((block_n, block_k), dtype)
                B_smem_wg1_1 = T.alloc_shared((block_n, block_k), dtype)
                C_local_wg0 = T.alloc_fragment((block_m, block_n), accum_dtype)
                C_local_wg1 = T.alloc_fragment((block_m, block_n), accum_dtype)

                # ── Scheduler SMEM ──
                s_cum = T.alloc_shared((num_experts + 1,), "int32")
                s_total = T.alloc_shared((1,), "int32")
                # Two consecutive tile IDs claimed per producer iteration.
                s_tile_pair = T.alloc_shared((2,), "int32")
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
                    A_smem_wg0_0: tilelang.layout.make_swizzled_layout(A_smem_wg0_0),
                    A_smem_wg0_1: tilelang.layout.make_swizzled_layout(A_smem_wg0_1),
                    B_smem_wg0_0: tilelang.layout.make_swizzled_layout(B_smem_wg0_0),
                    B_smem_wg0_1: tilelang.layout.make_swizzled_layout(B_smem_wg0_1),
                    A_smem_wg1_0: tilelang.layout.make_swizzled_layout(A_smem_wg1_0),
                    A_smem_wg1_1: tilelang.layout.make_swizzled_layout(A_smem_wg1_1),
                    B_smem_wg1_0: tilelang.layout.make_swizzled_layout(B_smem_wg1_0),
                    B_smem_wg1_1: tilelang.layout.make_swizzled_layout(B_smem_wg1_1),
                })

                # ── Per-WG producer/consumer barriers (arrive_count=128) ──
                ab_full_wg0_0 = T.alloc_barrier(arrive_count=128)
                ab_full_wg0_1 = T.alloc_barrier(arrive_count=128)
                ab_empty_wg0_0 = T.alloc_barrier(arrive_count=128)
                ab_empty_wg0_1 = T.alloc_barrier(arrive_count=128)
                ab_full_wg1_0 = T.alloc_barrier(arrive_count=128)
                ab_full_wg1_1 = T.alloc_barrier(arrive_count=128)
                ab_empty_wg1_0 = T.alloc_barrier(arrive_count=128)
                ab_empty_wg1_1 = T.alloc_barrier(arrive_count=128)

                # ── Phase counters: monotonic, never reset ──
                gi_prod_0 = T.alloc_var("int32", init=0)
                gi_prod_1 = T.alloc_var("int32", init=0)
                gi_cons_0 = T.alloc_var("int32", init=0)
                gi_cons_1 = T.alloc_var("int32", init=0)

                tx = T.get_thread_binding()

                # ════════════════════════════════════════════════════════
                # Producer WG: tx < 128
                # ════════════════════════════════════════════════════════
                # CUTLASS pingpong producer pattern: a *single* K-loop body
                # issues the WG0 TMA pair followed by the WG1 TMA pair at
                # each k, so WG1's first stage is ready right after WG0's
                # first stage — both consumers start in parallel.  The
                # previous "feed WG0 K-loop fully, then WG1 K-loop fully"
                # form forced WG1 to wait K_iters TMAs before its first
                # WGMMA could dispatch, defeating pingpong.
                if tx < 128:
                    T.dec_max_nreg(24)

                    if tx == 0:
                        s_cum[0] = T.int32(0)
                        for e in T.serial(num_experts):
                            s_cum[e + 1] = s_cum[e] + (true_sizes[e] + (block_m - 1)) // block_m
                        s_total[0] = s_cum[num_experts] * T.int32(_num_pid_n)
                    T.sync_threads()

                    for _iter in T.serial(_max_iters):
                        if tx == 0:
                            # atomic_add(+2) → atomic claim of both tiles in
                            # one transaction.  Two separate +1 adds would
                            # let outsider CTAs interleave and break tile-
                            # pair adjacency (Known trap #4).
                            s_tile_pair[0] = T.atomic_add(
                                tile_counter[0], T.int32(2), return_prev=True
                            )
                            s_tile_pair[1] = s_tile_pair[0] + T.int32(1)
                        T.sync_threads()

                        total = s_total[0]

                        # ── Resolve WG0 tile metadata into alloc_local ──
                        flat_id_0 = s_tile_pair[0]
                        if flat_id_0 < total:
                            if group_size_m == 1:
                                m_tile_0 = flat_id_0 // T.int32(_num_pid_n)
                                n_tile_0 = flat_id_0 % T.int32(_num_pid_n)
                            else:
                                num_pid_in_group = group_size_m * _num_pid_n
                                m_tiles_total = s_cum[num_experts]
                                pig_0 = flat_id_0 % T.int32(num_pid_in_group)
                                g_id_0 = flat_id_0 // T.int32(num_pid_in_group)
                                fp_m_0 = g_id_0 * T.int32(group_size_m)
                                agsm_0 = T.min(m_tiles_total - fp_m_0,
                                               T.int32(group_size_m))
                                m_tile_0 = fp_m_0 + pig_0 % agsm_0
                                n_tile_0 = pig_0 // agsm_0

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
                        flat_id_1 = s_tile_pair[1]
                        if flat_id_1 < total:
                            if group_size_m == 1:
                                m_tile_1 = flat_id_1 // T.int32(_num_pid_n)
                                n_tile_1 = flat_id_1 % T.int32(_num_pid_n)
                            else:
                                num_pid_in_group = group_size_m * _num_pid_n
                                m_tiles_total = s_cum[num_experts]
                                pig_1 = flat_id_1 % T.int32(num_pid_in_group)
                                g_id_1 = flat_id_1 // T.int32(num_pid_in_group)
                                fp_m_1 = g_id_1 * T.int32(group_size_m)
                                agsm_1 = T.min(m_tiles_total - fp_m_1,
                                               T.int32(group_size_m))
                                m_tile_1 = fp_m_1 + pig_1 % agsm_1
                                n_tile_1 = pig_1 // agsm_1

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

                            # WG0 stream
                            if v0[0] != 0:
                                if gi_prod_0 % 2 == 0:
                                    T.barrier_wait(ab_empty_wg0_0,
                                                   (gi_prod_0 // 2 + 1) % 2)
                                    T.tma_copy(
                                        A[ms0[0]:ms0[0] + block_m,
                                          k_start:k_start + block_k],
                                        A_smem_wg0_0, barrier=ab_full_wg0_0,
                                    )
                                    T.tma_copy(
                                        B[ex0[0],
                                          ns0[0]:ns0[0] + block_n,
                                          k_start:k_start + block_k],
                                        B_smem_wg0_0, barrier=ab_full_wg0_0,
                                    )
                                    T.barrier_arrive(ab_full_wg0_0)
                                else:
                                    T.barrier_wait(ab_empty_wg0_1,
                                                   (gi_prod_0 // 2 + 1) % 2)
                                    T.tma_copy(
                                        A[ms0[0]:ms0[0] + block_m,
                                          k_start:k_start + block_k],
                                        A_smem_wg0_1, barrier=ab_full_wg0_1,
                                    )
                                    T.tma_copy(
                                        B[ex0[0],
                                          ns0[0]:ns0[0] + block_n,
                                          k_start:k_start + block_k],
                                        B_smem_wg0_1, barrier=ab_full_wg0_1,
                                    )
                                    T.barrier_arrive(ab_full_wg0_1)
                                gi_prod_0 = gi_prod_0 + 1

                            # WG1 stream
                            if v1[0] != 0:
                                if gi_prod_1 % 2 == 0:
                                    T.barrier_wait(ab_empty_wg1_0,
                                                   (gi_prod_1 // 2 + 1) % 2)
                                    T.tma_copy(
                                        A[ms1[0]:ms1[0] + block_m,
                                          k_start:k_start + block_k],
                                        A_smem_wg1_0, barrier=ab_full_wg1_0,
                                    )
                                    T.tma_copy(
                                        B[ex1[0],
                                          ns1[0]:ns1[0] + block_n,
                                          k_start:k_start + block_k],
                                        B_smem_wg1_0, barrier=ab_full_wg1_0,
                                    )
                                    T.barrier_arrive(ab_full_wg1_0)
                                else:
                                    T.barrier_wait(ab_empty_wg1_1,
                                                   (gi_prod_1 // 2 + 1) % 2)
                                    T.tma_copy(
                                        A[ms1[0]:ms1[0] + block_m,
                                          k_start:k_start + block_k],
                                        A_smem_wg1_1, barrier=ab_full_wg1_1,
                                    )
                                    T.tma_copy(
                                        B[ex1[0],
                                          ns1[0]:ns1[0] + block_n,
                                          k_start:k_start + block_k],
                                        B_smem_wg1_1, barrier=ab_full_wg1_1,
                                    )
                                    T.barrier_arrive(ab_full_wg1_1)
                                gi_prod_1 = gi_prod_1 + 1

                # ════════════════════════════════════════════════════════
                # Consumer WG0: 128 ≤ tx < 256 — processes s_tile_pair[0]
                # ════════════════════════════════════════════════════════
                # Since cons0 and cons1 own *independent* SMEM buffers and
                # *independent* ab_full/ab_empty barrier pairs, the two math
                # WGs never collide on shared resources and no math-WG
                # ordering protocol is needed.  An earlier version added a
                # `math_wg_order_{0,1}` mbarrier pair (CUTLASS-style) but it
                # introduced a phase-tracking race: a single-bit mbarrier
                # parity cannot tolerate `arrive` running ahead of `wait` by
                # more than one iteration, which happens whenever the two
                # WGs' mainloop+epilogue durations diverge (e.g. one WG
                # claims an invalid tile and skips its mainloop).
                elif tx < 256:
                    T.inc_max_nreg(240)
                    T.sync_threads()

                    for _iter in T.serial(_max_iters):
                        T.sync_threads()

                        flat_id_0 = s_tile_pair[0]
                        total = s_total[0]
                        valid_0 = flat_id_0 < total

                        if valid_0:
                            if group_size_m == 1:
                                m_tile_0 = flat_id_0 // T.int32(_num_pid_n)
                                n_tile_0 = flat_id_0 % T.int32(_num_pid_n)
                            else:
                                num_pid_in_group = group_size_m * _num_pid_n
                                m_tiles_total = s_cum[num_experts]
                                pig_0 = flat_id_0 % T.int32(num_pid_in_group)
                                g_id_0 = flat_id_0 // T.int32(num_pid_in_group)
                                fp_m_0 = g_id_0 * T.int32(group_size_m)
                                agsm_0 = T.min(m_tiles_total - fp_m_0,
                                               T.int32(group_size_m))
                                m_tile_0 = fp_m_0 + pig_0 % agsm_0
                                n_tile_0 = pig_0 // agsm_0

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

                            T.clear(C_local_wg0)

                            for k in T.Pipelined(_k_iters, num_stages=0):
                                if gi_cons_0 % 2 == 0:
                                    T.barrier_wait(ab_full_wg0_0, gi_cons_0 // 2 % 2)
                                    T.wgmma_gemm(
                                        A_smem_wg0_0, B_smem_wg0_0, C_local_wg0,
                                        transpose_B=True,
                                        policy=T.GemmWarpPolicy.FullRow,
                                        clear_accum=(k == 0),
                                    )
                                    T.wait_wgmma(0)
                                    T.warpgroup_fence_operand(C_local_wg0, num_regs=64)
                                    T.barrier_arrive(ab_empty_wg0_0)
                                else:
                                    T.barrier_wait(ab_full_wg0_1, gi_cons_0 // 2 % 2)
                                    T.wgmma_gemm(
                                        A_smem_wg0_1, B_smem_wg0_1, C_local_wg0,
                                        transpose_B=True,
                                        policy=T.GemmWarpPolicy.FullRow,
                                        clear_accum=(k == 0),
                                    )
                                    T.wait_wgmma(0)
                                    T.warpgroup_fence_operand(C_local_wg0, num_regs=64)
                                    T.barrier_arrive(ab_empty_wg0_1)
                                gi_cons_0 = gi_cons_0 + 1

                            for i, j in T.Parallel(block_m, block_n):
                                if i < arows_0 and j < acols_0:
                                    C[m_start_0 + i, n_start_0 + j] = C_local_wg0[i, j]

                # ════════════════════════════════════════════════════════
                # Consumer WG1: tx ≥ 256 — processes s_tile_pair[1]
                # ════════════════════════════════════════════════════════
                else:
                    T.inc_max_nreg(240)
                    T.sync_threads()

                    for _iter in T.serial(_max_iters):
                        T.sync_threads()

                        flat_id_1 = s_tile_pair[1]
                        total = s_total[0]
                        valid_1 = flat_id_1 < total

                        if valid_1:
                            if group_size_m == 1:
                                m_tile_1 = flat_id_1 // T.int32(_num_pid_n)
                                n_tile_1 = flat_id_1 % T.int32(_num_pid_n)
                            else:
                                num_pid_in_group = group_size_m * _num_pid_n
                                m_tiles_total = s_cum[num_experts]
                                pig_1 = flat_id_1 % T.int32(num_pid_in_group)
                                g_id_1 = flat_id_1 // T.int32(num_pid_in_group)
                                fp_m_1 = g_id_1 * T.int32(group_size_m)
                                agsm_1 = T.min(m_tiles_total - fp_m_1,
                                               T.int32(group_size_m))
                                m_tile_1 = fp_m_1 + pig_1 % agsm_1
                                n_tile_1 = pig_1 // agsm_1

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

                            T.clear(C_local_wg1)

                            for k in T.Pipelined(_k_iters, num_stages=0):
                                if gi_cons_1 % 2 == 0:
                                    T.barrier_wait(ab_full_wg1_0, gi_cons_1 // 2 % 2)
                                    T.wgmma_gemm(
                                        A_smem_wg1_0, B_smem_wg1_0, C_local_wg1,
                                        transpose_B=True,
                                        policy=T.GemmWarpPolicy.FullRow,
                                        clear_accum=(k == 0),
                                    )
                                    T.wait_wgmma(0)
                                    T.warpgroup_fence_operand(C_local_wg1, num_regs=64)
                                    T.barrier_arrive(ab_empty_wg1_0)
                                else:
                                    T.barrier_wait(ab_full_wg1_1, gi_cons_1 // 2 % 2)
                                    T.wgmma_gemm(
                                        A_smem_wg1_1, B_smem_wg1_1, C_local_wg1,
                                        transpose_B=True,
                                        policy=T.GemmWarpPolicy.FullRow,
                                        clear_accum=(k == 0),
                                    )
                                    T.wait_wgmma(0)
                                    T.warpgroup_fence_operand(C_local_wg1, num_regs=64)
                                    T.barrier_arrive(ab_empty_wg1_1)
                                gi_cons_1 = gi_cons_1 + 1

                            for i, j in T.Parallel(block_m, block_n):
                                if i < arows_1 and j < acols_1:
                                    C[m_start_1 + i, n_start_1 + j] = C_local_wg1[i, j]

        return _gemm_main_v2

    return _func
