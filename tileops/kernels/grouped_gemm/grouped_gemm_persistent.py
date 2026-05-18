"""Persistent grouped GEMM kernel with global atomic work-stealing.

Single-kernel design (no separate tile scheduler):
  - 132 CTAs (one per H100 SM) form a fixed persistent grid
  - Each CTA atomically claims tiles from a global counter
  - Tile metadata (batch_id, row_offset) computed inline via prefix sum + binary search
  - K-aligned fast path uses manual 2-WG warp-specialized double-buffer
    (producer WG issues TMA, consumer WG runs WGMMA)
  - K-unaligned uses predicated path with automatic pipeline

Inspired by CUTLASS GroupedProblemVisitor: a single grid dispatches across all
grouped problems, with each CTA pulling work from a shared queue.

Trade-off vs the two-phase nopad kernel:
  + No tile_scheduler kernel launch / no scheduler output buffers
  + Better load balance for skewed token distributions
  - Atomic counter contention on a single global int32
  - Each CTA recomputes prefix sum + binary search (cheap, O(log E))
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

__all__ = [
    "GroupedGemmPersistentKernel",
]


_DEFAULT_CONFIG = {
    "block_m": 64,
    "block_n": 256,
    "block_k": 64,
    "num_stages": 2,
    "threads": 256,
    "group_size_m": 1,
}


@functools.lru_cache(maxsize=64)
def _persistent_grouped_gemm_kernel(
    numel: int,
    num_experts: int,
    N: int,
    K: int,
    dtype: str,
    sm_count: int,
    block_k: int,
):
    """Build a persistent grouped-GEMM JIT factory.

    Args:
        numel:       Total tight row count across all batches.
        num_experts: Number of batches E.
        N, K:        Output / input feature dims.
        dtype:       Activation dtype string ("float16" or "bfloat16").
        sm_count:    Number of CTAs to launch (one per SM, persistent grid).
        block_k:     K-tile size (needed to determine K-alignment path).
    """
    accum_dtype = "float"
    log2_up = max(1, math.ceil(math.log2(num_experts + 1)))

    _k_aligned = (K % block_k == 0)

    if _k_aligned:
        @tilelang.jit(
            out_idx=[],
            pass_configs={
                tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
                tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
            },
            compile_flags=["-O3", "-DENABLE_BF16", "-include", _ANCHOR_HELPER_PATH],
        )
        def _func(block_m, block_n, block_k, num_stages, threads, group_size_m):
            # K-aligned path is a manual 2-warpgroup WS double-buffer; barrier
            # arrive_counts (128 each) are bound to this layout, so threads
            # must be exactly 256.  Fail early if a caller overrides it.
            assert threads == 256, (
                f"K-aligned persistent grouped GEMM requires threads=256 "
                f"(1 producer WG + 1 consumer WG); got threads={threads}"
            )
            _num_pid_n = math.ceil(N / block_n)
            _max_tiles = numel // block_m + num_experts
            _total_ctas_ub = _max_tiles * _num_pid_n
            _max_iters = (_total_ctas_ub + sm_count - 1) // sm_count + 2
            _k_iters = K // block_k
            A_shape = (numel + block_m, K)

            @T.prim_func
            def _gemm_main(
                A: T.Tensor(A_shape, dtype),                       # type: ignore  # noqa: F821
                B: T.Tensor((num_experts, N, K), dtype),           # type: ignore  # noqa: F821
                true_sizes: T.Tensor((num_experts,), "int32"),     # noqa: F821
                true_offsets: T.Tensor((num_experts,), "int32"),   # noqa: F821
                C: T.Tensor((numel, N), dtype),                    # type: ignore  # noqa: F821
                tile_counter: T.Tensor((1,), "int32"),             # noqa: F821
            ):
                with T.Kernel(sm_count, threads=threads) as (pid,):
                    # ── Double-buffered SMEM ──
                    A_smem_0 = T.alloc_shared((block_m, block_k), dtype)
                    A_smem_1 = T.alloc_shared((block_m, block_k), dtype)
                    B_smem_0 = T.alloc_shared((block_n, block_k), dtype)
                    B_smem_1 = T.alloc_shared((block_n, block_k), dtype)
                    C_local = T.alloc_fragment((block_m, block_n), accum_dtype)

                    # ── Scheduler SMEM ──
                    s_cum = T.alloc_shared((num_experts + 1,), "int32")
                    s_total = T.alloc_shared((1,), "int32")
                    s_tile = T.alloc_shared((1,), "int32")
                    lo = T.alloc_local((1,), "int32")
                    hi = T.alloc_local((1,), "int32")

                    T.annotate_layout({
                        A_smem_0: tilelang.layout.make_swizzled_layout(A_smem_0),
                        A_smem_1: tilelang.layout.make_swizzled_layout(A_smem_1),
                        B_smem_0: tilelang.layout.make_swizzled_layout(B_smem_0),
                        B_smem_1: tilelang.layout.make_swizzled_layout(B_smem_1),
                    })

                    # ── Barriers: producer→consumer (full), consumer→producer (empty) ──
                    ab_full_0 = T.alloc_barrier(arrive_count=128)
                    ab_full_1 = T.alloc_barrier(arrive_count=128)
                    ab_empty_0 = T.alloc_barrier(arrive_count=128)
                    ab_empty_1 = T.alloc_barrier(arrive_count=128)

                    # ── Phase counters (monotonically increasing) ──
                    gi_prod = T.alloc_var("int32", init=0)
                    gi_cons = T.alloc_var("int32", init=0)

                    tx = T.get_thread_binding()

                    # ════════════════════════════════════════════════════════
                    # Producer WG: tx < 128
                    # ════════════════════════════════════════════════════════
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
                                s_tile[0] = T.atomic_add(tile_counter[0], 1, return_prev=True)
                            T.sync_threads()

                            flat_id = s_tile[0]
                            total = s_total[0]

                            if flat_id < total:
                                if group_size_m == 1:
                                    m_tile = flat_id // T.int32(_num_pid_n)
                                    n_tile = flat_id % T.int32(_num_pid_n)
                                else:
                                    num_pid_in_group = group_size_m * _num_pid_n
                                    m_tiles_total = s_cum[num_experts]
                                    pid_in_group = flat_id % T.int32(num_pid_in_group)
                                    group_id = flat_id // T.int32(num_pid_in_group)
                                    first_pid_m = group_id * T.int32(group_size_m)
                                    actual_gsm = T.min(m_tiles_total - first_pid_m,
                                                       T.int32(group_size_m))
                                    m_tile = first_pid_m + pid_in_group % actual_gsm
                                    n_tile = pid_in_group // actual_gsm

                                lo[0] = T.int32(0)
                                hi[0] = T.int32(num_experts - 1)
                                for _bs in T.serial(log2_up):
                                    mid = (lo[0] + hi[0]) >> T.int32(1)
                                    if s_cum[mid + 1] <= m_tile:
                                        lo[0] = mid + T.int32(1)
                                    else:
                                        hi[0] = mid
                                expert_id = lo[0]
                                row_in_expert = (m_tile - s_cum[expert_id]) * T.int32(block_m)
                                m_start = true_offsets[expert_id] + row_in_expert
                                n_start = n_tile * T.int32(block_n)

                                for k in T.Pipelined(_k_iters, num_stages=0):
                                    k_start = k * block_k
                                    if gi_prod % 2 == 0:
                                        T.barrier_wait(ab_empty_0, (gi_prod // 2 + 1) % 2)
                                        T.tma_copy(
                                            A[m_start:m_start + block_m,
                                              k_start:k_start + block_k],
                                            A_smem_0,
                                            barrier=ab_full_0,
                                        )
                                        T.tma_copy(
                                            B[expert_id,
                                              n_start:n_start + block_n,
                                              k_start:k_start + block_k],
                                            B_smem_0,
                                            barrier=ab_full_0,
                                        )
                                        T.barrier_arrive(ab_full_0)
                                    else:
                                        T.barrier_wait(ab_empty_1, (gi_prod // 2 + 1) % 2)
                                        T.tma_copy(
                                            A[m_start:m_start + block_m,
                                              k_start:k_start + block_k],
                                            A_smem_1,
                                            barrier=ab_full_1,
                                        )
                                        T.tma_copy(
                                            B[expert_id,
                                              n_start:n_start + block_n,
                                              k_start:k_start + block_k],
                                            B_smem_1,
                                            barrier=ab_full_1,
                                        )
                                        T.barrier_arrive(ab_full_1)
                                    gi_prod = gi_prod + 1

                    # ════════════════════════════════════════════════════════
                    # Consumer WG: tx >= 128
                    # ════════════════════════════════════════════════════════
                    else:
                        T.inc_max_nreg(240)

                        T.sync_threads()

                        for _iter in T.serial(_max_iters):
                            T.sync_threads()

                            flat_id = s_tile[0]
                            total = s_total[0]

                            if flat_id < total:
                                if group_size_m == 1:
                                    m_tile = flat_id // T.int32(_num_pid_n)
                                    n_tile = flat_id % T.int32(_num_pid_n)
                                else:
                                    num_pid_in_group = group_size_m * _num_pid_n
                                    m_tiles_total = s_cum[num_experts]
                                    pid_in_group = flat_id % T.int32(num_pid_in_group)
                                    group_id = flat_id // T.int32(num_pid_in_group)
                                    first_pid_m = group_id * T.int32(group_size_m)
                                    actual_gsm = T.min(m_tiles_total - first_pid_m,
                                                       T.int32(group_size_m))
                                    m_tile = first_pid_m + pid_in_group % actual_gsm
                                    n_tile = pid_in_group // actual_gsm

                                lo[0] = T.int32(0)
                                hi[0] = T.int32(num_experts - 1)
                                for _bs in T.serial(log2_up):
                                    mid = (lo[0] + hi[0]) >> T.int32(1)
                                    if s_cum[mid + 1] <= m_tile:
                                        lo[0] = mid + T.int32(1)
                                    else:
                                        hi[0] = mid
                                expert_id = lo[0]
                                row_in_expert = (m_tile - s_cum[expert_id]) * T.int32(block_m)
                                m_start = true_offsets[expert_id] + row_in_expert
                                n_start = n_tile * T.int32(block_n)
                                actual_rows = T.min(T.int32(block_m),
                                                    true_sizes[expert_id] - row_in_expert)
                                actual_cols = T.min(T.int32(block_n),
                                                    T.int32(N) - n_start)

                                T.clear(C_local)

                                for k in T.Pipelined(_k_iters, num_stages=0):
                                    if gi_cons % 2 == 0:
                                        T.barrier_wait(ab_full_0, gi_cons // 2 % 2)
                                        T.wgmma_gemm(
                                            A_smem_0, B_smem_0, C_local,
                                            transpose_B=True,
                                            policy=T.GemmWarpPolicy.FullRow,
                                            clear_accum=(k == 0),
                                        )
                                        T.wait_wgmma(0)
                                        T.warpgroup_fence_operand(C_local, num_regs=64)
                                        T.barrier_arrive(ab_empty_0)
                                    else:
                                        T.barrier_wait(ab_full_1, gi_cons // 2 % 2)
                                        T.wgmma_gemm(
                                            A_smem_1, B_smem_1, C_local,
                                            transpose_B=True,
                                            policy=T.GemmWarpPolicy.FullRow,
                                            clear_accum=(k == 0),
                                        )
                                        T.wait_wgmma(0)
                                        T.warpgroup_fence_operand(C_local, num_regs=64)
                                        T.barrier_arrive(ab_empty_1)
                                    gi_cons = gi_cons + 1

                                for i, j in T.Parallel(block_m, block_n):
                                    if i < actual_rows and j < actual_cols:
                                        C[m_start + i, n_start + j] = C_local[i, j]

            return _gemm_main

    else:
        @tilelang.jit(
            out_idx=[],
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
            compile_flags=["-O3", "-DENABLE_BF16"],
        )
        def _func(block_m, block_n, block_k, num_stages, threads, group_size_m):  # noqa: F811
            _num_pid_n = math.ceil(N / block_n)
            _max_tiles = numel // block_m + num_experts
            _total_ctas_ub = _max_tiles * _num_pid_n
            _max_iters = (_total_ctas_ub + sm_count - 1) // sm_count + 2
            A_shape = (numel, K)

            @T.prim_func
            def _gemm_main(
                A: T.Tensor(A_shape, dtype),                         # type: ignore  # noqa: F821
                B: T.Tensor((num_experts, N, K), dtype),             # type: ignore  # noqa: F821
                true_sizes: T.Tensor((num_experts,), "int32"),       # noqa: F821
                true_offsets: T.Tensor((num_experts,), "int32"),     # noqa: F821
                C: T.Tensor((numel, N), dtype),                      # type: ignore  # noqa: F821
                tile_counter: T.Tensor((1,), "int32"),               # noqa: F821
            ):
                with T.Kernel(sm_count, threads=threads) as (pid,):
                    A_shared = T.alloc_shared((block_m, block_k), dtype)
                    B_shared = T.alloc_shared((block_n, block_k), dtype)
                    C_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                    s_cum = T.alloc_shared((num_experts + 1,), "int32")
                    s_total = T.alloc_shared((1,), "int32")
                    s_tile = T.alloc_shared((1,), "int32")
                    s_expert = T.alloc_shared((1,), "int32")
                    lo = T.alloc_local((1,), "int32")
                    hi = T.alloc_local((1,), "int32")

                    T.annotate_layout({
                        A_shared: tilelang.layout.make_swizzled_layout(A_shared),
                        B_shared: tilelang.layout.make_swizzled_layout(B_shared),
                    })

                    tx = T.get_thread_binding()

                    if tx == 0:
                        s_cum[0] = T.int32(0)
                        for e in T.serial(num_experts):
                            s_cum[e + 1] = s_cum[e] + (true_sizes[e] + (block_m - 1)) // block_m
                        s_total[0] = s_cum[num_experts] * T.int32(_num_pid_n)
                    T.sync_threads()

                    for _iter in T.serial(_max_iters):
                        if tx == 0:
                            s_tile[0] = T.atomic_add(tile_counter[0], 1, return_prev=True)
                        T.sync_threads()

                        flat_id = s_tile[0]
                        total = s_total[0]

                        if flat_id < total:
                            if group_size_m == 1:
                                m_tile = flat_id // T.int32(_num_pid_n)
                                n_tile = flat_id % T.int32(_num_pid_n)
                            else:
                                num_pid_in_group = group_size_m * _num_pid_n
                                m_tiles_total = s_cum[num_experts]
                                pid_in_group = flat_id % T.int32(num_pid_in_group)
                                group_id = flat_id // T.int32(num_pid_in_group)
                                first_pid_m = group_id * T.int32(group_size_m)
                                actual_gsm = T.min(m_tiles_total - first_pid_m,
                                                   T.int32(group_size_m))
                                m_tile = first_pid_m + pid_in_group % actual_gsm
                                n_tile = pid_in_group // actual_gsm

                            lo[0] = T.int32(0)
                            hi[0] = T.int32(num_experts - 1)
                            for _bs in T.serial(log2_up):
                                mid = (lo[0] + hi[0]) >> T.int32(1)
                                if s_cum[mid + 1] <= m_tile:
                                    lo[0] = mid + T.int32(1)
                                else:
                                    hi[0] = mid
                            if tx == 0:
                                s_expert[0] = lo[0]
                            T.sync_threads()
                            expert_id = s_expert[0]
                            row_in_expert = (m_tile - s_cum[expert_id]) * T.int32(block_m)
                            m_start = true_offsets[expert_id] + row_in_expert
                            n_start = n_tile * T.int32(block_n)
                            actual_rows = T.min(T.int32(block_m),
                                                true_sizes[expert_id] - row_in_expert)
                            actual_cols = T.min(T.int32(block_n),
                                                T.int32(N) - n_start)

                            T.clear(C_local)

                            for k in T.Pipelined(T.ceildiv(K, block_k),
                                                 num_stages=num_stages):
                                for i, j in T.Parallel(block_m, block_k):
                                    A_shared[i, j] = T.if_then_else(
                                        i < actual_rows and j < K - k * block_k,
                                        A[m_start + i, k * block_k + j], 0)
                                for i, j in T.Parallel(block_n, block_k):
                                    B_shared[i, j] = T.if_then_else(
                                        j < K - k * block_k and i < actual_cols,
                                        B[expert_id, n_start + i, k * block_k + j], 0)
                                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                            for i, j in T.Parallel(block_m, block_n):
                                if i < actual_rows and j < actual_cols:
                                    C[m_start + i, n_start + j] = C_local[i, j]

            return _gemm_main

    return _func


class GroupedGemmPersistentKernel(Kernel):
    """Persistent grouped GEMM kernel with global atomic work-stealing.

    A single GPU kernel performs prefix sum, binary search, and GEMM in one
    launch — replacing the two-phase scheduler+GEMM design of the nopad
    grouped GEMM kernel.

    The kernel launches sm_count CTAs (one per SM); each CTA atomically claims
    tiles from a single global int32 counter until exhausted.

    Args:
        numel:       Total tight row count across all batches.
        num_experts: Number of batches E.
        N:           Output feature dimension.
        K:           Input feature dimension.
        dtype:       Activation / weight dtype.
        sm_count:    Number of CTAs (default 132 for H100; override per device).
        config:      Optional config dict; falls back to default_config.
        tune:        If True, run autotune over autotune_configs.

    Example:
        >>> sm = torch.cuda.get_device_properties(0).multi_processor_count
        >>> kernel = GroupedGemmPersistentKernel(
        ...     numel=16384, num_experts=128, N=4096, K=2048,
        ...     dtype=torch.bfloat16, sm_count=sm)
        >>> C = kernel(A, B, true_sizes, true_offsets)  # [numel, N]
    """

    supported_archs: list[int] = [90]

    def __init__(
        self,
        numel: int,
        num_experts: int,
        N: int,
        K: int,
        dtype: torch.dtype = torch.bfloat16,
        sm_count: int | None = None,
        config=None,
        tune: bool = False,
    ):
        super().__init__()
        self.numel = numel
        self.num_experts = num_experts
        self.N = N
        self.K = K
        self.dtype = dtype
        if sm_count is None:
            sm_count = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
        self.sm_count = sm_count
        self.init_config(config, tune)
        # Lazy-allocated persistent global tile counter; device is set on the
        # first forward() based on the input tensor's device so multi-GPU
        # callers (e.g. caller on cuda:1) don't get a cross-device error.
        self._tile_counter: torch.Tensor | None = None

    @property
    def default_config(self) -> dict:
        return dict(_DEFAULT_CONFIG)

    @property
    def autotune_configs(self) -> list[dict]:
        configs = []
        for block_m in (64, 128):
            for block_n in (128, 256):
                for block_k in (64, 128):
                    for num_stages in (2, 3, 4):
                        configs.append({
                            "block_m": block_m,
                            "block_n": block_n,
                            "block_k": block_k,
                            "num_stages": num_stages,
                            "threads": 256,
                            "group_size_m": 1,
                        })
        return configs

    def forward(
        self,
        A: torch.Tensor,             # [numel, K]
        B: torch.Tensor,             # [num_experts, N, K]
        true_sizes: torch.Tensor,    # [E] int32
        true_offsets: torch.Tensor,  # [E] int32
    ) -> torch.Tensor:
        """Run the persistent grouped GEMM.

        Args:
            A: [numel, K] tight permuted hidden states.
            B: [num_experts, N, K] per-batch weight matrix (NT: B^T applied).
            true_sizes: [E] int32 true token count per batch.
            true_offsets: [E] int32 tight start offset per batch in A.

        Returns:
            C: [numel, N] GEMM output.
        """
        # (Re)allocate the tile counter on the input's device so multi-GPU
        # callers work correctly.  Zero it every forward — it's the shared
        # atomic that every CTA increments to claim tiles.
        if self._tile_counter is None or self._tile_counter.device != A.device:
            self._tile_counter = torch.zeros(1, dtype=torch.int32, device=A.device)
        else:
            self._tile_counter.zero_()
        # Pre-allocate output and zero-fill so that M-tiles/N-tiles skipped by
        # the work-stealing loop (e.g. total_tiles == 0, ragged final M-tile)
        # contribute genuine zeros instead of uninitialised memory.
        C = torch.zeros(self.numel, self.N, dtype=self.dtype, device=A.device)
        # K-aligned fast path uses T.copy (TMA-eligible) with no row predicate.
        # The last M-tile of the last expert reads up to block_m rows starting
        # at numel — past the end of A.  Pad with block_m zero rows so the
        # T.copy lands in valid memory; the epilogue guard prevents writing
        # those zeros to C.  Overhead: block_m × K elements (tiny).
        #
        # TODO: replace F.pad with TMA's built-in OOB zero-fill once TileLang
        # exposes that knob via T.tma_copy.  F.pad allocates numel*K + block_m*K
        # bytes every forward; TMA OOB would drop the extra copy.
        block_m = self.config["block_m"]
        block_k = self.config["block_k"]
        if self.K % block_k == 0:
            A = F.pad(A, (0, 0, 0, block_m))
        gemm_fn = _persistent_grouped_gemm_kernel(
            self.numel, self.num_experts, self.N, self.K,
            self.dtype_str, self.sm_count, block_k,
        )(
            block_m,
            self.config["block_n"],
            block_k,
            self.config["num_stages"],
            self.config["threads"],
            self.config.get("group_size_m", 1),
        )
        gemm_fn(A, B, true_sizes, true_offsets, C, self._tile_counter)
        return C
