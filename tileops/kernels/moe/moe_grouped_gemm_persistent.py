"""MoE-specific persistent grouped GEMM kernel (NT layout).

Single-kernel persistent design: Grid = SM_count, each CTA runs a while loop
claiming tiles via atomicAdd on a global counter.  Eliminates the separate
tile-scheduler kernel and dead CTAs present in the two-phase nopad design.

Execution flow per CTA:
  1. [Init]   Warp 0 computes cum_tiles[E+1] in SMEM via in-kernel warp scan
              (ceil(E/32) rounds of __shfl_up_sync with inter-round carry).
              __syncthreads() makes s_cum visible to all warps.
  2. [Loop]   atomicAdd(tile_counter, 1) → flat_tile_id.
              Decode: m_tile_id = flat_tile_id // N_tiles,
                      n_tile_id = flat_tile_id % N_tiles.
  3. [Search] Binary search s_cum[0..E] for expert_id and row_in_expert.
  4. [GEMM]   Execute one M×N tile GEMM, write to C.
  5. Repeat from step 2 until flat_tile_id >= total_tiles.

Parameters (baked at compile time via lru_cache):
  numel       = T * top_k  (tight row count)
  num_experts = E
  N, K        = output/input feature dims
  dtype       = activation dtype string
  sm_count    = number of SMs on the target GPU (grid size)
"""

import functools
import itertools
import math

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["MoeGroupedGemmPersistentKernel"]

_DEFAULT_CONFIG = {
    "block_m": 64, "block_n": 256, "block_k": 64,
    "num_stages": 3, "threads": 128, "group_size_m": 1,
}


@functools.lru_cache(maxsize=64)
def _persistent_moe_gemm_kernel(
    numel: int, num_experts: int, N: int, K: int,
    dtype: str = "float16", sm_count: int = 132,
):
    """Persistent NT grouped GEMM kernel: Grid=sm_count, atomicAdd tile dispatch.

    Args:
        numel:       T * top_k tight row count.
        num_experts: E total experts.
        N:           Output feature dimension.
        K:           Input feature dimension.
        dtype:       Activation dtype string (e.g. "bfloat16").
        sm_count:    Number of SMs on target GPU (sets grid size).
    """
    accum_dtype = "float"
    # Number of warp-scan rounds needed to cover all E experts.
    _scan_rounds = math.ceil(num_experts / 32)
    # Static binary search depth.
    _log2_up = max(1, math.ceil(math.log2(num_experts + 1)))

    @tilelang.jit(
        out_idx=[2],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(block_m, block_n, block_k, num_stages, threads, group_size_m):  # noqa: ARG001
        _num_pid_n = math.ceil(N / block_n)
        _k_aligned = (K % block_k == 0)
        # A gets block_m padding rows on the TMA path (same as nopad kernel).
        A_shape = (numel + block_m, K) if _k_aligned else (numel, K)
        B_shape = (num_experts, N, K)
        C_shape = (numel, N)
        A_shared_shape = (block_m, block_k)
        B_shared_shape = (block_n, block_k)

        if _k_aligned:
            @T.prim_func
            def _gemm_main(
                A: T.Tensor(A_shape, dtype),                              # type: ignore
                B: T.Tensor(B_shape, dtype),                              # type: ignore
                C: T.Tensor(C_shape, dtype),                              # type: ignore
                true_sizes: T.Tensor([num_experts], "int32"),             # noqa: F821
                true_offsets: T.Tensor([num_experts], "int32"),           # noqa: F821
                tile_counter: T.Tensor([1], "int32"),                     # noqa: F821
            ):
                with T.Kernel(sm_count, threads=threads) as (_cta_id,):
                    # SMEM layout:
                    #   s_cum[E+1]      — persistent throughout kernel lifetime
                    #   A_shared / B_shared — pipeline buffers (after s_cum)
                    s_cum = T.alloc_shared([num_experts + 1], "int32")
                    A_shared = T.alloc_shared(A_shared_shape, dtype)
                    B_shared = T.alloc_shared(B_shared_shape, dtype)
                    C_local = T.alloc_fragment([block_m, block_n], accum_dtype)
                    T.annotate_layout({
                        A_shared: tilelang.layout.make_swizzled_layout(A_shared),
                        B_shared: tilelang.layout.make_swizzled_layout(B_shared),
                    })

                    tx = T.get_thread_binding()
                    lane_idx = tx % T.int32(32)

                    # ── Phase 1: In-kernel warp scan (Warp 0 only) ───────────
                    # Thread 0 initialises s_cum[0] = 0 (exclusive prefix sum base).
                    if tx == 0:
                        s_cum[0] = T.int32(0)

                    carry = T.alloc_local([1], "int32")
                    carry[0] = T.int32(0)

                    if tx < T.int32(32):  # warp 0
                        for r in T.serial(_scan_rounds):
                            e = r * T.int32(32) + lane_idx
                            # Read tile count for this expert; 0 for out-of-range lanes.
                            val = T.alloc_local([1], "int32")
                            if e < T.int32(num_experts):
                                val[0] = (true_sizes[e] + T.int32(block_m - 1)) // T.int32(block_m)
                            else:
                                val[0] = T.int32(0)

                            # Warp-level inclusive prefix sum via T.shfl_up.
                            for shift in T.serial(5):  # shifts: 1,2,4,8,16
                                s = T.int32(1) << shift
                                v = T.shfl_up(val[0], s)
                                if lane_idx >= s:
                                    val[0] = val[0] + v

                            # Guard: only write s_cum for in-range experts
                            # (OOB lanes contribute val=0 but must not write
                            # past s_cum[E], which would corrupt A/B shared buffers).
                            smem_idx = r * T.int32(32) + lane_idx + 1
                            if smem_idx <= T.int32(num_experts):
                                s_cum[smem_idx] = val[0] + carry[0]

                            # Broadcast thread-31's value as new carry.
                            carry[0] = T.call_extern("int32", "__shfl_sync",
                                                     0xFFFFFFFF, val[0], T.int32(31)) \
                                       + carry[0]

                    # All warps must wait until warp 0 has written the full s_cum
                    # (including s_cum[0]=0 and all scan rounds) before reading it.
                    T.sync_threads()

                    total_tiles = s_cum[num_experts] * T.int32(_num_pid_n)

                    # ── Phase 2: Persistent tile loop ────────────────────────
                    # Each iteration: atomicAdd → decode flat_tile_id → GEMM tile.
                    for _iter in T.serial(
                        # Static bound: at most ceil(max_possible_tiles / sm_count) + 1
                        (numel // block_m + num_experts) * _num_pid_n // sm_count + 2
                    ):
                        flat_id = T.alloc_local([1], "int32")
                        flat_id[0] = T.call_extern("int32", "atomicAdd",
                                                   T.address_of(tile_counter[0]), T.int32(1))
                        if flat_id[0] >= total_tiles:
                            T.break_loop()  # exit persistent loop

                        m_tile_id = flat_id[0] // T.int32(_num_pid_n)
                        n_tile_id = flat_id[0] % T.int32(_num_pid_n)

                        # Binary search: find expert owning m_tile_id.
                        lo = T.alloc_local([1], "int32")
                        hi = T.alloc_local([1], "int32")
                        lo[0] = T.int32(0)
                        hi[0] = T.int32(num_experts - 1)
                        for _bs in T.serial(_log2_up):
                            mid = (lo[0] + hi[0]) >> T.int32(1)
                            if s_cum[mid + 1] <= m_tile_id:
                                lo[0] = mid + T.int32(1)
                            else:
                                hi[0] = mid

                        expert_id = lo[0]
                        # row_offset: row index offset within this expert's token block.
                        # (m_tile_id - s_cum[expert_id]) gives tile index within expert,
                        # multiplied by block_m to convert to row count.
                        row_offset = (m_tile_id - s_cum[expert_id]) * T.int32(block_m)
                        m_start = true_offsets[expert_id] + row_offset
                        n_start = n_tile_id * T.int32(block_n)
                        actual_rows = T.min(T.int32(block_m),
                                            true_sizes[expert_id] - row_offset)
                        actual_cols = T.min(T.int32(block_n), T.int32(N) - n_start)
                        T.clear(C_local)

                        for k in T.Pipelined(T.ceildiv(K, block_k), num_stages=num_stages):
                            T.copy(A[m_start:m_start + block_m,
                                     k * block_k:(k + 1) * block_k], A_shared)
                            T.copy(B[expert_id, n_start:n_start + block_n,
                                     k * block_k:(k + 1) * block_k], B_shared)
                            T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                        for i, j in T.Parallel(block_m, block_n):
                            if i < actual_rows and j < actual_cols:
                                C[m_start + i, n_start + j] = C_local[i, j]

        else:
            # Non-TMA path (K not block_k-aligned): use T.if_then_else predicates.
            @T.prim_func
            def _gemm_main(                                               # noqa: F811
                A: T.Tensor(A_shape, dtype),                              # type: ignore
                B: T.Tensor(B_shape, dtype),                              # type: ignore
                C: T.Tensor(C_shape, dtype),                              # type: ignore
                true_sizes: T.Tensor([num_experts], "int32"),             # noqa: F821
                true_offsets: T.Tensor([num_experts], "int32"),           # noqa: F821
                tile_counter: T.Tensor([1], "int32"),                     # noqa: F821
            ):
                with T.Kernel(sm_count, threads=threads) as (_cta_id,):
                    s_cum = T.alloc_shared([num_experts + 1], "int32")
                    A_shared = T.alloc_shared(A_shared_shape, dtype)
                    B_shared = T.alloc_shared(B_shared_shape, dtype)
                    C_local = T.alloc_fragment([block_m, block_n], accum_dtype)
                    T.annotate_layout({
                        A_shared: tilelang.layout.make_swizzled_layout(A_shared),
                        B_shared: tilelang.layout.make_swizzled_layout(B_shared),
                    })

                    tx = T.get_thread_binding()
                    lane_idx = tx % T.int32(32)

                    if tx == 0:
                        s_cum[0] = T.int32(0)

                    carry = T.alloc_local([1], "int32")
                    carry[0] = T.int32(0)

                    if tx < T.int32(32):
                        for r in T.serial(_scan_rounds):
                            e = r * T.int32(32) + lane_idx
                            val = T.alloc_local([1], "int32")
                            if e < T.int32(num_experts):
                                val[0] = (true_sizes[e] + T.int32(block_m - 1)) // T.int32(block_m)
                            else:
                                val[0] = T.int32(0)
                            for shift in T.serial(5):
                                s = T.int32(1) << shift
                                v = T.shfl_up(val[0], s)
                                if lane_idx >= s:
                                    val[0] = val[0] + v
                            # Guard: only write s_cum for in-range experts
                            # (OOB lanes contribute val=0 but must not write
                            # past s_cum[E], which would corrupt A/B shared buffers).
                            if e < T.int32(num_experts):
                                s_cum[r * T.int32(32) + lane_idx + 1] = val[0] + carry[0]
                            carry[0] = T.call_extern("int32", "__shfl_sync",
                                                     0xFFFFFFFF, val[0], T.int32(31)) \
                                       + carry[0]

                    T.sync_threads()
                    total_tiles = s_cum[num_experts] * T.int32(_num_pid_n)

                    for _iter in T.serial(
                        (numel // block_m + num_experts) * _num_pid_n // sm_count + 2
                    ):
                        flat_id = T.alloc_local([1], "int32")
                        flat_id[0] = T.call_extern("int32", "atomicAdd",
                                                   T.address_of(tile_counter[0]), T.int32(1))
                        if flat_id[0] >= total_tiles:
                            T.break_loop()

                        m_tile_id = flat_id[0] // T.int32(_num_pid_n)
                        n_tile_id = flat_id[0] % T.int32(_num_pid_n)

                        lo = T.alloc_local([1], "int32")
                        hi = T.alloc_local([1], "int32")
                        lo[0] = T.int32(0)
                        hi[0] = T.int32(num_experts - 1)
                        for _bs in T.serial(_log2_up):
                            mid = (lo[0] + hi[0]) >> T.int32(1)
                            if s_cum[mid + 1] <= m_tile_id:
                                lo[0] = mid + T.int32(1)
                            else:
                                hi[0] = mid

                        expert_id = lo[0]
                        # row_offset: row index offset within this expert's token block.
                        # (m_tile_id - s_cum[expert_id]) gives tile index within expert,
                        # multiplied by block_m to convert to row count.
                        row_offset = (m_tile_id - s_cum[expert_id]) * T.int32(block_m)
                        m_start = true_offsets[expert_id] + row_offset
                        n_start = n_tile_id * T.int32(block_n)
                        actual_rows = T.min(T.int32(block_m),
                                            true_sizes[expert_id] - row_offset)
                        actual_cols = T.min(T.int32(block_n), T.int32(N) - n_start)
                        T.clear(C_local)

                        for k in T.Pipelined(T.ceildiv(K, block_k), num_stages=num_stages):
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


class MoeGroupedGemmPersistentKernel(Kernel):
    """Persistent NT grouped GEMM for MoE (Grid=SM_count, dynamic tile dispatch).

    Combines tile scheduling and GEMM into a single persistent kernel.
    Each CTA claims tiles via atomicAdd, eliminating dead CTAs and the
    separate tile-scheduler kernel overhead.

    Args:
        numel:       T * top_k tight row count.
        num_experts: Total number of experts E.
        N:           Output feature dimension.
        K:           Input feature dimension.
        dtype:       Activation and weight dtype.
        sm_count:    Number of SMs on the target GPU.
                     Use ``torch.cuda.get_device_properties(0).multi_processor_count``.
        config:      Optional config dict with block_m/n/k, num_stages, threads.
        tune:        Whether to autotune.

    Example:
        >>> sm = torch.cuda.get_device_properties(0).multi_processor_count
        >>> kernel = MoeGroupedGemmPersistentKernel(
        ...     numel=16384, num_experts=256, N=4096, K=2048,
        ...     dtype=torch.bfloat16, sm_count=sm)
        >>> C = kernel(A, B, true_sizes, true_offsets)  # [numel, N]
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        numel: int,
        num_experts: int,
        N: int,
        K: int,
        dtype: torch.dtype = torch.bfloat16,
        sm_count: int = 132,
        config=None,
        tune: bool = False,
    ):
        super().__init__()
        self.numel = numel
        self.num_experts = num_experts
        self.N = N
        self.K = K
        self.dtype = dtype
        self.sm_count = sm_count
        # tile_counter: persistent GMEM counter, reset to 0 before each kernel launch.
        self._tile_counter = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return dict(_DEFAULT_CONFIG)

    @property
    def autotune_configs(self) -> list[dict]:
        # Full search space matching nopad kernel; persistent overhead profile
        # may differ from nopad so we sweep block_n and threads as well.
        block_m = [32, 64, 128]
        block_n = [64, 128, 256]
        block_k = [32, 64, 128]
        num_stages = [1, 2, 3, 4, 5]
        threads = [128, 256]
        return [
            {"block_m": c[0], "block_n": c[1], "block_k": c[2],
             "num_stages": c[3], "threads": c[4], "group_size_m": 1}
            for c in itertools.product(block_m, block_n, block_k, num_stages, threads)
        ]

    def forward(
        self,
        A: torch.Tensor,             # [numel, K]
        B: torch.Tensor,             # [num_experts, N, K]
        true_sizes: torch.Tensor,    # [E] int32
        true_offsets: torch.Tensor,  # [E] int32
    ) -> torch.Tensor:
        """Run persistent GEMM: single kernel, Grid=SM_count, atomicAdd tile dispatch.

        Args:
            A: [numel, K] tight permuted hidden states.
            B: [num_experts, N, K] expert weight matrix (NT: B^T applied).
            true_sizes: [E] int32 true token count per expert.
            true_offsets: [E] int32 tight start offset per expert in A.

        Returns:
            C: [numel, N] GEMM output.
        """
        block_m = self.config["block_m"]
        block_k = self.config["block_k"]

        # TMA path: pad A with block_m zero rows to prevent OOB read on last expert's
        # last M-tile.  Middle-expert boundaries are safe because the C epilogue
        # predicate (i < actual_rows) prevents writing OOB data to C.
        if self.K % block_k == 0:
            A = torch.nn.functional.pad(A, (0, 0, 0, block_m))

        # Reset tile counter before kernel launch.
        # PyTorch's fill_() uses torch.cuda.current_stream() internally,
        # so both fill_ and gemm_fn launch on the same current stream by default.
        # WARNING: if the caller switches streams via torch.cuda.stream(s), both
        # this fill_ and the gemm_fn call below must be inside the same stream
        # context to avoid a race condition.
        self._tile_counter.fill_(0)

        gemm_fn = _persistent_moe_gemm_kernel(
            self.numel, self.num_experts, self.N, self.K,
            self.dtype_str, self.sm_count,
        )(
            block_m, self.config["block_n"], block_k,
            self.config["num_stages"], self.config["threads"],
            self.config.get("group_size_m", 1),
        )
        return gemm_fn(A, B, true_sizes, true_offsets, self._tile_counter)
