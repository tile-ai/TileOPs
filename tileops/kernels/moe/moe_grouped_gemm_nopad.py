"""MoE-specific no-padding grouped GEMM kernel (NT layout).

Zero CPU sync: total_tiles stays on GPU as a tensor and is read by each CTA
at runtime via an early-exit guard.  Grid = (max_tiles, ceildiv(N, block_n))
gives high CTA count for good warp occupancy and latency hiding.

Two-phase execution:
  Phase 1 - Tile scheduler (TileLang, single kernel):
    true_sizes[E] → tile_expert_ids[max_tiles], tile_row_offsets[max_tiles],
                    total_tiles[1]

    Internally uses two sub-phases:
      (a) Thread 0 runs a sequential prefix scan: cum_tiles[E+1] in SMEM,
          then writes total_tiles[0] = cum_tiles[E].
      (b) All threads in parallel run a binary search over cum_tiles[E+1].
          O(log E) per tile; static loop bound LOG2_UP (8 for E≤256).

  Phase 2 - NT grouped GEMM (max_tiles × ceildiv(N, block_n) blocks):
    A[numel, K] × B[E, N, K]^T → C[numel, N]

    Grid = (max_tiles, ceildiv(N, block_n)).  Each CTA reads total_tiles[0]
    from GPU memory and exits immediately if bx >= total_tiles[0] (dead CTA).
    Valid CTAs do exactly one M-tile.  Zero CPU sync; high CTA count for
    good warp occupancy and memory-latency hiding.

Parameters (all Python ints, baked in at compile time via lru_cache):
  numel         = T * top_k  (tight row count, no padding)
  max_tiles     = numel // block_m + num_experts  (conservative upper bound)
  num_experts   = E
  N, K          = output/input feature dimensions
  dtype         = activation dtype string

Performance (E=256, T=4096, block_m=64, H200):
  old persistent (132×8=1056 CTA): SM util 42%, warp active 12%
  new (768×8=6144 CTA, ~33% dead): larger grid → better warp occupancy
"""

import functools
import itertools
import math

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["MoeGroupedGemmNopadKernel"]

_DEFAULT_CONFIG = {"block_m": 64, "block_n": 256, "block_k": 64, "num_stages": 2, "threads": 128,
                   "group_size_m": 1}
_SCHED_THREADS = 256  # threads per block in the tile scheduler kernel


@functools.lru_cache(maxsize=32)
def _tile_scheduler_kernel(num_experts: int, max_tiles: int, block_m: int):
    """TileLang tile scheduler: true_sizes[E] → tile_expert_ids, tile_row_offsets, total_tiles.

    Single kernel, two sub-phases per block:
      (a) Thread 0: sequential prefix scan → cum_tiles[E+1] in SMEM,
          writes total_tiles[0] = cum_tiles[E].
      (b) All threads in parallel: O(log E) binary search per tile.

    Parameters baked in at compile time via lru_cache.

    Args:
        num_experts: E (total experts).
        max_tiles:   Conservative tile upper bound (numel // block_m + E).
        block_m:     Tile height in M dimension (compile-time constant).
    """
    # Static binary search depth: ceil(log2(E+1)) iterations suffice for E experts.
    log2_up = max(1, math.ceil(math.log2(num_experts + 1)))

    @tilelang.jit(out_idx=[1, 2, 3])
    def _func(threads: int):
        @T.prim_func
        def _sched_main(
            true_sizes: T.Tensor([num_experts], "int32"),         # noqa: F821
            tile_expert_ids: T.Tensor([max_tiles], "int32"),      # noqa: F821
            tile_row_offsets: T.Tensor([max_tiles], "int32"),     # noqa: F821
            total_tiles: T.Tensor([1], "int32"),                  # noqa: F821
        ):
            with T.Kernel(T.ceildiv(max_tiles, threads), threads=threads) as (bx,):
                s_cum = T.alloc_shared([num_experts + 1], "int32")
                lo = T.alloc_local([1], "int32")
                hi = T.alloc_local([1], "int32")

                tx = T.get_thread_binding()

                # Sub-phase (a): thread 0 computes exclusive prefix sum of
                # ceildiv(true_sizes[e], block_m) into SMEM.
                if tx == 0:
                    s_cum[0] = T.int32(0)
                    for e in T.serial(num_experts):
                        s_cum[e + 1] = s_cum[e] + (true_sizes[e] + (block_m - 1)) // block_m
                    if bx == 0:
                        total_tiles[0] = s_cum[num_experts]
                T.sync_threads()

                # Sub-phase (b): each thread assigns one tile via binary search.
                tile_id = bx * threads + tx
                if tile_id < max_tiles:
                    if tile_id < s_cum[num_experts]:
                        lo[0] = T.int32(0)
                        hi[0] = T.int32(num_experts - 1)
                        for _bs in T.serial(log2_up):
                            mid = (lo[0] + hi[0]) >> T.int32(1)
                            if s_cum[mid + 1] <= tile_id:
                                lo[0] = mid + T.int32(1)
                            else:
                                hi[0] = mid
                        tile_expert_ids[tile_id] = lo[0]
                        tile_row_offsets[tile_id] = (tile_id - s_cum[lo[0]]) * T.int32(block_m)
                    else:
                        tile_expert_ids[tile_id] = T.int32(-1)
                        tile_row_offsets[tile_id] = T.int32(0)

        return _sched_main

    return _func


@functools.lru_cache(maxsize=64)
def _moe_grouped_gemm_kernel(numel: int, max_tiles: int, num_experts: int,
                              N: int, K: int, dtype: str = "float16"):
    """NT grouped GEMM with GPU-side total_tiles early-exit guard (zero CPU sync).

    Grid = max_tiles * ceildiv(N, block_n) (1-D with GROUP_SIZE_M reordering).
    Each CTA maps its flat pid → (bx, by) via GROUP_SIZE_M tile ordering to
    improve B-weight L2 cache reuse.  Dead CTAs (bx >= total_tiles[0]) exit
    after one GPU-memory read.

    GROUP_SIZE_M (compile-time constant, baked into inner _func):
      Tiles are grouped so that group_size_m M-tiles process all N-columns
      before the next group starts.  Working set per group ≈ group_size_m/2
      experts × full B matrix, which fits in L2 and avoids the cross-wave
      eviction that a plain 2D grid (all M-tiles × one N-column per wave)
      would cause.

    A[numel, K] × B[E, N, K]^T → C[numel, N]

    Args:
        numel:       T * top_k tight row count.
        max_tiles:   Conservative tile upper bound (numel // block_m + E).
        num_experts: E.
        N, K:        Output / input feature dims.
        dtype:       Activation dtype string.
    """
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[2],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(block_m, block_n, block_k, num_stages, threads, group_size_m):
        A_shape = (numel, K)
        B_shape = (num_experts, N, K)
        C_shape = (numel, N)
        A_shared_shape = (block_m, block_k)
        B_shared_shape = (block_n, block_k)

        # Compile-time constants (all Python ints, baked in at JIT time).
        _k_aligned = (K % block_k == 0)
        _n_aligned = (N % block_n == 0)
        _b_copy_ok = _k_aligned and _n_aligned
        _num_pid_n = math.ceil(N / block_n)           # N-tile count
        _total_ctas = max_tiles * _num_pid_n          # 1-D grid size
        _num_pid_in_group = group_size_m * _num_pid_n  # CTAs per GROUP_SIZE_M group

        if _b_copy_ok:
            @T.prim_func
            def _gemm_main(
                A: T.Tensor(A_shape, dtype),                           # type: ignore
                B: T.Tensor(B_shape, dtype),                           # type: ignore
                C: T.Tensor(C_shape, dtype),                           # type: ignore
                tile_expert_ids: T.Tensor([max_tiles], "int32"),       # noqa: F821
                tile_row_offsets: T.Tensor([max_tiles], "int32"),      # noqa: F821
                true_offsets: T.Tensor([num_experts], "int32"),        # noqa: F821
                true_sizes: T.Tensor([num_experts], "int32"),          # noqa: F821
                total_tiles: T.Tensor([1], "int32"),                   # noqa: F821
            ):
                with T.Kernel(_total_ctas, threads=threads) as (pid,):
                    A_shared = T.alloc_shared(A_shared_shape, dtype)
                    B_shared = T.alloc_shared(B_shared_shape, dtype)
                    C_local = T.alloc_fragment([block_m, block_n], accum_dtype)

                    # M-major tile ordering: each M-tile processes all N-tiles
                    # before moving to the next M-tile, maximising A-tile reuse.
                    # For group_size_m=1 (default) this simplifies to a direct
                    # pid → (bx, by) mapping with compile-time constant divisor.
                    if group_size_m == 1:
                        bx = pid // T.int32(_num_pid_n)
                        by = pid % T.int32(_num_pid_n)
                    else:
                        pid_in_group = pid % T.int32(_num_pid_in_group)
                        group_id    = pid // T.int32(_num_pid_in_group)
                        first_pid_m = group_id * T.int32(group_size_m)
                        actual_gsm  = T.min(T.int32(max_tiles) - first_pid_m,
                                            T.int32(group_size_m))
                        bx = first_pid_m + pid_in_group % actual_gsm
                        by = pid_in_group // actual_gsm

                    if bx < total_tiles[0]:
                        expert_id    = tile_expert_ids[bx]
                        row_in_expert = tile_row_offsets[bx]
                        m_start      = true_offsets[expert_id] + row_in_expert
                        n_start      = by * T.int32(block_n)
                        actual_rows  = T.min(T.int32(block_m),
                                             true_sizes[expert_id] - row_in_expert)
                        T.clear(C_local)

                        for k in T.Pipelined(T.ceildiv(K, block_k), num_stages=num_stages):
                            for i, j in T.Parallel(block_m, block_k):
                                A_shared[i, j] = T.if_then_else(
                                    i < actual_rows,
                                    A[m_start + i, k * block_k + j], 0)
                            for i, j in T.Parallel(block_n, block_k):
                                B_shared[i, j] = B[expert_id, n_start + i,
                                                    k * block_k + j]
                            T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                        for i, j in T.Parallel(block_m, block_n):
                            if i < actual_rows:
                                C[m_start + i, n_start + j] = C_local[i, j]
        else:
            @T.prim_func
            def _gemm_main(                                            # noqa: F811
                A: T.Tensor(A_shape, dtype),                           # type: ignore
                B: T.Tensor(B_shape, dtype),                           # type: ignore
                C: T.Tensor(C_shape, dtype),                           # type: ignore
                tile_expert_ids: T.Tensor([max_tiles], "int32"),       # noqa: F821
                tile_row_offsets: T.Tensor([max_tiles], "int32"),      # noqa: F821
                true_offsets: T.Tensor([num_experts], "int32"),        # noqa: F821
                true_sizes: T.Tensor([num_experts], "int32"),          # noqa: F821
                total_tiles: T.Tensor([1], "int32"),                   # noqa: F821
            ):
                with T.Kernel(_total_ctas, threads=threads) as (pid,):
                    A_shared = T.alloc_shared(A_shared_shape, dtype)
                    B_shared = T.alloc_shared(B_shared_shape, dtype)
                    C_local = T.alloc_fragment([block_m, block_n], accum_dtype)

                    if group_size_m == 1:
                        bx = pid // T.int32(_num_pid_n)
                        by = pid % T.int32(_num_pid_n)
                    else:
                        pid_in_group = pid % T.int32(_num_pid_in_group)
                        group_id    = pid // T.int32(_num_pid_in_group)
                        first_pid_m = group_id * T.int32(group_size_m)
                        actual_gsm  = T.min(T.int32(max_tiles) - first_pid_m,
                                            T.int32(group_size_m))
                        bx = first_pid_m + pid_in_group % actual_gsm
                        by = pid_in_group // actual_gsm

                    if bx < total_tiles[0]:
                        expert_id    = tile_expert_ids[bx]
                        row_in_expert = tile_row_offsets[bx]
                        m_start      = true_offsets[expert_id] + row_in_expert
                        n_start      = by * T.int32(block_n)
                        actual_rows  = T.min(T.int32(block_m),
                                             true_sizes[expert_id] - row_in_expert)
                        actual_cols  = T.min(T.int32(block_n), T.int32(N) - n_start)
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


class MoeGroupedGemmNopadKernel(Kernel):
    """NT grouped GEMM for MoE without block_m-aligned padding.

    Two-phase execution: GPU tile scheduler → GEMM with dead-CTA early-exit.
    Grid = (max_tiles, ceildiv(N, block_n)); valid CTAs process one M-tile
    each; dead CTAs (bx >= total_tiles[0]) exit after one GPU-memory read.
    total_tiles is never copied to CPU — zero CPU sync.

    Args:
        numel: T * top_k, tight row count.
        num_experts: Total number of experts E.
        N: Output feature dimension (e.g. 2*ffn_size for gate+up).
        K: Input feature dimension (hidden_size).
        dtype: Activation and weight dtype.
        config: Optional config dict with block_m/n/k, num_stages, threads.
        tune: Whether to autotune.

    Example:
        >>> kernel = MoeGroupedGemmNopadKernel(numel=16384, num_experts=256, N=4096, K=2048,
        ...                               dtype=torch.bfloat16)
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
        config=None,
        tune: bool = False,
    ):
        super().__init__()
        self.numel = numel
        self.num_experts = num_experts
        self.N = N
        self.K = K
        self.dtype = dtype
        self.init_config(config, tune)

    def _max_tiles(self, block_m: int) -> int:
        return self.numel // block_m + self.num_experts

    def _max_tiles_per_expert(self, block_m: int) -> int:
        return (self.numel + block_m - 1) // block_m

    @property
    def default_config(self) -> dict:
        return dict(_DEFAULT_CONFIG)

    @property
    def autotune_configs(self) -> list[dict]:
        block_m = [32, 64, 128]
        block_n = [64, 128, 256]
        block_k = [32, 64, 128]
        num_stages = [1, 2, 3]
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
        """Run tile scheduler + GEMM with GPU-side dead-CTA early-exit.

        Phase 1: TileLang scheduler builds tile_expert_ids, tile_row_offsets,
                 and total_tiles[1] — all on GPU, no .item() needed.
        Phase 2: GEMM with grid = (max_tiles, N_tiles).
                 Each CTA reads total_tiles[0] from GPU; dead CTAs (bx >=
                 total_tiles[0]) exit immediately.  lru_cache key uses
                 (max_tiles, num_experts) — static Python ints, always reused.

        Args:
            A: [numel, K] tight permuted hidden states.
            B: [num_experts, N, K] expert weight matrix (NT: B^T applied).
            true_sizes: [E] int32 true token count per expert.
            true_offsets: [E] int32 tight start offset per expert in A.

        Returns:
            C: [numel, N] GEMM output.
        """
        block_m = self.config["block_m"]
        max_tiles = self._max_tiles(block_m)

        # Phase 1: build tile schedule — total_tiles stays on GPU (no .item())
        sched_fn = _tile_scheduler_kernel(self.num_experts, max_tiles, block_m)(_SCHED_THREADS)
        tile_expert_ids, tile_row_offsets, total_tiles_t = sched_fn(true_sizes)

        # Phase 2: GEMM with dead-CTA early-exit — 1D grid with GROUP_SIZE_M reordering
        gemm_fn = _moe_grouped_gemm_kernel(
            self.numel, max_tiles, self.num_experts, self.N, self.K, self.dtype_str
        )(
            block_m, self.config["block_n"], self.config["block_k"],
            self.config["num_stages"], self.config["threads"],
            self.config.get("group_size_m", 1),
        )
        return gemm_fn(A, B, tile_expert_ids, tile_row_offsets, true_offsets, true_sizes,
                       total_tiles_t)
