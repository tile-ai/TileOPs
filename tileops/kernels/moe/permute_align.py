"""MoE token-to-expert alignment kernel.

Converts topk_ids [total_tokens, top_k] into the three arrays needed by
MoE grouped GEMM:

  sorted_token_ids  [max_num_tokens_padded]  - flat token indices sorted by
                                               expert, padded with sentinel
  expert_ids        [num_blocks]             - expert index per GEMM block
  num_tokens_post_pad [1]                   - total padded token count

Algorithm (K1 align kernel + K2 multi-block scatter):

  Step 1 - count:
    * Zero s_counts in shared memory.
    * Count tokens per expert via shared-memory atomicAdd.

  Step 2 - warp-scan prefix-sum:
    * Warp-level inclusive scan (tvm_warp_shuffle_up) of padded counts.
    * Inter-warp exclusive scan (for->if pattern).
    * Writes cumsum[] to both shared s_cumsum and global output.
    * Writes num_tokens_post_pad.

  Step 3 - fill expert_ids (linear, no binary search):
    * Thread tx < num_experts owns blocks [cumsum[tx]/bs, cumsum[tx+1]/bs).
    * Writes expert_ids[blk] = tx directly — O(total_blocks) total work.

  Step 4 - sentinel fill (K1 only):
    * Fill sorted_token_ids with sentinel (numel).

  Step 5 - scatter (K2):
    * Multi-block scatter using global atomicAdd on cumsum buffer.

Reference: sgl-kernel/csrc/moe/moe_align_kernel.cu
"""

import math
import os
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

# Path to the CUDA helper header for the scatter kernel.
# Provides tl_atomic_add_offset() — a workaround for TileLang's codegen
# limitation that T.atomic_add(..., return_prev=True) does not support
# dynamic (indirect) global-memory indices.
_ATOMIC_HELPER_H = os.path.join(os.path.dirname(__file__), "_atomic_helper.h")

__all__ = ["MoePermuteAlignKernel"]

_THREADS = 1024
_SCATTER_THREADS = 256
_SMALL_NUMEL_THRESHOLD   = 1024
_SMALL_EXPERTS_THRESHOLD = 64
_FILL_THREADS            = 256


def _make_align_kernel(numel: int, num_experts: int, block_size: int):
    """K1: count → warp-scan → expert_ids → sentinel fill. Does NOT scatter."""
    max_padded = numel + (num_experts + 1) * (block_size - 1)
    max_num_blocks = math.ceil(max_padded / block_size)

    @tilelang.jit(out_idx=[], compile_flags=["-O3"])
    def _align(threads: int):

        @T.prim_func
        def _align_main(
            flat: T.Tensor([numel], "int32"),                    # noqa: F821
            sorted_token_ids: T.Tensor([max_padded], "int32"),   # noqa: F821
            expert_ids: T.Tensor([max_num_blocks], "int32"),     # noqa: F821
            num_tokens_post_pad: T.Tensor([1], "int32"),         # noqa: F821
            cumsum: T.Tensor([num_experts + 1], "int32"),        # noqa: F821
        ):
            with T.Kernel(1, threads=threads) as (_,):
                tx = T.get_thread_binding()
                num_warps = threads // 32

                # --- Shared memory ---
                s_counts    = T.alloc_shared([num_experts], "int32")
                s_vals      = T.alloc_shared([threads], "int32")
                s_warp_sum  = T.alloc_shared([num_warps], "int32")
                s_warp_excl = T.alloc_shared([num_warps], "int32")
                # s_total is a running accumulator used only by tx==0 to build
                # s_warp_excl during the inter-warp scan; its final value (grand
                # total) is derived independently at line 135 and not re-read.
                s_total     = T.alloc_shared([1], "int32")
                s_cumsum    = T.alloc_shared([num_experts + 1], "int32")

                # Step 1: zero s_counts
                for i in T.serial(T.ceildiv(num_experts, threads)):
                    idx = i * threads + tx
                    if idx < num_experts:
                        s_counts[idx] = T.int32(0)
                T.sync_threads()

                # Step 1: count tokens per expert
                for i in T.serial(T.ceildiv(numel, threads)):
                    idx = i * threads + tx
                    if idx < numel:
                        T.atomic_add(s_counts[flat[idx]], 1)
                T.sync_threads()

                # Step 2: warp-scan prefix-sum on padded counts
                lane    = tx % 32
                warp_id = tx // 32

                s_vals[tx] = (
                    T.ceildiv(s_counts[tx], block_size) * block_size
                    if tx < num_experts else T.int32(0)
                )
                T.sync_threads()

                # Intra-warp inclusive scan via shuffle_up
                # 5 rounds = log2(warp_size=32): each round doubles the scan distance
                for d in T.serial(5):
                    stride = 1 << d
                    up_val = T.tvm_warp_shuffle_up(
                        T.uint32(0xFFFFFFFF), s_vals[tx], stride, 32, 32
                    )
                    if lane >= stride:
                        s_vals[tx] = s_vals[tx] + up_val
                T.sync_threads()

                # Last lane of each warp records warp sum
                if lane == 31:
                    s_warp_sum[warp_id] = s_vals[tx]
                T.sync_threads()

                # Inter-warp exclusive scan (for->if pattern to write shared)
                for w in T.serial(num_warps):
                    if tx == 0:
                        if w == 0:
                            s_total[0] = T.int32(0)
                        s_warp_excl[w] = s_total[0]
                        s_total[0] = s_total[0] + s_warp_sum[w]
                T.sync_threads()

                # Convert inclusive scan -> exclusive cumsum entry
                own_padded = (
                    T.ceildiv(s_counts[tx], block_size) * block_size
                    if tx < num_experts else T.int32(0)
                )
                excl = s_vals[tx] - own_padded + s_warp_excl[warp_id]

                if tx < num_experts:
                    s_cumsum[tx] = excl
                    cumsum[tx] = excl
                if tx == num_experts - 1:
                    total = s_vals[tx] + s_warp_excl[warp_id]
                    s_cumsum[num_experts] = total
                    cumsum[num_experts] = total
                    num_tokens_post_pad[0] = total
                T.sync_threads()

                # Step 3: fill expert_ids linearly — no binary search.
                # Each thread tx < num_experts owns blocks [e_start, e_end).
                # Loop bound is max_num_blocks (worst case: all tokens to one
                # expert), guarded by `blk < e_end` to skip non-owned blocks.
                if tx < num_experts:
                    e_start = s_cumsum[tx]     // block_size
                    e_end   = s_cumsum[tx + 1] // block_size
                    for b in T.serial(max_num_blocks):
                        blk = e_start + b
                        if blk < e_end:
                            expert_ids[blk] = tx
                T.sync_threads()

                # Step 4: fill sentinel
                for i in T.serial(T.ceildiv(max_padded, threads)):
                    idx = i * threads + tx
                    if idx < max_padded:
                        sorted_token_ids[idx] = numel

        return _align_main

    return _align


def _make_scatter_kernel(numel: int, num_experts: int, block_size: int):
    """K2: multi-block scatter using global atomicAdd on cumsum buffer.

    cumsum[e] holds the current write offset for expert e (set by K1, then
    incremented here). CUDA stream order guarantees K1 finishes before K2.

    Reference: count_and_sort_expert_tokens_kernel in sgl-kernel.
    """
    max_padded = numel + (num_experts + 1) * (block_size - 1)
    scatter_blocks = min(math.ceil(numel / _SCATTER_THREADS), 65535)
    total_scatter_threads = scatter_blocks * _SCATTER_THREADS

    @tilelang.jit(out_idx=[], compile_flags=["-O3", "-include", _ATOMIC_HELPER_H])
    def _scatter():

        @T.prim_func
        def _scatter_main(
            flat: T.Tensor([numel], "int32"),                   # noqa: F821
            sorted_token_ids: T.Tensor([max_padded], "int32"),  # noqa: F821
            cumsum: T.Tensor([num_experts + 1], "int32"),       # noqa: F821
        ):
            with T.Kernel(scatter_blocks, threads=_SCATTER_THREADS) as (bid,):
                tx = T.get_thread_binding()
                gid = bid * _SCATTER_THREADS + tx
                for i in T.serial(T.ceildiv(numel, total_scatter_threads)):
                    idx = gid + i * total_scatter_threads
                    if idx < numel:
                        eid = flat[idx]
                        # T.atomic_add(..., return_prev=True) does not support dynamic
                        # global-memory indices (TileLang codegen limitation).
                        # Use tl_atomic_add_offset() from _atomic_helper.h instead.
                        slot = T.call_extern("int32", "tl_atomic_add_offset",
                                             T.address_of(cumsum[0]), eid, T.int32(1))
                        sorted_token_ids[slot] = idx

        return _scatter_main

    return _scatter


def _make_small_batch_kernel(numel: int, num_experts: int, block_size: int):
    """Single fused kernel for small batches (numel < 1024, num_experts <= 64).

    Uses per-worker private count rows to eliminate atomic conflicts in count
    and scatter phases. Concurrent sentinel fill (fill_threads) + compute
    (worker_threads) within the same block.

    Uses 0-indexed expert IDs throughout (unlike sgl-kernel which uses 1-indexed).
    Output format matches the large-batch path (K1+K2).

    Reference: moe_align_block_size_small_batch_expert_kernel in sgl-kernel.
    """
    max_padded     = numel + (num_experts + 1) * (block_size - 1)
    max_num_blocks = math.ceil(max_padded / block_size)
    worker_threads = max(num_experts, 32)
    total_threads  = _FILL_THREADS + worker_threads
    # tokens_cnts layout: (worker_threads+1) rows × num_experts cols (0-indexed).
    # Row k (k>=1): worker k-1's private counts per expert (0-indexed).
    # Row 0: reduce accumulator, initialised to 0 by expert threads.
    cnts_size = (worker_threads + 1) * num_experts

    @tilelang.jit(out_idx=[], compile_flags=["-O3"])
    def _small():

        @T.prim_func
        def _small_main(
            flat: T.Tensor([numel], "int32"),                    # noqa: F821
            sorted_token_ids: T.Tensor([max_padded], "int32"),   # noqa: F821
            expert_ids: T.Tensor([max_num_blocks], "int32"),     # noqa: F821
            num_tokens_post_pad: T.Tensor([1], "int32"),         # noqa: F821
        ):
            with T.Kernel(1, threads=total_threads) as (_,):
                tx = T.get_thread_binding()

                cumsum_s    = T.alloc_shared([num_experts + 1], "int32")
                tokens_cnts = T.alloc_shared([cnts_size], "int32")

                # fill_threads group: fill sentinel (concurrent with worker Phase 0)
                if tx < _FILL_THREADS:
                    for i in T.serial(T.ceildiv(max_padded, _FILL_THREADS)):
                        idx = i * _FILL_THREADS + tx
                        if idx < max_padded:
                            sorted_token_ids[idx] = numel

                # worker_threads group: Phase 0 — init private count row + count tokens
                if tx >= _FILL_THREADS:
                    wid = tx - _FILL_THREADS  # 0-indexed worker id
                    # Init row wid+1 (private row for this worker)
                    for i in T.serial(num_experts):
                        tokens_cnts[(wid + 1) * num_experts + i] = T.int32(0)
                    # Count tokens owned by this worker (grid-stride over numel)
                    for i in T.serial(T.ceildiv(numel, worker_threads)):
                        idx = wid + i * worker_threads
                        if idx < numel:
                            eid = flat[idx]  # 0-indexed
                            tokens_cnts[(wid + 1) * num_experts + eid] = (
                                tokens_cnts[(wid + 1) * num_experts + eid] + 1
                            )

                T.sync_threads()  # sync 1

                # worker_threads group: Phase 1 — column-wise inclusive prefix-sum reduce
                # Thread wid handles column wid (expert wid's counts across all workers)
                if tx >= _FILL_THREADS:
                    wid = tx - _FILL_THREADS
                    if wid < num_experts:
                        tokens_cnts[wid] = T.int32(0)  # init row 0 accumulator
                        for k in T.serial(worker_threads):
                            tokens_cnts[(k + 1) * num_experts + wid] = (
                                tokens_cnts[(k + 1) * num_experts + wid]
                                + tokens_cnts[k * num_experts + wid]
                            )
                        # After loop: tokens_cnts[worker_threads*E + wid] = total for expert wid

                T.sync_threads()  # sync 2

                # worker_threads group: Phase 2 — wid==0 builds exclusive prefix-sum cumsum
                if tx >= _FILL_THREADS:
                    wid = tx - _FILL_THREADS
                    if wid == 0:
                        cumsum_s[0] = T.int32(0)
                        for e in T.serial(num_experts):
                            cnt = tokens_cnts[worker_threads * num_experts + e]
                            cumsum_s[e + 1] = (
                                cumsum_s[e]
                                + T.ceildiv(cnt, block_size) * block_size
                            )
                        num_tokens_post_pad[0] = cumsum_s[num_experts]

                T.sync_threads()  # sync 3

                # worker_threads group: Phase 3a — expert_ids fill (0-indexed)
                if tx >= _FILL_THREADS:
                    wid = tx - _FILL_THREADS
                    if wid < num_experts:
                        e_start = cumsum_s[wid]     // block_size
                        e_end   = cumsum_s[wid + 1] // block_size
                        for b in T.serial(max_num_blocks):
                            blk = e_start + b
                            if blk < e_end:
                                expert_ids[blk] = wid  # 0-indexed expert id

                    # Phase 3b: scatter (no atomics — per-worker private rows)
                    # tokens_cnts[wid*E + eid] holds the running offset for worker wid
                    # into expert eid's slot range (starts at cumsum_s[eid])
                    for i in T.serial(T.ceildiv(numel, worker_threads)):
                        idx = wid + i * worker_threads
                        if idx < numel:
                            eid = flat[idx]  # 0-indexed
                            rank = tokens_cnts[wid * num_experts + eid]
                            slot = rank + cumsum_s[eid]
                            sorted_token_ids[slot] = idx
                            tokens_cnts[wid * num_experts + eid] = rank + 1

        return _small_main

    return _small


class MoePermuteAlignKernel(Kernel):
    """MoE token permutation and alignment kernel.

    Converts ``topk_ids`` into the three index arrays required by MoE grouped GEMM.

    Args:
        numel: Total number of (token, expert) assignments = total_tokens * top_k.
        num_experts: Number of experts.
        block_size: GEMM tile size (M dimension).
        config: Optional config dict (unused; kept for API consistency).

    Example:
        >>> kernel = MoePermuteAlignKernel(numel=32, num_experts=8, block_size=16)
        >>> sorted_ids, expert_ids, num_post_pad = kernel(topk_ids)
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        numel: int,
        num_experts: int,
        block_size: int,
        config: Optional[dict] = None,
    ):
        super().__init__()
        self.numel = numel
        self.num_experts = num_experts
        self.block_size = block_size

        self._align_fn = _make_align_kernel(numel, num_experts, block_size)
        self._scatter_fn = _make_scatter_kernel(numel, num_experts, block_size)
        self._small_batch_fn = (
            _make_small_batch_kernel(numel, num_experts, block_size)
            if numel < _SMALL_NUMEL_THRESHOLD and num_experts <= _SMALL_EXPERTS_THRESHOLD
            else None
        )

        self.init_config(config, tune=False)

    @property
    def default_config(self) -> dict:
        return {"threads": _THREADS}

    def forward(self, topk_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the alignment kernel.

        Args:
            topk_ids: [total_tokens, top_k] int32 tensor of expert indices (0-indexed).

        Returns:
            sorted_token_ids: [max_num_tokens_padded] int32
            expert_ids:       [num_blocks] int32
            num_tokens_post_pad: [1] int32
        """
        assert topk_ids.dtype == torch.int32, "topk_ids must be int32"
        assert topk_ids.is_cuda, "topk_ids must be on CUDA"
        assert topk_ids.numel() == self.numel, (
            f"Expected numel={self.numel}, got {topk_ids.numel()}"
        )

        flat = topk_ids.flatten().contiguous()
        max_padded      = self.numel + (self.num_experts + 1) * (self.block_size - 1)
        max_num_blocks  = math.ceil(max_padded / self.block_size)
        dev             = flat.device

        sorted_token_ids    = torch.empty(max_padded,     dtype=torch.int32, device=dev)
        expert_ids          = torch.empty(max_num_blocks, dtype=torch.int32, device=dev)
        num_tokens_post_pad = torch.empty(1,              dtype=torch.int32, device=dev)

        if self._small_batch_fn is not None:
            fn = self._small_batch_fn()
            fn(flat, sorted_token_ids, expert_ids, num_tokens_post_pad)
        else:
            cumsum  = torch.empty(self.num_experts + 1, dtype=torch.int32, device=dev)
            threads = self.config["threads"]
            align_fn = self._align_fn(threads)
            align_fn(flat, sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum)
            scatter_fn = self._scatter_fn()
            scatter_fn(flat, sorted_token_ids, cumsum)

        return sorted_token_ids, expert_ids, num_tokens_post_pad
