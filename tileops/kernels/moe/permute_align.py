"""MoE token-to-expert alignment kernel.

Converts topk_ids [total_tokens, top_k] into the three arrays needed by
MoE grouped GEMM:

  sorted_token_ids  [max_num_tokens_padded]  - flat token indices sorted by
                                               expert, padded with sentinel
  expert_ids        [num_blocks]             - expert index per GEMM block
  num_tokens_post_pad [1]                   - total padded token count

Algorithm (single fused kernel):

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

  Step 4 - sentinel fill + scatter sort:
    * Fill sorted_token_ids with sentinel (numel).
    * Reuse s_cumsum as s_offset; each thread atomically claims its slot.

Reference: sgl-kernel/csrc/moe/moe_align_kernel.cu
"""

import math
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["MoePermuteAlignKernel"]

_THREADS = 1024


def _make_fused_kernel(numel: int, num_experts: int, block_size: int):
    """Single fused kernel: count → warp-scan → expert_ids → sentinel fill → sort."""
    max_padded = numel + (num_experts + 1) * (block_size - 1)
    max_num_blocks = math.ceil(max_padded / block_size)

    @tilelang.jit(out_idx=[], compile_flags=["-O3"])
    def _fused(threads: int):

        @T.prim_func
        def _fused_main(
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
                T.sync_threads()

                # Step 4: scatter token indices (reuse s_cumsum as s_offset)
                for i in T.serial(T.ceildiv(numel, threads)):
                    idx = i * threads + tx
                    if idx < numel:
                        eid  = flat[idx]
                        slot = T.atomic_add(s_cumsum[eid], 1, return_prev=True)
                        sorted_token_ids[slot] = idx

        return _fused_main

    return _fused


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

        self._fused_fn = _make_fused_kernel(numel, num_experts, block_size)

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
        threads = self.config["threads"]

        max_padded = self.numel + (self.num_experts + 1) * (self.block_size - 1)
        max_num_blocks = math.ceil(max_padded / self.block_size)

        sorted_token_ids    = torch.empty(max_padded, dtype=torch.int32, device=flat.device)
        expert_ids          = torch.empty(max_num_blocks, dtype=torch.int32, device=flat.device)
        num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device=flat.device)
        cumsum              = torch.empty(self.num_experts + 1, dtype=torch.int32,
                                         device=flat.device)

        fused_fn = self._fused_fn(threads)
        fused_fn(flat, sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum)

        return sorted_token_ids, expert_ids, num_tokens_post_pad
