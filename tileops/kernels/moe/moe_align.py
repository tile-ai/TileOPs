"""MoE token-to-expert alignment kernel.

Converts topk_ids [total_tokens, top_k] into the three arrays needed by
MoE grouped GEMM:

  sorted_token_ids  [max_num_tokens_padded]  - flat token indices sorted by
                                               expert, padded with sentinel
  expert_ids        [num_blocks]             - expert index per GEMM block
  num_tokens_post_pad [1]                   - total padded token count

Algorithm:

  TileLang Kernel 1 - count:
    * Count tokens per expert via global-memory atomicAdd.

  TileLang Kernel 2 - align:
    * Thread 0 computes exclusive prefix-sum of padded counts.
    * Writes expert_ids (binary search) and num_tokens_post_pad.
    * Copies padded prefix-sum into cumsum[] for the sort step.
    * Fills sorted_token_ids with sentinel (numel).

  PyTorch sort step:
    * torch.argsort(topk_ids, stable=True) gives expert-sorted token order.
    * Computes each token's slot = cumsum[eid] + local_rank_within_expert.
    * Scatters token indices into sorted_token_ids.
    (TileLang does not support dynamic-index atomic_add with return_prev on
    global memory, so this step is implemented in PyTorch.)

Reference: sgl-kernel/csrc/moe/moe_align_kernel.cu
"""

import math
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["moe_align_kernel"]

_THREADS = 1024


def _make_count_kernel(numel: int, num_experts: int):
    """Kernel 1: count tokens per expert into counts[] via atomicAdd."""

    @tilelang.jit(out_idx=[], compile_flags=["-O3"])
    def _count(threads: int):

        @T.prim_func
        def _count_main(
            topk_ids: T.Tensor([numel], "int32"),  # noqa: F821
            counts: T.Tensor([num_experts], "int32"),  # noqa: F821
        ):
            with T.Kernel(T.ceildiv(numel, threads), threads=threads) as (bx,):
                tx = T.get_thread_binding()
                idx = bx * threads + tx
                if idx < numel:
                    T.atomic_add(counts[topk_ids[idx]], 1)

        return _count_main

    return _count


def _make_align_kernel(numel: int, num_experts: int, block_size: int):
    """Kernel 2: prefix-sum + expert_ids + sentinel fill + write cumsum."""
    max_padded = numel + (num_experts + 1) * (block_size - 1)
    max_num_blocks = math.ceil(max_padded / block_size)

    @tilelang.jit(out_idx=[], compile_flags=["-O3"])
    def _align(threads: int):

        @T.prim_func
        def _align_main(
            counts: T.Tensor([num_experts], "int32"),  # noqa: F821
            sorted_token_ids: T.Tensor([max_padded], "int32"),  # noqa: F821
            expert_ids: T.Tensor([max_num_blocks], "int32"),  # noqa: F821
            num_tokens_post_pad: T.Tensor([1], "int32"),  # noqa: F821
            cumsum: T.Tensor([num_experts + 1], "int32"),  # noqa: F821
        ):
            with T.Kernel(1, threads=threads) as (_,):
                tx = T.get_thread_binding()

                s_prefix = T.alloc_shared([num_experts + 1], "int32")

                # Thread 0: serial prefix-sum of padded counts
                if tx == 0:
                    s_prefix[0] = 0
                    for e in T.serial(num_experts):
                        padded = T.ceildiv(counts[e], block_size) * block_size
                        s_prefix[e + 1] = s_prefix[e] + padded
                    num_tokens_post_pad[0] = s_prefix[num_experts]

                T.sync_threads()

                # Write padded cumsum to global (used by PyTorch sort step)
                for i in T.serial(T.ceildiv(num_experts + 1, threads)):
                    idx = i * threads + tx
                    if idx <= num_experts:
                        cumsum[idx] = s_prefix[idx]

                # Fill expert_ids via binary search
                total_padded = s_prefix[num_experts]
                num_blocks = T.ceildiv(total_padded, block_size)
                for i in T.serial(T.ceildiv(max_num_blocks, threads)):
                    blk = i * threads + tx
                    if blk < num_blocks:
                        block_start = blk * block_size
                        lo = T.alloc_var("int32")
                        hi = T.alloc_var("int32")
                        result = T.alloc_var("int32")
                        lo = 0
                        hi = num_experts - 1
                        result = num_experts - 1
                        for _ in T.serial(num_experts):
                            mid = (lo + hi) // 2
                            if s_prefix[mid] <= block_start and block_start < s_prefix[mid + 1]:
                                result = mid
                                lo = hi + 1  # break
                            elif block_start < s_prefix[mid]:
                                hi = mid - 1
                            else:
                                lo = mid + 1
                        expert_ids[blk] = result

                # Fill sorted_token_ids with sentinel (numel)
                for i in T.serial(T.ceildiv(max_padded, threads)):
                    idx = i * threads + tx
                    if idx < max_padded:
                        sorted_token_ids[idx] = numel

        return _align_main

    return _align


def _sort_tokens(
    flat: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    counts: torch.Tensor,
    cumsum: torch.Tensor,
    numel: int,
    num_experts: int,
) -> None:
    """PyTorch sort step: scatter token indices into sorted_token_ids.

    Computes each token's slot = cumsum[eid] + local_rank_within_expert,
    where local_rank is derived from the unpadded counts prefix-sum.
    """
    order = torch.argsort(flat.long(), stable=True)  # expert-sorted token indices
    sorted_eids = flat[order]  # expert id for each sorted position

    # Unpadded exclusive prefix-sum for computing local ranks
    cumsum_unpadded = torch.zeros(num_experts + 1, dtype=torch.int32, device=flat.device)
    cumsum_unpadded[1:] = counts.cumsum(0)

    local_rank = (
        torch.arange(numel, dtype=torch.int32, device=flat.device)
        - cumsum_unpadded[sorted_eids.long()]
    )
    slots = (cumsum[sorted_eids.long()] + local_rank).long()
    sorted_token_ids[slots] = order.int()


class moe_align_kernel(Kernel):
    """MoE token alignment kernel.

    Converts ``topk_ids`` into the three index arrays required by MoE grouped GEMM.

    Args:
        numel: Total number of (token, expert) assignments = total_tokens * top_k.
        num_experts: Number of experts.
        block_size: GEMM tile size (M dimension).
        config: Optional config dict (unused; kept for API consistency).

    Example:
        >>> kernel = moe_align_kernel(numel=32, num_experts=8, block_size=16)
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

        self._count_fn = _make_count_kernel(numel, num_experts)
        self._align_fn = _make_align_kernel(numel, num_experts, block_size)

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

        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=flat.device)
        sorted_token_ids = torch.empty(max_padded, dtype=torch.int32, device=flat.device)
        expert_ids = torch.empty(max_num_blocks, dtype=torch.int32, device=flat.device)
        num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device=flat.device)
        cumsum = torch.zeros(self.num_experts + 1, dtype=torch.int32, device=flat.device)

        # Kernel 1: count tokens per expert
        count_fn = self._count_fn(threads)
        count_fn(flat, counts)

        # Kernel 2: prefix-sum + expert_ids + sentinel fill
        align_fn = self._align_fn(threads)
        align_fn(counts, sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum)

        # Sort step: scatter token indices (PyTorch, stable argsort)
        _sort_tokens(flat, sorted_token_ids, counts, cumsum, self.numel, self.num_experts)

        return sorted_token_ids, expert_ids, num_tokens_post_pad
