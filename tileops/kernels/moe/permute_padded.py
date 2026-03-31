"""MoE token permute kernel (cutlass path).

Counting sort + padded gather: routes tokens to expert-contiguous layout
with block_m-aligned padding for NT grouped-GEMM.

Phase 1 (1 block, 1024 threads):
  - Count tokens per expert
  - Exclusive prefix-sum → expert_first_token_offset [E+1] (int64)
  - Compute padded layout: padded_offsets [E], padded_sizes [E]
  - Build permuted_idx [T*K] and scatter indices via atomic scatter:
      slot_to_padded [T*K]: permuted pos → padded slot
      fwd_idx [T*K]: original flat pos → padded slot

Phase 2 (N blocks, 16 threads):
  - Gather hidden_states rows into perm_h_pad (pre-zeroed) at padded positions
  - Each thread loads 8 bf16 (128-bit) via uint4 vectorized ld.global

Outputs:
  perm_h_pad                [padded_batch_sum, H]  tokens in padded expert-contiguous order
  padded_offsets            [E]                    int32 padded start per expert
  padded_sizes              [E]                    int32 block_m-aligned size per expert
  expert_first_token_offset [E+1]                  int64 non-padded exclusive prefix-sum
  fwd_idx                   [T*K]                  int32 forward mapping: flat pos → padded slot
"""

import os
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

_ATOMIC_HELPER_H = os.path.join(os.path.dirname(__file__), "_atomic_helper.h")

__all__ = ["MoePermutePaddedKernel"]

_SCAN_THREADS = 1024


def _make_scan_kernel(numel: int, num_experts: int, top_k: int, block_m: int):
    """Phase 1: count → prefix-sum → padded layout → scatter indices."""

    @tilelang.jit(out_idx=[], compile_flags=["-O3", "-include", _ATOMIC_HELPER_H])
    def _scan(threads: int):

        @T.prim_func
        def _scan_main(
            flat_ids: T.Tensor([numel], "int32"),                              # noqa: F821
            expert_first_token_offset: T.Tensor([num_experts + 1], "int64"),  # noqa: F821
            padded_offsets: T.Tensor([num_experts], "int32"),                  # noqa: F821
            padded_sizes: T.Tensor([num_experts], "int32"),                    # noqa: F821
            permuted_idx: T.Tensor([numel], "int32"),                          # noqa: F821
            slot_to_padded: T.Tensor([numel], "int32"),                        # noqa: F821
            fwd_idx: T.Tensor([numel], "int32"),                               # noqa: F821
            write_offsets: T.Tensor([num_experts], "int32"),                   # noqa: F821
        ):
            with T.Kernel(1, threads=threads) as (_,):
                tx = T.get_thread_binding()

                s_counts = T.alloc_shared([num_experts], "int32")
                s_offset = T.alloc_shared([num_experts + 1], "int64")
                s_base = T.alloc_shared([num_experts], "int32")
                s_padded_starts = T.alloc_shared([num_experts + 1], "int32")

                # Step 1: zero counts
                for i in T.serial(T.ceildiv(num_experts, threads)):
                    idx = i * threads + tx
                    if idx < num_experts:
                        s_counts[idx] = T.int32(0)
                T.sync_threads()

                # Step 2: count tokens per expert
                for i in T.serial(T.ceildiv(numel, threads)):
                    idx = i * threads + tx
                    if idx < numel:
                        T.atomic_add(s_counts[flat_ids[idx]], 1)
                T.sync_threads()

                # Step 3: exclusive prefix-sum → s_offset [E+1]
                if tx == 0:
                    s_offset[0] = T.int64(0)
                    for e in T.serial(num_experts):
                        s_offset[e + 1] = s_offset[e] + T.Cast(T.int64, s_counts[e])
                T.sync_threads()

                # Step 3.5: compute block_m-aligned padded layout (thread 0)
                if tx == 0:
                    s_padded_starts[0] = T.int32(0)
                    for e in T.serial(num_experts):
                        s_base[e] = T.Cast(T.int32, s_offset[e])
                        cnt = s_counts[e]
                        ps = ((cnt + T.int32(block_m - 1)) // T.int32(block_m)) * T.int32(block_m)
                        padded_sizes[e] = ps
                        padded_offsets[e] = s_padded_starts[e]
                        s_padded_starts[e + 1] = s_padded_starts[e] + ps
                T.sync_threads()

                # Step 4: write expert_first_token_offset and init write_offsets
                for i in T.serial(T.ceildiv(num_experts + 1, threads)):
                    idx = i * threads + tx
                    if idx <= num_experts:
                        expert_first_token_offset[idx] = s_offset[idx]
                for i in T.serial(T.ceildiv(num_experts, threads)):
                    idx = i * threads + tx
                    if idx < num_experts:
                        write_offsets[idx] = s_base[idx]
                T.sync_threads()

                # Step 5: scatter — assign each (token, topk) pair to its expert slot
                for i in T.serial(T.ceildiv(numel, threads)):
                    idx = i * threads + tx
                    if idx < numel:
                        eid = flat_ids[idx]
                        slot = T.call_extern("int32", "tl_atomic_add_offset",
                                             T.address_of(write_offsets[0]), eid, T.int32(1))
                        # permuted_idx[slot] = token row (idx // top_k) for gather
                        # within = intra-expert position of this slot
                        # padded_slot = padded start of expert + within
                        permuted_idx[slot] = idx // T.int32(top_k)
                        within = slot - s_base[eid]
                        padded_slot = s_padded_starts[eid] + within
                        slot_to_padded[slot] = padded_slot
                        fwd_idx[idx] = padded_slot

        return _scan_main

    return _scan


def _make_gather_kernel(
    num_tokens: int, numel: int, padded_batch_sum: int, hidden_size: int, dtype: str
):
    """Phase 2: gather hidden_states rows into perm_h_pad at padded positions.

    One block per slot. Each thread handles 8 elements (128-bit uint4 load/store).
    threads = hidden_size // 8, capped at 1024.
    """
    VEC = 8  # 8 x bf16/fp16 = 128 bits → uint4 ld.global
    threads = min(1024, hidden_size // VEC)
    while threads > 0 and hidden_size % threads != 0:
        threads -= 1
    threads = max(threads, 1)

    @tilelang.jit(out_idx=[], compile_flags=["-O3", "-DENABLE_BF16"])
    def _gather():

        @T.prim_func
        def _gather_main(
            hidden_states: T.Tensor([num_tokens, hidden_size], dtype),          # noqa: F821
            permuted_idx: T.Tensor([numel], "int32"),                           # noqa: F821
            slot_to_padded: T.Tensor([numel], "int32"),                         # noqa: F821
            perm_h_pad: T.Tensor([padded_batch_sum, hidden_size], dtype),       # noqa: F821
        ):
            with T.Kernel(numel, threads=threads) as (slot,):
                frag = T.alloc_fragment([hidden_size], dtype)
                T.copy(hidden_states[permuted_idx[slot], 0:hidden_size], frag)
                T.copy(frag, perm_h_pad[slot_to_padded[slot], 0:hidden_size])

        return _gather_main

    return _gather


class MoePermutePaddedKernel(Kernel):
    """MoE token permute kernel with block_m-aligned padding (cutlass path).

    Sorts tokens into expert-contiguous order with block_m padding alignment
    for NT grouped-GEMM, and gathers hidden_states into the padded buffer.

    Args:
        num_tokens: Number of input tokens T.
        top_k: Number of experts selected per token K.
        num_experts: Total number of experts E.
        hidden_size: Hidden dimension H.
        dtype: Data type of hidden_states.
        block_m: Block size for expert row-start alignment (default: 64).
        config: Optional config dict.

    Example:
        >>> kernel = MoePermutePaddedKernel(num_tokens=4, top_k=2, num_experts=8, hidden_size=128)
        >>> perm_h_pad, p_offsets, p_sizes, offsets, fwd_idx = kernel(hidden_states, topk_ids)
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        num_tokens: int,
        top_k: int,
        num_experts: int,
        hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
        block_m: int = 64,
        config: Optional[dict] = None,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.block_m = block_m
        self.numel = num_tokens * top_k
        self._padded_batch_sum = self.numel + num_experts * (block_m - 1)

        self._scan_fn = _make_scan_kernel(self.numel, num_experts, top_k, block_m)
        self._gather_fn = _make_gather_kernel(
            num_tokens, self.numel, self._padded_batch_sum, hidden_size, self.dtype_str
        )

        self.init_config(config, tune=False)

    @property
    def default_config(self) -> dict:
        return {"threads": _SCAN_THREADS}

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run moe_permute with padded output.

        Args:
            hidden_states: [T, H] input activations.
            topk_ids: [T, K] int32 expert assignments.

        Returns:
            perm_h_pad:                [padded_batch_sum, H] padded hidden states
            padded_offsets:            [E] int32 padded start per expert
            padded_sizes:              [E] int32 block_m-aligned sizes per expert
            expert_first_token_offset: [E+1] int64 non-padded exclusive prefix-sum
            fwd_idx:                   [T*K] int32 forward mapping: flat_idx → padded slot
        """
        assert topk_ids.dtype == torch.int32
        assert hidden_states.is_cuda and topk_ids.is_cuda
        assert topk_ids.numel() == self.numel

        dev = hidden_states.device
        flat_ids = topk_ids.flatten().contiguous()

        expert_first_token_offset = torch.empty(self.num_experts + 1, dtype=torch.int64, device=dev)
        padded_offsets = torch.empty(self.num_experts, dtype=torch.int32, device=dev)
        padded_sizes = torch.empty(self.num_experts, dtype=torch.int32, device=dev)
        permuted_idx = torch.empty(self.numel, dtype=torch.int32, device=dev)
        slot_to_padded = torch.empty(self.numel, dtype=torch.int32, device=dev)
        fwd_idx = torch.empty(self.numel, dtype=torch.int32, device=dev)
        write_offsets = torch.empty(self.num_experts, dtype=torch.int32, device=dev)

        threads = self.config["threads"]
        scan_fn = self._scan_fn(threads)
        scan_fn(
            flat_ids, expert_first_token_offset, padded_offsets, padded_sizes,
            permuted_idx, slot_to_padded, fwd_idx, write_offsets,
        )

        perm_h_pad = torch.zeros(
            (self._padded_batch_sum, self.hidden_size), dtype=self.dtype, device=dev
        )
        gather_fn = self._gather_fn()
        gather_fn(hidden_states, permuted_idx, slot_to_padded, perm_h_pad)

        return perm_h_pad, padded_offsets, padded_sizes, expert_first_token_offset, fwd_idx
