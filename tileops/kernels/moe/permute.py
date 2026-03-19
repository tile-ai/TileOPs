"""MoE token permute kernel (cutlass path).

Counting sort + gather: routes tokens to expert-contiguous layout without padding.

Phase 1 (1 block, 1024 threads):
  - Count tokens per expert
  - Exclusive prefix-sum → expert_first_token_offset [E+1] (int64)
  - Build permuted_idx [T*K] and inv_permuted_idx [T*K] via atomic scatter

Phase 2 (N blocks, 16 threads):
  - Gather hidden_states rows into permuted_hidden_states using permuted_idx
  - Each thread loads 8 bf16 (128-bit) via uint4 vectorized ld.global

Outputs:
  permuted_hidden_states    [T*K, H]   tokens in expert-contiguous order
  expert_first_token_offset [E+1]      int64 start offset per expert
  inv_permuted_idx          [T*K]      int32 inverse mapping: permuted pos → original flat pos
"""

import os
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

_ATOMIC_HELPER_H = os.path.join(os.path.dirname(__file__), "_atomic_helper.h")

__all__ = ["MoePermuteKernel"]

_SCAN_THREADS = 1024


def _make_scan_kernel(numel: int, num_experts: int, top_k: int):
    """Phase 1: count → prefix-sum → scatter indices (build permuted_idx + inv_permuted_idx)."""

    @tilelang.jit(out_idx=[], compile_flags=["-O3", "-include", _ATOMIC_HELPER_H])
    def _scan(threads: int):

        @T.prim_func
        def _scan_main(
            flat_ids: T.Tensor([numel], "int32"),                        # noqa: F821
            expert_first_token_offset: T.Tensor([num_experts + 1], "int64"),  # noqa: F821
            permuted_idx: T.Tensor([numel], "int32"),                    # noqa: F821
            inv_permuted_idx: T.Tensor([numel], "int32"),                # noqa: F821
            write_offsets: T.Tensor([num_experts], "int32"),             # noqa: F821
        ):
            with T.Kernel(1, threads=threads) as (_,):
                tx = T.get_thread_binding()

                s_counts = T.alloc_shared([num_experts], "int32")
                s_offset = T.alloc_shared([num_experts + 1], "int64")

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

                # Step 4: write expert_first_token_offset and init write_offsets
                for i in T.serial(T.ceildiv(num_experts + 1, threads)):
                    idx = i * threads + tx
                    if idx <= num_experts:
                        expert_first_token_offset[idx] = s_offset[idx]
                for i in T.serial(T.ceildiv(num_experts, threads)):
                    idx = i * threads + tx
                    if idx < num_experts:
                        write_offsets[idx] = T.Cast(T.int32, s_offset[idx])
                T.sync_threads()

                # Step 5: scatter — assign each (token, topk) pair to its expert slot
                for i in T.serial(T.ceildiv(numel, threads)):
                    idx = i * threads + tx
                    if idx < numel:
                        eid = flat_ids[idx]
                        slot = T.call_extern("int32", "tl_atomic_add_offset",
                                             T.address_of(write_offsets[0]), eid, T.int32(1))
                        # permuted_idx[slot] = token row (idx // top_k) for gather
                        # inv_permuted_idx[slot] = flat index for unpermute
                        permuted_idx[slot] = idx // T.int32(top_k)
                        inv_permuted_idx[slot] = idx

        return _scan_main

    return _scan


def _make_gather_kernel(num_tokens: int, numel: int, hidden_size: int, dtype: str):
    """Phase 2: gather hidden_states rows into permuted_hidden_states.

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
            hidden_states: T.Tensor([num_tokens, hidden_size], dtype),      # noqa: F821
            permuted_idx: T.Tensor([numel], "int32"),                       # noqa: F821
            permuted_hidden_states: T.Tensor([numel, hidden_size], dtype),  # noqa: F821
        ):
            with T.Kernel(numel, threads=threads) as (slot,):
                frag = T.alloc_fragment([hidden_size], dtype)
                T.copy(hidden_states[permuted_idx[slot], 0:hidden_size], frag)
                T.copy(frag, permuted_hidden_states[slot, 0:hidden_size])

        return _gather_main

    return _gather


class MoePermuteKernel(Kernel):
    """MoE token permute kernel (cutlass path).

    Sorts tokens into expert-contiguous order and gathers hidden_states.

    Args:
        num_tokens: Number of input tokens T.
        top_k: Number of experts selected per token K.
        num_experts: Total number of experts E.
        hidden_size: Hidden dimension H.
        dtype: Data type of hidden_states.
        config: Optional config dict.

    Example:
        >>> kernel = MoePermuteKernel(num_tokens=4, top_k=2, num_experts=8, hidden_size=128)
        >>> perm_h, offsets, inv_idx = kernel(hidden_states, topk_ids)
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        num_tokens: int,
        top_k: int,
        num_experts: int,
        hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
        config: Optional[dict] = None,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.numel = num_tokens * top_k

        self._scan_fn = _make_scan_kernel(self.numel, num_experts, top_k)
        self._gather_fn = _make_gather_kernel(num_tokens, self.numel, hidden_size, self.dtype_str)

        self.init_config(config, tune=False)

    @property
    def default_config(self) -> dict:
        return {"threads": _SCAN_THREADS}

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run moe_permute.

        Args:
            hidden_states: [T, H] input activations.
            topk_ids: [T, K] int32 expert assignments.

        Returns:
            permuted_hidden_states:    [T*K, H]
            expert_first_token_offset: [E+1] int64
            inv_permuted_idx:          [T*K] int32
        """
        assert topk_ids.dtype == torch.int32
        assert hidden_states.is_cuda and topk_ids.is_cuda
        assert topk_ids.numel() == self.numel

        dev = hidden_states.device
        flat_ids = topk_ids.flatten().contiguous()

        expert_first_token_offset = torch.empty(self.num_experts + 1, dtype=torch.int64, device=dev)
        permuted_idx = torch.empty(self.numel, dtype=torch.int32, device=dev)
        inv_permuted_idx = torch.empty(self.numel, dtype=torch.int32, device=dev)
        write_offsets = torch.empty(self.num_experts, dtype=torch.int32, device=dev)

        threads = self.config["threads"]
        scan_fn = self._scan_fn(threads)
        scan_fn(flat_ids, expert_first_token_offset, permuted_idx, inv_permuted_idx, write_offsets)

        permuted_hidden_states = torch.empty(
            (self.numel, self.hidden_size), dtype=self.dtype, device=dev
        )
        gather_fn = self._gather_fn()
        gather_fn(hidden_states, permuted_idx, permuted_hidden_states)

        return permuted_hidden_states, expert_first_token_offset, inv_permuted_idx
