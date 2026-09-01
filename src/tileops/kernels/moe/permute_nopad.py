"""Tight no-pad materialization kernel for staged MoE PrePermute.

The scan counts local expert assignments, writes physical segment ends, and
builds the forward-to-materialized inverse indices. A second kernel gathers
hidden-state rows into tight expert-contiguous order.
"""

from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["MoePrePermuteNopadKernel"]

_SCAN_THREADS = 1024


def _make_scan_kernel_nopad(numel: int, num_experts: int, top_k: int):
    """Count local assignments and produce staged physical-PSUM metadata."""

    @tilelang.jit(out_idx=[], compile_flags=["-O3"])
    def _scan(threads: int):
        @T.prim_func
        def _scan_main(
            flat_ids: T.Tensor([numel], "int32"),
            physical_ends: T.Tensor([num_experts], "int32"),
            permuted_idx: T.Tensor([numel], "int32"),
            inverse_indices: T.Tensor([numel], "int32"),
            write_offsets: T.Tensor([num_experts], "int32"),
        ):
            with T.Kernel(1, threads=threads) as (_,):
                tx = T.get_thread_binding()
                counts = T.alloc_shared([num_experts], "int32")
                offsets = T.alloc_shared([num_experts + 1], "int64")
                slot_buf = T.alloc_local([1], "int32")

                for i in T.serial(T.ceildiv(num_experts, threads)):
                    idx = i * threads + tx
                    if idx < num_experts:
                        counts[idx] = T.int32(0)
                T.sync_threads()

                for i in T.serial(T.ceildiv(numel, threads)):
                    idx = i * threads + tx
                    if idx < numel:
                        T.atomic_add(counts[flat_ids[idx]], 1)
                T.sync_threads()

                if tx == 0:
                    offsets[0] = T.int64(0)
                    for expert in T.serial(num_experts):
                        offsets[expert + 1] = offsets[expert] + T.Cast(T.int64, counts[expert])
                T.sync_threads()

                for i in T.serial(T.ceildiv(num_experts, threads)):
                    idx = i * threads + tx
                    if idx < num_experts:
                        physical_ends[idx] = T.Cast(T.int32, offsets[idx + 1])
                        write_offsets[idx] = T.Cast(T.int32, offsets[idx])
                T.sync_threads()

                for i in T.serial(T.ceildiv(numel, threads)):
                    idx = i * threads + tx
                    if idx < numel:
                        expert = flat_ids[idx]
                        slot_buf[0] = T.atomic_add(
                            write_offsets[expert], T.int32(1), return_prev=True
                        )
                        slot = slot_buf[0]
                        permuted_idx[slot] = idx // T.int32(top_k)
                        inverse_indices[idx] = slot

        return _scan_main

    return _scan


def _make_gather_kernel_nopad(num_tokens: int, numel: int, hidden_size: int, dtype: str):
    """Gather hidden-state rows into tight expert-contiguous order."""
    vector = 8
    threads = min(1024, hidden_size // vector)
    while threads > 0 and hidden_size % threads != 0:
        threads -= 1
    threads = max(threads, 1)
    rows_per_block = 8
    grid = (numel + rows_per_block - 1) // rows_per_block

    @tilelang.jit(out_idx=[], compile_flags=["-O3", "-DENABLE_BF16"])
    def _gather():
        @T.prim_func
        def _gather_main(
            hidden_states: T.Tensor([num_tokens, hidden_size], dtype),
            permuted_idx: T.Tensor([numel], "int32"),
            expert_input: T.Tensor([numel, hidden_size], dtype),
        ):
            with T.Kernel(grid, threads=threads) as (bid,):
                for row in T.serial(rows_per_block):
                    slot = bid * rows_per_block + row
                    if slot < numel:
                        T.copy(
                            hidden_states[permuted_idx[slot], 0:hidden_size],
                            expert_input[slot, 0:hidden_size],
                        )

        return _gather_main

    return _gather


class MoePrePermuteNopadKernel(Kernel):
    """Materialize local assignments into a tight physical-PSUM layout."""

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        num_tokens: int,
        top_k: int,
        num_experts: int,
        hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.numel = num_tokens * top_k
        self._scan_fn = _make_scan_kernel_nopad(self.numel, num_experts, top_k)
        self._gather_fn = _make_gather_kernel_nopad(
            num_tokens, self.numel, hidden_size, self.dtype_str
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"threads": _SCAN_THREADS}

    def forward(
        self,
        hidden_states: torch.Tensor,
        local_expert_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return expert input, physical segment ends, and inverse indices."""
        assert local_expert_ids.dtype == torch.int32
        assert hidden_states.is_cuda and local_expert_ids.is_cuda
        assert local_expert_ids.numel() == self.numel

        device = hidden_states.device
        flat_ids = local_expert_ids.flatten().contiguous()
        physical_ends = torch.empty(self.num_experts, dtype=torch.int32, device=device)
        permuted_idx = torch.empty(self.numel, dtype=torch.int32, device=device)
        inverse_indices = torch.empty(self.numel, dtype=torch.int32, device=device)
        write_offsets = torch.empty(self.num_experts, dtype=torch.int32, device=device)

        scan_fn = self._scan_fn(self.config["threads"])
        scan_fn(flat_ids, physical_ends, permuted_idx, inverse_indices, write_offsets)

        expert_input = torch.empty((self.numel, self.hidden_size), dtype=self.dtype, device=device)
        self._gather_fn()(hidden_states, permuted_idx, expert_input)
        return expert_input, physical_ends, inverse_indices
