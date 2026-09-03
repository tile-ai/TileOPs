"""Layout-specialized contiguous materialization for staged MoE PrePermute."""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.call_spec import PrePermuteCall

__all__ = ["MoePrePermuteContiguousKernel"]

_SCAN_THREADS = 1024
_SUPPORTED_LAYOUTS = frozenset(("tight_physical_psum", "aligned_per_row"))


def _fused_tight_plan(num_tokens: int, numel: int, hidden_size: int) -> bool:
    """Whether one cooperative tight launch beats the scan/gather pair.

    Tiny routing problems benefit directly from losing a launch.  At production
    widths, decode benefits as well, while prefill spends more time holding the
    cooperative grid across the barrier than the removed launch costs.  The
    thresholds are the H200 sweep boundary, following the measured planner style
    used by the fused split-row softmax path.
    """
    return numel <= 64 or (num_tokens <= 512 and hidden_size >= 2048)


def _make_tight_scan_body(numel: int, num_experts: int, top_k: int, threads: int):
    """Shared tight count/prefix/scatter body for split and fused launches."""

    @T.macro
    def scan(
        flat_ids,
        physical_ends,
        permuted_idx,
        inverse_indices,
        write_offsets,
        counts,
        offsets,
        slot_buf,
    ):
        tx = T.get_thread_binding()
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
                slot_buf[0] = T.atomic_add(write_offsets[expert], T.int32(1), return_prev=True)
                slot = slot_buf[0]
                permuted_idx[slot] = idx // T.int32(top_k)
                inverse_indices[idx] = slot

    return scan


def _make_tight_scan(numel: int, num_experts: int, top_k: int):
    """Count assignments and produce tight physical-PSUM metadata."""

    @tilelang.jit(out_idx=[], compile_flags=["-O3"])
    def _scan(threads: int):
        scan = _make_tight_scan_body(numel, num_experts, top_k, threads)

        @T.prim_func
        def _scan_main(
            flat_ids: T.Tensor([numel], "int32"),
            physical_ends: T.Tensor([num_experts], "int32"),
            permuted_idx: T.Tensor([numel], "int32"),
            inverse_indices: T.Tensor([numel], "int32"),
            write_offsets: T.Tensor([num_experts], "int32"),
        ):
            with T.Kernel(1, threads=threads) as (_,):
                counts = T.alloc_shared([num_experts], "int32")
                offsets = T.alloc_shared([num_experts + 1], "int64")
                slot_buf = T.alloc_local([1], "int32")
                scan(
                    flat_ids,
                    physical_ends,
                    permuted_idx,
                    inverse_indices,
                    write_offsets,
                    counts,
                    offsets,
                    slot_buf,
                )

        return _scan_main

    return _scan


def _make_aligned_per_row_scan(
    num_tokens: int,
    numel: int,
    num_experts: int,
    top_k: int,
    alignment: int,
    capacity: int,
):
    """Produce aligned slots, per-row expert IDs, and inverse indices."""
    owner_search_steps = max(1, num_experts.bit_length())

    @tilelang.jit(out_idx=[], compile_flags=["-O3"])
    def _scan(threads: int):
        @T.prim_func
        def _scan_main(
            flat_ids: T.Tensor([numel], "int32"),
            row_expert_ids: T.Tensor([capacity], "int32"),
            permuted_idx: T.Tensor([capacity], "int32"),
            inverse_indices: T.Tensor([numel], "int32"),
            write_offsets: T.Tensor([num_experts], "int32"),
        ):
            with T.Kernel(1, threads=threads) as (_,):
                tx = T.get_thread_binding()
                counts = T.alloc_shared([num_experts], "int32")
                offsets = T.alloc_shared([num_experts + 1], "int32")
                slot_buf = T.alloc_local([1], "int32")
                owner_buf = T.alloc_local([1], "int32")
                lower_buf = T.alloc_local([1], "int32")
                upper_buf = T.alloc_local([1], "int32")
                middle_buf = T.alloc_local([1], "int32")

                for i in T.serial(T.ceildiv(num_experts, threads)):
                    idx = i * threads + tx
                    if idx < num_experts:
                        counts[idx] = T.int32(0)
                for i in T.serial(T.ceildiv(capacity, threads)):
                    row = i * threads + tx
                    if row < capacity:
                        row_expert_ids[row] = T.int32(num_experts)
                        permuted_idx[row] = T.int32(num_tokens)
                T.sync_threads()

                for i in T.serial(T.ceildiv(numel, threads)):
                    idx = i * threads + tx
                    if idx < numel:
                        T.atomic_add(counts[flat_ids[idx]], 1)
                T.sync_threads()

                if tx == 0:
                    offsets[0] = T.int32(0)
                    for expert in T.serial(num_experts):
                        physical_size = (
                            (counts[expert] + T.int32(alignment - 1)) // T.int32(alignment)
                        ) * T.int32(alignment)
                        offsets[expert + 1] = offsets[expert] + physical_size
                T.sync_threads()

                for i in T.serial(T.ceildiv(num_experts, threads)):
                    expert = i * threads + tx
                    if expert < num_experts:
                        write_offsets[expert] = offsets[expert]
                for i in T.serial(T.ceildiv(capacity, threads)):
                    row = i * threads + tx
                    if row < capacity:
                        owner_buf[0] = T.int32(num_experts)
                        if row < offsets[num_experts]:
                            lower_buf[0] = T.int32(0)
                            upper_buf[0] = T.int32(num_experts)
                            for _ in T.serial(owner_search_steps):
                                middle_buf[0] = (lower_buf[0] + upper_buf[0]) // T.int32(2)
                                if row < offsets[middle_buf[0] + 1]:
                                    upper_buf[0] = middle_buf[0]
                                else:
                                    lower_buf[0] = middle_buf[0] + T.int32(1)
                            owner_buf[0] = lower_buf[0]
                        row_expert_ids[row] = owner_buf[0]
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


def _make_gather(
    num_tokens: int,
    physical_rows: int,
    hidden_size: int,
    dtype: str,
    *,
    zero_fill: bool,
):
    """Gather real rows and optionally zero-fill padding and capacity rows."""
    vector = 8
    threads = min(1024, hidden_size // vector)
    while threads > 0 and hidden_size % threads != 0:
        threads -= 1
    threads = max(threads, 1)
    rows_per_block = 8
    grid = (physical_rows + rows_per_block - 1) // rows_per_block

    @tilelang.jit(out_idx=[], compile_flags=["-O3", "-DENABLE_BF16"])
    def _gather():
        @T.prim_func
        def _gather_main(
            hidden_states: T.Tensor([num_tokens, hidden_size], dtype),
            permuted_idx: T.Tensor([physical_rows], "int32"),
            expert_input: T.Tensor([physical_rows, hidden_size], dtype),
        ):
            with T.Kernel(grid, threads=threads) as (bid,):
                tx = T.get_thread_binding()
                for local_row in T.serial(rows_per_block):
                    row = bid * rows_per_block + local_row
                    if row < physical_rows:
                        source = permuted_idx[row]
                        if not zero_fill:
                            T.copy(
                                hidden_states[source, 0:hidden_size],
                                expert_input[row, 0:hidden_size],
                            )
                        else:
                            if source < num_tokens:
                                T.copy(
                                    hidden_states[source, 0:hidden_size],
                                    expert_input[row, 0:hidden_size],
                                )
                            else:
                                for i in T.serial(T.ceildiv(hidden_size, threads)):
                                    column = i * threads + tx
                                    if column < hidden_size:
                                        expert_input[row, column] = T.Cast(dtype, 0)

        return _gather_main

    return _gather


@functools.lru_cache(maxsize=64)
def _make_fused_tight(
    num_tokens: int,
    numel: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    dtype: str,
    grid: int,
    rows_per_block: int,
):
    """Build tight metadata and gather rows in one cooperative launch."""
    vector = 8
    gather_threads = min(1024, hidden_size // vector)
    while gather_threads > 0 and hidden_size % gather_threads != 0:
        gather_threads -= 1
    scan_threads = 1 << min(10, max(0, numel - 1).bit_length())
    threads = max(gather_threads, scan_threads, 1)
    scan = _make_tight_scan_body(numel, num_experts, top_k, threads)

    @tilelang.jit(out_idx=[], compile_flags=["-O3", "-DENABLE_BF16"])
    def _fused():
        @T.prim_func
        def _fused_main(
            hidden_states: T.Tensor([num_tokens, hidden_size], dtype),
            flat_ids: T.Tensor([numel], "int32"),
            physical_ends: T.Tensor([num_experts], "int32"),
            permuted_idx: T.Tensor([numel], "int32"),
            inverse_indices: T.Tensor([numel], "int32"),
            write_offsets: T.Tensor([num_experts], "int32"),
            expert_input: T.Tensor([numel, hidden_size], dtype),
        ):
            with T.Kernel(grid, threads=threads) as (bid,):
                counts = T.alloc_shared([num_experts], "int32")
                offsets = T.alloc_shared([num_experts + 1], "int64")
                slot_buf = T.alloc_local([1], "int32")

                if bid == 0:
                    scan(
                        flat_ids,
                        physical_ends,
                        permuted_idx,
                        inverse_indices,
                        write_offsets,
                        counts,
                        offsets,
                        slot_buf,
                    )

                T.sync_grid()

                for local_row in T.serial(rows_per_block):
                    row = bid * rows_per_block + local_row
                    if row < numel:
                        T.copy(
                            hidden_states[permuted_idx[row], 0:hidden_size],
                            expert_input[row, 0:hidden_size],
                        )

        return _fused_main

    return _fused


class MoePrePermuteContiguousKernel(Kernel):
    """Build one contiguous PrePermute specialization from ``call.layout``."""

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def applies(cls, call: PrePermuteCall) -> bool:
        layout = call.layout
        return getattr(
            layout, "selection_key", None
        ) in _SUPPORTED_LAYOUTS and call.input_dtype in (torch.bfloat16, torch.float16)

    def __init__(
        self,
        call: PrePermuteCall,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        layout = call.layout
        self.layout_key = getattr(layout, "selection_key", "")
        if self.layout_key not in _SUPPORTED_LAYOUTS:
            raise ValueError(f"unsupported contiguous PrePermute layout: {self.layout_key!r}")
        self.num_tokens = call.num_tokens
        self.top_k = call.top_k
        self.num_experts = call.num_experts
        self.hidden_size = call.hidden_size
        self.dtype = call.input_dtype
        self.h200 = call.h200
        self.sm_count = call.sm_count
        self.numel = call.num_tokens * call.top_k
        self.alignment = getattr(layout, "alignment", 1)
        self.capacity = (
            self.numel
            if self.layout_key == "tight_physical_psum"
            else self.numel + self.num_experts * (self.alignment - 1)
        )

        if self.layout_key == "tight_physical_psum":
            self._scan_fn = _make_tight_scan(self.numel, self.num_experts, self.top_k)
        else:
            self._scan_fn = _make_aligned_per_row_scan(
                self.num_tokens,
                self.numel,
                self.num_experts,
                self.top_k,
                self.alignment,
                self.capacity,
            )
        self._gather_fn = _make_gather(
            self.num_tokens,
            self.capacity,
            self.hidden_size,
            self.dtype_str,
            zero_fill=self.layout_key == "aligned_per_row",
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
        """Return expert input, layout metadata, and inverse indices."""
        assert local_expert_ids.dtype == torch.int32
        assert hidden_states.is_cuda and local_expert_ids.is_cuda
        assert local_expert_ids.numel() == self.numel

        device = hidden_states.device
        flat_ids = local_expert_ids.flatten().contiguous()
        metadata_rows = (
            self.num_experts if self.layout_key == "tight_physical_psum" else self.capacity
        )
        layout_metadata = torch.empty(metadata_rows, dtype=torch.int32, device=device)
        permuted_idx = torch.empty(self.capacity, dtype=torch.int32, device=device)
        inverse_indices = torch.empty(self.numel, dtype=torch.int32, device=device)
        write_offsets = torch.empty(self.num_experts, dtype=torch.int32, device=device)

        expert_input = torch.empty(
            (self.capacity, self.hidden_size), dtype=self.dtype, device=device
        )
        use_fused_tight = (
            self.layout_key == "tight_physical_psum"
            and self.h200
            and _fused_tight_plan(self.num_tokens, self.numel, self.hidden_size)
        )
        if use_fused_tight:
            rows_per_block = max(1, (self.numel + self.sm_count - 1) // self.sm_count)
            grid = (self.numel + rows_per_block - 1) // rows_per_block
            _make_fused_tight(
                self.num_tokens,
                self.numel,
                self.num_experts,
                self.top_k,
                self.hidden_size,
                self.dtype_str,
                grid,
                rows_per_block,
            )()(
                hidden_states,
                flat_ids,
                layout_metadata,
                permuted_idx,
                inverse_indices,
                write_offsets,
                expert_input,
            )
        else:
            self._scan_fn(self.config["threads"])(
                flat_ids,
                layout_metadata,
                permuted_idx,
                inverse_indices,
                write_offsets,
            )
            self._gather_fn()(hidden_states, permuted_idx, expert_input)
        return expert_input, layout_metadata, inverse_indices
