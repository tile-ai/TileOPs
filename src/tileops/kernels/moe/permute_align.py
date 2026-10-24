"""MoE token-to-expert alignment kernel.

The kernel picks its path when it is built: a call with fewer than 1024 routes and at most one
warp of experts takes one fused small-batch kernel; any other call takes the align kernel and,
when it has routes, the scatter kernel after it. Both paths return:

  sorted_token_ids  [max_num_tokens_padded]  - route indices grouped by expert, each expert's
                                               routes first in its range, numel elsewhere
  expert_ids        [num_blocks]             - the expert of each block of the padded length,
                                               num_experts past it
  num_tokens_post_pad [1]                   - total padded route count

Reference: sglang/sgl-kernel/csrc/moe/moe_align_kernel.cu
"""

import functools
import math
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import MAX_BLOCK_THREADS
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.moe.call_spec import PermuteAlignCall, PermuteAlignFwdInterface
from tileops.utils import WARP_LANES, WARP_SHUFFLE_STAGES

__all__ = ["MoEPermuteAlignKernel"]


@functools.lru_cache(maxsize=32)
def _make_align_kernel(numel: int, num_experts: int, block_size: int):
    """The align kernel; the scatter kernel fills the routed slots after it.

    Writes num_tokens_post_pad, each expert's range start into cumsum, expert_ids, and numel
    into every slot of sorted_token_ids.
    """
    max_padded = numel + (num_experts + 1) * (block_size - 1)
    max_num_blocks = math.ceil(max_padded / block_size)

    @tilelang.jit(out_idx=[], compile_flags=["-O3"])
    def _align(threads: int):
        @T.prim_func
        def _align_main(
            flat: T.Tensor([numel], "int32"),
            sorted_token_ids: T.Tensor([max_padded], "int32"),
            expert_ids: T.Tensor([max_num_blocks], "int32"),
            num_tokens_post_pad: T.Tensor([1], "int32"),
            cumsum: T.Tensor([num_experts + 1], "int32"),
        ):
            with T.Kernel(2, threads=threads) as (bx,):
                tx = T.get_thread_binding()
                num_warps = threads // WARP_LANES
                num_chunks = T.ceildiv(num_experts, threads)

                s_counts = T.alloc_shared([num_experts], "int32")
                s_vals = T.alloc_shared([threads], "int32")
                s_warp_sum = T.alloc_shared([num_warps], "int32")
                s_warp_excl = T.alloc_shared([num_warps], "int32")
                # A chunk's padded total, and the totals of the chunks before it.
                s_total = T.alloc_shared([1], "int32")
                s_carry = T.alloc_shared([1], "int32")
                s_cumsum = T.alloc_shared([num_experts + 1], "int32")

                # The scatter writes the routed slots in a later launch, so this fill races
                # no write.
                if bx == 1:
                    for i in T.serial(T.ceildiv(max_padded, threads)):
                        idx = i * threads + tx
                        if idx < max_padded:
                            sorted_token_ids[idx] = numel
                else:
                    for i in T.serial(T.ceildiv(num_experts, threads)):
                        idx = i * threads + tx
                        if idx < num_experts:
                            s_counts[idx] = T.int32(0)
                    T.sync_threads()

                    for i in T.serial(T.ceildiv(numel, threads)):
                        idx = i * threads + tx
                        if idx < numel:
                            T.atomic_add(s_counts[flat[idx]], 1)
                    T.sync_threads()

                    lane = tx % WARP_LANES
                    warp_id = tx // WARP_LANES

                    if tx == 0:
                        s_carry[0] = T.int32(0)
                    for c in T.serial(num_chunks):
                        expert = c * threads + tx
                        own_padded = T.if_then_else(
                            expert < num_experts,
                            T.ceildiv(s_counts[expert], block_size) * block_size,
                            T.int32(0),
                        )
                        s_vals[tx] = own_padded
                        T.sync_threads()

                        for d in T.serial(WARP_SHUFFLE_STAGES):
                            stride = 1 << d
                            up_val = T.tvm_warp_shuffle_up(
                                T.uint32((1 << WARP_LANES) - 1),
                                s_vals[tx],
                                stride,
                                WARP_LANES,
                                WARP_LANES,
                            )
                            if lane >= stride:
                                s_vals[tx] = s_vals[tx] + up_val
                        T.sync_threads()

                        if lane == WARP_LANES - 1:
                            s_warp_sum[warp_id] = s_vals[tx]
                        T.sync_threads()

                        # Nesting is deliberate: the loop stays outside the tx == 0 guard.
                        for w in T.serial(num_warps):
                            if tx == 0:
                                if w == 0:
                                    s_total[0] = T.int32(0)
                                s_warp_excl[w] = s_total[0]
                                s_total[0] = s_total[0] + s_warp_sum[w]
                        T.sync_threads()

                        excl = s_vals[tx] - own_padded + s_warp_excl[warp_id] + s_carry[0]
                        if expert < num_experts:
                            s_cumsum[expert] = excl
                            cumsum[expert] = excl
                        T.sync_threads()
                        if tx == 0:
                            s_carry[0] = s_carry[0] + s_total[0]
                        T.sync_threads()

                    if tx == 0:
                        s_cumsum[num_experts] = s_carry[0]
                        cumsum[num_experts] = s_carry[0]
                        num_tokens_post_pad[0] = s_carry[0]
                    T.sync_threads()

                    # A block past the padded length names no expert.
                    for i in T.serial(T.ceildiv(max_num_blocks, threads)):
                        blk = i * threads + tx
                        if blk >= s_carry[0] // block_size and blk < max_num_blocks:
                            expert_ids[blk] = num_experts

                    e_start = T.alloc_var("int32")
                    e_blocks = T.alloc_var("int32")
                    for c in T.serial(num_chunks):
                        expert = c * threads + tx
                        if expert < num_experts:
                            e_start = s_cumsum[expert] // block_size
                            e_blocks = s_cumsum[expert + 1] // block_size - e_start
                            for b in T.serial(e_blocks):
                                expert_ids[e_start + b] = expert

        return _align_main

    return _align


@functools.lru_cache(maxsize=32)
def _make_scatter_kernel(numel: int, num_experts: int, block_size: int):
    """The scatter kernel: writes each route's index into the front of its expert's range.

    Takes each expert's next free slot from cumsum as the align kernel left it, so it runs
    after the align kernel on the same stream, and advances it.
    """
    if numel < 1:
        raise ValueError(f"the scatter needs at least one route, got numel={numel}")
    max_padded = numel + (num_experts + 1) * (block_size - 1)
    # Timed against 256 and 1024 on the H200 manifest rows: fastest at T = 4096, and within
    # 0.2 us of the fastest on the others.
    threads = 512
    scatter_blocks = math.ceil(numel / threads)
    expert_rounds = math.ceil(num_experts / threads)

    @tilelang.jit(out_idx=[], compile_flags=["-O3"])
    def _scatter():
        @T.prim_func
        def _scatter_main(
            flat: T.Tensor([numel], "int32"),
            sorted_token_ids: T.Tensor([max_padded], "int32"),
            cumsum: T.Tensor([num_experts + 1], "int32"),
        ):
            with T.Kernel(scatter_blocks, threads=threads) as (bid,):
                tx = T.get_thread_binding()
                idx = bid * threads + tx
                # The reserved starts get their own buffer: updating s_count in place makes
                # TileLang put a sync inside the expert guard, which deadlocks past one round.
                s_count = T.alloc_shared([num_experts], "int32")
                s_base = T.alloc_shared([num_experts], "int32")
                eid = T.alloc_var("int32")
                rank = T.alloc_var("int32")

                for i in T.serial(expert_rounds):
                    expert = i * threads + tx
                    if expert < num_experts:
                        s_count[expert] = T.int32(0)
                eid = T.if_then_else(idx < numel, flat[T.min(idx, numel - 1)], -1)
                T.sync_threads()

                if eid >= 0:
                    rank = T.atomic_add(s_count[eid], T.int32(1), return_prev=True)
                T.sync_threads()

                for i in T.serial(expert_rounds):
                    expert = i * threads + tx
                    if expert < num_experts:
                        routed = s_count[expert]
                        if routed > 0:
                            s_base[expert] = T.atomic_add(cumsum[expert], routed, return_prev=True)
                T.sync_threads()

                if eid >= 0:
                    sorted_token_ids[s_base[eid] + rank] = idx

        return _scatter_main

    return _scatter


@functools.lru_cache(maxsize=32)
def _make_small_batch_kernel(numel: int, num_experts: int, block_size: int):
    """The small-batch kernel: one block whose worker lanes count and scatter the routes
    while the other threads write the padding.

    Each worker counts into its own row, so neither the count nor the scatter takes an
    atomic. Writes the same outputs as the align and scatter kernels.

    Reference: moe_align_block_size_small_batch_expert_kernel in sgl-kernel.

    Raises:
        ValueError: ``num_experts`` is more than one warp of workers.
    """
    # One warp of workers: past it, TileLang can put a named barrier inside the workers' guarded
    # loops that a warp skipping the last iteration never reaches.
    worker_threads = WARP_LANES
    if num_experts > worker_threads:
        raise ValueError(
            f"the small-batch kernel takes at most {worker_threads} experts, got {num_experts}"
        )
    max_padded = numel + (num_experts + 1) * (block_size - 1)
    max_num_blocks = math.ceil(max_padded / block_size)
    # Threads that write the sentinel fill while the workers count and scatter.
    fill_threads = 256
    total_threads = fill_threads + worker_threads
    # tokens_cnts: row k + 1 holds worker k's counts; after the prefix sum over rows, row k
    # holds worker k's first rank per expert and the last row the totals.
    cnts_size = (worker_threads + 1) * num_experts

    @tilelang.jit(out_idx=[], compile_flags=["-O3"])
    def _small():
        @T.prim_func
        def _small_main(
            flat: T.Tensor([numel], "int32"),
            sorted_token_ids: T.Tensor([max_padded], "int32"),
            expert_ids: T.Tensor([max_num_blocks], "int32"),
            num_tokens_post_pad: T.Tensor([1], "int32"),
        ):
            with T.Kernel(1, threads=total_threads) as (_,):
                tx = T.get_thread_binding()

                cumsum_s = T.alloc_shared([num_experts + 1], "int32")
                tokens_cnts = T.alloc_shared([cnts_size], "int32")

                if tx < fill_threads:
                    for i in T.serial(T.ceildiv(max_padded, fill_threads)):
                        idx = i * fill_threads + tx
                        if idx < max_padded:
                            sorted_token_ids[idx] = numel

                if tx >= fill_threads:
                    wid = tx - fill_threads
                    for i in T.serial(num_experts):
                        tokens_cnts[(wid + 1) * num_experts + i] = T.int32(0)
                    for i in T.serial(T.ceildiv(numel, worker_threads)):
                        idx = wid + i * worker_threads
                        if idx < numel:
                            eid = flat[idx]
                            tokens_cnts[(wid + 1) * num_experts + eid] = (
                                tokens_cnts[(wid + 1) * num_experts + eid] + 1
                            )

                T.sync_threads()

                if tx >= fill_threads:
                    wid = tx - fill_threads
                    if wid < num_experts:
                        tokens_cnts[wid] = T.int32(0)
                        for k in T.serial(worker_threads):
                            tokens_cnts[(k + 1) * num_experts + wid] = (
                                tokens_cnts[(k + 1) * num_experts + wid]
                                + tokens_cnts[k * num_experts + wid]
                            )

                T.sync_threads()

                if tx >= fill_threads:
                    wid = tx - fill_threads
                    if wid == 0:
                        cumsum_s[0] = T.int32(0)
                        for e in T.serial(num_experts):
                            cnt = tokens_cnts[worker_threads * num_experts + e]
                            cumsum_s[e + 1] = cumsum_s[e] + T.ceildiv(cnt, block_size) * block_size
                        num_tokens_post_pad[0] = cumsum_s[num_experts]

                T.sync_threads()

                # A block past the padded length names no expert.
                if tx < fill_threads:
                    for i in T.serial(T.ceildiv(max_num_blocks, fill_threads)):
                        blk = i * fill_threads + tx
                        if blk >= cumsum_s[num_experts] // block_size and blk < max_num_blocks:
                            expert_ids[blk] = num_experts

                if tx >= fill_threads:
                    wid = tx - fill_threads
                    e_start = T.alloc_var("int32")
                    e_blocks = T.alloc_var("int32")
                    if wid < num_experts:
                        e_start = cumsum_s[wid] // block_size
                        e_blocks = cumsum_s[wid + 1] // block_size - e_start
                        for b in T.serial(e_blocks):
                            expert_ids[e_start + b] = wid

                    for i in T.serial(T.ceildiv(numel, worker_threads)):
                        idx = wid + i * worker_threads
                        if idx < numel:
                            eid = flat[idx]
                            rank = tokens_cnts[wid * num_experts + eid]
                            slot = rank + cumsum_s[eid]
                            sorted_token_ids[slot] = idx
                            tokens_cnts[wid * num_experts + eid] = rank + 1

        return _small_main

    return _small


class MoEPermuteAlignKernel(Kernel, PermuteAlignFwdInterface):
    """MoE token permutation and alignment kernel.

    Converts ``topk_ids`` into the three index arrays required by MoE grouped GEMM.

    Args:
        numel: Total number of (token, expert) assignments = total_tokens * top_k.
        num_experts: Number of experts.
        block_size: GEMM tile size (M dimension).
        config: Optional config dict with "threads", the align kernel's threads per block: a
            multiple of 32 in [32, 1024]. Its count, scan and fill loops take ``threads`` items
            per round, so fewer threads only add rounds; this kernel defines no
            ``autotune_configs``.
        device_index: The device this kernel is built for.

    Raises:
        ValueError: ``config`` states a thread count outside that range.

    Note:
        ``dtype`` does not apply. Every tensor this kernel reads or writes is
        an int32 index array whose element type is fixed by the MoE grouped
        GEMM index contract, so there is no free element type to select.

    Example:
        ```python linenums="1"
        kernel = MoEPermuteAlignKernel(numel=32, num_experts=8, block_size=16)
        sorted_ids, expert_ids, num_post_pad = kernel(topk_ids)
        ```
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def entry_for(cls, call: PermuteAlignCall) -> Entry:
        """One build per routed count, expert count, tile height and device."""
        index = call.device.index if call.device is not None else None
        identity = (call.num_routes, call.num_experts, call.block_size, index)
        return identity, lambda: cls(
            call.num_routes, call.num_experts, call.block_size, device_index=index
        )

    def __init__(
        self,
        numel: int,
        num_experts: int,
        block_size: int,
        config: Optional[dict] = None,
        *,
        device_index: Optional[int] = None,
    ):
        super().__init__(device_index=device_index)
        self.numel = numel
        self.num_experts = num_experts
        self.block_size = block_size

        # Fewer routes than this, with at most a warp of experts, take the small-batch kernel.
        small_numel_threshold = 1024
        self._small_batch_fn = (
            _make_small_batch_kernel(numel, num_experts, block_size)
            if numel < small_numel_threshold and num_experts <= WARP_LANES
            else None
        )
        if self._small_batch_fn is None:
            self._align_fn = _make_align_kernel(numel, num_experts, block_size)
            # The scatter kernel takes at least one route.
            self._scatter_fn = (
                _make_scatter_kernel(numel, num_experts, block_size) if numel else None
            )
        else:
            self._align_fn = None
            self._scatter_fn = None

        self.init_config(config)
        # The align kernel's scan totals whole warps only, so the experts a partial warp holds
        # would drop out of it.
        threads = self.config["threads"]
        if threads % WARP_LANES or not WARP_LANES <= threads <= MAX_BLOCK_THREADS:
            raise ValueError(
                f"threads must be a multiple of {WARP_LANES} in [{WARP_LANES}, "
                f"{MAX_BLOCK_THREADS}], got {threads}"
            )

    @property
    def default_config(self) -> dict:
        # Fewer threads only add rounds to the align kernel's loops.
        return {"threads": MAX_BLOCK_THREADS}

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
        self._require_devices(topk_ids=topk_ids)
        assert topk_ids.numel() == self.numel, (
            f"Expected numel={self.numel}, got {topk_ids.numel()}"
        )

        flat = topk_ids.flatten().contiguous()
        max_padded = self.numel + (self.num_experts + 1) * (self.block_size - 1)
        max_num_blocks = math.ceil(max_padded / self.block_size)
        dev = flat.device

        sorted_token_ids = torch.empty(max_padded, dtype=torch.int32, device=dev)
        expert_ids = torch.empty(max_num_blocks, dtype=torch.int32, device=dev)
        num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device=dev)

        if self._small_batch_fn is not None:
            fn = self._small_batch_fn()
            fn(flat, sorted_token_ids, expert_ids, num_tokens_post_pad)
        else:
            cumsum = torch.empty(self.num_experts + 1, dtype=torch.int32, device=dev)
            threads = self.config["threads"]
            align_fn = self._align_fn(threads)
            align_fn(flat, sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum)
            if self._scatter_fn is not None:
                self._scatter_fn()(flat, sorted_token_ids, cumsum)

        return sorted_token_ids, expert_ids, num_tokens_post_pad
