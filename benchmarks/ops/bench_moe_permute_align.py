"""Benchmark for MoePermuteAlignOp vs Triton baseline.

Triton baseline is adapted from SGLang's moe_align_block_size implementation:
  sglang/sgl-kernel/benchmark/bench_moe_align_block_size.py

Usage:
    conda run -n tileops python -m pytest benchmarks/ops/bench_permute_align.py -vvs
    conda run -n tileops python benchmarks/ops/bench_permute_align.py
"""

import math
from typing import Optional, Tuple

import pytest
import torch
import triton
import triton.language as tl

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_moe_permute_align import MoePermuteAlignFixture, MoePermuteAlignTest
from tileops.ops.moe import MoePermuteAlignOp


# ---------------------------------------------------------------------------
# Triton baseline (adapted from SGLang, no sgl_kernel dependency)
# ---------------------------------------------------------------------------


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@triton.jit
def _stage1_count(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = pid * tokens_per_thread
    off_c = (pid + 1) * num_experts
    for i in range(tokens_per_thread):
        if start_idx + i < numel:
            idx = tl.load(topk_ids_ptr + start_idx + i)
            cnt = tl.load(tokens_cnts_ptr + off_c + idx)
            tl.store(tokens_cnts_ptr + off_c + idx, cnt + 1)


@triton.jit
def _stage2_reduce(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tl.program_id(0)
    last = 0
    for i in range(1, num_experts + 1):
        cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
        last = last + cnt
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last)


@triton.jit
def _stage3_cumsum(
    num_tokens_post_pad_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
):
    last = 0
    off = num_experts * num_experts
    for i in range(1, num_experts + 1):
        cnt = tl.load(tokens_cnts_ptr + off + i - 1)
        last = last + tl.cdiv(cnt, block_size) * block_size
        tl.store(cumsum_ptr + i, last)
    tl.store(num_tokens_post_pad_ptr, last)


@triton.jit
def _stage4_scatter(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = tl.load(cumsum_ptr + pid)
    end_idx = tl.load(cumsum_ptr + pid + 1)
    for i in range(start_idx, end_idx, block_size):
        tl.store(expert_ids_ptr + i // block_size, pid)
    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts
    for i in range(start_idx, tl.minimum(start_idx + tokens_per_thread, numel)):
        expert_id = tl.load(topk_ids_ptr + i)
        rank = tl.load(tokens_cnts_ptr + off_t + expert_id)
        slot = rank + tl.load(cumsum_ptr + expert_id)
        tl.store(sorted_token_ids_ptr + slot, i)
        tl.store(tokens_cnts_ptr + off_t + expert_id, rank + 1)


def _triton_permute_align(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    numel = topk_ids.numel()
    tokens_per_thread = _ceil_div(numel, num_experts)
    grid = (num_experts,)
    tokens_cnts = torch.zeros(
        (num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device
    )
    cumsum = torch.zeros(num_experts + 1, dtype=torch.int32, device=topk_ids.device)
    _stage1_count[grid](topk_ids, tokens_cnts, num_experts, numel, tokens_per_thread)
    _stage2_reduce[grid](tokens_cnts, num_experts)
    _stage3_cumsum[(1,)](num_tokens_post_pad, tokens_cnts, cumsum, num_experts, block_size)
    _stage4_scatter[grid](
        topk_ids, sorted_token_ids, expert_ids, tokens_cnts, cumsum,
        num_experts, block_size, numel, tokens_per_thread,
    )


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------


class MoePermuteAlignBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        return None

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        numel = t.total_tokens * t.top_k
        max_padded = numel + (t.num_experts + 1) * (t.block_size - 1)
        max_num_blocks = math.ceil(max_padded / t.block_size)
        # topk_ids read + sorted_token_ids write + expert_ids write + num_post_pad write
        return (numel + max_padded + max_num_blocks + 1) * 4  # int32 = 4 bytes


# ---------------------------------------------------------------------------
# Triton callable wrapper (allocates output tensors fresh each call)
# ---------------------------------------------------------------------------


class _TritonBaseline:
    """Wraps the 4-stage Triton baseline so it can be profiled like an op."""

    def __init__(self, num_experts: int, block_size: int, numel: int):
        self.num_experts = num_experts
        self.block_size = block_size
        max_padded = numel + (num_experts + 1) * (block_size - 1)
        self.max_padded = max_padded
        self.max_num_blocks = math.ceil(max_padded / block_size)
        self.numel = numel

    def __call__(self, topk_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = topk_ids.device
        sorted_ids = torch.empty(self.max_padded, dtype=torch.int32, device=dev)
        sorted_ids.fill_(self.numel)
        expert_ids = torch.empty(self.max_num_blocks, dtype=torch.int32, device=dev)
        num_post_pad = torch.empty(1, dtype=torch.int32, device=dev)
        _triton_permute_align(
            topk_ids, self.num_experts, self.block_size,
            sorted_ids, expert_ids, num_post_pad,
        )
        return sorted_ids, expert_ids, num_post_pad


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


@MoePermuteAlignFixture
def test_permute_align_bench(
    total_tokens: int, top_k: int, num_experts: int, block_size: int
) -> None:
    test = MoePermuteAlignTest(total_tokens, top_k, num_experts, block_size)
    bm = MoePermuteAlignBenchmark(test)
    inputs = test.gen_inputs()
    numel = total_tokens * top_k

    # TileOPs
    op = MoePermuteAlignOp(numel, num_experts, block_size)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("permute_align", locals(), result, tag="tileops")

    # Triton baseline
    triton_fn = _TritonBaseline(num_experts, block_size, numel)
    result_bl = bm.profile(triton_fn, *inputs)
    BenchmarkReport.record("permute_align", locals(), result_bl, tag="triton")

    speedup = result_bl["latency_ms"] / result["latency_ms"]
    print(
        f"\n[{total_tokens}tok, top{top_k}, E={num_experts}, bs={block_size}] "
        f"TileOPs: {result['latency_ms']*1e3:.1f}us  "
        f"Triton: {result_bl['latency_ms']*1e3:.1f}us  "
        f"speedup: {speedup:.2f}x"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
