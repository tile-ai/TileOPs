"""Benchmark for MoePermuteAlignOp vs Triton and sgl-kernel baselines.

Baselines:
  - Triton: adapted from SGLang's moe_align_block_size (4-stage fallback)
      sglang/sgl-kernel/benchmark/bench_moe_align_block_size.py
  - sgl-kernel (optional): SGLang's production CUDA kernel; only runs when
      sgl_kernel is installed (`pip install sgl-kernel`).

Usage:
    conda run -n tileops python -m pytest benchmarks/ops/bench_moe_permute_align.py -vvs
    conda run -n tileops python benchmarks/ops/bench_moe_permute_align.py
"""

import math
from typing import Optional

import pytest
import torch
import triton
import triton.language as tl

try:
    from sgl_kernel import moe_align_block_size as _sgl_moe_align_block_size
    _SGL_KERNEL_AVAILABLE = True
except ImportError:
    _SGL_KERNEL_AVAILABLE = False

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport, _shared_cupti_session
from tests.ops.test_moe_permute_align import MoePermuteAlignTest
from tests.test_base import FixtureBase
from tileops.ops.moe import MoePermuteAlignOp

# ---------------------------------------------------------------------------
# CUPTI warmup fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def warmup_cupti():
    """Pre-initialize the CUPTI profiler once per session.

    The first torch.profiler.profile() call with CUDA activity tracking
    incurs a one-time initialization cost.  If this happens inside do_bench's
    estimation phase, estimate_ms is inflated and n_repeat is computed as 1,
    causing the measured latency to include initialization overhead.
    """
    if True:  # bench_kernel manages its own profiler; no external warmup needed
        return
    dummy = torch.empty(1, device="cuda")
    schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=schedule,
    ) as prof:
        for _ in range(2):
            dummy.zero_()
            prof.step()
    torch.cuda.synchronize()


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
# Benchmark fixture (production-scale configs)
# ---------------------------------------------------------------------------


class MoePermuteAlignBenchFixture(FixtureBase):
    """Production-scale configs for throughput benchmarking.

    Columns: total_tokens, top_k, num_experts, block_size
    """
    PARAMS = [
        ("total_tokens, top_k, num_experts, block_size", [
            # ── Mixtral-8x7B style: 8 experts, top_k=2 ──────────────────────
            (512,  2,   8,   16),
            (2048, 2,   8,   16),
            (4096, 2,   8,   16),
            # ── DeepSeek-MoE style: 64 experts, top_k=6 ─────────────────────
            (512,  6,  64,   64),
            (2048, 6,  64,   64),
            (4096, 6,  64,  128),
            (8192, 6,  64,  128),
            # ── Large MoE: 256 experts, top_k=6 ─────────────────────────────
            (2048, 6, 256,  128),
            (8192, 6, 256,  128),
        ]),
    ]


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
# Benchmark test
# ---------------------------------------------------------------------------


@MoePermuteAlignBenchFixture
def test_permute_align_bench(
    total_tokens: int, top_k: int, num_experts: int, block_size: int
) -> None:
    numel = total_tokens * top_k
    test = MoePermuteAlignTest(total_tokens, top_k, num_experts, block_size)
    bm = MoePermuteAlignBenchmark(test)
    inputs = test.gen_inputs()

    # TileOPs
    op = MoePermuteAlignOp(numel, num_experts, block_size)

    # Warmup: trigger JIT compilation before timed profiling
    op(*inputs)
    torch.cuda.synchronize()

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # Triton baseline
    dev = inputs[0].device
    max_padded = numel + (num_experts + 1) * (block_size - 1)
    max_num_blocks = math.ceil(max_padded / block_size)

    def _triton_fn(topk_ids):
        sorted_ids = torch.empty(max_padded, dtype=torch.int32, device=dev)
        sorted_ids.fill_(numel)
        expert_ids = torch.empty(max_num_blocks, dtype=torch.int32, device=dev)
        num_post_pad = torch.empty(1, dtype=torch.int32, device=dev)
        _triton_permute_align(topk_ids, num_experts, block_size,
                              sorted_ids, expert_ids, num_post_pad)
        return sorted_ids, expert_ids, num_post_pad

    # Warmup Triton baseline
    _triton_fn(*inputs)
    torch.cuda.synchronize()

    result_bl = bm.profile(_triton_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="triton")

    # sgl-kernel baseline (optional — only runs when sgl_kernel is installed)
    if _SGL_KERNEL_AVAILABLE:
        def _sgl_fn(topk_ids):
            sorted_ids = torch.empty(max_padded, dtype=torch.int32, device=dev)
            sorted_ids.fill_(numel)
            expert_ids = torch.empty(max_num_blocks, dtype=torch.int32, device=dev)
            num_post_pad = torch.empty(1, dtype=torch.int32, device=dev)
            cumsum_buf = torch.empty(num_experts + 1, dtype=torch.int32, device=dev)
            _sgl_moe_align_block_size(topk_ids, num_experts, block_size,
                                      sorted_ids, expert_ids, num_post_pad,
                                      cumsum_buf)
            return sorted_ids, expert_ids, num_post_pad

        # Warmup sgl-kernel baseline
        _sgl_fn(*inputs)
        torch.cuda.synchronize()

        result_sgl = bm.profile(_sgl_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_sgl, tag="sgl-kernel")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
