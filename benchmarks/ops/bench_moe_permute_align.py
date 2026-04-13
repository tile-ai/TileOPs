"""Benchmark for MoePermuteAlignFwdOp vs Triton and sgl-kernel baselines.

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

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.manifest import eval_roofline, load_workloads
from tileops.ops.moe import MoePermuteAlignFwdOp
from workloads.moe import MoePermuteAlignTest

_OP_NAME = "MoePermuteAlignFwdOp"

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

    _roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            t = self.workload
            self._roofline_cache = eval_roofline(
                _OP_NAME,
                total_tokens=t.total_tokens,
                top_k=t.top_k,
                num_experts=t.num_experts,
                block_size=t.block_size,
            )
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


# ---------------------------------------------------------------------------
# Manifest-driven parametrize
# ---------------------------------------------------------------------------


def _manifest_params():
    """Convert manifest workloads to pytest params."""
    params = []
    for w in load_workloads(_OP_NAME):
        label = w.get("label", "unlabeled")
        for dtype_str in w["dtypes"]:
            params.append(pytest.param(
                w["total_tokens"], w["top_k"], w["num_experts"], w["block_size"],
                id=f"{label}-{dtype_str}",
            ))
    return params


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "total_tokens, top_k, num_experts, block_size",
    _manifest_params(),
)
def test_permute_align_bench(
    total_tokens: int, top_k: int, num_experts: int, block_size: int
) -> None:
    numel = total_tokens * top_k
    test = MoePermuteAlignTest(total_tokens, top_k, num_experts, block_size)
    bm = MoePermuteAlignBenchmark(test)
    inputs = test.gen_inputs()

    # TileOPs
    op = MoePermuteAlignFwdOp(numel, num_experts, block_size)

    # Warmup: trigger JIT compilation before timed profiling
    op(*inputs)
    torch.cuda.synchronize()

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # Triton baseline
    dev = inputs[0].device
    max_padded = numel + (num_experts + 1) * (block_size - 1)
    max_num_blocks = math.ceil(max_padded / block_size)

    sorted_ids = torch.empty(max_padded, dtype=torch.int32, device=dev)
    expert_ids = torch.empty(max_num_blocks, dtype=torch.int32, device=dev)
    num_post_pad = torch.empty(1, dtype=torch.int32, device=dev)

    def _triton_fn(topk_ids):
        sorted_ids.fill_(numel)
        _triton_permute_align(topk_ids, num_experts, block_size,
                              sorted_ids, expert_ids, num_post_pad)
        return sorted_ids, expert_ids, num_post_pad

    # Warmup Triton baseline
    _triton_fn(*inputs)
    torch.cuda.synchronize()

    result_bl = bm.profile(_triton_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="triton")

    # sgl-kernel baseline (optional -- only runs when sgl_kernel is installed)
    if _SGL_KERNEL_AVAILABLE:
        sorted_ids_sgl = torch.empty(max_padded, dtype=torch.int32, device=dev)
        expert_ids_sgl = torch.empty(max_num_blocks, dtype=torch.int32, device=dev)
        num_post_pad_sgl = torch.empty(1, dtype=torch.int32, device=dev)
        cumsum_buf = torch.empty(num_experts + 1, dtype=torch.int32, device=dev)

        def _sgl_fn(topk_ids):
            sorted_ids_sgl.fill_(numel)
            _sgl_moe_align_block_size(topk_ids, num_experts, block_size,
                                      sorted_ids_sgl, expert_ids_sgl, num_post_pad_sgl,
                                      cumsum_buf)
            return sorted_ids_sgl, expert_ids_sgl, num_post_pad_sgl

        # Warmup sgl-kernel baseline
        _sgl_fn(*inputs)
        torch.cuda.synchronize()

        result_sgl = bm.profile(_sgl_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_sgl, tag="sgl-kernel")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
