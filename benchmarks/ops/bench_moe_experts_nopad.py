"""Benchmark for MoEExpertsNopadFwdOp and MoEExpertsPaddedFwdOp (expert GEMM layer only).

Measures the permute + grouped-GEMM + unpermute pipeline without routing.
Both nopad and padded layouts are benchmarked side-by-side.

Workloads match the manifest entries for MoEExpertsNopadFwdOp and
MoEExpertsPaddedFwdOp (shared workload set):

  Model              T     H     F     E    K
  Qwen3-235B-A22B   512  7168  2048  128   8   (decode)
  Qwen3-235B-A22B  4096  7168  2048  128   8   (prefill)
  DeepSeek-V3       512  7168  2048  256   8   (decode)
  DeepSeek-V3      4096  7168  2048  256   8   (prefill)

Usage:
    conda run -n tileops python -m pytest benchmarks/ops/bench_moe_experts_nopad.py -vvs
"""

from typing import Optional

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.manifest import load_workloads
from tileops.ops.moe import MoEExpertsNopadFwdOp, MoEExpertsPaddedFwdOp
from workloads.workload_base import WorkloadBase

_OP_NAME = "MoEExpertsNopadFwdOp"


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------


class MoEExpertsTest(WorkloadBase):
    def __init__(self, num_tokens, num_experts, top_k, hidden_size, ffn_size, dtype):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.dtype = dtype

    def gen_inputs(self):
        torch.manual_seed(42)
        dev = "cuda"
        hidden = torch.randn(self.num_tokens, self.hidden_size, dtype=self.dtype, device=dev)
        w1 = torch.randn(self.num_experts, self.ffn_size * 2, self.hidden_size, dtype=self.dtype, device=dev) * 0.02
        w2 = torch.randn(self.num_experts, self.hidden_size, self.ffn_size, dtype=self.dtype, device=dev) * 0.02
        topk_weights = torch.softmax(
            torch.randn(self.num_tokens, self.top_k, dtype=torch.float32, device=dev), dim=-1
        )
        topk_ids = torch.randint(0, self.num_experts, (self.num_tokens, self.top_k), dtype=torch.int32, device=dev)
        return hidden, w1, w2, topk_weights, topk_ids

    def ref_program(self, *args):
        return None


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------


class MoEExpertsBenchmark(BenchmarkBase[MoEExpertsTest]):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        return t.num_tokens * t.top_k * 6 * t.ffn_size * t.hidden_size

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        elem = 2  # bfloat16
        weights = t.num_experts * 3 * t.ffn_size * t.hidden_size * elem
        tokens = 2 * t.num_tokens * t.hidden_size * elem
        return weights + tokens


# ---------------------------------------------------------------------------
# Manifest-driven parametrize
# ---------------------------------------------------------------------------


def _manifest_params():
    params = []
    for w in load_workloads(_OP_NAME):
        label = w.get("label", "unlabeled")
        for dtype_str in w["dtypes"]:
            params.append(pytest.param(
                w["num_tokens"], w["num_experts"], w["top_k"],
                w["hidden_size"], w["ffn_size"],
                id=f"{label}-{dtype_str}",
            ))
    return params


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_tokens, num_experts, top_k, hidden_size, ffn_size",
    _manifest_params(),
)
def test_moe_experts_nopad_bench(
    num_tokens: int, num_experts: int, top_k: int, hidden_size: int, ffn_size: int,
) -> None:
    dtype = torch.bfloat16
    test = MoEExpertsTest(num_tokens, num_experts, top_k, hidden_size, ffn_size, dtype)
    hidden, w1, w2, topk_weights, topk_ids = test.gen_inputs()
    bm = MoEExpertsBenchmark(test)

    kwargs = dict(
        num_tokens=num_tokens, num_experts=num_experts, top_k=top_k,
        hidden_size=hidden_size, ffn_size=ffn_size, dtype=dtype,
    )
    output = torch.empty(num_tokens, hidden_size, dtype=dtype, device="cuda")
    ws1 = torch.empty(0, device="cuda")
    ws2 = torch.empty(0, device="cuda")

    # -- TileOPs nopad --------------------------------------------------------
    nopad = MoEExpertsNopadFwdOp(**kwargs)

    def _nopad_fn(hidden, w1, w2, topk_weights, topk_ids):
        nopad.apply(output, hidden, w1, w2, topk_weights, topk_ids, num_experts, None, ws1, ws2)
        return output

    _nopad_fn(hidden, w1, w2, topk_weights, topk_ids)  # warmup / JIT compile
    torch.cuda.synchronize()

    result = bm.profile(_nopad_fn, hidden, w1, w2, topk_weights, topk_ids)
    BenchmarkReport.record(nopad, locals(), result, tag="tileops-nopad")

    # -- TileOPs padded -------------------------------------------------------
    padded = MoEExpertsPaddedFwdOp(**kwargs)

    def _padded_fn(hidden, w1, w2, topk_weights, topk_ids):
        padded.apply(output, hidden, w1, w2, topk_weights, topk_ids, num_experts, None, ws1, ws2)
        return output

    _padded_fn(hidden, w1, w2, topk_weights, topk_ids)  # warmup / JIT compile
    torch.cuda.synchronize()

    result_pad = bm.profile(_padded_fn, hidden, w1, w2, topk_weights, topk_ids)
    BenchmarkReport.record(padded, locals(), result_pad, tag="tileops-padded")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
