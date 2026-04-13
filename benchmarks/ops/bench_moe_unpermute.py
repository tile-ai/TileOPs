"""Benchmark for MoeUnpermuteFwdOp.

Baselines:
  - vLLM moe_unpermute (optional): vLLM's CUDA kernel.
  - PyTorch reference: gather + weighted view+sum.

Note: vLLM uses inv_permuted_idx (reverse mapping) while TileOPs uses fwd_idx
(forward mapping); inv_permuted_idx is derived from fwd_idx before benchmarking.

Real model configurations:
  Model              H     K
  Kimi K2          7168   8
  DeepSeek-V3      7168   8
  Qwen3-235B-A22B  7168   8
  Qwen3-30B-A3B    3072   8

Usage:
    conda run -n tileops python -m pytest benchmarks/ops/bench_moe_unpermute.py -vvs
    conda run -n tileops python benchmarks/ops/bench_moe_unpermute.py
"""

from typing import Optional

import pytest
import torch

try:
    from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import moe_unpermute
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.manifest import eval_roofline, load_workloads
from tileops.ops.moe import MoeUnpermuteFwdOp
from workloads.moe import MoeUnpermuteTest

_OP_NAME = "MoeUnpermuteFwdOp"

# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------


class MoeUnpermuteBenchmark(BenchmarkBase):

    _roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            t = self.workload
            elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
            self._roofline_cache = eval_roofline(
                _OP_NAME,
                total_tokens=t.total_tokens,
                top_k=t.top_k,
                hidden_size=t.hidden_size,
                elem_bytes=elem_bytes,
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
                w["total_tokens"], w["top_k"], w["hidden_size"],
                id=f"{label}-{dtype_str}",
            ))
    return params


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "total_tokens, top_k, hidden_size",
    _manifest_params(),
)
def test_moe_unpermute_bench(total_tokens: int, top_k: int, hidden_size: int) -> None:
    dtype = torch.bfloat16
    test = MoeUnpermuteTest(total_tokens, top_k, hidden_size, dtype)
    bm = MoeUnpermuteBenchmark(test)
    mm2_pad, fwd_idx, topk_weights = test.gen_inputs()

    # TileOPs
    op = MoeUnpermuteFwdOp(total_tokens, top_k, hidden_size, dtype)
    op(mm2_pad, fwd_idx, topk_weights)  # warmup / JIT compile
    torch.cuda.synchronize()

    result = bm.profile(op, mm2_pad, fwd_idx, topk_weights)
    BenchmarkReport.record("moe_unpermute", locals(), result, tag="tileops")

    # vLLM baseline (optional)
    if _VLLM_AVAILABLE:
        # vLLM uses inv_permuted_idx (reverse mapping: padded_slot -> flat_idx)
        # Compute from fwd_idx (forward mapping: flat_idx -> padded_slot)
        numel = total_tokens * top_k
        inv_permuted_idx = torch.empty(numel, dtype=torch.int32, device=fwd_idx.device)
        inv_permuted_idx[fwd_idx.long()] = torch.arange(numel, dtype=torch.int32, device=fwd_idx.device)
        out_vllm = torch.empty(total_tokens, hidden_size, dtype=mm2_pad.dtype, device=mm2_pad.device)

        def _vllm_fn(mm2_pad, fwd_idx, topk_weights):
            moe_unpermute(out_vllm, mm2_pad, topk_weights, inv_permuted_idx)
            return out_vllm

        _vllm_fn(mm2_pad, fwd_idx, topk_weights)  # warmup
        torch.cuda.synchronize()

        result_vllm = bm.profile(_vllm_fn, mm2_pad, fwd_idx, topk_weights)
        BenchmarkReport.record("moe_unpermute", locals(), result_vllm, tag="vllm")
    else:
        # Fallback: PyTorch vectorized baseline (gather + weighted sum)
        fwd_idx_long = fwd_idx.long()
        topk_weights_f32 = topk_weights.float()

        def _torch_fn(mm2_pad, fwd_idx, topk_weights):
            gathered = mm2_pad[fwd_idx_long].float()                  # [T*K, H]
            weighted_sum = (gathered.view(total_tokens, top_k, hidden_size)
                            * topk_weights_f32.unsqueeze(-1)).sum(dim=1)  # [T, H]
            return weighted_sum.to(mm2_pad.dtype)

        _torch_fn(mm2_pad, fwd_idx, topk_weights)  # warmup
        torch.cuda.synchronize()

        result_torch = bm.profile(_torch_fn, mm2_pad, fwd_idx, topk_weights)
        BenchmarkReport.record("moe_unpermute", locals(), result_torch, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
