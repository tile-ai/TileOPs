"""Benchmark for MoeGroupedGemmNopadFwdOp (tight, no-pad grouped GEMM).

Baseline:
  - PyTorch reference: per-expert NT matmul loop (`a_e @ b[e].T`).

Workload shapes come from the manifest entry's `workloads` (via
`load_workloads`); the benchmark reports TileOPs latency alongside the
manifest-derived roofline (`op.eval_roofline()`).

Usage:
    conda run -n tileops python -m pytest benchmarks/ops/bench_moe_grouped_gemm_nopad.py -vvs
    conda run -n tileops python benchmarks/ops/bench_moe_grouped_gemm_nopad.py
"""

from typing import Optional

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.manifest import load_workloads
from tileops.ops.moe import MoeGroupedGemmNopadFwdOp
from workloads.workload_base import WorkloadBase

_OP_NAME = "MoeGroupedGemmNopadFwdOp"


class MoeGroupedGemmNopadTest(WorkloadBase):
    """Manifest-shaped inputs: tight A, expert weights B, uniform per-expert sizes."""

    def __init__(self, numel: int, num_experts: int, n: int, k: int, dtype: torch.dtype):
        self.numel = numel
        self.num_experts = num_experts
        self.n = n
        self.k = k
        self.dtype = dtype

    def gen_inputs(self):
        torch.manual_seed(42)
        dev = "cuda"
        base = max(1, self.numel // self.num_experts)
        sizes = torch.full((self.num_experts,), base, dtype=torch.int32, device=dev)
        sizes[-1] = self.numel - base * (self.num_experts - 1)
        offsets = torch.zeros(self.num_experts, dtype=torch.int32, device=dev)
        offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
        a = torch.randn(self.numel, self.k, dtype=self.dtype, device=dev) * 0.02
        b = torch.randn(self.num_experts, self.n, self.k, dtype=self.dtype, device=dev) * 0.02
        return a, b, sizes, offsets

    def ref_program(self, *args):
        return None


class MoeGroupedGemmNopadBenchmark(BenchmarkBase[MoeGroupedGemmNopadTest]):

    _roofline_cache: Optional[tuple[float, float]] = None

    def __init__(self, test, op):
        super().__init__(test)
        self._op = op

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = self._op.eval_roofline()
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _manifest_params():
    """Convert manifest workloads to pytest params (numel, E, N, K, dtype)."""
    params = []
    for w in load_workloads(_OP_NAME):
        label = w.get("label", "unlabeled")
        for dtype_str in w["dtypes"]:
            params.append(pytest.param(
                w["numel"], w["num_experts"], w["n"], w["k"], dtype_str,
                id=f"{label}-{dtype_str}",
            ))
    return params


@pytest.mark.parametrize(
    "numel, num_experts, n, k, dtype_str",
    _manifest_params(),
)
def test_moe_grouped_gemm_nopad_bench(
    numel: int, num_experts: int, n: int, k: int, dtype_str: str,
) -> None:
    dtype = _DTYPE_MAP[dtype_str]
    test = MoeGroupedGemmNopadTest(numel, num_experts, n, k, dtype)
    a, b, true_sizes, true_offsets = test.gen_inputs()

    op = MoeGroupedGemmNopadFwdOp(numel, num_experts, n, k, dtype=dtype)
    bm = MoeGroupedGemmNopadBenchmark(test, op)

    # Warmup: trigger JIT compilation before timed profiling.
    op(a, b, true_sizes, true_offsets)
    torch.cuda.synchronize()

    result = bm.profile(op, a, b, true_sizes, true_offsets)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # PyTorch baseline: per-expert NT matmul.
    sizes_l = true_sizes.tolist()
    offsets_l = true_offsets.tolist()

    def _torch_fn(a, b, true_sizes, true_offsets):
        out = torch.empty(numel, n, dtype=dtype, device=a.device)
        for e in range(num_experts):
            size_e = sizes_l[e]
            if size_e == 0:
                continue
            off_e = offsets_l[e]
            out[off_e:off_e + size_e] = a[off_e:off_e + size_e] @ b[e].T
        return out

    _torch_fn(a, b, true_sizes, true_offsets)  # warmup
    torch.cuda.synchronize()

    result_torch = bm.profile(_torch_fn, a, b, true_sizes, true_offsets)
    BenchmarkReport.record(op, locals(), result_torch, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
