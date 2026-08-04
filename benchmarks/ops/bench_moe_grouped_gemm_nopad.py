"""Benchmark for MoeGroupedGemmNopadFwdOp (tight, no-pad grouped GEMM).

Baseline:
  - PyTorch reference: per-expert NT matmul loop (`a_e @ b[e].T`).

Workload shapes come from the manifest entry's `workloads` (via
`load_workloads`); the benchmark reports TileOPs latency alongside the
manifest-derived roofline (`op.eval_roofline()`).
"""

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.moe import MoeGroupedGemmNopadFwdOp
from workloads.moe import MoeGroupedGemmNopadWorkload

_OP_NAME = "MoeGroupedGemmNopadFwdOp"


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
    workload = MoeGroupedGemmNopadWorkload(numel, num_experts, n, k, dtype)
    a, b, true_sizes, true_offsets = workload.gen_inputs()

    op = MoeGroupedGemmNopadFwdOp(numel, num_experts, n, k)
    bm = ManifestBenchmark(_OP_NAME, op, workload)

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
