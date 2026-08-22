"""Benchmark for MoeGroupedGemmNopadFwdOp (tight, no-pad grouped GEMM).

Baseline:
  - PyTorch reference: per-expert NT matmul loop (`a_e @ b[e].T`).

Workload shapes come from the manifest entry's `workloads` (via
`load_workloads`); the benchmark reports TileOPs latency alongside the
manifest-derived roofline (`op.eval_roofline()`).
"""

import pytest
import torch

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import ManifestBenchmark, fields, workload_params
from tileops.manifest import load_workloads
from tileops.ops.moe import MoeGroupedGemmNopadFwdOp
from workloads.moe import MoeGroupedGemmNopadWorkload

_OP_NAME = "MoeGroupedGemmNopadFwdOp"


@pytest.mark.parametrize(
    "numel, num_experts, n, k, dtype",
    workload_params(
        load_workloads(_OP_NAME), fields("numel", "num_experts", "n", "k", dtype_last=True)
    ),
)
def test_moe_grouped_gemm_nopad_bench(
    numel: int,
    num_experts: int,
    n: int,
    k: int,
    dtype: torch.dtype,
) -> None:
    workload = MoeGroupedGemmNopadWorkload(numel, num_experts, n, k, dtype)
    a, b, true_sizes, true_offsets = workload.gen_inputs()

    op = MoeGroupedGemmNopadFwdOp(numel, num_experts, n, k)
    bm = ManifestBenchmark(_OP_NAME, op, workload)

    # Warmup: trigger JIT compilation before timed profiling.
    op(a, b, true_sizes, true_offsets)
    torch.cuda.synchronize()

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
            out[off_e : off_e + size_e] = a[off_e : off_e + size_e] @ b[e].T
        return out

    _torch_fn(a, b, true_sizes, true_offsets)  # warmup
    torch.cuda.synchronize()

    bm.compare(
        {
            "tileops": op,
            "torch-ref": _torch_fn,
            TORCH_COMPILE_TAG: compiled_reference(_torch_fn),
        },
        a,
        b,
        true_sizes,
        true_offsets,
        record_as=op,
        params=locals(),
    )
