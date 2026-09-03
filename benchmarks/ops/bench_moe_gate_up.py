"""Benchmark for MoeGateUpFwdOp (gate/up GEMM and its gated activation).

Baselines:
  - `tileops-fused-act` and `tileops-separate-act`: both implementations of this
    role, built directly, so the region between them can be re-taken on another
    device or dtype. `tileops` is whichever of them the op selects.
  - `torch-ref`: per-expert NT matmul loop, then silu_and_mul.

Workload shapes come from the manifest entry's `workloads` (via `load_workloads`);
the benchmark reports TileOPs latency alongside the manifest-derived roofline
(`op.eval_roofline()`).
"""

import pytest
import torch
import torch.nn.functional as F

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import ManifestBenchmark, fields, workload_params
from tileops.kernels.moe import (
    MoeGroupedGemmPersistent3WGFusedActKernel,
    MoeGroupedGemmSeparateActKernel,
)
from tileops.manifest import load_workloads
from tileops.ops.moe.routed_expert.gate_up import MoeGateUpFwdOp
from workloads.moe import MoeGroupedGemmNopadWorkload


@pytest.mark.parametrize(
    "numel, num_experts, ffn, k, dtype",
    workload_params(
        load_workloads(MoeGateUpFwdOp), fields("numel", "num_experts", "ffn", "k", dtype_last=True)
    ),
)
def test_moe_gate_up_bench(
    numel: int,
    num_experts: int,
    ffn: int,
    k: int,
    dtype: torch.dtype,
) -> None:
    cap = torch.cuda.get_device_capability()
    if cap[0] < 9:
        pytest.skip(f"SM90 required for the 3WG fused-activation kernel; got SM{cap[0]}{cap[1]}.")

    workload = MoeGroupedGemmNopadWorkload(numel, num_experts, 2 * ffn, k, dtype)
    a, b, true_sizes, true_offsets = workload.gen_inputs()

    op = MoeGateUpFwdOp(numel, num_experts, ffn, k)
    bm = ManifestBenchmark(op, workload)

    op(a, b, true_sizes, true_offsets)
    torch.cuda.synchronize()

    sizes_l = true_sizes.tolist()
    offsets_l = true_offsets.tolist()

    def _torch_fn(a, b, true_sizes, true_offsets):
        out = torch.empty(numel, ffn, dtype=dtype, device=a.device)
        for e in range(num_experts):
            size_e = sizes_l[e]
            if size_e == 0:
                continue
            off_e = offsets_l[e]
            gate_up = a[off_e : off_e + size_e] @ b[e].T
            out[off_e : off_e + size_e] = F.silu(gate_up[:, :ffn]) * gate_up[:, ffn:]
        return out

    _torch_fn(a, b, true_sizes, true_offsets)  # warmup
    torch.cuda.synchronize()

    # Both implementations of this role, so the region between them stays measurable.
    forced = {}
    for tag, cls in (
        ("tileops-fused-act", MoeGroupedGemmPersistent3WGFusedActKernel),
        ("tileops-separate-act", MoeGroupedGemmSeparateActKernel),
    ):
        kernel = cls(numel, num_experts, ffn, k, dtype=dtype, activation="silu_and_mul")
        kernel(a, b, true_sizes, true_offsets)  # warmup
        torch.cuda.synchronize()
        forced[tag] = kernel

    bm.compare(
        {
            "tileops": op,
            **forced,
            "torch-ref": _torch_fn,
            TORCH_COMPILE_TAG: compiled_reference(_torch_fn),
        },
        a,
        b,
        true_sizes,
        true_offsets,
    )
