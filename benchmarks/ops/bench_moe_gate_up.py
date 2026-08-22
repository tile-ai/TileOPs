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
from benchmarks.benchmark_base import ManifestBenchmark
from tileops.kernels.moe import (
    MoeGroupedGemmPersistent3WGFusedActKernel,
    MoeGroupedGemmSeparateActKernel,
)
from tileops.manifest import load_workloads
from tileops.ops.moe.routed_expert.gate_up import MoeGateUpFwdOp
from workloads.moe import MoeGroupedGemmNopadWorkload

_OP_NAME = "MoeGateUpFwdOp"

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _manifest_params():
    """Convert manifest workloads to pytest params (numel, E, ffn, K, dtype)."""
    params = []
    for w in load_workloads(_OP_NAME):
        label = w.get("label", "unlabeled")
        for dtype_str in w["dtypes"]:
            params.append(
                pytest.param(
                    w["numel"],
                    w["num_experts"],
                    w["ffn"],
                    w["k"],
                    dtype_str,
                    id=f"{label}-{dtype_str}",
                )
            )
    return params


@pytest.mark.parametrize(
    "numel, num_experts, ffn, k, dtype_str",
    _manifest_params(),
)
def test_moe_gate_up_bench(
    numel: int,
    num_experts: int,
    ffn: int,
    k: int,
    dtype_str: str,
) -> None:
    cap = torch.cuda.get_device_capability()
    if cap[0] < 9:
        pytest.skip(f"SM90 required for the 3WG fused-activation kernel; got SM{cap[0]}{cap[1]}.")

    dtype = _DTYPE_MAP[dtype_str]
    workload = MoeGroupedGemmNopadWorkload(numel, num_experts, 2 * ffn, k, dtype)
    a, b, true_sizes, true_offsets = workload.gen_inputs()

    op = MoeGateUpFwdOp(numel, num_experts, ffn, k)
    bm = ManifestBenchmark(_OP_NAME, op, workload)

    # Warmup: trigger JIT compilation before timed profiling.
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
        record_as=op,
        params=locals(),
    )
