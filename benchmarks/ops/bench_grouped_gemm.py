"""Benchmarks for the grouped GEMM op.

Workload shapes, dtypes, and transpose layouts come from the ops manifest;
per-variant roofline FLOP and byte counts come from the op's
``eval_roofline()`` via :class:`ManifestBenchmark`.
"""

import pytest
import torch

from benchmarks.baselines import (
    QUACK_TAG,
    TORCH_COMPILE_TAG,
    compiled_reference,
    quack_op,
)
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops import GroupedGemmFwdOp
from workloads.gemm import (
    GroupedGemmWorkload,
)


def _torch_grouped_mm(workload: GroupedGemmWorkload, inputs: tuple):
    """``torch._grouped_mm`` over the same groups, or None where it cannot take them.

    Reads B as ``[groups, K, N]`` and takes cumulative group ends, both built here
    rather than inside the timed callable.
    """
    if not hasattr(torch, "_grouped_mm"):
        return None
    if workload.transpose_a and workload.transpose_b:
        return None

    sizes = torch.tensor(workload.batch_sizes_list, device=inputs[0].device, dtype=torch.int32)
    offsets = torch.cumsum(sizes, dim=0).to(torch.int32)

    if workload.transpose_a:

        def fn(a, b, *_):
            return torch._grouped_mm(a.t(), b, offs=offsets)
    elif workload.transpose_b:
        b_kn = inputs[1].transpose(1, 2).contiguous()

        def fn(a, _b, *_):
            return torch._grouped_mm(a, b_kn, offs=offsets)
    else:

        def fn(a, b, *_):
            return torch._grouped_mm(a, b, offs=offsets)

    return fn


def _compiled_grouped_mm(workload: GroupedGemmWorkload, inputs: tuple):
    """Specialize the compiled baseline to this workload's fixed group boundaries."""
    sizes = inputs[2].tolist()
    starts = inputs[3].tolist()
    bounds = tuple((start, start + size) for start, size in zip(starts, sizes, strict=True))

    def fn(a, b, *_):
        outputs = []
        for i, (start, end) in enumerate(bounds):
            if workload.transpose_a:
                rhs = b[:, start:end].t() if workload.transpose_b else b[start:end]
                outputs.append(torch.mm(a[start:end].t(), rhs))
            else:
                rhs = b[i].t().contiguous() if workload.transpose_b else b[i]
                outputs.append(torch.mm(a[start:end], rhs))
        return torch.stack(outputs) if workload.transpose_a else torch.cat(outputs)

    return compiled_reference(fn)


@pytest.mark.parametrize("call", manifest_calls(GroupedGemmFwdOp))
def test_grouped_gemm_bench(call) -> None:
    workload = GroupedGemmWorkload.from_call(call)
    inputs = workload.gen_inputs()

    op = GroupedGemmFwdOp(**call.arguments({}), tune=True)
    bm = ManifestBenchmark(op, workload)

    functors = {
        "tileops": op,
        "torch-ref": workload.ref_program,
        TORCH_COMPILE_TAG: _compiled_grouped_mm(workload, inputs),
    }
    grouped_mm_fn = _torch_grouped_mm(workload, inputs)
    if grouped_mm_fn is not None:
        functors["torch"] = grouped_mm_fn
    quack_gemm = quack_op("gemm", "quack.gemm_interface")
    offsets = torch.tensor(
        [0, *torch.tensor(workload.batch_sizes_list).cumsum(0).tolist()],
        device=inputs[0].device,
        dtype=torch.int32,
    )

    def quack_fn(a, b, *_):
        if workload.transpose_a:
            rhs = b.T if workload.transpose_b else b
            return quack_gemm(a.T, rhs, cu_seqlens_k=offsets)
        rhs = b.transpose(-1, -2) if workload.transpose_b else b
        return quack_gemm(a, rhs, cu_seqlens_m=offsets)

    functors[QUACK_TAG] = quack_fn
    bm.compare(functors, *inputs)
