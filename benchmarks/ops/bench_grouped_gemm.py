"""Benchmarks for the grouped GEMM op.

Workload shapes, dtypes, and transpose layouts come from the ops manifest;
per-variant roofline FLOP and byte counts come from the op's
``eval_roofline()`` via :class:`ManifestBenchmark`. The composed
forward+backward case keeps a local roofline because it aggregates four
GEMM launches, which no single manifest workload describes.
"""

import pytest
import torch

from benchmarks.baselines import (
    TORCH_COMPILE_TAG,
    assert_matches_reference,
    compiled_reference,
    reference_tolerance,
)
from benchmarks.benchmark_base import (
    ManifestBenchmark,
    fields,
    workload_params,
)
from tileops.manifest import load_workloads
from tileops.ops import GroupedGemmFwdOp
from workloads.grouped_gemm import (
    GroupedGemmWorkload,
)

# Autotuning is a bench-run policy, not a workload property; manifest
# workloads do not carry it.
_TUNE = True


# Test functions

_GROUPED_GEMM_OP = "GroupedGemmFwdOp"
_GROUPED_GEMM_PARAMS = workload_params(
    load_workloads(_GROUPED_GEMM_OP),
    fields("batch_sum", "batch_count", "n", "k", "dtype", "transpose_a", "transpose_b"),
    smoke_first=True,
)


def _torch_grouped_mm(test: GroupedGemmWorkload, inputs: tuple):
    """``torch._grouped_mm`` over the same groups, or None where it cannot take them.

    Reads B as ``[groups, K, N]`` and takes cumulative group ends, both built here
    rather than inside the timed callable.
    """
    if not hasattr(torch, "_grouped_mm"):
        return None
    if test.transpose_a and test.transpose_b:
        return None

    sizes = torch.tensor(test.batch_sizes_list, device=inputs[0].device, dtype=torch.int32)
    offsets = torch.cumsum(sizes, dim=0).to(torch.int32)

    if test.transpose_a:

        def fn(a, b, *_):
            return torch._grouped_mm(a.t(), b, offs=offsets)
    elif test.transpose_b:
        b_kn = inputs[1].transpose(1, 2).contiguous()

        def fn(a, _b, *_):
            return torch._grouped_mm(a, b_kn, offs=offsets)
    else:

        def fn(a, b, *_):
            return torch._grouped_mm(a, b, offs=offsets)

    return fn


@pytest.mark.parametrize(
    "batch_sum, batch_count, N, K, dtype, transpose_a, transpose_b",
    _GROUPED_GEMM_PARAMS,
)
def test_grouped_gemm_bench(
    batch_sum: int,
    batch_count: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    transpose_a: bool,
    transpose_b: bool,
) -> None:
    layout = ("T" if transpose_a else "N") + ("T" if transpose_b else "N")

    test = GroupedGemmWorkload(batch_sum, batch_count, N, K, dtype, transpose_a, transpose_b)
    inputs = test.gen_inputs()

    op = GroupedGemmFwdOp(transpose_a=transpose_a, transpose_b=transpose_b, tune=_TUNE)
    bm = ManifestBenchmark(op, test)

    functors = {
        "tileops": op,
        "torch-ref": test.ref_program,
        TORCH_COMPILE_TAG: compiled_reference(test.ref_program),
    }
    grouped_mm_fn = _torch_grouped_mm(test, inputs)
    if grouped_mm_fn is not None:
        assert_matches_reference(
            grouped_mm_fn, test.ref_program, *inputs, **reference_tolerance(dtype)
        )
        functors["torch"] = grouped_mm_fn
    # Rows are named by the op, with the layout among their params: a row named
    # for the layout leaves the op it measured out of the report.
    bm.compare(functors, *inputs, record_as=op, params=locals())
