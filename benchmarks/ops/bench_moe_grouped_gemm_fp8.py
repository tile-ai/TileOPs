"""Benchmark the block-scaled FP8 masked M-grouped GEMM against DeepGEMM's masked kernel.

DeepGEMM is the only library baseline. FlashInfer's groupwise FP8 grouped GEMM has kernels
only for SM100 and SM12x; on SM90 it goes through cutile, which copies ``m_indptr`` to the
host (a synchronisation inside the timed call) and has no masked layout, so H200 has no
second baseline for these rows.

Both are checked against the workload's reference on each expert's valid rows only, which is
all the op defines; the rows past a count are undefined for either.
"""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import DEEPGEMM_TAG, deepgemm_op
from tileops.ops.moe import MoEGroupedGemmFP8FwdOp


def _deepgemm_masked(case) -> bench.Implementation:
    """``deep_gemm.m_grouped_fp8_gemm_nt_masked`` on the case's tensors.

    Its SM90 kernel reads ``a_scale`` M-major with TMA-aligned rows, and converts any other
    layout on every call; the conversion is done once here, outside the timer, as a caller
    holding scales in that layout would. ``expected_m`` is the rounded mean valid count, the
    hint its tile heuristic takes, read once off the metadata. The output is a private
    buffer the call overwrites.
    """
    import deep_gemm

    gemm = deepgemm_op("m_grouped_fp8_gemm_nt_masked")
    a, a_scale, b, b_scale, masked_m = case.inputs
    a_scale_mn_major = deep_gemm.get_mn_major_tma_aligned_tensor(a_scale)
    expected_m = round(masked_m.float().mean().item())
    out = torch.empty(*a.shape[:-1], b.shape[1], dtype=torch.bfloat16, device=a.device)

    def run(a_, a_scale_, b_, b_scale_, masked_m_):
        gemm((a_, a_scale_), (b_, b_scale_), out, masked_m_, expected_m)
        return out

    return bench.Implementation(run=run, args=(a, a_scale_mn_major, b, b_scale, masked_m))


@pytest.mark.parametrize("case", bench.cases(MoEGroupedGemmFP8FwdOp), ids=lambda case: case.id)
def test_moe_grouped_gemm_fp8_bench(case) -> None:
    op = MoEGroupedGemmFP8FwdOp(**case.arguments)
    bench.Runner(op, case).compare({DEEPGEMM_TAG: _deepgemm_masked(case)})
