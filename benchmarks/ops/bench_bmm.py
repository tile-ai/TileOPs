"""Benchmark TileOPs batched matmul and its FP8 and W8A16 variants, one case per manifest
call, against cuBLAS, FlagGems and FlashInfer."""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import (
    FLAGGEMS_TAG,
    QUACK_TAG,
    baseline_module,
    flaggems_op,
    quack_op,
)
from tileops.gemm import BMMFP8FwdOp, BMMFwdOp, BMMW8A16FwdOp
from workloads.gemm import BMMFP8Workload, dequantize_block_scaled


def _flashinfer_bmm_fp8_row(
    workload: BMMFP8Workload, *inputs: torch.Tensor
) -> bench.Implementation | None:
    """FlashInfer's cuDNN ``bmm_fp8`` on ``a``, ``b`` as a ``[B, K, N]`` view and the per-tensor
    scales, or ``None`` when it cannot run, which drops its tag; a wrong result fails the case.
    """

    def run(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor):
        flashinfer = baseline_module("flashinfer", "flashinfer")
        return flashinfer.bmm_fp8(a, b, sa, sb, dtype=workload.out_dtype, backend="cudnn")

    try:
        run(*inputs)
    except (ImportError, RuntimeError) as exc:
        print(f"  [skip] flashinfer-bmm-fp8: {str(exc).splitlines()[0]}")
        return None
    return bench.Implementation(run=run, args=inputs)


@pytest.mark.parametrize("case", bench.cases(BMMFwdOp))
def test_bmm_bench(case) -> None:
    op = BMMFwdOp(**case.arguments)
    op.request_tune()
    quack_gemm = quack_op("gemm", "quack.gemm_interface")

    def quack_fn(a, b):
        return quack_gemm(a, b)

    bench.Runner(op, case).compare(
        {
            FLAGGEMS_TAG: flaggems_op("bmm"),
            QUACK_TAG: quack_fn,
            "torch-cublas": case.torch,
        }
    )


@pytest.mark.parametrize("case", bench.cases(BMMFP8FwdOp))
def test_bmm_fp8_bench(case) -> None:
    """Both orders of ``b``: ``[B, K, N]`` reaches the kernel through a transpose,
    ``[B, N, K]`` (``trans_b``) lies K-innermost already."""
    workload = case.workload
    a, b, scale_a, scale_b = case.inputs
    b_kn = b.transpose(-2, -1) if workload.trans_b else b
    op = BMMFP8FwdOp(**case.arguments)
    op.request_tune()
    implementations = {"torch-fp32-ref": case.reference}
    row = _flashinfer_bmm_fp8_row(workload, a, b_kn, scale_a, scale_b)
    if row is not None:
        implementations["flashinfer-bmm-fp8"] = row
    bench.Runner(op, case).compare(implementations)


@pytest.mark.parametrize("case", bench.cases(BMMW8A16FwdOp))
def test_bmm_w8a16_bench(case) -> None:
    """Against cuBLAS's bfloat16 batched matmul on the weight dequantized to bfloat16 once
    beforehand, as DeepSeek-V4-Flash's inference holds ``wo_a``."""
    activation, weight, weight_scale = case.inputs
    dequantized = dequantize_block_scaled(weight, weight_scale).to(torch.bfloat16)
    bench.Runner(BMMW8A16FwdOp(**case.arguments), case).compare(
        {
            "torch-dequantized-matmul": bench.Implementation(
                run=torch.bmm, args=(activation, dequantized.transpose(1, 2).contiguous())
            )
        }
    )
