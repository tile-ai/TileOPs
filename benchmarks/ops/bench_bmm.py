"""Benchmark TileOPs batched matmul and its FP8 variant, one case per manifest call, against cuBLAS, FlagGems and FlashInfer."""

from typing import Optional

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import FLAGGEMS_TAG, QUACK_TAG, flaggems_op, quack_op
from tileops.ops import BmmFP8FwdOp, BmmFwdOp
from workloads.gemm import BmmFP8Workload


def _flashinfer_bmm_fp8_per_tensor_ref(
    workload: BmmFP8Workload,
    a: torch.Tensor,
    b_kmajor: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    import flashinfer

    if a.dtype != torch.float8_e4m3fn or b_kmajor.dtype != torch.float8_e4m3fn:
        raise ValueError("FlashInfer bmm_fp8 baseline requires float8_e4m3fn.")
    if workload.out_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("FlashInfer bmm_fp8 baseline requires bfloat16 / float16 output.")
    if scale_a.dim() != 0 or scale_b.dim() != 0:
        raise ValueError(
            "FlashInfer bmm_fp8 baseline requires 0-D per-tensor scales, "
            f"got {tuple(scale_a.shape)} / {tuple(scale_b.shape)}"
        )
    return flashinfer.bmm_fp8(
        a,
        b_kmajor,
        scale_a,
        scale_b,
        dtype=workload.out_dtype,
        backend="cudnn",
    )


def _flashinfer_bmm_fp8_row(
    workload: BmmFP8Workload, *inputs: torch.Tensor
) -> Optional[bench.Implementation]:
    """The flashinfer entry for this case, or ``None`` when it cannot serve it.

    Preferred, not selected: a flashinfer row that cannot run drops its tag
    rather than failing the case. If flashinfer runs but disagrees with the
    reference, that is a correctness signal and should fail the benchmark.

    Args:
        workload: The case's workload.
        *inputs: ``a``, ``b`` as a ``[B, K, N]`` view, ``scale_a``, ``scale_b``, as
            flashinfer takes them.

    Returns:
        The implementation, or ``None`` when flashinfer cannot serve the case.
    """

    def run(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor):
        return _flashinfer_bmm_fp8_per_tensor_ref(workload, a, b, sa, sb)

    try:
        run(*inputs)
    except (ImportError, RuntimeError) as exc:
        print(f"  [skip] flashinfer-bmm-fp8: {str(exc).splitlines()[0]}")
        return None
    return bench.Implementation(run=run, args=inputs)


@pytest.mark.parametrize("case", bench.cases(BmmFwdOp), ids=lambda case: case.id)
def test_bmm_bench(case) -> None:
    op = BmmFwdOp(**case.arguments)
    op.autotune()
    quack_gemm = quack_op("gemm", "quack.gemm_interface")

    def quack_fn(a, b):
        return quack_gemm(a, b)

    bench.Runner(op, case).compare(
        {
            FLAGGEMS_TAG: flaggems_op("bmm"),
            QUACK_TAG: quack_fn,
            "torch-cublas": case.reference,
        }
    )


@pytest.mark.parametrize("case", bench.cases(BmmFP8FwdOp), ids=lambda case: case.id)
def test_bmm_fp8_bench(case) -> None:
    """Both orders of ``b``: ``[B, K, N]`` reaches the kernel through a transpose,
    ``[B, N, K]`` (``trans_b``) lies K-innermost already."""
    workload = case.workload
    a, b, scale_a, scale_b = case.inputs
    b_kn = b.transpose(-2, -1) if workload.trans_b else b
    op = BmmFP8FwdOp(**case.arguments)
    op.autotune()
    implementations = {"torch-fp32-ref": case.reference}
    row = _flashinfer_bmm_fp8_row(workload, a, b_kn, scale_a, scale_b)
    if row is not None:
        implementations["flashinfer-bmm-fp8"] = row
    bench.Runner(op, case).compare(implementations)
