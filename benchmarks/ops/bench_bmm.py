from typing import Optional

import pytest
import torch

from benchmarks.baselines import (
    FLAGGEMS_TAG,
    assert_matches_reference,
    flaggems_op,
    reference_tolerance,
)
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops import BmmFp8FwdOp, BmmFwdOp
from workloads.bmm import BmmFp8Workload, BmmWorkload

# The tolerance tests/ops/test_bmm.py holds the FP8 op to against the same reference.
_FP8_ATOL = 2e-2
_FP8_RTOL = 2e-2


def _flashinfer_bmm_fp8_per_tensor_ref(
    workload: BmmFp8Workload,
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


def _flashinfer_bmm_fp8_row(workload: BmmFp8Workload, *inputs: torch.Tensor) -> Optional[tuple]:
    """The flashinfer entry for this case, or ``None`` when it cannot serve it.

    Preferred, not selected: a flashinfer row that cannot run drops its tag
    rather than failing the case. If flashinfer runs but disagrees with the
    reference, that is a correctness signal and should fail the benchmark.

    Args:
        workload: The case being timed, which states the reference and tolerance.
        *inputs: ``a``, ``b`` as a ``[B, K, N]`` view, ``scale_a``, ``scale_b``, as
            flashinfer takes them.

    Returns:
        A ``(callable, args)`` pair for :meth:`ManifestBenchmark.compare`.
    """

    def run(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor):
        return _flashinfer_bmm_fp8_per_tensor_ref(workload, a, b, sa, sb)

    def reference(a: torch.Tensor, b_kn: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor):
        # The reference takes b in the op's layout; flashinfer takes the [B, K, N] view.
        return workload.ref_program(a, b_kn.transpose(-2, -1) if workload.trans_b else b_kn, sa, sb)

    try:
        assert_matches_reference(
            run,
            reference,
            *inputs,
            # cuDNN accumulates the fp8 products in another order, so agreement is bounded by
            # the fp8 inputs, not the output dtype: fp16 output measured 1.04e-3 off at K=1024.
            atol=_FP8_ATOL,
            rtol=_FP8_RTOL,
        )
    except (ImportError, RuntimeError) as exc:
        print(f"  [skip] flashinfer-bmm-fp8: {str(exc).splitlines()[0]}")
        return None
    except AssertionError as exc:
        raise AssertionError(
            f"flashinfer-bmm-fp8 disagrees with the reference: {str(exc).splitlines()[0]}"
        ) from exc
    return run, inputs


@pytest.mark.parametrize("call", manifest_calls(BmmFwdOp))
def test_bmm_bench(call) -> None:
    workload = BmmWorkload.from_call(call)
    a, b = workload.gen_inputs()

    op = BmmFwdOp(**call.arguments({}), tune=True)
    bm = ManifestBenchmark(op, workload)

    flaggems_bmm = flaggems_op("bmm")
    assert_matches_reference(
        flaggems_bmm, workload.ref_program, a, b, **reference_tolerance(a.dtype)
    )

    bm.compare(
        {
            "tileops": op,
            FLAGGEMS_TAG: flaggems_bmm,
            "torch-cublas": workload.ref_program,
        },
        a,
        b,
    )


@pytest.mark.parametrize("call", manifest_calls(BmmFp8FwdOp))
def test_bmm_fp8_bench(call) -> None:
    """Both orders of ``b``: ``[B, K, N]`` reaches the kernel through a transpose,
    ``[B, N, K]`` (``trans_b``) lies K-innermost already."""
    workload = BmmFp8Workload.from_call(call)
    a, b, scale_a, scale_b = workload.gen_inputs()
    # The [B, K, N] logical view of b: a copy of the row-major operand, or a zero-copy
    # view of the K-innermost one, which is flashinfer's column-major contract.
    b_kn = b.transpose(-2, -1) if workload.trans_b else b

    op = BmmFp8FwdOp(**call.arguments({}), tune=True)
    bm = ManifestBenchmark(op, workload)
    functors = {
        "tileops": (op, (a, b, scale_a, scale_b)),
        "torch-fp32-ref": (workload.ref_program, (a, b, scale_a, scale_b)),
    }

    row = _flashinfer_bmm_fp8_row(workload, a, b_kn, scale_a, scale_b)
    if row is not None:
        functors["flashinfer-bmm-fp8"] = row

    bm.compare(functors)
