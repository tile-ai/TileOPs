from typing import Optional

import pytest
import torch

from benchmarks.baselines import (
    FLAGGEMS_TAG,
    assert_matches_reference,
    flaggems_op,
    reference_tolerance,
)
from benchmarks.benchmark_base import ManifestBenchmark, fields, workload_params
from tileops.manifest import load_workloads
from tileops.ops import BmmFp8FwdOp, BmmFwdOp
from workloads.bmm import BmmFp8Workload, BmmWorkload


class BmmFp8BenchmarkWorkload(BmmFp8Workload):
    def torch_fp32_bmm_ref(self, *inputs: torch.Tensor) -> torch.Tensor:
        a, b, scale_a, scale_b = inputs
        if scale_a.dim() != 0 or scale_b.dim() != 0:
            raise ValueError(
                "BmmFp8 benchmark baseline requires per-tensor 0-D scales, "
                f"got {tuple(scale_a.shape)} / {tuple(scale_b.shape)}"
            )
        a_f = a.float() * scale_a
        b_f = b.float() * scale_b
        return torch.bmm(a_f, b_f).to(self.out_dtype)


def _flashinfer_bmm_fp8_per_tensor_ref(
    workload: BmmFp8BenchmarkWorkload,
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
    workload: BmmFp8BenchmarkWorkload, *inputs: torch.Tensor
) -> Optional[tuple]:
    """The flashinfer entry for this case, or ``None`` when it cannot serve it.

    Preferred, not selected: a flashinfer row that cannot run drops its tag
    rather than failing the case. If flashinfer runs but disagrees with the
    reference, that is a correctness signal and should fail the benchmark.

    Args:
        workload: The case being timed, which states the reference and tolerance.
        *inputs: ``a``, ``b``, ``scale_a``, ``scale_b`` as flashinfer takes them.

    Returns:
        A ``(callable, args)`` pair for :meth:`ManifestBenchmark.compare`.
    """

    def run(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor):
        return _flashinfer_bmm_fp8_per_tensor_ref(workload, a, b, sa, sb)

    try:
        assert_matches_reference(
            run,
            workload.torch_fp32_bmm_ref,
            *inputs,
            **reference_tolerance(workload.out_dtype),
        )
    except (ImportError, RuntimeError) as exc:
        print(f"  [skip] flashinfer-bmm-fp8: {str(exc).splitlines()[0]}")
        return None
    except AssertionError as exc:
        raise AssertionError(
            f"flashinfer-bmm-fp8 disagrees with the reference: {str(exc).splitlines()[0]}"
        ) from exc
    return run, inputs


@pytest.mark.parametrize(
    "batch, m, n, k, dtype",
    workload_params(load_workloads(BmmFwdOp), fields("b", "m", "n", "k", dtype_last=True)),
)
def test_bmm_bench(batch: int, m: int, n: int, k: int, dtype: torch.dtype) -> None:
    workload = BmmWorkload(batch, m, n, k, dtype)
    a, b = workload.gen_inputs()

    op = BmmFwdOp(tune=True)
    bm = ManifestBenchmark(op, workload)

    # eval_roofline() is read lazily after profiling, by which point
    # forward() has bound the dims.

    flaggems_bmm = flaggems_op("bmm")
    assert_matches_reference(flaggems_bmm, torch.bmm, a, b, **reference_tolerance(a.dtype))

    bm.compare(
        {
            "tileops": op,
            FLAGGEMS_TAG: flaggems_bmm,
            "torch-cublas": torch.bmm,
        },
        a,
        b,
    )


@pytest.mark.parametrize(
    "batch, m, n, k, dtype",
    workload_params(load_workloads(BmmFp8FwdOp), fields("b", "m", "n", "k", dtype_last=True)),
)
def test_bmm_fp8_kn_bench(
    batch: int,
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
) -> None:
    """The [B, K, N] order, which the kernel reaches through a transpose."""
    out_dtype = torch.bfloat16
    workload = BmmFp8BenchmarkWorkload(batch, m, n, k, dtype, out_dtype=out_dtype)
    a, b_kn, scale_a, scale_b = workload.gen_inputs()

    op = BmmFp8FwdOp(out_dtype=out_dtype, tune=True)
    bm = ManifestBenchmark(op, workload)
    functors = {
        "tileops": (op, (a, b_kn, scale_a, scale_b)),
        "torch-fp32-ref": (workload.torch_fp32_bmm_ref, (a, b_kn, scale_a, scale_b)),
    }

    # b_kn carries [B, K, N] row-major; flashinfer's contract asks for that shape
    # column-major, which is the other bench.
    row = _flashinfer_bmm_fp8_row(workload, a, b_kn, scale_a, scale_b)
    if row is not None:
        functors["flashinfer-bmm-fp8"] = row

    bm.compare(functors)


@pytest.mark.parametrize(
    "batch, m, n, k, dtype",
    workload_params(load_workloads(BmmFp8FwdOp), fields("b", "m", "n", "k", dtype_last=True)),
)
def test_bmm_fp8_nk_bench(
    batch: int,
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
) -> None:
    out_dtype = torch.bfloat16
    workload = BmmFp8BenchmarkWorkload(batch, m, n, k, dtype, out_dtype=out_dtype)
    a, b_kn, scale_a, scale_b = workload.gen_inputs()
    b_nk = b_kn.transpose(-2, -1).contiguous()  # [B, N, K], K-innermost
    b_kmajor = b_nk.transpose(-2, -1)  # [B, K, N] view, zero-copy

    op = BmmFp8FwdOp(out_dtype=out_dtype, trans_b=True, tune=True)
    bm = ManifestBenchmark(op, workload)
    functors = {
        "tileops": (op, (a, b_nk, scale_a, scale_b)),
        "torch-fp32-ref": (workload.torch_fp32_bmm_ref, (a, b_kn, scale_a, scale_b)),
    }

    row = _flashinfer_bmm_fp8_row(workload, a, b_kmajor, scale_a, scale_b)
    if row is not None:
        functors["flashinfer-bmm-fp8"] = row

    bm.compare(functors)
