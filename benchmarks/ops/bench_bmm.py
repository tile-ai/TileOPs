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
from tileops.ops import BmmFp8KNFwdOp, BmmFp8NKFwdOp, BmmFwdOp
from workloads.bmm import BmmFp8Workload, BmmWorkload

_OP_NAME = "BmmFwdOp"
_FP8_KN_OP_NAME = "BmmFp8KNFwdOp"
_FP8_NK_OP_NAME = "BmmFp8NKFwdOp"


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


@pytest.mark.parametrize(
    "batch, m, n, k, dtype",
    workload_params(load_workloads(_OP_NAME), fields("b", "m", "n", "k", dtype_last=True)),
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
        params=locals(),
    )


@pytest.mark.parametrize(
    "batch, m, n, k, dtype",
    workload_params(load_workloads(_FP8_KN_OP_NAME), fields("b", "m", "n", "k", dtype_last=True)),
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

    op = BmmFp8KNFwdOp(out_dtype=out_dtype, tune=True)
    bm = ManifestBenchmark(op, workload)
    functors = {
        "tileops": (op, (a, b_kn, scale_a, scale_b)),
        "torch-fp32-ref": (workload.torch_fp32_bmm_ref, (a, b_kn, scale_a, scale_b)),
    }

    def flashinfer_fn(a_, b_, sa_, sb_):
        return _flashinfer_bmm_fp8_per_tensor_ref(workload, a_, b_, sa_, sb_)

    # b_kn is already [B, K, N], the order flashinfer's bmm_fp8 reads.
    try:
        flashinfer_fn(a, b_kn, scale_a, scale_b)
    except (ImportError, RuntimeError) as exc:
        print(f"  [skip] flashinfer-bmm-fp8: {exc}")
    else:
        functors["flashinfer-bmm-fp8"] = (flashinfer_fn, (a, b_kn, scale_a, scale_b))

    bm.compare(functors, params=locals())


@pytest.mark.parametrize(
    "batch, m, n, k, dtype",
    workload_params(load_workloads(_FP8_NK_OP_NAME), fields("b", "m", "n", "k", dtype_last=True)),
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

    # Fast path: feed [B, N, K] (K-innermost) using BmmFp8NKFwdOp.
    op = BmmFp8NKFwdOp(out_dtype=out_dtype, tune=True)
    bm = ManifestBenchmark(op, workload)
    functors = {
        "tileops": (op, (a, b_nk, scale_a, scale_b)),
        "torch-fp32-ref": (workload.torch_fp32_bmm_ref, (a, b_kn, scale_a, scale_b)),
    }

    def flashinfer_fn(a_, b_, sa_, sb_):
        return _flashinfer_bmm_fp8_per_tensor_ref(workload, a_, b_, sa_, sb_)

    # flashinfer is optional and shape-sensitive. Probe it once and drop only
    # its row when it cannot run; skipping the case would take the op's own
    # numbers down with it.
    try:
        flashinfer_fn(a, b_kmajor, scale_a, scale_b)
    except (ImportError, RuntimeError) as exc:
        print(f"  [skip] flashinfer-bmm-fp8: {exc}")
    else:
        functors["flashinfer-bmm-fp8"] = (flashinfer_fn, (a, b_kmajor, scale_a, scale_b))

    bm.compare(functors, params=locals())
