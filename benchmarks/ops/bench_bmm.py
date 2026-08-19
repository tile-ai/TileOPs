import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops import BmmFp8NKOp, BmmFp8Op, BmmFwdOp
from workloads.bmm import BmmFp8Workload, BmmWorkload

_OP_NAME = "BmmFwdOp"
_FP8_KN_OP_NAME = "BmmFp8Op"
_FP8_OP_NAME = "BmmFp8NKOp"

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}


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
    workload: BmmFp8BenchmarkWorkload, a: torch.Tensor, b_kmajor: torch.Tensor,
    scale_a: torch.Tensor, scale_b: torch.Tensor,
) -> torch.Tensor:

    import flashinfer

    if a.dtype != torch.float8_e4m3fn or b_kmajor.dtype != torch.float8_e4m3fn:
        raise ValueError(
            "FlashInfer bmm_fp8 baseline requires float8_e4m3fn.")
    if workload.out_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            "FlashInfer bmm_fp8 baseline requires bfloat16 / float16 output.")
    if scale_a.dim() != 0 or scale_b.dim() != 0:
        raise ValueError(
            "FlashInfer bmm_fp8 baseline requires 0-D per-tensor scales, "
            f"got {tuple(scale_a.shape)} / {tuple(scale_b.shape)}"
        )
    return flashinfer.bmm_fp8(
        a, b_kmajor, scale_a, scale_b,
        dtype=workload.out_dtype, backend="cudnn",
    )


def _manifest_params() -> list:
    """Convert manifest workloads to pytest params (batch, m, n, k, dtype)."""
    params = []
    for w in load_workloads(_OP_NAME):
        label = w.get("label", "unlabeled")
        for dtype_str in w["dtypes"]:
            params.append(pytest.param(
                w["b"], w["m"], w["n"], w["k"], dtype_str,
                id=f"{label}-{dtype_str}",
            ))
    return params


def _fp8_params(workloads) -> list:
    params = []
    for w in workloads:
        label = w.get("label", "unlabeled")
        for dtype_str in w["dtypes"]:
            params.append(pytest.param(
                w["b"], w["m"], w["n"], w["k"], dtype_str,
                id=f"{label}-{dtype_str}",
            ))
    return params


@pytest.mark.parametrize("batch, m, n, k, dtype_str", _manifest_params())
def test_bmm_bench(batch: int, m: int, n: int, k: int, dtype_str: str) -> None:
    dtype = _DTYPE_MAP[dtype_str]
    workload = BmmWorkload(batch, m, n, k, dtype)
    a, b = workload.gen_inputs()

    op = BmmFwdOp(tune=True)
    bm = ManifestBenchmark(_OP_NAME, op, workload)

    # eval_roofline() is read lazily after profiling, by which point
    # forward() has bound the dims.

    bm.compare({"tileops": op, "torch-cublas": torch.bmm}, a, b, record_as=op, params=locals())


@pytest.mark.parametrize(
    "batch, m, n, k, dtype_str", _fp8_params(load_workloads(_FP8_KN_OP_NAME))
)
def test_bmm_fp8_kn_bench(
    batch: int, m: int, n: int, k: int, dtype_str: str,
) -> None:
    """The [B, K, N] order, which the kernel reaches through a transpose."""
    dtype = _DTYPE_MAP[dtype_str]
    out_dtype = torch.bfloat16
    workload = BmmFp8BenchmarkWorkload(batch, m, n, k, dtype, out_dtype=out_dtype)
    a, b_kn, scale_a, scale_b = workload.gen_inputs()

    op = BmmFp8Op(out_dtype=out_dtype, tune=True)
    bm = ManifestBenchmark(_FP8_KN_OP_NAME, op, workload)
    bm.compare(
        {"tileops": (op, (a, b_kn, scale_a, scale_b))},
        record_as=op,
        params=locals(),
    )


@pytest.mark.parametrize(
    "batch, m, n, k, dtype_str", _fp8_params(load_workloads(_FP8_OP_NAME))
)
def test_bmm_fp8_bench(
    batch: int, m: int, n: int, k: int, dtype_str: str,
) -> None:
    dtype = _DTYPE_MAP[dtype_str]
    out_dtype = torch.bfloat16
    workload = BmmFp8BenchmarkWorkload(batch, m, n, k, dtype, out_dtype=out_dtype)
    a, b_kn, scale_a, scale_b = workload.gen_inputs()
    b_nk = b_kn.transpose(-2, -1).contiguous()      # [B, N, K], K-innermost
    b_kmajor = b_nk.transpose(-2, -1)               # [B, K, N] view, zero-copy

    # Fast path: feed [B, N, K] (K-innermost) using BmmFp8NKOp.
    op = BmmFp8NKOp(out_dtype=out_dtype, tune=True)
    bm = ManifestBenchmark(_FP8_OP_NAME, op, workload)
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

    bm.compare(functors, record_as=op, params=locals())
