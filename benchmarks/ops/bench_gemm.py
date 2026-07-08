from typing import Optional

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tests.ops.test_gemm import GemmFp8Test, GemmTest
from tileops.manifest import load_workloads
from tileops.ops import GemmFp8Op, GemmOp

_OP_NAME = "GemmOp"
_FP8_OP_NAME = "GemmFp8Op"

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}


class GemmBenchmark(BenchmarkBase[GemmTest]):
    """Reads FLOP/byte counts from the Op's manifest-derived roofline.

    `GemmOp` is input-inferred, so `eval_roofline()` is valid only after a
    forward has bound `m/n/k/dtype`; the benchmark calls it lazily.
    """

    _roofline_cache: Optional[tuple[float, float]] = None

    def __init__(self, test: GemmTest, op: GemmOp):
        super().__init__(test)
        self._op = op

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            flops, mem_bytes = self._op.eval_roofline()
            self._roofline_cache = (float(flops), float(mem_bytes))
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


def _flashinfer_fp8_blockscale_ref(test: GemmFp8Test, *inputs: torch.Tensor) -> torch.Tensor:
    from flashinfer.gemm import fp8_blockscale_gemm_sm90

    a, b, scale_a, scale_b = inputs[:4]
    if len(inputs) == 5:
        raise ValueError("FlashInfer FP8 blockscale GEMM baseline does not support bias.")
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        raise ValueError("FlashInfer FP8 blockscale GEMM baseline requires float8_e4m3fn.")
    if test.out_dtype != torch.bfloat16:
        raise ValueError("FlashInfer FP8 blockscale GEMM baseline requires bfloat16 output.")
    if test.k % 128 != 0:
        raise ValueError("FlashInfer FP8 blockscale GEMM baseline requires k divisible by 128.")
    if scale_a.shape != (test.m, test.k // 128) or scale_b.shape != (test.n, test.k // 128):
        raise ValueError(
            "FlashInfer FP8 blockscale GEMM baseline requires exact "
            f"scale shapes {(test.m, test.k // 128)} and {(test.n, test.k // 128)}, "
            f"got {tuple(scale_a.shape)} and {tuple(scale_b.shape)}"
        )
    return fp8_blockscale_gemm_sm90(a, b, scale_a, scale_b, out_dtype=test.out_dtype)


def _prepare_flashinfer_fp8_per_tensor(
    test: GemmFp8Test, *inputs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    import flashinfer

    a, b, scale_a, scale_b = inputs[:4]
    if len(inputs) == 5:
        raise ValueError("FlashInfer FP8 per-tensor GEMM baseline does not support bias.")
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        raise ValueError("FlashInfer FP8 per-tensor GEMM baseline requires float8_e4m3fn.")
    if test.out_dtype != torch.bfloat16:
        raise ValueError("FlashInfer FP8 per-tensor GEMM baseline requires bfloat16 output.")
    if scale_a.shape != (1, 1) or scale_b.shape != (1, 1):
        raise ValueError(
            "FlashInfer FP8 per-tensor GEMM baseline requires (1, 1) scales, "
            f"got {tuple(scale_a.shape)} and {tuple(scale_b.shape)}"
        )
    prepared_b = flashinfer.prepare_low_latency_gemm_weights(b, {})
    alpha = (scale_a * scale_b).reshape(())
    return prepared_b, alpha


class GemmFp8Benchmark(BenchmarkBase[GemmFp8Test]):
    _roofline_cache: Optional[tuple[float, float]] = None

    def __init__(self, test: GemmFp8Test, op: GemmFp8Op):
        super().__init__(test)
        self._op = op

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            flops, mem_bytes = self._op.eval_roofline()
            self._roofline_cache = (float(flops), float(mem_bytes))
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


def _manifest_params() -> list:
    """Convert manifest workloads to pytest params (m, n, k, trans_a, trans_b, dtype)."""
    params = []
    for w in load_workloads(_OP_NAME):
        label = w.get("label", "unlabeled")
        trans_a = bool(w.get("trans_a", False))
        trans_b = bool(w.get("trans_b", True))
        for dtype_str in w["dtypes"]:
            params.append(pytest.param(
                w["m"], w["n"], w["k"], trans_a, trans_b, dtype_str,
                id=f"{label}-{dtype_str}",
            ))
    return params


def _manifest_fp8_params() -> list:
    params = []
    for w in load_workloads(_FP8_OP_NAME):
        label = w.get("label", "unlabeled")
        for dtype_str in w["dtypes"]:
            params.append(pytest.param(
                w["m"], w["n"], w["k"], w["scale_mode"], dtype_str,
                id=f"{label}-{dtype_str}",
            ))
    return params


@pytest.mark.parametrize("m, n, k, trans_a, trans_b, dtype_str", _manifest_params())
def test_gemm_bench(
    m: int, n: int, k: int, trans_a: bool, trans_b: bool, dtype_str: str,
) -> None:
    dtype = _DTYPE_MAP[dtype_str]
    test = GemmTest(m, n, k, dtype, trans_a, trans_b)
    a, b = test.gen_inputs()

    op = GemmOp(trans_a=trans_a, trans_b=trans_b)
    bm = GemmBenchmark(test, op)

    # The benchmark framework warms up internally; eval_roofline() is read
    # lazily after profiling, by which point forward() has bound the dims.
    result = bm.profile(op, a, b)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, a, b)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-cublas")


@pytest.mark.parametrize("m, n, k, scale_mode, dtype_str", _manifest_fp8_params())
def test_gemm_fp8_bench(
    m: int, n: int, k: int, scale_mode: str, dtype_str: str,
) -> None:
    dtype = _DTYPE_MAP[dtype_str]
    out_dtype = torch.bfloat16
    test = GemmFp8Test(m, n, k, dtype, scale_mode, out_dtype=out_dtype)
    inputs = test.gen_inputs()

    op = GemmFp8Op(out_dtype=out_dtype)
    bm = GemmFp8Benchmark(test, op)

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-scaled-mm")

    flashinfer = pytest.importorskip("flashinfer")
    if scale_mode == "per_tensor":
        prepared_b, alpha = _prepare_flashinfer_fp8_per_tensor(test, *inputs)
        try:
            result_flashinfer = bm.profile(
                lambda a: flashinfer.mm_fp8(a, prepared_b, alpha, out_dtype=out_dtype),
                inputs[0],
            )
        except RuntimeError as exc:
            reason = str(exc).splitlines()[0]
            print(f"  [skip] flashinfer-mm-fp8: {reason}")
            return
        BenchmarkReport.record(op, locals(), result_flashinfer, tag="flashinfer-mm-fp8")
        return

    if scale_mode == "block128":
        result_flashinfer = bm.profile(
            lambda *args: _flashinfer_fp8_blockscale_ref(test, *args), *inputs)
        BenchmarkReport.record(op, locals(), result_flashinfer, tag="flashinfer-fp8-blockscale-sm90")
        return

    raise ValueError(f"unsupported FP8 GEMM scale_mode for benchmark: {scale_mode!r}")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
