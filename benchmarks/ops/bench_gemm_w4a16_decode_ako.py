from __future__ import annotations

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport
from benchmarks.ops.gemm_w4a16_benchmark_utils import (
    DECODE_AKO_CASES,
    StaticGemmW4A16Benchmark,
    W4A16BenchmarkCase,
    dense_a16_memory_bytes,
    make_marlin_w4a16_callable,
    marlin_w4a16_memory_bytes,
    w4a16_memory_bytes,
)
from tests.ops.test_gemm import GemmW4A16Test
from tileops.ops import GemmW4A16Op


@pytest.mark.parametrize(
    "case",
    [pytest.param(case, id=case.label) for case in DECODE_AKO_CASES],
)
def test_gemm_w4a16_decode_ako_bench(case: W4A16BenchmarkCase) -> None:
    """AKO W4A16 decode optimization suite.

    This is intentionally not a complete public manifest benchmark. The suite
    fixes M=1, FP16 activation/output, affine UINT4 weights, group size 128,
    FP32 accumulation, pre-packed weights, and TileOps' cold-cache bench_kernel
    protocol. Quantization and repacking are outside the timed region.

    The long-K case (1,8192,81920) is the main mechanism stress test: it exceeds
    L2 by a wide margin, elongates the K dependency chain, and makes load,
    depack, compute overlap, activation reuse, and split-K policy visible in
    latency. The first three cases are retained so the stress test does not
    stand in for common model layers by itself.
    """
    dtype = torch.float16
    m, n, k = case.m, case.n, case.k
    scenario = case.scenario
    purpose = case.purpose
    weight_metadata_mib = case.weight_metadata_mib

    test = GemmW4A16Test(m, n, k, dtype)
    inputs = test.gen_inputs()
    op = GemmW4A16Op()

    w4_bm = StaticGemmW4A16Benchmark(test, w4a16_memory_bytes(m, n, k, dtype=dtype))
    result = w4_bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops-w4a16")

    dense_bm = StaticGemmW4A16Benchmark(test, dense_a16_memory_bytes(m, n, k, dtype=dtype))
    result_dense = dense_bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_dense, tag="torch-dequantized-a16")

    for reduce_mode, use_fp32_reduce in (("fp32", True), ("fp16", False)):
        try:
            marlin, marlin_inputs = make_marlin_w4a16_callable(
                m, n, k, use_fp32_reduce=use_fp32_reduce
            )
        except (ImportError, ModuleNotFoundError) as exc:
            print(f"  [skip] marlin-{reduce_mode}: {exc}")
            continue
        marlin_bm = StaticGemmW4A16Benchmark(test, marlin_w4a16_memory_bytes(m, n, k, dtype=dtype))
        actual = marlin(*marlin_inputs)
        if actual.shape != (m, n) or not torch.isfinite(actual).all():
            raise RuntimeError("Marlin W4A16 baseline smoke check failed")
        torch.cuda.synchronize()
        result_marlin = marlin_bm.profile(marlin, *marlin_inputs)
        BenchmarkReport.record(op, locals(), result_marlin, tag=f"marlin-{reduce_mode}")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
