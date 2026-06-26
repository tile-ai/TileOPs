"""Benchmark for dense GemmOp.

Workload shapes come from the manifest entry's `workloads` (via
`load_workloads`); the benchmark reports TileOPs latency alongside the
manifest-derived roofline (`op.eval_roofline()`), with a cuBLAS
(`torch.matmul`) baseline.
"""

from typing import Optional

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.manifest import load_workloads
from tileops.ops import GemmOp
from workloads.gemm import GemmTest

_OP_NAME = "GemmOp"

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


class _GemmTestBaseline(GemmTest):
    """Adds baseline ref_program for benchmark profiling."""

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.trans_a:
            a = a.T
        if self.trans_b:
            b = b.T
        return torch.matmul(a, b)


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


@pytest.mark.parametrize("m, n, k, trans_a, trans_b, dtype_str", _manifest_params())
def test_gemm_bench(
    m: int, n: int, k: int, trans_a: bool, trans_b: bool, dtype_str: str,
) -> None:
    dtype = _DTYPE_MAP[dtype_str]
    test = _GemmTestBaseline(m, n, k, dtype, trans_a, trans_b)
    a, b = test.gen_inputs()

    op = GemmOp(trans_a=trans_a, trans_b=trans_b)
    bm = GemmBenchmark(test, op)

    # The benchmark framework warms up internally; eval_roofline() is read
    # lazily after profiling, by which point forward() has bound the dims.
    result = bm.profile(op, a, b)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, a, b)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-cublas")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
