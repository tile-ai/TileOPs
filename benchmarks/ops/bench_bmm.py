from typing import Optional

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tests.ops.test_bmm import BmmTest
from tileops.manifest import load_workloads
from tileops.ops import BmmFwdOp

_OP_NAME = "BmmFwdOp"

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


class BmmBenchmark(BenchmarkBase[BmmTest]):
    """Reads FLOP/byte counts from the Op's manifest-derived roofline.

    ``BmmFwdOp`` is input-inferred, so ``eval_roofline()`` is valid only
    after a forward has bound ``batch/m/n/k/dtype``; the benchmark calls it
    lazily.
    """

    _roofline_cache: Optional[tuple[float, float]] = None

    def __init__(self, test: BmmTest, op: BmmFwdOp):
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


@pytest.mark.parametrize("batch, m, n, k, dtype_str", _manifest_params())
def test_bmm_bench(batch: int, m: int, n: int, k: int, dtype_str: str) -> None:
    dtype = _DTYPE_MAP[dtype_str]
    test = BmmTest(batch, m, n, k, dtype)
    a, b = test.gen_inputs()

    op = BmmFwdOp(tune=True)
    bm = BmmBenchmark(test, op)

    # eval_roofline() is read lazily after profiling, by which point
    # forward() has bound the dims.
    result = bm.profile(op, a, b)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, a, b)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-cublas")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
