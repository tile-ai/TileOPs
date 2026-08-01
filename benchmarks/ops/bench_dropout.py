"""Benchmarks for DropoutOp.

Profiles TileOPs dropout vs torch.nn.functional.dropout on DNN-realistic shapes.
Uses p=0.5 (default) as representative drop rate.
"""

from math import prod

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark, workloads_to_params
from tileops.ops.dropout import DropoutOp
from workloads.elementwise import ShapedRandnWorkload

_OP_NAME = "DropoutOp"


class DropoutBenchmarkWorkload(ShapedRandnWorkload):
    def __init__(self, shape: tuple, dtype, p: float = 0.5):
        super().__init__(shape, dtype)
        self.p = p

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, p=self.p, training=True)

@pytest.mark.parametrize("shape, dtype", workloads_to_params(_OP_NAME))
def test_dropout_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = DropoutBenchmarkWorkload(shape, dtype)
    (x,) = test.gen_inputs()

    op = DropoutOp(p=test.p, seed=42)
    bm = ManifestBenchmark(_OP_NAME, op, test)
    result = bm.profile(op, x)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, x)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
