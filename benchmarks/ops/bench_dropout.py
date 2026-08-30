"""Benchmarks for DropoutFwdOp.

Profiles TileOPs dropout vs torch.nn.functional.dropout on DNN-realistic shapes.
Uses p=0.5 (default) as representative drop rate.

torch eager and inductor are the two competitors: flag_gems' ``dropout`` raises on
this torch version.
"""

import pytest
import torch
import torch.nn.functional as F

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import ManifestBenchmark, workloads_to_params
from tileops.ops.dropout import DropoutFwdOp
from workloads.elementwise import ShapedRandnWorkload


class DropoutBenchmarkWorkload(ShapedRandnWorkload):
    def __init__(self, shape: tuple, dtype, p: float = 0.5):
        super().__init__(shape, dtype)
        self.p = p

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, p=self.p, training=True)


@pytest.mark.parametrize("shape, dtype", workloads_to_params(DropoutFwdOp))
def test_dropout_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = DropoutBenchmarkWorkload(shape, dtype)
    (x,) = test.gen_inputs()

    op = DropoutFwdOp(p=test.p, seed=42)
    bm = ManifestBenchmark(op, test)

    bm.compare(
        {
            "tileops": op,
            "torch": test.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(test.ref_program),
        },
        x,
    )
