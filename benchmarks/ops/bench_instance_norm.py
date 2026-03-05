from typing import Optional, Tuple

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_instance_norm import InstanceNormFixture, InstanceNormTest
from tileops.ops import InstanceNormOp


class InstanceNormBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        numel = 1
        for dim in t.shape:
            numel *= dim
        # Approximate: mean + var + normalize + affine
        return float(6 * numel)

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        numel = 1
        for dim in t.shape:
            numel *= dim
        c = t.shape[1]
        io_bytes = 2 * numel * t.dtype.itemsize
        affine_bytes = 2 * c * t.dtype.itemsize if t.affine else 0
        return float(io_bytes + affine_bytes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@InstanceNormFixture
def test_instance_norm_bench(shape: Tuple[int, ...], dtype: torch.dtype, eps: float, affine: bool,
                             non_contiguous: bool, tune: bool) -> None:
    test = InstanceNormTest(shape, dtype, eps, affine, non_contiguous)
    bm = InstanceNormBenchmark(test)
    inputs = test.gen_inputs()

    op = InstanceNormOp(num_channels=shape[1], eps=eps, affine=affine, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("instance_norm", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("instance_norm", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
