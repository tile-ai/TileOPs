import math
from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_fft import FFTTest
from tests.test_base import FixtureBase
from tileops.ops import FFTC2COp


class FFTBenchmarkFixture(FixtureBase):
    PARAMS = [
        ("n, dtype, tune", [
            (4096, torch.complex64, True),
            (8192, torch.complex64, True),
            (16384, torch.complex64, True),
            (32768, torch.complex64, True),
            (4096, torch.complex128, True),
            (8192, torch.complex128, True),
            (16384, torch.complex128, True),
            (32768, torch.complex128, True),
        ]),
    ]


class FFTBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        n = self.test.n
        return 5.0 * n * math.log2(n)

    def calculate_memory(self) -> Optional[float]:
        n = self.test.n
        dtype = self.test.dtype
        return 2 * n * torch.empty(1, dtype=dtype).element_size()


@FFTBenchmarkFixture
def test_fft_bench(n: int, dtype: torch.dtype, tune: bool) -> None:
    test = FFTTest(n, dtype)
    bm = FFTBenchmark(test)
    inputs = test.gen_inputs()

    op = FFTC2COp(n, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-cufft")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
