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
            (64, torch.complex64, True),
            (128, torch.complex64, True),
            (256, torch.complex64, True),
            (512, torch.complex64, True),
            (1024, torch.complex64, True),
        ]),
    ]


class FFTBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        n = self.test.n
        # Cooley-Tukey FFT: log2(N) stages, each with N/2 butterflies.
        # One butterfly = 1 complex mul (6 FLOPs) + 2 complex adds/subs (4 FLOPs) = 10 FLOPs.
        # Total: (N/2) * log2(N) * 10 = 5 * N * log2(N) FLOPs.
        import math
        return 5.0 * n * math.log2(n)

    def calculate_memory(self) -> Optional[float]:
        n = self.test.n
        dtype = self.test.dtype
        # Read N complex + write N complex.
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
    BenchmarkReport.record(op, locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
