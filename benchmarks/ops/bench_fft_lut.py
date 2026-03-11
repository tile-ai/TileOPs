import math
from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_fft_lut import FFTLUTTest
from tests.test_base import FixtureBase
from tileops.ops import FFTC2CLUTOp, FFTC2COp


class FFTLUTBenchmarkFixture(FixtureBase):
    PARAMS = [
        ("n, dtype, tune", [
            (64, torch.complex64, True),
            (128, torch.complex64, True),
            (256, torch.complex64, True),
            (512, torch.complex64, True),
            (1024, torch.complex64, True),
            (4096, torch.complex64, True),
            (16384, torch.complex64, True),
        ]),
    ]


class FFTLUTBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        n = self.test.n
        # Cooley-Tukey FFT: (N/2) * log2(N) butterflies, each 10 real FLOPs.
        # One butterfly = 1 complex mul (6 FLOPs) + 2 complex adds/subs (4 FLOPs) = 10 FLOPs.
        # Total: 5 * N * log2(N) FLOPs.
        return 5.0 * n * math.log2(n)

    def calculate_memory(self) -> Optional[float]:
        n = self.test.n
        dtype = self.test.dtype
        # Read N complex + write N complex.
        return 2 * n * torch.empty(1, dtype=dtype).element_size()


@FFTLUTBenchmarkFixture
def test_fft_lut_bench(n: int, dtype: torch.dtype, tune: bool) -> None:
    test = FFTLUTTest(n, dtype)
    bm = FFTLUTBenchmark(test)
    inputs = test.gen_inputs()

    op_lut = FFTC2CLUTOp(n, dtype=dtype, tune=tune)
    result_lut = bm.profile(op_lut, *inputs)
    BenchmarkReport.record("fft_c2c_lut", locals(), result_lut, tag="tileops-lut")

    op_base = FFTC2COp(n, dtype=dtype, tune=tune)
    result_base = bm.profile(op_base, *inputs)
    BenchmarkReport.record("fft_c2c_lut", locals(), result_base, tag="tileops-base")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("fft_c2c_lut", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
