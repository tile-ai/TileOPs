import math
from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport, CuptiSession
from tests.ops.test_fft import FFTTest
from tests.test_base import FixtureBase
from tileops.ops import FFTC2COp


class FFTBenchmarkFixture(FixtureBase):
    PARAMS = [
        ("n, dtype, tune, batch_shape", [
            (4096, torch.complex64, True, ()),
            (16384, torch.complex64, True, ()),
            (65536, torch.complex64, True, ()),
            (262144, torch.complex64, True, ()),
            (1048576, torch.complex64, True, ()),
            (4096, torch.complex64, True, (64,)),
            (4096, torch.complex64, True, (256,)),
            (1024, torch.complex64, True, (1024,)),
            (4096, torch.complex128, True, ()),
            (65536, torch.complex128, True, ()),
            (4096, torch.complex128, True, (64,)),
        ]),
    ]


class FFTBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        n = self.test.n
        batch = math.prod(self.test.batch_shape) if self.test.batch_shape else 1
        return batch * 5.0 * n * math.log2(n)

    def calculate_memory(self) -> Optional[float]:
        n = self.test.n
        dtype = self.test.dtype
        batch = math.prod(self.test.batch_shape) if self.test.batch_shape else 1
        return batch * 2 * n * torch.empty(1, dtype=dtype).element_size()


@FFTBenchmarkFixture
def test_fft_bench(n: int, dtype: torch.dtype, tune: bool, batch_shape: tuple) -> None:
    test = FFTTest(n, dtype, batch_shape=batch_shape)
    bm = FFTBenchmark(test)
    inputs = test.gen_inputs()

    op = FFTC2COp(n, dtype=dtype, tune=tune)

    # Warmup: trigger JIT compilation before timed profiling
    op(*inputs)
    torch.cuda.synchronize()

    with CuptiSession():
        result = bm.profile(op, *inputs)
        BenchmarkReport.record(op, locals(), result, tag="tileops")

        result_bl = bm.profile(test.ref_program, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch-cufft")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
