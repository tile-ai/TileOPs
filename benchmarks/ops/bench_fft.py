import pytest
import torch

from benchmarks.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.ops import FFTC2COp
from workloads.fft import FFTWorkload

_OP_NAME = "FFTC2COp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_OP_NAME))
def test_fft_bench(shape: tuple, dtype: torch.dtype) -> None:
    n = shape[-1]
    batch_shape = shape[:-1]
    test = FFTWorkload(n, dtype, batch_shape=batch_shape)
    inputs = test.gen_inputs()

    op = FFTC2COp(tune=True)

    # Warmup: trigger JIT compilation before timed profiling
    op(*inputs)
    torch.cuda.synchronize()

    bm = ManifestBenchmark(_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-cufft")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
