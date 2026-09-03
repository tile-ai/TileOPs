import pytest
import torch

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import (
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.ops import FFTC2CFwdOp
from workloads.fft import FFTWorkload


@pytest.mark.parametrize("shape, dtype", workloads_to_params(FFTC2CFwdOp))
def test_fft_bench(shape: tuple, dtype: torch.dtype) -> None:
    n = shape[-1]
    batch_shape = shape[:-1]
    test = FFTWorkload(n, dtype, batch_shape=batch_shape)
    inputs = test.gen_inputs()

    op = FFTC2CFwdOp(tune=True)

    op(*inputs)
    torch.cuda.synchronize()

    bm = ManifestBenchmark(op, test)

    bm.compare(
        {
            "tileops": op,
            "torch-cufft": test.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(test.ref_program),
        },
        *inputs,
    )
