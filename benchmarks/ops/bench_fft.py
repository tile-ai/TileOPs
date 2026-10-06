"""Benchmark the TileOPs complex-to-complex FFT, one case per manifest call, against cuFFT through torch."""

import pytest

from benchmarks import api as bench
from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from tileops.ops import FFTC2CFwdOp


@pytest.mark.parametrize("case", bench.cases(FFTC2CFwdOp), ids=lambda case: case.id)
def test_fft_bench(case) -> None:
    op = FFTC2CFwdOp(**case.arguments, tune=True)
    bench.Runner(op, case).compare(
        {
            "torch-cufft": case.reference,
            TORCH_COMPILE_TAG: compiled_reference(case.reference),
        }
    )
