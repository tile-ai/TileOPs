"""Benchmark the TileOPs complex-to-complex FFT, one case per manifest call, against cuFFT through torch."""

import pytest
import torch

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from benchmarks.verification import Exact
from tileops.ops import FFTC2CFwdOp
from workloads.fft import FFTWorkload


@pytest.mark.parametrize("call", manifest_calls(FFTC2CFwdOp))
def test_fft_bench(call) -> None:
    shape, dtype = call.tensors["input"]
    workload = FFTWorkload(shape[-1], getattr(torch, dtype), batch_shape=shape[:-1])
    inputs = workload.gen_inputs()

    op = FFTC2CFwdOp(**call.arguments({}), tune=True)

    op(*inputs)
    torch.cuda.synchronize()

    bm = ManifestBenchmark(op, workload)

    bm.compare(
        {
            "tileops": op,
            "torch-cufft": workload.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
        },
        *inputs,
        # The dtype table carries no complex entry, and a whole-sequence float32
        # accumulation does not reach its float32 row. These are the tolerances
        # tests/ops/test_fft.py asserts this op at.
        evidence=dict.fromkeys(
            ("tileops", "torch-cufft", TORCH_COMPILE_TAG),
            Exact(rtol=1e-4, atol=1e-4) if dtype == "complex64" else Exact(rtol=1e-8, atol=1e-8),
        ),
    )
