"""Benchmark the TileOPs complex-to-complex FFT, one case per manifest call, against cuFFT through torch."""

import pytest
import torch

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
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
    )
