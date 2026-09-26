import pytest
import torch

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops import FFTC2CFwdOp
from workloads.fft import FFTWorkload


@pytest.mark.parametrize("call", manifest_calls(FFTC2CFwdOp))
def test_fft_bench(call) -> None:
    shape, dtype = call.tensors["input"]
    test = FFTWorkload(shape[-1], getattr(torch, dtype), batch_shape=shape[:-1])
    inputs = test.gen_inputs()

    op = FFTC2CFwdOp(**call.arguments({}), tune=True)

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
