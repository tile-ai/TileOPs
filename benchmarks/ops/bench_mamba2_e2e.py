"""End-to-end Mamba-2 SSD forward benchmark: TileOPs vs mamba_ssm official.

Benchmarks the full Mamba-2 SSD forward pass (DaCumsum → CBProducer → SSDChunkState →
SSDStatePassing → SSDChunkScan), one case per manifest call, against
mamba_chunk_scan_combined from the official mamba_ssm library (Triton baseline) and
the PyTorch reference.
"""

import pytest

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops.mamba.mamba2_fwd import Mamba2FwdOp
from workloads.mamba2_e2e import Mamba2FwdCall

# Optional mamba_ssm Triton baseline
try:
    from mamba_ssm.ops.triton.ssd_combined import (
        mamba_chunk_scan_combined as _mamba_chunk_scan_combined,
    )
except ImportError:
    _mamba_chunk_scan_combined = None


@pytest.mark.parametrize("call", manifest_calls(Mamba2FwdOp))
def test_mamba2_fwd_bench(call):
    test = Mamba2FwdCall(call)
    inputs = test.gen_inputs()
    x, dt, A, B, C, dt_bias, initial_states = inputs
    op = Mamba2FwdOp(**test.arguments())
    bm = ManifestBenchmark(op, test)
    functors = {"tileops": op}

    # Only the five leading tensors are positional, so every path gets identical clone
    # treatment; the optional tensors are captured.
    reference_args = (x, dt, A, B, C)
    if _mamba_chunk_scan_combined is not None:

        def _mamba_wrapper(x, dt, A, B, C):
            return _mamba_chunk_scan_combined(
                x,
                dt,
                A,
                B,
                C,
                op.chunk_size,
                dt_bias=dt_bias,
                dt_softplus=op.dt_softplus,
                initial_states=initial_states,
                return_final_states=True,
            )

        functors["mamba"] = (_mamba_wrapper, reference_args)

    def _torch_wrapper(x, dt, A, B, C):
        return test.ref_program(x, dt, A, B, C, dt_bias, initial_states)

    functors["torch-ref"] = (_torch_wrapper, reference_args)
    functors[TORCH_COMPILE_TAG] = (compiled_reference(_torch_wrapper), reference_args)

    bm.compare(functors, *inputs)
