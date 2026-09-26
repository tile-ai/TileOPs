"""Benchmarks for the Engram gate-conv and decode ops.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from each op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.

One ``test_*_bench`` per op, so every op this file is declared the benchmark
of records a row of its own.
"""

import pytest
import torch

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops.sequence_modeling.engram import EngramGateConvBwdOp, EngramGateConvFwdOp
from tileops.ops.sequence_modeling.engram_decode import EngramDecodeFwdOp
from workloads.engram import (
    EngramDecodeWorkload,
    EngramGateConvBwdWorkload,
    EngramGateConvFwdWorkload,
)

# Autotuning is a bench-run policy, not a workload property; manifest
# workloads do not carry it.
_TUNE = True


def _dtype(call, tensor: str) -> torch.dtype:
    return getattr(torch, call.tensors[tensor][1])


@pytest.mark.parametrize("call", manifest_calls(EngramGateConvFwdOp))
def test_engram_gate_conv_fwd_bench(call):
    params = call.arguments({})
    test = EngramGateConvFwdWorkload(**params, dtype=_dtype(call, "H"))
    inputs = test.gen_inputs()

    op = EngramGateConvFwdOp(**params, tune=_TUNE)
    bm = ManifestBenchmark(op, test)

    bm.compare(
        {
            "tileops": op,
            "torch-ref": test.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(test.ref_program),
        },
        *inputs,
    )


@pytest.mark.parametrize("call", manifest_calls(EngramGateConvBwdOp))
def test_engram_gate_conv_bwd_bench(call):
    params = call.arguments({})
    test = EngramGateConvBwdWorkload(**params, dtype=_dtype(call, "dY"))
    inputs = test.gen_inputs()

    op = EngramGateConvBwdOp(**params, tune=_TUNE)
    bm = ManifestBenchmark(op, test)

    @torch.enable_grad()
    def ref_with_grad(*args):
        return test.ref_program(*args)

    bm.compare(
        {
            "tileops": op,
            "torch": ref_with_grad,
            TORCH_COMPILE_TAG: compiled_reference(ref_with_grad),
        },
        *inputs,
    )


@pytest.mark.parametrize("call", manifest_calls(EngramDecodeFwdOp))
def test_engram_decode_bench(call):
    params = call.arguments({})
    test = EngramDecodeWorkload(**params, dtype=_dtype(call, "e_t"), conv_len=call.ix["L"])
    inputs = test.gen_inputs()

    op = EngramDecodeFwdOp(**params, tune=_TUNE)
    bm = ManifestBenchmark(op, test)

    bm.compare(
        {
            "tileops": op,
            "torch-ref": test.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(test.ref_program),
        },
        *inputs,
    )
