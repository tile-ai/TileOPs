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
from workloads.numerics import Exact, zeroed_input
from workloads.sequence_modeling.engram import (
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
    workload = EngramGateConvFwdWorkload(**params, dtype=_dtype(call, "H"))
    inputs = workload.gen_inputs()

    op = EngramGateConvFwdOp(**params, tune=_TUNE)
    bm = ManifestBenchmark(op, workload)
    # The Engram unit-test contract includes low-precision saved intermediates.
    checked = Exact(
        rtol=0.1,
        atol=0.1 if _dtype(call, "H") == torch.float16 else 0.2,
        controls=(zeroed_input(0, "first-input-zeroed"),),
    )

    bm.compare(
        {
            "tileops": op,
            "torch-ref": workload.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
        },
        *inputs,
        evidence={"tileops": checked, TORCH_COMPILE_TAG: checked},
    )


@pytest.mark.parametrize("call", manifest_calls(EngramGateConvBwdOp))
def test_engram_gate_conv_bwd_bench(call):
    params = call.arguments({})
    workload = EngramGateConvBwdWorkload(**params, dtype=_dtype(call, "dY"))
    inputs = workload.gen_inputs()

    op = EngramGateConvBwdOp(**params, tune=_TUNE)
    bm = ManifestBenchmark(op, workload)
    # The Engram unit-test contract includes low-precision saved intermediates.
    checked = Exact(
        rtol=0.2,
        atol=0.2 if _dtype(call, "dY") == torch.float16 else 0.3,
        controls=(zeroed_input(0, "first-input-zeroed"),),
    )

    @torch.enable_grad()
    def ref_with_grad(*args):
        return workload.ref_program(*args)

    # No torch-compile tag: the reference calls ``requires_grad_()`` on the intermediates it
    # returns gradients for and runs ``backward`` over them, which dynamo splits into seven
    # graphs, so the row would time six eager segments under a tag that says compiled.
    bm.compare(
        {"tileops": op, "torch": ref_with_grad},
        *inputs,
        evidence={"tileops": checked, "torch": checked},
    )


@pytest.mark.parametrize("call", manifest_calls(EngramDecodeFwdOp))
def test_engram_decode_bench(call):
    params = call.arguments({})
    workload = EngramDecodeWorkload(**params, dtype=_dtype(call, "e_t"), conv_len=call.ix["L"])
    inputs = workload.gen_inputs()

    op = EngramDecodeFwdOp(**params, tune=_TUNE)
    bm = ManifestBenchmark(op, workload)
    # The Engram unit-test contract includes low-precision saved intermediates.
    checked = Exact(
        rtol=0.05,
        atol=0.05 if _dtype(call, "e_t") == torch.float16 else 0.1,
        controls=(zeroed_input(0, "first-input-zeroed"),),
    )

    bm.compare(
        {
            "tileops": op,
            "torch-ref": workload.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
        },
        *inputs,
        evidence={"tileops": checked, TORCH_COMPILE_TAG: checked},
    )
