"""Benchmarks for the Engram gate-conv and decode ops.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from each op's ``eval_roofline()``.

One ``test_*_bench`` per op, so every op this file is declared the benchmark
of records a row of its own.
"""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from tileops.ops.sequence_modeling.engram import EngramGateConvBwdOp, EngramGateConvFwdOp
from tileops.ops.sequence_modeling.engram_decode import EngramDecodeFwdOp

# Autotuning is a bench-run policy, not a workload property; manifest
# workloads do not carry it.
_TUNE = True


@pytest.mark.parametrize("case", bench.cases(EngramGateConvFwdOp), ids=lambda case: case.id)
def test_engram_gate_conv_fwd_bench(case):
    op = EngramGateConvFwdOp(**case.arguments, tune=_TUNE)
    bench.Runner(op, case).compare(
        {
            "torch-ref": case.reference,
            TORCH_COMPILE_TAG: compiled_reference(case.reference),
        }
    )


@pytest.mark.parametrize("case", bench.cases(EngramGateConvBwdOp), ids=lambda case: case.id)
def test_engram_gate_conv_bwd_bench(case):
    op = EngramGateConvBwdOp(**case.arguments, tune=_TUNE)
    reference = case.reference

    @torch.enable_grad()
    def ref_with_grad(*args):
        return reference(*args)

    bench.Runner(op, case).compare({"torch": ref_with_grad})


@pytest.mark.parametrize("case", bench.cases(EngramDecodeFwdOp), ids=lambda case: case.id)
def test_engram_decode_bench(case):
    op = EngramDecodeFwdOp(**case.arguments, tune=_TUNE)
    bench.Runner(op, case).compare(
        {
            "torch-ref": case.reference,
            TORCH_COMPILE_TAG: compiled_reference(case.reference),
        }
    )
