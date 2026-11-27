"""Benchmarks for the Engram gate-conv and decode ops."""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from tileops._ops.sequence_modeling.engram import EngramGateConvBwdOp, EngramGateConvFwdOp
from tileops._ops.sequence_modeling.engram_decode import EngramDecodeFwdOp


def _bench_against_reference(op_cls, case) -> None:
    op = op_cls(**case.arguments)
    op.request_tune()
    bench.Runner(op, case).compare(
        {
            "torch-ref": case.reference,
            TORCH_COMPILE_TAG: compiled_reference(case.reference),
        }
    )


@pytest.mark.parametrize("case", bench.cases(EngramGateConvFwdOp))
def test_engram_gate_conv_fwd_bench(case):
    _bench_against_reference(EngramGateConvFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(EngramGateConvBwdOp))
def test_engram_gate_conv_bwd_bench(case):
    op = EngramGateConvBwdOp(**case.arguments)
    op.request_tune()
    bench.Runner(op, case).compare({"torch": torch.enable_grad()(case.reference)})


@pytest.mark.parametrize("case", bench.cases(EngramDecodeFwdOp))
def test_engram_decode_bench(case):
    _bench_against_reference(EngramDecodeFwdOp, case)
