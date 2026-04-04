import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.engram_fwd import EngramGateConvFwdOp
from workloads.ops.engram_fwd import (
    EngramGateConvFwdTest as _EngramGateConvFwdTestWorkload,
)
from workloads.ops.engram_fwd import (
    ref_engram_gate_conv_fwd,
)


class EngramGateConvFwdTest(_EngramGateConvFwdTestWorkload, TestBase):
    def ref_program(self, H, k, v, rms_w_h, rms_w_v, conv_w):
        return ref_engram_gate_conv_fwd(H, k, v, rms_w_h, rms_w_v, conv_w, self.eps)


class EngramGateConvFwdFixture(FixtureBase):
    PARAMS = [
        ("M, seq_len, d, dtype, tune", [
            pytest.param(1, 32, 256, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(2, 64, 512, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1, 128, 256, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2, 16, 256, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


@EngramGateConvFwdFixture
def test_engram_gate_conv_fwd(M, seq_len, d, dtype, tune):
    test = EngramGateConvFwdTest(M, seq_len, d, dtype)
    op = EngramGateConvFwdOp(M, seq_len, d, dtype, tune=tune)
    inputs = test.gen_inputs()
    atol = 1e-1 if dtype == torch.float16 else 2e-1
    rtol = 1e-1
    test.check(op, *inputs, atol=atol, rtol=rtol)
