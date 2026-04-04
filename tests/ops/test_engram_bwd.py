import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.engram_bwd import EngramGateConvBwdOp
from workloads.ops.engram_bwd import EngramGateConvBwdTest as _EngramGateConvBwdTestWorkload
from workloads.ops.engram_bwd import ref_engram_gate_conv_bwd


class EngramGateConvBwdTest(_EngramGateConvBwdTestWorkload, TestBase):
    def ref_program(self, dY, H, k, v, rms_w_h, rms_w_v, conv_w,
                    vhat, alpha, rrms_h, rrms_k, rrms_v):
        return ref_engram_gate_conv_bwd(
            dY, H, k, v, rms_w_h, rms_w_v, conv_w,
            vhat, alpha, rrms_h, rrms_k, rrms_v, self.eps,
        )


def _ref_rmsnorm(x, w, eps=1e-6):
    x_f = x.float()
    rrms = (x_f ** 2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    normed = x_f * rrms * w.float()
    return normed, rrms.squeeze(-1)


class EngramGateConvBwdFixture(FixtureBase):
    PARAMS = [
        ("M, seq_len, d, dtype, tune", [
            pytest.param(1, 32, 256, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(2, 64, 512, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1, 128, 256, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2, 16, 256, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


@EngramGateConvBwdFixture
def test_engram_gate_conv_bwd(M, seq_len, d, dtype, tune):
    test = EngramGateConvBwdTest(M, seq_len, d, dtype)
    op = EngramGateConvBwdOp(M, seq_len, d, dtype, tune=tune)
    inputs = test.gen_inputs()
    atol = 2e-1 if dtype == torch.float16 else 3e-1
    rtol = 2e-1
    test.check(op, *inputs, atol=atol, rtol=rtol)
