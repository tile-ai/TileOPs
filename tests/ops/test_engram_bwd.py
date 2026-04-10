import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops.engram import EngramGateConvBwdOp
from workloads.ops.engram_bwd import CONV_KERNEL_SIZE
from workloads.ops.engram_bwd import EngramGateConvBwdTest as _EngramGateConvBwdTestWorkload


def ref_engram_gate_conv_bwd(dY, H, k, v, rms_w_h, rms_w_v, conv_w,
                              vhat, alpha, rrms_h, rrms_k, rrms_v, eps=1e-6):
    """PyTorch reference backward via autograd."""
    M, T, d = H.shape

    H_ag = H.float().detach().requires_grad_(True)
    k_ag = k.float().detach().requires_grad_(True)
    v_ag = v.float().detach().requires_grad_(True)
    w_h_ag = rms_w_h.float().detach().requires_grad_(True)
    w_v_ag = rms_w_v.float().detach().requires_grad_(True)
    cw_ag = conv_w.float().detach().requires_grad_(True)

    def _rmsnorm(x, w):
        return x * (x ** 2).mean(dim=-1, keepdim=True).add(eps).rsqrt() * w

    h_norm = _rmsnorm(H_ag, w_h_ag)
    k_norm = _rmsnorm(k_ag, w_h_ag)

    dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
    alpha_ag = torch.sigmoid(dot / (d ** 0.5))
    v_hat_ag = alpha_ag * v_ag
    v_hat_norm = _rmsnorm(v_hat_ag, w_v_ag)

    v_perm = v_hat_norm.permute(0, 2, 1)
    v_padded = F.pad(v_perm, (CONV_KERNEL_SIZE - 1, 0))
    cw_expanded = cw_ag.T.unsqueeze(1)
    conv_out = F.conv1d(v_padded, cw_expanded, groups=d).permute(0, 2, 1)
    Y_ag = F.silu(conv_out) + v_hat_ag

    Y_ag.backward(dY.float())

    return (
        H_ag.grad.to(H.dtype),
        k_ag.grad.to(H.dtype),
        v_ag.grad.to(H.dtype),
        w_h_ag.grad,
        w_v_ag.grad,
        cw_ag.grad,
    )


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
