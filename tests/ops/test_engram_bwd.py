import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops.engram_bwd import EngramGateConvBwdOp

CONV_KERNEL_SIZE = 4


def _ref_rmsnorm(x, w, eps=1e-6):
    x_f = x.float()
    rrms = (x_f ** 2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    normed = x_f * rrms * w.float()
    return normed, rrms.squeeze(-1)


def ref_engram_gate_conv_bwd(dY, H, k, v, rms_w_h, rms_w_v, conv_w,
                              vhat, alpha, rrms_h, rrms_k, rrms_v, eps=1e-6):
    """PyTorch reference backward via autograd."""
    M, T, d = H.shape

    # Re-run forward with autograd to get exact gradients
    H_ag = H.float().detach().requires_grad_(True)
    k_ag = k.float().detach().requires_grad_(True)
    v_ag = v.float().detach().requires_grad_(True)
    w_h_ag = rms_w_h.float().detach().requires_grad_(True)
    w_v_ag = rms_w_v.float().detach().requires_grad_(True)
    cw_ag = conv_w.float().detach().requires_grad_(True)

    # Forward (autograd-friendly)
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

    # Backward
    Y_ag.backward(dY.float())

    return (
        H_ag.grad.to(H.dtype),
        k_ag.grad.to(H.dtype),
        v_ag.grad.to(H.dtype),
        w_h_ag.grad,   # fp32
        w_v_ag.grad,   # fp32
        cw_ag.grad,    # fp32
    )


class EngramGateConvBwdFixture(FixtureBase):
    PARAMS = [
        ("M, seq_len, d, dtype, tune", [
            pytest.param(1, 32, 256, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 64, 512, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1, 128, 256, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2, 16, 256, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


class EngramGateConvBwdTest(TestBase):
    def __init__(self, M, seq_len, d, dtype, eps=1e-6):
        self.M = M
        self.seq_len = seq_len
        self.d = d
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self):
        """Generate inputs including saved intermediates from a reference forward."""
        H = torch.randn(self.M, self.seq_len, self.d, dtype=self.dtype, device="cuda")
        k = torch.randn(self.M, self.seq_len, self.d, dtype=self.dtype, device="cuda") * 0.1
        v = torch.randn(self.M, self.seq_len, self.d, dtype=self.dtype, device="cuda") * 0.1
        rms_w_h = torch.ones(self.d, dtype=self.dtype, device="cuda")
        rms_w_v = torch.ones(self.d, dtype=self.dtype, device="cuda")
        conv_w = torch.randn(CONV_KERNEL_SIZE, self.d, dtype=self.dtype, device="cuda") * 0.02
        dY = torch.randn(self.M, self.seq_len, self.d, dtype=self.dtype, device="cuda") * 0.1

        # Compute saved intermediates via reference forward
        def _rmsnorm(x, w):
            x_f = x.float()
            rrms = (x_f ** 2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
            return x_f * rrms * w.float(), rrms.squeeze(-1)

        h_norm, rrms_h = _rmsnorm(H, rms_w_h)
        k_norm, rrms_k = _rmsnorm(k, rms_w_h)
        dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
        alpha = torch.sigmoid(dot / (self.d ** 0.5))
        v_hat = alpha * v.float()
        _, rrms_v = _rmsnorm(v_hat.to(self.dtype), rms_w_v)

        vhat = v_hat.to(self.dtype)
        alpha_squeezed = alpha.squeeze(-1).float()
        rrms_h = rrms_h.float()
        rrms_k = rrms_k.float()
        rrms_v = rrms_v.float()

        return (dY, H, k, v, rms_w_h, rms_w_v, conv_w,
                vhat, alpha_squeezed, rrms_h, rrms_k, rrms_v)

    def ref_program(self, dY, H, k, v, rms_w_h, rms_w_v, conv_w,
                    vhat, alpha, rrms_h, rrms_k, rrms_v):
        return ref_engram_gate_conv_bwd(
            dY, H, k, v, rms_w_h, rms_w_v, conv_w,
            vhat, alpha, rrms_h, rrms_k, rrms_v, self.eps,
        )


@EngramGateConvBwdFixture
def test_engram_gate_conv_bwd(M, seq_len, d, dtype, tune):
    test = EngramGateConvBwdTest(M, seq_len, d, dtype)
    op = EngramGateConvBwdOp(M, seq_len, d, dtype, tune=tune)
    inputs = test.gen_inputs()
    atol = 2e-1 if dtype == torch.float16 else 3e-1
    rtol = 2e-1
    test.check(op, *inputs, atol=atol, rtol=rtol)
