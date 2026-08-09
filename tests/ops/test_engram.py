import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.engram import EngramGateConvBwdOp, EngramGateConvFwdOp
from tileops.ops.engram_decode import EngramDecodeOp
from workloads.engram import (
    EngramDecodeWorkload,
    EngramGateConvBwdWorkload,
    EngramGateConvFwdWorkload,
    engram_decode_step_torch,
)


class EngramGateConvFwdTest(EngramGateConvFwdWorkload, TestBase):
    pass


class EngramGateConvFwdFixture(FixtureBase):
    PARAMS = [
        ("M, seq_len, d, dtype, tune", [
            pytest.param(1, 32, 256, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(1, 32, 256, torch.bfloat16, False, marks=pytest.mark.smoke),
            pytest.param(2, 64, 512, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 16, 256, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


@EngramGateConvFwdFixture
def test_engram_gate_conv_fwd(M, seq_len, d, dtype, tune):
    test = EngramGateConvFwdTest(M, seq_len, d, dtype)
    op = EngramGateConvFwdOp(M, seq_len, d, tune=tune)
    inputs = test.gen_inputs()
    atol = 1e-1 if dtype == torch.float16 else 2e-1
    rtol = 1e-1
    test.check(op, *inputs, atol=atol, rtol=rtol)


class EngramGateConvBwdTest(EngramGateConvBwdWorkload, TestBase):
    pass


def _ref_rmsnorm(x, w, eps=1e-6):
    x_f = x.float()
    rrms = (x_f ** 2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    normed = x_f * rrms * w.float()
    return normed, rrms.squeeze(-1)


class EngramGateConvBwdFixture(FixtureBase):
    PARAMS = [
        ("M, seq_len, d, dtype, tune", [
            pytest.param(1, 32, 256, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(1, 32, 256, torch.bfloat16, False, marks=pytest.mark.smoke),
            pytest.param(2, 64, 512, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 16, 256, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2, 512, 256, torch.float16, False, marks=pytest.mark.full),
        ]),
    ]


@EngramGateConvBwdFixture
def test_engram_gate_conv_bwd(M, seq_len, d, dtype, tune):
    test = EngramGateConvBwdTest(M, seq_len, d, dtype)
    op = EngramGateConvBwdOp(M, seq_len, d, tune=tune)
    inputs = test.gen_inputs()
    atol = 2e-1 if dtype == torch.float16 else 3e-1
    rtol = 2e-1
    test.check(op, *inputs, atol=atol, rtol=rtol)

    # A data race varies run to run; allclose can still pass, so require two runs to match.
    run1 = [o.clone() for o in op(*inputs)]
    run2 = [o.clone() for o in op(*inputs)]
    for i, name in ((0, "dH"), (1, "dk"), (2, "dv")):
        max_err = (run1[i].float() - run2[i].float()).abs().max()
        assert torch.equal(run1[i], run2[i]), \
            f"{name} non-deterministic across runs (data race): max_err={max_err:.4e}"


class EngramDecodeTest(EngramDecodeWorkload, TestBase):
    pass


class EngramDecodeFixture(FixtureBase):
    PARAMS = [
        # (batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype, tune)
        ("batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype, tune", [
            pytest.param(1, 512, 256, 12, 4, 3, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(1, 512, 256, 12, 4, 3, torch.bfloat16, False, marks=pytest.mark.smoke),
            pytest.param(4, 1024, 512, 20, 4, 5, torch.float16, False, marks=pytest.mark.full),
            pytest.param(8, 512, 256, 18, 4, 3, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


@EngramDecodeFixture
def test_engram_decode(batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype, tune):
    test = EngramDecodeTest(batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype)
    op = EngramDecodeOp(
        batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, tune=tune,
    )
    inputs = test.gen_inputs()
    atol = 5e-2 if dtype == torch.float16 else 1e-1
    rtol = 5e-2
    test.check(op, *inputs, atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_engram_decode_multi_step():
    """Verify multi-step decode with growing conv_state and dilated conv."""
    B, d_mem, d = 2, 256, 256
    conv_kernel_size = 4
    dilation = 3
    max_conv_len = dilation * (conv_kernel_size - 1)  # = 9, minimum required
    dtype = torch.float16
    eps = 1e-6

    torch.manual_seed(123)
    W_K = torch.randn(d_mem, d, dtype=dtype, device="cuda") * 0.02
    W_V = torch.randn(d_mem, d, dtype=dtype, device="cuda") * 0.02
    rms_w_h = torch.ones(d, dtype=dtype, device="cuda")
    rms_w_v = torch.ones(d, dtype=dtype, device="cuda")
    conv_w = torch.randn(conv_kernel_size, d, dtype=dtype, device="cuda") * 0.02

    op = EngramDecodeOp(B, d_mem, d, max_conv_len, conv_kernel_size, dilation)

    # Start with empty conv_state (like empty KV cache)
    conv_state = torch.zeros(B, 0, d, dtype=dtype, device="cuda")
    conv_state_ref = conv_state.clone()

    num_steps = max_conv_len + 8  # go past growing phase into steady state
    for step in range(num_steps):
        e_t = torch.randn(B, d_mem, dtype=dtype, device="cuda") * 0.1
        h_t = torch.randn(B, d, dtype=dtype, device="cuda")

        y_op, conv_state = op(e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w)
        y_ref, conv_state_ref = engram_decode_step_torch(
            e_t, h_t, conv_state_ref, W_K, W_V, rms_w_h, rms_w_v, conv_w,
            max_conv_len, dilation, eps,
        )

        y_err = (y_op.float() - y_ref.float()).abs().max().item()
        # Compare valid portion of conv_state
        ref_len = conv_state_ref.shape[1]
        op_state_valid = conv_state[:, -ref_len:, :]
        s_err = (op_state_valid.float() - conv_state_ref.float()).abs().max().item()

        assert y_err < 0.1, f"Step {step}: y max_err={y_err:.6f}"
        assert s_err < 0.05, f"Step {step}: state max_err={s_err:.6f}"



if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
