import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.engram_decode import EngramDecodeOp
from workloads.ops.engram_decode import EngramDecodeTest as _EngramDecodeTestWorkload
from workloads.ops.engram_decode import engram_decode_step_torch


class EngramDecodeTest(_EngramDecodeTestWorkload, TestBase):
    def ref_program(self, e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w):
        y_ref, state_ref = engram_decode_step_torch(
            e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w,
            self.max_conv_len, self.dilation, self.eps,
        )
        return y_ref, state_ref


class EngramDecodeFixture(FixtureBase):
    PARAMS = [
        # (batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype, tune)
        ("batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype, tune", [
            pytest.param(1, 512, 256, 12, 4, 3, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(4, 1024, 512, 20, 4, 5, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1, 256, 256, 9, 4, 3, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(8, 512, 256, 18, 4, 3, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


@EngramDecodeFixture
def test_engram_decode(batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype, tune):
    test = EngramDecodeTest(batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype)
    op = EngramDecodeOp(
        batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype, tune=tune,
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

    op = EngramDecodeOp(B, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype)

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

    print(f"Multi-step decode test passed ({num_steps} steps, w={conv_kernel_size}, δ={dilation}).")
