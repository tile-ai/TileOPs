from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.ops.engram import EngramGateConvBwdOp, EngramGateConvFwdOp
from tileops.ops.engram_decode import EngramDecodeOp
from workloads.engram import (
    CONV_KERNEL_SIZE,
    EngramDecodeTest,
    EngramGateConvBwdTest,
    EngramGateConvFwdTest,
)


def _rmsnorm(x, w, eps=1e-6):
    """Returns (normed, rrms)."""
    x_f = x.float()
    rrms = (x_f ** 2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    normed = x_f * rrms * w.float()
    return normed, rrms.squeeze(-1)


def engram_gate_conv_fwd_torch(H, k, v, rms_w_h, rms_w_v, conv_w, eps=1e-6):
    """PyTorch reference for Engram GateConv forward."""
    M, T, d = H.shape

    h_norm, rrms_h = _rmsnorm(H, rms_w_h, eps)
    k_norm, rrms_k = _rmsnorm(k, rms_w_h, eps)

    dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
    alpha = torch.sigmoid(dot / (d ** 0.5))

    v_hat = alpha * v.float()

    v_hat_norm, rrms_v = _rmsnorm(v_hat.to(H.dtype), rms_w_v, eps)

    v_perm = v_hat_norm.float().permute(0, 2, 1)
    v_padded = F.pad(v_perm, (CONV_KERNEL_SIZE - 1, 0))
    conv_w_expanded = conv_w.float().T.unsqueeze(1)
    conv_out = F.conv1d(v_padded, conv_w_expanded, groups=d).permute(0, 2, 1)

    Y = F.silu(conv_out) + v_hat.float()

    return (
        Y.to(H.dtype),
        v_hat.to(H.dtype),
        alpha.squeeze(-1).float(),
        rrms_h.float(),
        rrms_k.float(),
        rrms_v.float(),
    )


class EngramGateConvFwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        M, T, d = t.M, t.seq_len, t.d
        # 2x RMSNorm(d): ~4d each -> 8*M*T*d
        # dot product (d): 2*M*T*d
        # sigmoid: ~10*M*T
        # gated mul: M*T*d
        # RMSNorm(v_hat): 4*M*T*d
        # conv (kernel=4): 4*2*M*T*d
        # SiLU: ~10*M*T
        # residual add: M*T*d
        return M * T * (8 * d + 2 * d + d + 4 * d + 8 * d + d) + 20 * M * T

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        M, T, d = t.M, t.seq_len, t.d
        elem = torch.tensor([], dtype=t.dtype).element_size()
        # Read: H + k + v (3*M*T*d) + weights (2*d + 4*d)
        # Write: Y + vhat (2*M*T*d) + alpha + rrms*3 (4*M*T * 4bytes)
        return (5 * M * T * d) * elem + 4 * M * T * 4 + 6 * d * elem


_ENGRAM_GATE_CONV_FWD_BENCH_PARAMS = [
    pytest.param(1, 32, 256, torch.float16, True, id="fp16-small"),
    pytest.param(2, 64, 512, torch.float16, True, id="fp16-mainstream"),
    pytest.param(1, 128, 256, torch.bfloat16, True, id="bf16-long-seq"),
    pytest.param(2, 16, 256, torch.bfloat16, True, id="bf16-batched"),
]


@pytest.mark.parametrize("M, seq_len, d, dtype, tune", _ENGRAM_GATE_CONV_FWD_BENCH_PARAMS)
def test_engram_gate_conv_fwd_bench(M, seq_len, d, dtype, tune):
    test = EngramGateConvFwdTest(M, seq_len, d, dtype)
    bm = EngramGateConvFwdBenchmark(test)
    inputs = test.gen_inputs()

    op = EngramGateConvFwdOp(M, seq_len, d, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline(*args):
        return engram_gate_conv_fwd_torch(*args)
    result_bl = bm.profile(baseline, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


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


class _EngramGateConvBwdTestBaseline(EngramGateConvBwdTest):
    """Adds baseline ref_program for benchmark profiling."""

    def ref_program(self, dY, H, k, v, rms_w_h, rms_w_v, conv_w,
                    vhat, alpha, rrms_h, rrms_k, rrms_v):
        return ref_engram_gate_conv_bwd(
            dY, H, k, v, rms_w_h, rms_w_v, conv_w,
            vhat, alpha, rrms_h, rrms_k, rrms_v, self.eps,
        )


class EngramGateConvBwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        M, T, d = t.M, t.seq_len, t.d
        fwd_flops = M * T * (8 * d + 2 * d + d + 4 * d + 8 * d + d) + 20 * M * T
        return int(fwd_flops * 2.5)

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        M, T, d = t.M, t.seq_len, t.d
        elem = torch.tensor([], dtype=t.dtype).element_size()
        read_bytes = 5 * M * T * d * elem + 6 * d * elem + 4 * M * T * 4
        write_bytes = 3 * M * T * d * elem + 10 * d * 4 + M * T * d * 4
        return read_bytes + write_bytes


_ENGRAM_GATE_CONV_BWD_BENCH_PARAMS = [
    pytest.param(1, 32, 256, torch.float16, True, id="fp16-small"),
    pytest.param(2, 64, 512, torch.float16, True, id="fp16-mainstream"),
    pytest.param(1, 128, 256, torch.bfloat16, True, id="bf16-long-seq"),
    pytest.param(2, 16, 256, torch.bfloat16, True, id="bf16-batched"),
]


@pytest.mark.parametrize("M, seq_len, d, dtype, tune", _ENGRAM_GATE_CONV_BWD_BENCH_PARAMS)
def test_engram_gate_conv_bwd_bench(M, seq_len, d, dtype, tune):
    test = _EngramGateConvBwdTestBaseline(M, seq_len, d, dtype)
    bm = EngramGateConvBwdBenchmark(test)
    inputs = test.gen_inputs()

    op = EngramGateConvBwdOp(M, seq_len, d, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    @torch.enable_grad()
    def ref_with_grad(*args):
        return test.ref_program(*args)

    result_bl = bm.profile(ref_with_grad, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


def _rmsnorm_decode(x, w, eps=1e-6):
    x_f = x.float()
    rrms = (x_f ** 2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return (x_f * rrms * w.float()), rrms


def engram_decode_step_torch(
    e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w,
    max_conv_len, dilation, eps=1e-6,
):
    """PyTorch reference for a single decode step with dilated causal conv."""
    B, d = h_t.shape
    w = conv_w.shape[0]
    L = conv_state.shape[1]

    k = e_t.float() @ W_K.float()
    v = e_t.float() @ W_V.float()

    h_norm, _ = _rmsnorm_decode(h_t.unsqueeze(1), rms_w_h)
    k_norm, _ = _rmsnorm_decode(k.unsqueeze(1).to(h_t.dtype), rms_w_h)
    h_norm = h_norm.squeeze(1)
    k_norm = k_norm.squeeze(1)

    dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
    alpha = torch.sigmoid(dot / (d ** 0.5))
    v_hat = alpha * v

    v_hat_norm, _ = _rmsnorm_decode(v_hat.unsqueeze(1).to(h_t.dtype), rms_w_v)
    v_hat_norm = v_hat_norm.squeeze(1)

    if max_conv_len > L:
        padded_state = F.pad(conv_state.float(), (0, 0, max_conv_len - L, 0))
    else:
        padded_state = conv_state.float()

    conv_out = torch.zeros(B, d, device=h_t.device)
    for p in range(w - 1):
        state_idx = max_conv_len - (w - 1 - p) * dilation
        if 0 <= state_idx < max_conv_len:
            conv_out += conv_w[p].float().unsqueeze(0) * padded_state[:, state_idx, :]
    conv_out += conv_w[w - 1].float().unsqueeze(0) * v_hat_norm

    if max_conv_len > L:
        new_conv_state = torch.cat([
            conv_state,
            v_hat_norm.unsqueeze(1).to(conv_state.dtype),
        ], dim=1)
    else:
        new_conv_state = torch.cat([
            conv_state[:, 1:, :],
            v_hat_norm.unsqueeze(1).to(conv_state.dtype),
        ], dim=1)

    y_t = F.silu(conv_out) + v_hat
    return y_t.to(h_t.dtype), new_conv_state


class EngramDecodeBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        B, d_mem, d, w = t.batch, t.d_mem, t.d, t.conv_kernel_size
        # GEMV: 2 * B * d_mem * d (k) + 2 * B * d_mem * d (v)
        # 2x RMSNorm(d): ~4d each -> 8*B*d
        # dot product: 2*B*d, sigmoid: ~10*B, gated mul: B*d
        # RMSNorm(v_hat): 4*B*d
        # dilated conv (w taps): w*2*B*d
        # SiLU + residual: ~10*B + B*d
        return (4 * B * d_mem * d
                + B * (8 * d + 2 * d + d + 4 * d + w * 2 * d + d)
                + 20 * B)

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        B, d_mem, d, mcl, w = t.batch, t.d_mem, t.d, t.max_conv_len, t.conv_kernel_size
        elem = torch.tensor([], dtype=t.dtype).element_size()
        # Read: e_t (B*d_mem) + h_t (B*d) + conv_state (B*mcl*d) + W_K,W_V (2*d_mem*d)
        #        + weights (2*d + w*d)
        # Write: y_t (B*d) + new_conv_state (B*mcl*d)
        return (B * d_mem + B * d + 2 * B * mcl * d + 2 * d_mem * d
                + 2 * d + w * d + B * d) * elem


_ENGRAM_DECODE_BENCH_PARAMS = [
    pytest.param(1, 512, 256, 12, 4, 3, torch.float16, True, id="fp16-mainstream"),
    pytest.param(4, 1024, 512, 20, 4, 5, torch.float16, True, id="fp16-large"),
    pytest.param(8, 512, 256, 18, 4, 3, torch.bfloat16, True, id="bf16-batched"),
]


@pytest.mark.parametrize(
    "batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype, tune",
    _ENGRAM_DECODE_BENCH_PARAMS,
)
def test_engram_decode_bench(batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype, tune):
    test = EngramDecodeTest(batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype)
    bm = EngramDecodeBenchmark(test)
    inputs = test.gen_inputs()

    op = EngramDecodeOp(
        batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype, tune=tune,
    )
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline(*args):
        return engram_decode_step_torch(*args, max_conv_len=max_conv_len, dilation=dilation)
    result_bl = bm.profile(baseline, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")
