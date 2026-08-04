"""Benchmarks for the Engram gate-conv and decode ops.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from each op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.

One ``test_*_bench`` per op, so the validator's L4 AST check can tie each
``load_workloads("<OpName>")`` call to its manifest entry.
"""

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workload_field_params,
)
from tileops.manifest import load_workloads
from tileops.ops.engram import EngramGateConvBwdOp, EngramGateConvFwdOp
from tileops.ops.engram_decode import EngramDecodeOp
from workloads.engram import (
    CONV_KERNEL_SIZE,
    EngramDecodeWorkload,
    EngramGateConvBwdWorkload,
    EngramGateConvFwdWorkload,
)

# Autotuning is a bench-run policy, not a workload property; manifest
# workloads do not carry it.
_TUNE = True


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


_ENGRAM_GATE_CONV_FWD_OP = "EngramGateConvFwdOp"
_ENGRAM_GATE_CONV_FWD_PARAMS = workload_field_params(
    load_workloads(_ENGRAM_GATE_CONV_FWD_OP), ("M", "seq_len", "d", "dtype"),
)


@pytest.mark.parametrize("M, seq_len, d, dtype", _ENGRAM_GATE_CONV_FWD_PARAMS)
def test_engram_gate_conv_fwd_bench(M, seq_len, d, dtype):
    test = EngramGateConvFwdWorkload(M, seq_len, d, dtype)
    inputs = test.gen_inputs()

    op = EngramGateConvFwdOp(M, seq_len, d, tune=_TUNE)
    bm = ManifestBenchmark(_ENGRAM_GATE_CONV_FWD_OP, op, test)
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


class EngramGateConvBwdTestBaseline(EngramGateConvBwdWorkload):
    """Adds baseline ref_program for benchmark profiling."""

    def ref_program(self, dY, H, k, v, rms_w_h, rms_w_v, conv_w,
                    vhat, alpha, rrms_h, rrms_k, rrms_v):
        return ref_engram_gate_conv_bwd(
            dY, H, k, v, rms_w_h, rms_w_v, conv_w,
            vhat, alpha, rrms_h, rrms_k, rrms_v, self.eps,
        )


_ENGRAM_GATE_CONV_BWD_OP = "EngramGateConvBwdOp"
_ENGRAM_GATE_CONV_BWD_PARAMS = workload_field_params(
    load_workloads(_ENGRAM_GATE_CONV_BWD_OP), ("M", "seq_len", "d", "dtype"),
)


@pytest.mark.parametrize("M, seq_len, d, dtype", _ENGRAM_GATE_CONV_BWD_PARAMS)
def test_engram_gate_conv_bwd_bench(M, seq_len, d, dtype):
    test = EngramGateConvBwdTestBaseline(M, seq_len, d, dtype)
    inputs = test.gen_inputs()

    op = EngramGateConvBwdOp(M, seq_len, d, tune=_TUNE)
    bm = ManifestBenchmark(_ENGRAM_GATE_CONV_BWD_OP, op, test)
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


_ENGRAM_DECODE_OP = "EngramDecodeOp"
_ENGRAM_DECODE_PARAMS = workload_field_params(
    load_workloads(_ENGRAM_DECODE_OP),
    ("batch", "d_mem", "d", "max_conv_len", "conv_kernel_size", "dilation", "dtype"),
)


@pytest.mark.parametrize(
    "batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype",
    _ENGRAM_DECODE_PARAMS,
)
def test_engram_decode_bench(batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype):
    test = EngramDecodeWorkload(batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype)
    inputs = test.gen_inputs()

    op = EngramDecodeOp(
        batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, tune=_TUNE,
    )
    bm = ManifestBenchmark(_ENGRAM_DECODE_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline(*args):
        return engram_decode_step_torch(*args, max_conv_len=max_conv_len, dilation=dilation)
    result_bl = bm.profile(baseline, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
