from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops.engram_decode import EngramDecodeOp
from workloads.ops.engram_decode import EngramDecodeTest


def _rmsnorm(x, w, eps=1e-6):
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

    h_norm, _ = _rmsnorm(h_t.unsqueeze(1), rms_w_h)
    k_norm, _ = _rmsnorm(k.unsqueeze(1).to(h_t.dtype), rms_w_h)
    h_norm = h_norm.squeeze(1)
    k_norm = k_norm.squeeze(1)

    dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
    alpha = torch.sigmoid(dot / (d ** 0.5))
    v_hat = alpha * v

    v_hat_norm, _ = _rmsnorm(v_hat.unsqueeze(1).to(h_t.dtype), rms_w_v)
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


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
