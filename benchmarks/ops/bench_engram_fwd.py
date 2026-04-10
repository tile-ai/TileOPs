from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops.engram import EngramGateConvFwdOp
from workloads.ops.engram_fwd import CONV_KERNEL_SIZE, EngramGateConvFwdTest


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


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
