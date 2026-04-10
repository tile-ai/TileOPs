from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops.engram import EngramGateConvBwdOp
from workloads.ops.engram_bwd import CONV_KERNEL_SIZE, EngramGateConvBwdTest


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


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
