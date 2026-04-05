from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops.ssd_decode import SsdDecodeOp
from workloads.ops.ssd_decode import SsdDecodeFixture, SsdDecodeTest


def ssd_decode_ref(
    A: torch.Tensor,      # (H,)          float32
    dt: torch.Tensor,     # (B, H)        float32
    x: torch.Tensor,      # (B, H, P)     any dtype
    B_in: torch.Tensor,   # (B, G, N)     any dtype
    C_in: torch.Tensor,   # (B, G, N)     any dtype
    state: torch.Tensor,  # (B, H, P, N)  float32  -- updated in-place
) -> torch.Tensor:
    """PyTorch reference for ssd_decode (benchmark-local copy)."""
    B, H = dt.shape
    G = B_in.shape[1]
    heads_per_group = H // G

    dA = torch.exp(dt.float() * A.float())

    head_idx = torch.arange(H, device=B_in.device) // heads_per_group
    B_heads = B_in.float()[:, head_idx, :]
    C_heads = C_in.float()[:, head_idx, :]

    dBx = (
        dt.float()[:, :, None, None]
        * x.float()[:, :, :, None]
        * B_heads[:, :, None, :]
    )

    new_state = dA[:, :, None, None] * state.float() + dBx
    state.copy_(new_state)

    y_out = torch.einsum("bhpn,bhn->bhp", state.float(), C_heads)
    return y_out


class SsdDecodeBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        b, h, p, n = t.batch, t.n_heads, t.d_head, t.d_state
        # State update: dA * old_s + dt * x * B  -> 3 muls + 1 add per (b,h,p,n)
        # Output accum: new_s * C                 -> 1 mul + 1 add per (b,h,p,n)
        # Total: 6 * b * h * p * n
        return float(6 * b * h * p * n)

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        b, h, p, n, g = t.batch, t.n_heads, t.d_head, t.d_state, t.n_groups
        f32 = torch.float32.itemsize
        dtype_bytes = self.workload.dtype.itemsize
        # Reads: A(h) + dt(b,h) + x(b,h,p) + B_in(b,g,n) + C_in(b,g,n) + state(b,h,p,n)
        reads = (
            h * f32
            + b * h * f32
            + b * h * p * dtype_bytes
            + 2 * b * g * n * dtype_bytes
            + b * h * p * n * f32
        )
        # Writes: state(b,h,p,n) + y_out(b,h,p)
        writes = (b * h * p * n + b * h * p) * f32
        return float(reads + writes)


@SsdDecodeFixture
def test_ssd_decode_bench(batch, n_heads, d_head, d_state, n_groups, dtype, tune):
    test = SsdDecodeTest(batch, n_heads, d_head, d_state, n_groups, dtype)
    bm = SsdDecodeBenchmark(test)
    A, dt, x, B_in, C_in, state = test.gen_inputs()

    # Clone state before each profile run so both start from identical initial
    # conditions (op mutates state in-place across iterations).
    state_for_op = state.clone()
    state_bl = state.clone()

    op = SsdDecodeOp(batch, n_heads, d_head, d_state, n_groups, dtype, tune=tune)
    result = bm.profile(op, A, dt, x, B_in, C_in, state_for_op)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline(A, dt, x, B_in, C_in, state):
        return ssd_decode_ref(A, dt, x, B_in, C_in, state)

    result_bl = bm.profile(baseline, A, dt, x, B_in, C_in, state_bl)
    BenchmarkReport.record("ssd_decode", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
