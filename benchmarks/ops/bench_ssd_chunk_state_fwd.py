from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops.ssd_chunk_state import SsdChunkStateFwdOp
from workloads.ops.ssd_chunk_state_fwd import SsdChunkStateFwdFixture, SsdChunkStateFwdTest


def ssd_chunk_state_fwd_ref(
    x: torch.Tensor,
    Bmat: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    n_groups: int,
    seq_idx=None,
) -> torch.Tensor:
    """PyTorch reference for ssd_chunk_state_fwd (benchmark-local copy)."""
    b, seq_len, h, p = x.shape
    _, _, c, Q = dt.shape
    n = Bmat.shape[-1]
    heads_per_group = h // n_groups

    x_chunked = x.float().reshape(b, c, Q, h, p)
    B_chunked = Bmat.float().reshape(b, c, Q, n_groups, n)
    B_heads = B_chunked[:, :, :, torch.arange(h) // heads_per_group, :]

    dA = dA_cumsum.float().permute(0, 2, 1, 3)
    dA_end = dA[:, :, :, -1:]
    decay = torch.exp(torch.clamp(dA_end - dA, max=0.0))

    dt_chunked = dt.float().permute(0, 2, 1, 3)
    weight = decay * dt_chunked

    if seq_idx is not None:
        seq_chunked = seq_idx.reshape(b, c, Q)
        seq_end = seq_chunked[..., -1:]
        same = (seq_chunked == seq_end).unsqueeze(3)
        weight = weight * same.permute(0, 1, 3, 2)

    w = weight.permute(0, 1, 3, 2).unsqueeze(-1).unsqueeze(-1)
    contrib = w * B_heads.unsqueeze(-1) * x_chunked.unsqueeze(-2)
    out = contrib.sum(dim=2)
    return out.permute(0, 1, 2, 4, 3)


class SsdChunkStateFwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        b, c, Q, h, p, n = (
            t.batch, t.num_chunks, t.chunk_len,
            t.n_heads, t.d_head, t.d_state,
        )
        # For each (b, c, h) block we do a rank-1 outer-product accumulation
        # over Q positions: Q * (p + n) multiply-adds, giving Q * p * n * 2 FLOPs
        # (treating the outer product as n*p MACs per position).
        flops = b * c * h * Q * n * p * 2
        return float(flops)

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        b, c, Q, h, p, n, g = (
            t.batch, t.num_chunks, t.chunk_len,
            t.n_heads, t.d_head, t.d_state, t.n_groups,
        )
        seq_len = c * Q
        elem = torch.tensor([], dtype=t.dtype).element_size()
        # Reads (input dtype): x + Bmat
        reads = (
            b * seq_len * h * p      # x
            + b * seq_len * g * n    # Bmat
        ) * elem
        # Reads (float32): dt + dA_cumsum
        reads += b * h * c * Q * 4 * 2
        # Writes (float32): out
        writes = b * c * h * n * p * 4
        return float(reads + writes)


@SsdChunkStateFwdFixture
def test_ssd_chunk_state_fwd_bench(
    batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune, has_seq_idx,
):
    test = SsdChunkStateFwdTest(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, has_seq_idx,
    )
    bm = SsdChunkStateFwdBenchmark(test)
    inputs = test.gen_inputs()

    op = SsdChunkStateFwdOp(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype,
        has_seq_idx=has_seq_idx, tune=tune,
    )
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline(x, Bmat, dt, dA_cumsum, seq_idx):
        return ssd_chunk_state_fwd_ref(x, Bmat, dt, dA_cumsum, n_groups=n_groups, seq_idx=seq_idx)
    result_bl = bm.profile(baseline, *inputs)
    BenchmarkReport.record("ssd_chunk_state_fwd", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
