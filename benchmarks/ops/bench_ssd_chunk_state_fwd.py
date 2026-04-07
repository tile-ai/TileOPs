from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops.ssd_chunk_state_fwd import SsdChunkStateFwdOp
from workloads.ops.ssd_chunk_state_fwd import SsdChunkStateFwdTest


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

try:
    from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_state_fwd
except ImportError:
    _chunk_state_fwd = None


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


_SSD_CHUNK_STATE_FWD_BENCH_PARAMS = [
    pytest.param(1, 2,  64, 4,  64,  32, 1, torch.float16,  False, False, id="b1-c2-L64-h4-p64-n32-g1-fp16"),
    pytest.param(2, 4,  64, 8,  64,  64, 2, torch.float16,  False, False, id="b2-c4-L64-h8-p64-n64-g2-fp16"),
    pytest.param(1, 2, 128, 4, 128,  32, 1, torch.bfloat16, False, False, id="b1-c2-L128-h4-p128-n32-g1-bf16"),
    pytest.param(2, 2,  64, 4,  64,  32, 2, torch.bfloat16, False, False, id="b2-c2-L64-h4-p64-n32-g2-bf16"),
    pytest.param(2, 4,  64, 8,  64,  64, 2, torch.float16,  False, True,  id="b2-c4-L64-h8-p64-n64-g2-seqidx-fp16"),
]


@pytest.mark.parametrize(
    "batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune, has_seq_idx",
    _SSD_CHUNK_STATE_FWD_BENCH_PARAMS,
)
def test_ssd_chunk_state_fwd_bench(
    batch: int, num_chunks: int, chunk_len: int, n_heads: int, d_head: int,
    d_state: int, n_groups: int, dtype: torch.dtype, tune: bool, has_seq_idx: bool,
) -> None:
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

    if _chunk_state_fwd is not None:
        x, Bmat, dt, dA_cumsum, seq_idx = inputs

        def mamba_fwd():
            # mamba_ssm expects (b, c, h, L) layout; TileOPs uses (b, h, c, L)
            return _chunk_state_fwd(
                Bmat.contiguous(),
                x.contiguous(),
                dt.permute(0, 2, 1, 3).contiguous(),
                dA_cumsum.permute(0, 2, 1, 3).contiguous(),
                seq_idx=seq_idx,
            )

        result_mamba = bm.profile(mamba_fwd)
        BenchmarkReport.record(op, locals(), result_mamba, tag="mamba")
    else:
        def baseline(x, Bmat, dt, dA_cumsum, seq_idx):
            return ssd_chunk_state_fwd_ref(x, Bmat, dt, dA_cumsum, n_groups=n_groups, seq_idx=seq_idx)
        result_bl = bm.profile(baseline, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
