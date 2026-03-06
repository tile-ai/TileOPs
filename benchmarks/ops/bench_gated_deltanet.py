from typing import Optional, Tuple

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_gated_deltanet_fwd import GatedDeltaNetFwdFixture, GatedDeltaNetFwdTest
from tests.test_base import FixtureBase
from tileops.ops import GatedDeltaNetBwdOp, GatedDeltaNetFwdOp

# =============================================================================
# Forward benchmark
# =============================================================================

class GatedDeltaNetFwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        # Approximate: two matmuls per chunk (attn + state update) + compute_w_u
        flops = 2.0 * B * H * S * DK * DV
        return flops

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        elem = t.dtype.itemsize
        # q, k: B*H*S*DK; v, o: B*H*S*DV; g, beta: B*H*S
        return B * H * S * (2 * DK + 2 * DV + 2) * elem


@GatedDeltaNetFwdFixture
def test_gated_deltanet_fwd_bench(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GatedDeltaNetFwdTest(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    bm = GatedDeltaNetFwdBenchmark(test)
    inputs = test.gen_inputs()

    op = GatedDeltaNetFwdOp(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("gated_deltanet_fwd", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("gated_deltanet_fwd", locals(), result_bl, tag="baseline")


# =============================================================================
# Backward benchmark
# =============================================================================

class GatedDeltaNetBwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune", [
            (2, 64, 2, 64, 64, 32, torch.float32, False),
            (1, 128, 4, 64, 64, 32, torch.float32, False),
            (2, 64, 2, 64, 64, 32, torch.float16, False),
            (1, 128, 4, 64, 64, 32, torch.float16, False),
            (2, 64, 2, 64, 64, 32, torch.bfloat16, False),
            (1, 128, 4, 64, 64, 32, torch.bfloat16, False),
        ]),
    ]


class GatedDeltaNetBwdBenchmark(BenchmarkBase):

    def __init__(self, batch: int, heads: int, seq_len: int, dim_k: int, dim_v: int,
                 chunk_size: int, dtype: torch.dtype) -> None:
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.chunk_size = chunk_size
        self.dtype = dtype

    def calculate_flops(self) -> Optional[float]:
        B, H, S, DK, DV = self.batch, self.heads, self.seq_len, self.dim_k, self.dim_v
        flops = 4.0 * B * H * S * DK * DV
        return flops

    def calculate_memory(self) -> Optional[float]:
        B, H, S, DK, DV = self.batch, self.heads, self.seq_len, self.dim_k, self.dim_v
        elem = self.dtype.itemsize
        # do, q, k: B*H*S*DK; v: B*H*S*DV; g, beta: B*H*S; outputs: dq,dk,dv,dg,dbeta
        return B * H * S * (3 * DK + 2 * DV + 2 + 2 * DK + DV + 2) * elem


@GatedDeltaNetBwdFixture
def test_gated_deltanet_bwd_bench(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    B, H, S, DK, DV, BC = batch, heads, seq_len, dim_k, dim_v, chunk_size
    op = GatedDeltaNetBwdOp(B, H, S, DK, DV, BC, dtype, tune=tune)

    q = torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * 0.1
    k = torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * 0.1
    v = torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1
    g = -torch.rand(B, H, S, device="cuda", dtype=dtype)
    beta = torch.rand(B, H, S, device="cuda", dtype=dtype) * 0.5
    do = torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1

    bm = GatedDeltaNetBwdBenchmark(B, H, S, DK, DV, BC, dtype)
    result = bm.profile(op.forward, do, q, k, v, g, beta)
    BenchmarkReport.record("gated_deltanet_bwd", locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
