from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_gated_deltanet_bwd import GatedDeltaNetBwdTest
from tests.ops.test_gated_deltanet_fwd import GatedDeltaNetFwdTest
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


class GatedDeltaNetFwdBenchFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune", [
            (2, 1024, 4, 64, 64, 32, torch.float32, False),
            (2, 2048, 4, 64, 64, 32, torch.float32, False),
            (2, 4096, 4, 64, 64, 32, torch.float32, False),
            (2, 1024, 4, 64, 64, 32, torch.float16, False),
            (2, 2048, 4, 64, 64, 32, torch.float16, False),
            (2, 4096, 4, 64, 64, 32, torch.float16, False),
            (2, 1024, 4, 64, 64, 32, torch.bfloat16, False),
            (2, 2048, 4, 64, 64, 32, torch.bfloat16, False),
            (2, 4096, 4, 64, 64, 32, torch.bfloat16, False),
        ]),
    ]


@GatedDeltaNetFwdBenchFixture
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
            (2, 1024, 4, 64, 64, 32, torch.float32, False),
            (2, 2048, 4, 64, 64, 32, torch.float32, False),
            (2, 4096, 4, 64, 64, 32, torch.float32, False),
            (2, 1024, 4, 64, 64, 32, torch.float16, False),
            (2, 2048, 4, 64, 64, 32, torch.float16, False),
            (2, 4096, 4, 64, 64, 32, torch.float16, False),
            (2, 1024, 4, 64, 64, 32, torch.bfloat16, False),
            (2, 2048, 4, 64, 64, 32, torch.bfloat16, False),
            (2, 4096, 4, 64, 64, 32, torch.bfloat16, False),
        ]),
    ]


class GatedDeltaNetBwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        flops = 4.0 * B * H * S * DK * DV
        return flops

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        elem = t.dtype.itemsize
        # Inputs: q,k (2*DK), v,do (2*DV), g,beta (2); Outputs: dq,dk (2*DK), dv (DV), dg,dbeta (2)
        return B * H * S * (4 * DK + 3 * DV + 4) * elem


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
    test = GatedDeltaNetBwdTest(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    bm = GatedDeltaNetBwdBenchmark(test)
    inputs = test.gen_inputs()

    op = GatedDeltaNetBwdOp(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype, tune=tune)
    result = bm.profile(op.forward, *inputs)
    BenchmarkReport.record("gated_deltanet_bwd", locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
