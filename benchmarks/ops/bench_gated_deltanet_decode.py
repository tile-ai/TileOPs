from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_gated_deltanet_decode import GatedDeltaNetDecodeTest
from tests.test_base import FixtureBase
from tileops.ops import GatedDeltaNetDecodeOp


class GatedDeltaNetDecodeBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        B, H, DK, DV = t.batch, t.heads, t.dim_k, t.dim_v
        # Two matvecs: S@k and S@q -> 2 * B*H*DK*DV each (multiply + add)
        # dot product q.k -> B*H*DK
        # state update outer product -> B*H*DK*DV
        return 2.0 * B * H * (2 * DK * DV + DK * DV + DK)

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        B, H, DK, DV = t.batch, t.heads, t.dim_k, t.dim_v
        elem = 4  # float32
        # Read: q(DK) + k(DK) + v(DV) + g(1) + beta(1) + state(DK*DV)
        # Write: o(DV) + new_state(DK*DV)
        return B * H * (2 * DK + DV + 2 + 2 * DK * DV + DV) * elem


class GatedDeltaNetDecodeBenchFixture(FixtureBase):
    PARAMS = [
        ("batch, heads, dim_k, dim_v, dtype", [
            # Typical MiMo-like configs
            (1, 32, 128, 128, torch.float32),
            (1, 32, 128, 128, torch.bfloat16),
            (8, 32, 128, 128, torch.float32),
            (8, 32, 128, 128, torch.bfloat16),
            (16, 32, 128, 128, torch.float32),
            (16, 32, 128, 128, torch.bfloat16),
            (32, 32, 128, 128, torch.float32),
            (32, 32, 128, 128, torch.bfloat16),
            # Smaller head dim
            (1, 32, 64, 64, torch.float32),
            (8, 32, 64, 64, torch.float32),
            (32, 32, 64, 64, torch.float32),
            # Larger batch (continuous batching scenario)
            (64, 32, 128, 128, torch.float32),
            (64, 32, 128, 128, torch.bfloat16),
        ]),
    ]


@GatedDeltaNetDecodeBenchFixture
def test_gated_deltanet_decode_bench(
    batch: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    dtype: torch.dtype,
) -> None:
    test = GatedDeltaNetDecodeTest(batch, heads, dim_k, dim_v, dtype)
    bm = GatedDeltaNetDecodeBenchmark(test)
    inputs = test.gen_inputs()

    op = GatedDeltaNetDecodeOp(batch, heads, dim_k, dim_v, dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("gated_deltanet_decode", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("gated_deltanet_decode", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
