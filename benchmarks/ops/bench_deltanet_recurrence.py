from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import DeltaNetDecodeOp
from workloads.base import FixtureBase
from workloads.ops.deltanet_recurrence import DeltaNetDecodeTest


class DeltaNetDecodeBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        B, H, DK, DV = t.batch, t.heads, t.dim_k, t.dim_v
        # Two matvecs: S@k and S@q -> 2 * B*H*DK*DV each (multiply + add)
        # dot product q.k -> B*H*DK
        # state update outer product -> B*H*DK*DV
        return 2.0 * B * H * (2 * DK * DV + DK * DV + DK)

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        B, H, DK, DV = t.batch, t.heads, t.dim_k, t.dim_v
        elem = t.dtype.itemsize
        # Read: q(DK) + k(DK) + v(DV) + beta(1) + state(DK*DV)
        # Write: o(DV) + new_state(DK*DV)
        return B * H * (2 * DK + DV + 1 + 2 * DK * DV + DV) * elem


class DeltaNetDecodeBenchFixture(FixtureBase):
    PARAMS = [
        ("batch, heads, dim_k, dim_v, dtype", [
            pytest.param(1, 32, 128, 128, torch.float32, marks=pytest.mark.smoke),
            pytest.param(1, 32, 128, 128, torch.bfloat16, marks=pytest.mark.full),
            pytest.param(8, 32, 128, 128, torch.float32, marks=pytest.mark.full),
            pytest.param(8, 32, 128, 128, torch.bfloat16, marks=pytest.mark.full),
            pytest.param(16, 32, 128, 128, torch.float32, marks=pytest.mark.full),
            pytest.param(16, 32, 128, 128, torch.bfloat16, marks=pytest.mark.full),
            pytest.param(32, 32, 128, 128, torch.float32, marks=pytest.mark.full),
            pytest.param(32, 32, 128, 128, torch.bfloat16, marks=pytest.mark.full),
            pytest.param(1, 32, 64, 64, torch.float32, marks=pytest.mark.full),
            pytest.param(8, 32, 64, 64, torch.float32, marks=pytest.mark.full),
            pytest.param(32, 32, 64, 64, torch.float32, marks=pytest.mark.full),
            pytest.param(64, 32, 128, 128, torch.float32, marks=pytest.mark.nightly),
            pytest.param(64, 32, 128, 128, torch.bfloat16, marks=pytest.mark.nightly),
        ]),
    ]


@DeltaNetDecodeBenchFixture
def test_deltanet_decode_bench(
    batch: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    dtype: torch.dtype,
) -> None:
    test = DeltaNetDecodeTest(batch, heads, dim_k, dim_v, dtype)
    bm = DeltaNetDecodeBenchmark(test)
    inputs = test.gen_inputs()

    op = DeltaNetDecodeOp(batch, heads, dim_k, dim_v, dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
