from typing import Optional, Tuple

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import DeltaNetDecodeOp
from workloads.base import FixtureBase
from workloads.deltanet_recurrence import DeltaNetDecodeTest


def deltanet_decode_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for single-step delta rule (ungated)."""
    q, k, v = q.float(), k.float(), v.float()
    beta = beta.float()
    state = state.float()

    old_val = torch.einsum("bhkv,bhk->bhv", state, k)
    beta_unsq = beta.unsqueeze(-1)
    v_new = beta_unsq * (v - old_val)

    o_inter = torch.einsum("bhkv,bhk->bhv", state, q)
    qk_dot = torch.einsum("bhk,bhk->bh", q, k).unsqueeze(-1)
    o_intra = qk_dot * v_new
    o = o_inter + o_intra

    new_state = state + k.unsqueeze(-1) * v_new.unsqueeze(-2)

    return o, new_state


class _DeltaNetDecodeTestBaseline(DeltaNetDecodeTest):
    """Adds baseline ref_program for benchmark profiling."""

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        o, new_state = deltanet_decode_torch(q, k, v, beta, state)
        return o.to(self.dtype), new_state.to(self.dtype)


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
    test = _DeltaNetDecodeTestBaseline(batch, heads, dim_k, dim_v, dtype)
    bm = DeltaNetDecodeBenchmark(test)
    inputs = test.gen_inputs()

    op = DeltaNetDecodeOp(batch, heads, dim_k, dim_v, dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
