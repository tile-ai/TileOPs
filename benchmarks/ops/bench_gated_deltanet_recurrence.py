from typing import Optional, Tuple

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import GatedDeltaNetDecodeOp
from workloads.base import FixtureBase
from workloads.ops.gated_deltanet_recurrence import (
    GatedDeltaNetDecodeTest,
    gated_deltanet_decode_torch,
)


class _GatedDeltaNetDecodeTestBaseline(GatedDeltaNetDecodeTest):
    """Adds baseline ref_program for benchmark profiling."""

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        o, new_state = gated_deltanet_decode_torch(q, k, v, g, beta, state)
        return o.to(self.dtype), new_state.to(self.dtype)

try:
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
except ImportError:
    fused_recurrent_gated_delta_rule = None


class GatedDeltaNetDecodeBenchmark(BenchmarkBase):

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
        # Read: q(DK) + k(DK) + v(DV) + g(1) + beta(1) + state(DK*DV)
        # Write: o(DV) + new_state(DK*DV)
        return B * H * (2 * DK + DV + 2 + 2 * DK * DV + DV) * elem


class GatedDeltaNetDecodeBenchFixture(FixtureBase):
    PARAMS = [
        ("batch, heads, dim_k, dim_v, dtype", [
            (1, 32, 128, 128, torch.float32),
            (1, 32, 128, 128, torch.bfloat16),
            (8, 32, 128, 128, torch.float32),
            (8, 32, 128, 128, torch.bfloat16),
            (16, 32, 128, 128, torch.float32),
            (16, 32, 128, 128, torch.bfloat16),
            (32, 32, 128, 128, torch.float32),
            (32, 32, 128, 128, torch.bfloat16),
            (1, 32, 64, 64, torch.float32),
            (8, 32, 64, 64, torch.float32),
            (32, 32, 64, 64, torch.float32),
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
    test = _GatedDeltaNetDecodeTestBaseline(batch, heads, dim_k, dim_v, dtype)
    bm = GatedDeltaNetDecodeBenchmark(test)
    inputs = test.gen_inputs()

    op = GatedDeltaNetDecodeOp(batch, heads, dim_k, dim_v, dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    if fused_recurrent_gated_delta_rule is not None:
        # --- FLA: fused_recurrent_gated_delta_rule with T=1 ---
        q, k, v, g, beta, state = inputs
        q_fla = q.unsqueeze(1)       # [B, H, DK] -> [B, 1, H, DK]
        k_fla = k.unsqueeze(1)
        v_fla = v.unsqueeze(1)
        g_fla = g.unsqueeze(1)       # [B, H] -> [B, 1, H]
        beta_fla = beta.unsqueeze(1)

        state_fla = state.contiguous()

        def fla_decode():
            return fused_recurrent_gated_delta_rule(
                q_fla, k_fla, v_fla, g=g_fla, beta=beta_fla,
                initial_state=state_fla,
                output_final_state=True,
            )

        result_fla = bm.profile(fla_decode)
        BenchmarkReport.record(op, locals(), result_fla, tag="fla")
    else:
        # --- Torch reference baseline ---
        result_bl = bm.profile(test.ref_program, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
