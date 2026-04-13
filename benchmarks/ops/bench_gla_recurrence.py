"""Benchmark: TileOPs GLA decode vs FLA fused_recurrent_gla (T=1).

Compares single-step decode latency across batch sizes, dimensions, and dtypes.

When FLA is not installed, benchmarks still run using a pure-torch reference
implementation as baseline, so CI is never blocked by a missing optional dependency.
"""
from typing import Optional, Tuple

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import GLADecodeOp
from workloads.base import FixtureBase
from workloads.gla_recurrence import GLADecodeTest


def gla_decode_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    state: torch.Tensor,
    scale: float = -1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for single-step GLA recurrence."""
    DK = q.shape[-1]
    if scale <= 0:
        scale = DK ** -0.5

    q, k, v = q.float(), k.float(), v.float()
    gk = gk.float()
    state = state.float()

    alpha = torch.exp(gk)
    new_state = alpha.unsqueeze(-1) * state + k.unsqueeze(-1) * v.unsqueeze(-2)
    o = scale * torch.einsum("bhk,bhkv->bhv", q, new_state)

    return o, new_state


class _GLADecodeTestBaseline(GLADecodeTest):
    """Adds baseline ref_program for benchmark profiling."""

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gk: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        o, new_state = gla_decode_torch(q, k, v, gk, state, self.scale)
        return o.to(self.dtype), new_state.to(self.dtype)

try:
    from fla.ops.gla import fused_recurrent_gla
except ImportError:
    fused_recurrent_gla = None


class GLADecodeBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        B, H, DK, DV = t.batch, t.heads, t.dim_k, t.dim_v
        # One matvec: S @ q_gated -> B*H*DK*DV (multiply + add)
        # dot product q.k -> B*H*DK
        # state update: element-wise scale + outer product -> B*H*DK*DV
        return 2.0 * B * H * (DK * DV + DK * DV + DK)

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        B, H, DK, DV = t.batch, t.heads, t.dim_k, t.dim_v
        elem = t.dtype.itemsize
        # Read: q(DK) + k(DK) + v(DV) + gk(DK) + state(DK*DV)
        # Write: o(DV) + new_state(DK*DV)
        return B * H * (3 * DK + DV + 2 * DK * DV + DV) * elem


class GLADecodeBenchFixture(FixtureBase):
    PARAMS = [
        ("batch, heads, dim_k, dim_v, dtype", [
            (1, 32, 64, 64, torch.float32),
            (1, 32, 128, 128, torch.float32),
            (1, 32, 128, 128, torch.float16),
            (1, 32, 128, 128, torch.bfloat16),
            (8, 32, 128, 128, torch.float32),
            (8, 32, 128, 128, torch.float16),
            (8, 32, 128, 128, torch.bfloat16),
            (16, 32, 128, 128, torch.float32),
            (16, 32, 128, 128, torch.float16),
            (16, 32, 128, 128, torch.bfloat16),
            (32, 32, 128, 128, torch.float32),
            (32, 32, 128, 128, torch.float16),
            (32, 32, 128, 128, torch.bfloat16),
            (64, 32, 128, 128, torch.float32),
            (64, 32, 128, 128, torch.float16),
            (64, 32, 128, 128, torch.bfloat16),
        ]),
    ]


@GLADecodeBenchFixture
def test_gla_decode_bench(
    batch: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    dtype: torch.dtype,
) -> None:
    scale = dim_k ** -0.5
    test = _GLADecodeTestBaseline(batch, heads, dim_k, dim_v, dtype, scale=scale)
    bm = GLADecodeBenchmark(test)
    inputs = test.gen_inputs()

    # --- TileOPs ---
    op = GLADecodeOp(batch, heads, dim_k, dim_v, scale=scale, dtype=dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    if fused_recurrent_gla is not None:
        # --- FLA: fused_recurrent_gla with T=1 ---
        q, k, v, gk, state = inputs
        q_fla = q.unsqueeze(1)
        k_fla = k.unsqueeze(1)
        v_fla = v.unsqueeze(1)
        gk_fla = gk.unsqueeze(1)

        def fla_decode():
            return fused_recurrent_gla(
                q_fla, k_fla, v_fla, gk=gk_fla,
                scale=scale, initial_state=state.contiguous(),
                output_final_state=True,
            )

        result_fla = bm.profile(fla_decode)
        BenchmarkReport.record(op, locals(), result_fla, tag="fla")
    else:
        # --- Torch reference baseline ---
        result_bl = bm.profile(test.ref_program, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
