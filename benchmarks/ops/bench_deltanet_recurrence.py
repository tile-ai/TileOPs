from typing import Optional

import torch

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import (
    BenchmarkBase,
    ManifestBenchmark,
    then_dtype,
    workload_params,
)
from tileops.manifest import load_workloads
from tileops.ops import DeltaNetDecodeFwdOp
from workloads.linear_attention import DeltaNetDecodeWorkload
from workloads.workload_base import FixtureBase

_OP_NAME = "DeltaNetDecodeFwdOp"


def deltanet_decode_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
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


class DeltaNetDecodeBenchmark(BenchmarkBase[DeltaNetDecodeWorkload]):
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
        (
            "batch, heads, dim_k, dim_v, dtype, tune",
            workload_params(
                load_workloads(_OP_NAME),
                then_dtype(
                    lambda w: (w["q_shape"][0], w["q_shape"][1], w["q_shape"][2], w["v_shape"][2]),
                    tune=False,
                ),
            ),
        ),
    ]


@DeltaNetDecodeBenchFixture
def test_deltanet_decode_bench(
    batch: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = DeltaNetDecodeWorkload(batch, heads, dim_k, dim_v, dtype)
    inputs = test.gen_inputs()

    op = DeltaNetDecodeFwdOp(tune=tune)
    bm = ManifestBenchmark(_OP_NAME, op, test)

    bm.compare(
        {
            "tileops": op,
            "torch": test.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(test.ref_program),
        },
        *inputs,
        record_as=op,
        params=locals(),
    )
