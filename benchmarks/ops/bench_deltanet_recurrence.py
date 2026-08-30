import torch

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import (
    ManifestBenchmark,
    then_dtype,
    workload_params,
)
from tileops.manifest import load_workloads
from tileops.ops import DeltaNetDecodeFwdOp
from workloads.linear_attention import DeltaNetDecodeWorkload
from workloads.workload_base import FixtureBase


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


class DeltaNetDecodeBenchFixture(FixtureBase):
    PARAMS = [
        (
            "batch, heads, dim_k, dim_v, dtype, tune",
            workload_params(
                load_workloads(DeltaNetDecodeFwdOp),
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
    bm = ManifestBenchmark(op, test)

    bm.compare(
        {
            "tileops": op,
            "torch": test.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(test.ref_program),
        },
        *inputs,
    )
