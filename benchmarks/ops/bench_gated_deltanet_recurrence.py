import torch

from benchmarks.benchmark_base import (
    ManifestBenchmark,
    then_dtype,
    workload_params,
)
from tileops.manifest import load_workloads
from tileops.ops import GatedDeltaNetDecodeFwdOp
from workloads.linear_attention import GatedDeltaNetDecodeWorkload
from workloads.workload_base import FixtureBase


def gated_deltanet_decode_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for single-step gated delta rule."""
    q, k, v = q.float(), k.float(), v.float()
    g, beta = g.float(), beta.float()
    state = state.float()

    alpha = torch.exp(g)
    old_val = torch.einsum("bhkv,bhk->bhv", state, k)

    beta_unsq = beta.unsqueeze(-1)
    alpha_unsq = alpha.unsqueeze(-1)
    v_new = beta_unsq * v - alpha_unsq * beta_unsq * old_val

    o_inter = alpha_unsq * torch.einsum("bhkv,bhk->bhv", state, q)
    qk_dot = torch.einsum("bhk,bhk->bh", q, k).unsqueeze(-1)
    o_intra = qk_dot * v_new
    o = o_inter + o_intra

    new_state = alpha_unsq.unsqueeze(-1) * state + k.unsqueeze(-1) * v_new.unsqueeze(-2)

    return o, new_state


try:
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
except ImportError:
    fused_recurrent_gated_delta_rule = None


class GatedDeltaNetDecodeBenchFixture(FixtureBase):
    PARAMS = [
        (
            "batch, heads, dim_k, dim_v, dtype, tune",
            workload_params(
                load_workloads(GatedDeltaNetDecodeFwdOp),
                then_dtype(
                    lambda w: (w["q_shape"][0], w["q_shape"][1], w["q_shape"][2], w["v_shape"][2]),
                    tune=False,
                ),
            ),
        ),
    ]


@GatedDeltaNetDecodeBenchFixture
def test_gated_deltanet_decode_bench(
    batch: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GatedDeltaNetDecodeWorkload(batch, heads, dim_k, dim_v, dtype)
    inputs = test.gen_inputs()

    op = GatedDeltaNetDecodeFwdOp(tune=tune)
    bm = ManifestBenchmark(op, test)
    functors = {"tileops": op}

    if fused_recurrent_gated_delta_rule is not None:
        # --- FLA: fused_recurrent_gated_delta_rule with T=1 ---
        q, k, v, g, beta, state = inputs
        q_fla = q.unsqueeze(1)  # [B, H, DK] -> [B, 1, H, DK]
        k_fla = k.unsqueeze(1)
        v_fla = v.unsqueeze(1)
        g_fla = g.unsqueeze(1)  # [B, H] -> [B, 1, H]
        beta_fla = beta.unsqueeze(1)

        state_fla = state.contiguous()

        def fla_decode():
            return fused_recurrent_gated_delta_rule(
                q_fla,
                k_fla,
                v_fla,
                g=g_fla,
                beta=beta_fla,
                initial_state=state_fla,
                output_final_state=True,
            )

        functors["fla"] = (fla_decode, ())
    else:
        # --- Torch reference baseline ---
        functors["torch-ref"] = test.ref_program

    bm.compare(functors, *inputs)
