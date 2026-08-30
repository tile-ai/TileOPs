"""Benchmark: TileOPs GLA decode vs FLA fused_recurrent_gla (T=1).

Compares single-step decode latency across batch sizes, dimensions, and dtypes.

When FLA is not installed, benchmarks still run using a pure-torch reference
implementation as baseline, so CI is never blocked by a missing optional dependency.
"""

import torch

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import (
    ManifestBenchmark,
    then_dtype,
    workload_params,
)
from tileops.manifest import load_workloads
from tileops.ops import GLADecodeFwdOp
from workloads.linear_attention import GLADecodeWorkload
from workloads.workload_base import FixtureBase


def gla_decode_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    state: torch.Tensor,
    scale: float = -1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for single-step GLA recurrence."""
    DK = q.shape[-1]
    if scale <= 0:
        scale = DK**-0.5

    q, k, v = q.float(), k.float(), v.float()
    gk = gk.float()
    state = state.float()

    alpha = torch.exp(gk)
    new_state = alpha.unsqueeze(-1) * state + k.unsqueeze(-1) * v.unsqueeze(-2)
    o = scale * torch.einsum("bhk,bhkv->bhv", q, new_state)

    return o, new_state


try:
    from fla.ops.gla import fused_recurrent_gla
except ImportError:
    fused_recurrent_gla = None


class GLADecodeBenchFixture(FixtureBase):
    PARAMS = [
        (
            "batch, heads, dim_k, dim_v, scale, dtype, tune",
            workload_params(
                load_workloads(GLADecodeFwdOp),
                then_dtype(
                    lambda w: (
                        w["q_shape"][0],
                        w["q_shape"][1],
                        w["q_shape"][2],
                        w["v_shape"][2],
                        w.get("scale", w["q_shape"][2] ** -0.5),
                    ),
                    tune=False,
                ),
            ),
        ),
    ]


@GLADecodeBenchFixture
def test_gla_decode_bench(
    batch: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    scale: float,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GLADecodeWorkload(batch, heads, dim_k, dim_v, dtype, scale=scale)
    inputs = test.gen_inputs()

    # --- TileOPs ---
    op = GLADecodeFwdOp(scale=scale, tune=tune)
    bm = ManifestBenchmark(op, test)
    functors = {"tileops": op}

    if fused_recurrent_gla is not None:
        # --- FLA: fused_recurrent_gla with T=1 ---
        q, k, v, gk, state = inputs
        q_fla = q.unsqueeze(1)
        k_fla = k.unsqueeze(1)
        v_fla = v.unsqueeze(1)
        gk_fla = gk.unsqueeze(1)

        def fla_decode():
            return fused_recurrent_gla(
                q_fla,
                k_fla,
                v_fla,
                gk=gk_fla,
                scale=scale,
                initial_state=state.contiguous(),
                output_final_state=True,
            )

        functors["fla"] = (fla_decode, ())

    functors["torch"] = test.ref_program
    functors[TORCH_COMPILE_TAG] = compiled_reference(test.ref_program)

    bm.compare(functors, *inputs)
