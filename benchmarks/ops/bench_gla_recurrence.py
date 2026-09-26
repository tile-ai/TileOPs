"""Benchmark: TileOPs GLA decode vs FLA fused_recurrent_gla (T=1).

Compares single-step decode latency across batch sizes, dimensions, and dtypes.

When FLA is not installed, benchmarks still run using a pure-torch reference
implementation as baseline, so CI is never blocked by a missing optional dependency.
"""

import pytest

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops import GLADecodeFwdOp
from workloads.linear_attention import GLADecodeCall

try:
    from fla.ops.gla import fused_recurrent_gla
except ImportError:
    fused_recurrent_gla = None


@pytest.mark.parametrize("call", manifest_calls(GLADecodeFwdOp))
def test_gla_decode_bench(call) -> None:
    test = GLADecodeCall(call)
    inputs = test.gen_inputs()
    op = GLADecodeFwdOp(**test.arguments())
    bm = ManifestBenchmark(op, test)
    functors = {"tileops": op}

    if fused_recurrent_gla is not None:
        # --- FLA: fused_recurrent_gla with T=1 ---
        q, k, v, gk, state = inputs
        q_fla, k_fla, v_fla, gk_fla = (t.unsqueeze(1) for t in (q, k, v, gk))

        def fla_decode():
            o, new_state = fused_recurrent_gla(
                q_fla,
                k_fla,
                v_fla,
                gk=gk_fla,
                scale=op.scale,
                initial_state=state.contiguous(),
                output_final_state=True,
            )
            return o.squeeze(1), new_state.to(state.dtype)

        functors["fla"] = (fla_decode, ())

    functors["torch"] = test.ref_program
    functors[TORCH_COMPILE_TAG] = compiled_reference(test.ref_program)

    bm.compare(functors, *inputs)
