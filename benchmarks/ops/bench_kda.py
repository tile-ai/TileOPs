"""Benchmark Kimi Delta Attention (KDA), one case per manifest call, against FLA."""

import pytest

from benchmarks import api as bench
from benchmarks.baselines import baseline_module
from tileops.ops import KDAFwdOp


@pytest.mark.parametrize("case", bench.cases(KDAFwdOp), ids=lambda case: case.id)
def test_kda_fwd_bench(case) -> None:
    # FIXME(staged-rollout): the raw-gate KDA row is skipped
    #
    # Broken invariant: a benchmark times every manifest call of an implemented op.
    # Why: no in-tree KDA kernel computes the decay from A_log and dt_bias, and the row
    #   exists because the manifest requires a row passing each optional input.
    # Cleanup: when an in-tree KDA kernel serves use_gate_in_kernel=True, delete this skip.
    params = case.params
    if params["use_gate_in_kernel"]:
        pytest.skip("no in-tree KDA kernel serves use_gate_in_kernel=True")
    op = KDAFwdOp(**case.arguments)

    def fla_fn(q, k, v, g, beta, state, cu, cu_cpu, a_log, dt_bias):
        fla_kda = baseline_module("fla.ops.kda", "fla")
        chunk_kda, fused_recurrent_kda = fla_kda.chunk_kda, fla_kda.fused_recurrent_kda

        # FLA caches derived metadata by tensor identity; snapshot the live call.
        cu = None if cu is None else cu.clone()
        cu_cpu = None if cu_cpu is None else cu_cpu.clone()
        arguments = dict(
            scale=params["scale"],
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=params["use_qk_l2norm_in_kernel"],
            use_beta_sigmoid_in_kernel=params["use_beta_sigmoid_in_kernel"],
            allow_neg_eigval=params["allow_neg_eigval"],
            state_v_first=params["state_v_first"],
            cu_seqlens=cu,
        )
        if q.shape[1] == 1:
            return fused_recurrent_kda(q, k, v, g, beta, **arguments)
        return chunk_kda(q, k, v, g, beta, cu_seqlens_cpu=cu_cpu, **arguments)

    bench.Runner(op, case).compare({"fla": fla_fn})
