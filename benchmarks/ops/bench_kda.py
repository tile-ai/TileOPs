"""Benchmark Kimi Delta Attention (KDA), one case per manifest call, against FLA."""

import pytest

from benchmarks import api as bench
from benchmarks.baselines import baseline_module
from tileops.linear_attention import KDAFwdOp


@pytest.mark.parametrize("case", bench.cases(KDAFwdOp), ids=lambda case: case.id)
def test_kda_fwd_bench(case) -> None:
    params = case.params
    op = KDAFwdOp(**case.arguments)

    def fla_fn(q, k, v, g, beta, state, cu, cu_cpu, a_log, dt_bias):
        fla_kda = baseline_module("fla.ops.kda", "fla")
        chunk_kda, fused_recurrent_kda = fla_kda.chunk_kda, fla_kda.fused_recurrent_kda

        # FLA caches derived metadata by tensor identity; snapshot the live call.
        cu = None if cu is None else cu.clone()
        cu_cpu = None if cu_cpu is None else cu_cpu.clone()
        arguments = dict(
            scale=params["scale"],
            use_gate_in_kernel=params["use_gate_in_kernel"],
            A_log=a_log,
            dt_bias=dt_bias,
            lower_bound=params["lower_bound"],
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
        lower_bound = params["lower_bound"]
        # FLA applies the lower-bound gate either way; safe_gate only selects its faster
        # intra-chunk path, which FLA accepts for a lower bound in [-5, 0) alone.
        safe_gate = (
            params["use_gate_in_kernel"] and lower_bound is not None and -5 <= lower_bound < 0
        )
        return chunk_kda(q, k, v, g, beta, cu_seqlens_cpu=cu_cpu, safe_gate=safe_gate, **arguments)

    bench.Runner(op, case).compare({"fla": fla_fn})
