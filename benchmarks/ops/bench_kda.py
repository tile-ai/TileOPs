"""Benchmark Kimi Delta Attention (KDA), one case per manifest call, against FLA."""

import pytest

from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops import KDAFwdOp
from workloads.linear_attention.kda import KDAFwdCall


@pytest.mark.parametrize("call", manifest_calls(KDAFwdOp))
def test_kda_fwd_bench(call) -> None:
    # FIXME(staged-rollout): the raw-gate KDA row is skipped
    #
    # Broken invariant: a benchmark times every manifest call of an implemented op.
    # Why: no in-tree KDA kernel computes the decay from A_log and dt_bias, and the row
    #   exists because the manifest requires a row passing each optional input.
    # Cleanup: when an in-tree KDA kernel serves use_gate_in_kernel=True, delete this skip.
    if call.ix["use_gate_in_kernel"]:
        pytest.skip("no in-tree KDA kernel serves use_gate_in_kernel=True")
    workload = KDAFwdCall(call)
    inputs = workload.gen_inputs()
    op = KDAFwdOp(**workload.arguments())

    def fla_fn(q, k, v, g, beta, state, cu, cu_cpu, a_log, dt_bias):
        from fla.ops.kda import chunk_kda, fused_recurrent_kda

        # FLA caches derived metadata by tensor identity; snapshot the live call.
        cu = None if cu is None else cu.clone()
        cu_cpu = None if cu_cpu is None else cu_cpu.clone()
        arguments = dict(
            scale=call.ix["scale"],
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=call.ix["use_qk_l2norm_in_kernel"],
            use_beta_sigmoid_in_kernel=call.ix["use_beta_sigmoid_in_kernel"],
            allow_neg_eigval=call.ix["allow_neg_eigval"],
            state_v_first=call.ix["state_v_first"],
            cu_seqlens=cu,
        )
        if q.shape[1] == 1:
            return fused_recurrent_kda(q, k, v, g, beta, **arguments)
        return chunk_kda(q, k, v, g, beta, cu_seqlens_cpu=cu_cpu, **arguments)

    ManifestBenchmark(op, workload).compare({"tileops": op, "fla": fla_fn}, *inputs)
