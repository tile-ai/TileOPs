"""FLA receives the same gate semantics and recurrent state as the KDA op."""

from types import SimpleNamespace

import pytest
import torch

from benchmarks.ops import bench_kda


@pytest.mark.parametrize("seqlen,lower_bound", [(1, None), (65, -0.25)])
def test_fla_kda_receives_gate_parameters(monkeypatch, seqlen, lower_bound):
    params = dict(
        scale=0.125,
        use_gate_in_kernel=True,
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False,
        state_v_first=False,
    )
    case = SimpleNamespace(params=params, arguments=params)
    q = torch.zeros(1, seqlen, 2, 64)
    state = torch.zeros(2, 2, 64, 64)
    cu = torch.tensor([0, 1, seqlen], dtype=torch.int64)
    a_log, dt_bias = torch.ones(2), torch.ones(128)
    inputs = (q, q, q, q, q[..., 0], state, cu, cu.clone(), a_log, dt_bias)
    calls = []

    def baseline(name):
        def run(*args, **kwargs):
            calls.append((name, args, kwargs))
            return q, state

        return run

    monkeypatch.setattr(bench_kda, "KDAFwdOp", lambda **kwargs: object())
    monkeypatch.setattr(
        bench_kda,
        "baseline_module",
        lambda *args: SimpleNamespace(
            chunk_kda=baseline("chunk"), fused_recurrent_kda=baseline("recurrent")
        ),
    )

    class Runner:
        def __init__(self, op, case):
            pass

        def compare(self, implementations):
            implementations["fla"](*inputs)

    monkeypatch.setattr(bench_kda.bench, "Runner", Runner)
    bench_kda.test_kda_fwd_bench(case)
    name, args, kwargs = calls.pop()
    assert name == ("recurrent" if seqlen == 1 else "chunk")
    assert all(actual is expected for actual, expected in zip(args, inputs[:5], strict=True))
    assert kwargs["use_gate_in_kernel"] is True
    assert kwargs["A_log"] is a_log
    assert kwargs["dt_bias"] is dt_bias
    assert kwargs["lower_bound"] == lower_bound
    assert kwargs["initial_state"] is state
    assert kwargs["output_final_state"] is True
    assert kwargs["scale"] == params["scale"]
    assert kwargs["use_qk_l2norm_in_kernel"] is True
    for name in ("use_beta_sigmoid_in_kernel", "allow_neg_eigval", "state_v_first"):
        assert kwargs[name] is False
    assert "safe_gate" not in kwargs
    torch.testing.assert_close(kwargs["cu_seqlens"], cu)
    assert kwargs["cu_seqlens"] is not cu
    if seqlen != 1:
        torch.testing.assert_close(kwargs["cu_seqlens_cpu"], inputs[7])
        assert kwargs["cu_seqlens_cpu"] is not inputs[7]
