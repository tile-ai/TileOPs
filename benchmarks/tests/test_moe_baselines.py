"""The CUTLASS adapter's packing, tuning lifetime and supported input domain."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from benchmarks import moe_baselines


@pytest.fixture
def cutlass(monkeypatch):
    state = {"mode": "outer", "buckets": (7,), "calls": [], "fail": False}
    acts = SimpleNamespace(Swiglu="silu", Geglu="gelu", SwigluBias="swigluoai")

    @contextmanager
    def autotune(mode, *, tuning_buckets):
        before = state["mode"], state["buckets"]
        state["mode"], state["buckets"] = mode, tuning_buckets
        try:
            yield
        finally:
            state["mode"], state["buckets"] = before

    def run(*args, **kwargs):
        state["calls"].append((state["mode"], state["buckets"], args, kwargs))
        if state["fail"] and state["mode"]:
            raise RuntimeError("tactic profiling failed")
        return (torch.zeros_like(args[0]),)

    entries = {
        "fused_moe.cutlass_fused_moe": run,
        "fused_moe.cutlass_fused_moe_workspace_size": lambda *args, **kwargs: 16,
        "fused_moe.ActivationType": acts,
        "autotune": autotune,
    }
    monkeypatch.setattr(moe_baselines, "flashinfer_op", lambda name, *args: entries[name])
    return state


def _inputs(activation, biased=False, *, tokens=100):
    experts, hidden, ffn, top_k = 2, 8, 16, 2
    x = torch.zeros(tokens, hidden, dtype=torch.bfloat16)
    w1 = torch.arange(experts * 2 * ffn * hidden, dtype=torch.float32).view(
        experts, 2 * ffn, hidden
    )
    w2 = torch.zeros(experts, hidden, ffn)
    b1 = (
        torch.arange(experts * 2 * ffn, dtype=torch.float32).view(experts, 2 * ffn)
        if biased
        else None
    )
    b2 = torch.zeros(experts, hidden) if biased else None
    weights = torch.ones(tokens, top_k)
    ids = torch.zeros(tokens, top_k, dtype=torch.int32)
    baseline = moe_baselines.flashinfer_experts(x, w1, w2, top_k, activation, b1)
    return baseline, (x, w1, w2, weights, ids, b1, b2)


@pytest.mark.parametrize("activation", ["silu_and_mul", "gelu_and_mul", "swigluoai"])
@pytest.mark.parametrize("biased", [False, True])
def test_cutlass_packs_each_activation_weights_and_bias(cutlass, activation, biased):
    """CUTLASS receives up rows followed by gate rows, in the same order for each bias."""
    baseline, inputs = _inputs(activation, biased)
    baseline(*inputs)
    _, _, args, kwargs = cutlass["calls"][-1]
    ffn = inputs[2].shape[-1]
    order = (
        list(range(1, 2 * ffn, 2)) + list(range(0, 2 * ffn, 2))
        if activation == "swigluoai"
        else list(range(ffn, 2 * ffn)) + list(range(ffn))
    )
    torch.testing.assert_close(args[3], inputs[1][:, order])
    if biased:
        torch.testing.assert_close(kwargs["fc1_expert_biases"], inputs[5][:, order])
    else:
        assert kwargs["fc1_expert_biases"] is None
    assert kwargs["fc2_expert_biases"] is inputs[6]
    assert (
        kwargs["activation_type"]
        == {"silu_and_mul": "silu", "gelu_and_mul": "gelu", "swigluoai": "swigluoai"}[activation]
    )
    if activation == "swigluoai":
        for name, value in (("swiglu_alpha", 1.702), ("swiglu_beta", 1.0), ("swiglu_limit", 7.0)):
            torch.testing.assert_close(kwargs[name], torch.full((2,), value))


@pytest.mark.parametrize("activation", ["silu_and_mul", "gelu_and_mul", "swigluoai"])
@pytest.mark.parametrize("tokens", [100, 8193], ids=["off-grid", "above-default-bucket-limit"])
def test_cutlass_tunes_once_and_restores_the_outer_context(cutlass, activation, tokens):
    """Profiling and every lookup use the exact token bucket without leaking tuning state."""
    baseline, inputs = _inputs(activation, tokens=tokens)
    outputs = [baseline(*inputs) for _ in range(3)]
    assert [call[0] for call in cutlass["calls"]] == [True, False, False, False]
    assert all(call[1] == (inputs[0].shape[0],) for call in cutlass["calls"])
    assert cutlass["mode"] == "outer" and cutlass["buckets"] == (7,)
    assert all(torch.equal(outputs[0], output) for output in outputs[1:])


@pytest.mark.parametrize(
    "fault",
    [
        "tokens",
        "tokens-over",
        "hidden-width",
        "hidden-dtype",
        "hidden-device",
        "down-width",
        "down-dtype",
        "down-device",
        "weights",
        "bias",
        "routes-lower",
        "routes-upper",
        "routing-weights",
        "routing-ids",
    ],
)
def test_cutlass_refuses_an_input_outside_its_captured_call(cutlass, fault):
    """A different token count or repacked tensor is refused before calling CUTLASS."""
    baseline, original = _inputs("swigluoai", True)
    inputs = list(original)
    if fault == "tokens":
        inputs[0] = inputs[0][:-1]
    elif fault == "tokens-over":
        inputs[0] = torch.cat((inputs[0], inputs[0][:1]))
    elif fault == "hidden-width":
        inputs[0] = inputs[0][:, :-1]
    elif fault == "hidden-dtype":
        inputs[0] = inputs[0].float()
    elif fault == "hidden-device":
        inputs[0] = inputs[0].to("meta")
    elif fault == "down-width":
        inputs[2] = inputs[2][:, :, :-1]
    elif fault == "down-dtype":
        inputs[2] = inputs[2].to(torch.bfloat16)
    elif fault == "down-device":
        inputs[2] = inputs[2].to("meta")
    elif fault in ("routes-lower", "routes-upper"):
        for index in (3, 4):
            inputs[index] = (
                inputs[index][:, :-1]
                if fault == "routes-lower"
                else torch.cat((inputs[index], inputs[index][:, :1]), dim=1)
            )
    elif fault in ("routing-weights", "routing-ids"):
        index = 3 if fault == "routing-weights" else 4
        inputs[index] = inputs[index][:-1]
    else:
        index = {"weights": 1, "bias": 5}[fault]
        inputs[index] = inputs[index].clone()
    with pytest.raises(AssertionError):
        baseline(*inputs)
    assert cutlass["calls"] == []


def test_cutlass_failed_tuning_restores_context_and_remains_untuned(cutlass):
    """A profiling failure restores the outer context and permits a complete retry."""
    baseline, inputs = _inputs("silu_and_mul")
    cutlass["fail"] = True
    with pytest.raises(RuntimeError, match="tactic profiling failed"):
        baseline(*inputs)
    assert cutlass["mode"] == "outer" and cutlass["buckets"] == (7,)
    cutlass["fail"] = False
    baseline(*inputs)
    baseline(*inputs)
    assert [call[0] for call in cutlass["calls"]] == [True, True, False, False]
    assert cutlass["mode"] == "outer" and cutlass["buckets"] == (7,)


@pytest.mark.parametrize(
    "gate_bias,down_bias,accepted",
    [(False, False, True), (True, True, True), (True, False, False), (False, True, False)],
)
def test_cutlass_requires_both_biases_or_neither(gate_bias, down_bias, accepted):
    """The bias domain contains both-present and both-absent calls only."""
    x, down = torch.empty(1, 8, dtype=torch.bfloat16), torch.empty(2, 8, 16)
    bias = torch.empty(1)
    reason = moe_baselines.flashinfer_experts_skip_reason(
        x, down, bias if gate_bias else None, bias if down_bias else None
    )
    assert (reason is None) is accepted


@pytest.mark.parametrize(
    "hidden,ffn,accepted",
    [(8, 8, True), (16, 16, True), (7, 8, False), (9, 8, False), (8, 7, False), (8, 9, False)],
)
def test_cutlass_alignment_checks_each_projection_width(hidden, ffn, accepted):
    """Both projection widths must fill whole 16-byte TMA units."""
    x = torch.empty(1, hidden, dtype=torch.bfloat16)
    down = torch.empty(2, hidden, ffn)
    reason = moe_baselines.flashinfer_experts_skip_reason(x, down)
    assert (reason is None) is accepted


@pytest.mark.cuda_only
@pytest.mark.parametrize("activation", ["silu_and_mul", "gelu_and_mul", "swigluoai"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cutlass_adapter_computes_each_activation_with_biases(activation, dtype):
    """Each supported activation and operand dtype matches the independent routed reference."""
    from workloads.device import run_device, run_device_available
    from workloads.moe import moe_verification, ref_routed_experts
    from workloads.numerics import compare_outputs

    if not run_device_available() or torch.device(run_device()).type != "cuda":
        pytest.skip("CUTLASS requires a CUDA run device")
    tokens, experts, hidden, ffn, top_k = 100, 4, 64, 128, 2
    device = run_device()
    x = torch.randn(tokens, hidden, dtype=dtype, device=device)
    w1 = torch.randn(experts, 2 * ffn, hidden, dtype=dtype, device=device) * 0.5
    w2 = torch.randn(experts, hidden, ffn, dtype=dtype, device=device) * 0.002
    b1 = torch.randn(experts, 2 * ffn, dtype=dtype, device=device)
    b2 = torch.randn(experts, hidden, dtype=dtype, device=device) * 0.05
    ids = torch.randint(experts, (tokens, top_k), dtype=torch.int32, device=device)
    weights = torch.rand(tokens, top_k, device=device)
    weights = weights / weights.sum(-1, keepdim=True)
    baseline = moe_baselines.flashinfer_experts(x, w1, w2, top_k, activation, b1)
    got = baseline(x, w1, w2, weights, ids, b1, b2)
    args = (x, w1, w2, weights, ids, activation)
    biases = {"b_gate_up": b1, "b_down": b2}
    compare_outputs(got, ref_routed_experts(*args, **biases), moe_verification(1))
