"""The CUTLASS adapter's weight packing, tuning and captured-call guard."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from benchmarks import moe_baselines


@pytest.fixture
def cutlass(monkeypatch):
    state = {"mode": None, "calls": [], "fail": False}
    acts = SimpleNamespace(Swiglu="silu", Geglu="gelu", SwigluBias="swigluoai")

    @contextmanager
    def autotune(mode, *, tuning_buckets):
        state["mode"], state["buckets"] = mode, tuning_buckets
        yield

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


@pytest.mark.parametrize(
    "activation, biased",
    [("silu_and_mul", False), ("swigluoai", True)],
    ids=["stacked", "interleaved-biased"],
)
def test_cutlass_receives_up_rows_before_gate_rows(cutlass, activation, biased):
    """Stacked and interleaved gate/up weights, and the gate/up bias, reach CUTLASS as up then gate."""
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
    if activation == "swigluoai":
        for name, value in (("swiglu_alpha", 1.702), ("swiglu_beta", 1.0), ("swiglu_limit", 7.0)):
            torch.testing.assert_close(kwargs[name], torch.full((2,), value))


def test_cutlass_tunes_its_token_count_once_and_looks_it_up(cutlass):
    """The first call profiles the exact token-count bucket; every call then looks it up."""
    baseline, inputs = _inputs("silu_and_mul", tokens=100)
    for _ in range(3):
        baseline(*inputs)
    assert [call[:2] for call in cutlass["calls"]] == [(True, (100,))] + [(False, (100,))] * 3


def test_cutlass_retunes_after_a_failed_profile(cutlass):
    """A profiling failure leaves the adapter untuned, so the next call profiles again."""
    baseline, inputs = _inputs("silu_and_mul")
    cutlass["fail"] = True
    with pytest.raises(RuntimeError, match="tactic profiling failed"):
        baseline(*inputs)
    cutlass["fail"] = False
    baseline(*inputs)
    assert [call[0] for call in cutlass["calls"]] == [True, True, False]


@pytest.mark.parametrize("fault", ["weights", "tokens", "routes"])
def test_cutlass_refuses_a_call_other_than_the_captured_one(cutlass, fault):
    """Other gate/up tensors, another token count or another route count fail before CUTLASS runs."""
    baseline, original = _inputs("swigluoai", True)
    inputs = list(original)
    if fault == "weights":
        inputs[1] = inputs[1].clone()
    elif fault == "tokens":
        inputs[0] = inputs[0][:-1]
    else:
        inputs[3], inputs[4] = inputs[3][:, :-1], inputs[4][:, :-1]
    with pytest.raises(AssertionError):
        baseline(*inputs)
    assert cutlass["calls"] == []
