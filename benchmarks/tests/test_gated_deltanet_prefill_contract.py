"""Contract checks for the Gated DeltaNet prefill external baseline."""

import torch

from benchmarks.ops import bench_gated_deltanet_prefill as bench


def test_fla_prefill_baseline_times_contract_equivalent_state_cast(monkeypatch) -> None:
    seen = {}

    def fake_fla(q, k, v, g, beta, **kwargs):
        seen["args"] = (q, k, v, g, beta)
        seen["kwargs"] = kwargs
        return torch.zeros_like(v), torch.zeros(
            q.shape[0], q.shape[2], q.shape[-1], v.shape[-1], dtype=torch.float32
        )

    monkeypatch.setattr(bench, "chunk_gated_delta_rule", fake_fla)
    baseline = bench._fla_prefill_fwd()
    inputs = (
        torch.randn(1, 8, 2, 4, dtype=torch.float16),
        torch.randn(1, 8, 2, 4, dtype=torch.float16),
        torch.randn(1, 8, 2, 4, dtype=torch.float16),
        torch.randn(1, 8, 2, dtype=torch.float16),
        torch.rand(1, 8, 2, dtype=torch.float16),
    )

    o, final_state = baseline(*inputs)

    assert seen["args"] == inputs
    assert seen["kwargs"] == {
        "scale": 1.0,
        "initial_state": None,
        "output_final_state": True,
    }
    assert o.dtype == torch.float16
    assert final_state.dtype == torch.float16


def test_fla_prefill_baseline_is_optional(monkeypatch) -> None:
    monkeypatch.setattr(bench, "chunk_gated_delta_rule", None)
    assert bench._fla_prefill_fwd() is None
