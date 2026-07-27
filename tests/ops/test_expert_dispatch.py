"""Correctness tests for tight local and DeepEP dispatch adapters."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tileops.ops.moe import (
    DeepEPDispatchAdapter,
    DispatchedExpertMLPFwdOp,
    ExpertDispatchResult,
    LocalDispatchHandle,
    LocalExpertDispatcher,
)


def _routing(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    *,
    hidden_size: int = 16,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = (
        torch.arange(num_tokens * hidden_size, device="cuda", dtype=torch.float32)
        .reshape(num_tokens, hidden_size)
        .to(dtype)
    )
    scores = torch.rand(num_tokens, num_experts, device="cuda")
    topk_ids = scores.topk(top_k, dim=-1).indices.to(torch.int32)
    weights = torch.softmax(torch.randn(num_tokens, top_k, device="cuda"), dim=-1)
    return hidden, topk_ids, weights


@pytest.mark.smoke
@pytest.mark.parametrize("num_tokens,top_k,num_experts", [(5, 2, 4), (3, 1, 8)])
def test_local_dispatch_is_tight_and_reversible(
    num_tokens: int,
    top_k: int,
    num_experts: int,
) -> None:
    hidden, topk_ids, weights = _routing(num_tokens, top_k, num_experts)
    result = LocalExpertDispatcher(num_experts).dispatch(hidden, topk_ids, weights)

    assert isinstance(result, ExpertDispatchResult)
    assert isinstance(result.combine_handle, LocalDispatchHandle)
    assert result.batch.layout == "tight"
    assert result.batch.capacity == num_tokens * top_k
    assert result.batch.expert_offsets.dtype == torch.int32
    assert result.routing_weights.dtype == torch.float32

    flat_ids = topk_ids.flatten()
    counts = torch.bincount(flat_ids.to(torch.int64), minlength=num_experts)
    expected_offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.int64, device="cuda"),
            counts.cumsum(0),
        )
    ).to(torch.int32)
    torch.testing.assert_close(result.batch.expert_offsets, expected_offsets)

    mapping = result.combine_handle.forward_mapping.to(torch.int64)
    source_rows = torch.arange(num_tokens, device="cuda").unsqueeze(1).expand(-1, top_k).flatten()
    torch.testing.assert_close(result.batch.hidden[mapping], hidden[source_rows])
    torch.testing.assert_close(result.routing_weights[mapping], weights.flatten())


@pytest.mark.smoke
def test_local_dispatch_reference_combine_applies_weights_once() -> None:
    hidden, topk_ids, weights = _routing(7, 3, 6)
    result = LocalExpertDispatcher(6).dispatch(hidden, topk_ids, weights)
    handle = result.combine_handle
    assert isinstance(handle, LocalDispatchHandle)

    # Stand in for an expert function that preserves row order.
    expert_output = result.batch.hidden.float() * 2.0
    pair_output = expert_output[handle.forward_mapping.to(torch.int64)]
    combined = (
        pair_output.reshape(handle.num_tokens, handle.top_k, -1) * weights.unsqueeze(-1)
    ).sum(dim=1)
    reference = hidden.float() * 2.0 * weights.sum(dim=1, keepdim=True)
    torch.testing.assert_close(combined, reference)


@pytest.mark.smoke
def test_local_dispatch_output_is_consumed_directly_by_m5() -> None:
    num_tokens, top_k, num_experts = 4, 2, 4
    hidden_size, ffn_size = 128, 128
    hidden, topk_ids, weights = _routing(num_tokens, top_k, num_experts, hidden_size=hidden_size)
    hidden = torch.randn_like(hidden) * 0.1
    result = LocalExpertDispatcher(num_experts).dispatch(hidden, topk_ids, weights)
    w_gate_up = (
        torch.randn(
            num_experts,
            2 * ffn_size,
            hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.02
    )
    w_down = (
        torch.randn(
            num_experts,
            hidden_size,
            ffn_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.02
    )
    op = DispatchedExpertMLPFwdOp(
        num_pairs=result.batch.capacity,
        num_experts=num_experts,
        hidden_size=hidden_size,
        ffn_size=ffn_size,
        dtype=torch.bfloat16,
    )
    output = op.forward_batch(result.batch, w_gate_up, w_down)

    reference = torch.empty_like(output.hidden)
    offsets = result.batch.expert_offsets.cpu().tolist()
    for expert in range(num_experts):
        start, end = offsets[expert], offsets[expert + 1]
        gate_up = result.batch.hidden[start:end].float() @ w_gate_up[expert].float().T
        activated = F.silu(gate_up[:, :ffn_size]) * gate_up[:, ffn_size:]
        reference[start:end] = (activated @ w_down[expert].float().T).to(reference.dtype)

    assert output.valid_rows is not None
    torch.testing.assert_close(output.valid_rows, result.batch.valid_rows)
    torch.testing.assert_close(
        output.hidden,
        reference,
        rtol=2e-2,
        atol=2e-2,
    )


class _FakeEvent:
    def __init__(self) -> None:
        self.waited = False

    def current_stream_wait(self) -> None:
        self.waited = True


class _FakeDeepEPBuffer:
    def __init__(self, num_local_experts: int, capacity_tail: int = 3) -> None:
        self.num_local_experts = num_local_experts
        self.capacity_tail = capacity_tail
        self.kwargs = None
        self.event = _FakeEvent()
        self.recv_x = None

    def dispatch(self, hidden_states, **kwargs):
        self.kwargs = kwargs
        cached_handle = kwargs["handle"]
        topk_ids = kwargs["topk_idx"] if cached_handle is None else cached_handle.topk_idx
        weights = kwargs["topk_weights"]
        flat_ids = topk_ids.flatten()
        order = torch.argsort(flat_ids, stable=True)
        source_rows = (
            torch.arange(hidden_states.shape[0], device=hidden_states.device)
            .unsqueeze(1)
            .expand_as(topk_ids)
            .flatten()
        )
        valid_x = hidden_states[source_rows[order]]
        valid_weights = weights.flatten()[order]
        self.recv_x = torch.empty(
            valid_x.shape[0] + self.capacity_tail,
            hidden_states.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self.recv_x[: valid_x.shape[0]].copy_(valid_x)
        recv_weights = torch.empty(
            valid_weights.shape[0] + self.capacity_tail,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        recv_weights[: valid_weights.shape[0]].copy_(valid_weights)
        counts = torch.bincount(flat_ids, minlength=self.num_local_experts).to(torch.int32)
        handle = cached_handle or SimpleNamespace(
            do_expand=True,
            expert_alignment=1,
            topk_idx=topk_ids,
            psum_num_recv_tokens_per_expert=counts.cumsum(0, dtype=torch.int32),
        )
        return self.recv_x, None, recv_weights, handle, self.event


@pytest.mark.smoke
def test_deepep_adapter_uses_unpadded_expanded_layout_without_copy() -> None:
    num_experts = 4
    hidden, topk_ids, weights = _routing(5, 2, num_experts)
    topk_ids = topk_ids.to(torch.int64)
    buffer = _FakeDeepEPBuffer(num_experts)
    offsets_buffer = torch.empty(num_experts + 1, dtype=torch.int32, device="cuda")
    result = DeepEPDispatchAdapter(
        buffer,
        num_experts=num_experts,
        num_local_experts=num_experts,
        num_max_tokens_per_rank=32,
        num_sms=12,
    ).dispatch(
        hidden,
        topk_ids,
        weights,
        expert_offsets=offsets_buffer,
    )

    assert buffer.kwargs is not None
    assert buffer.kwargs["do_expand"] is True
    assert buffer.kwargs["expert_alignment"] == 1
    assert buffer.kwargs["do_cpu_sync"] is False
    assert buffer.kwargs["async_with_compute_stream"] is True
    assert buffer.kwargs["num_sms"] == 12
    assert buffer.kwargs["topk_idx"] is topk_ids
    assert buffer.kwargs["handle"] is None
    assert buffer.event.waited
    assert result.batch.hidden is buffer.recv_x
    assert result.batch.expert_offsets is offsets_buffer
    assert result.batch.capacity == topk_ids.numel() + buffer.capacity_tail
    assert result.batch.valid_rows.item() == topk_ids.numel()

    counts = torch.bincount(topk_ids.flatten().to(torch.int64), minlength=num_experts)
    expected_offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.int64, device="cuda"),
            counts.cumsum(0),
        )
    ).to(torch.int32)
    torch.testing.assert_close(result.batch.expert_offsets, expected_offsets)


@pytest.mark.smoke
def test_deepep_adapter_reuses_expanded_decode_handle() -> None:
    num_experts = 4
    hidden, topk_ids, weights = _routing(5, 2, num_experts)
    topk_ids = topk_ids.to(torch.int64)
    buffer = _FakeDeepEPBuffer(num_experts)
    adapter = DeepEPDispatchAdapter(
        buffer,
        num_experts=num_experts,
        num_local_experts=num_experts,
        num_max_tokens_per_rank=32,
    )
    first = adapter.dispatch(hidden, topk_ids, weights)
    cached_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device="cuda")
    second = adapter.dispatch(
        hidden + 1,
        None,
        weights,
        expert_offsets=cached_offsets,
        cached_handle=first.combine_handle,
    )

    assert buffer.kwargs["topk_idx"] is None
    assert buffer.kwargs["handle"] is first.combine_handle
    assert second.combine_handle is first.combine_handle
    assert second.batch.expert_offsets is cached_offsets
    torch.testing.assert_close(second.batch.expert_offsets, first.batch.expert_offsets)


@pytest.mark.smoke
def test_deepep_adapter_does_not_hide_topk_dtype_conversion() -> None:
    hidden, topk_ids, weights = _routing(4, 2, 4)
    adapter = DeepEPDispatchAdapter(
        _FakeDeepEPBuffer(4),
        num_experts=4,
        num_local_experts=4,
        num_max_tokens_per_rank=16,
    )
    with pytest.raises(ValueError, match="topk_ids_dtype"):
        adapter.dispatch(hidden, topk_ids, weights)


@pytest.mark.smoke
def test_deepep_adapter_rejects_non_device_prefix_sum() -> None:
    hidden, topk_ids, weights = _routing(4, 2, 4)
    topk_ids = topk_ids.to(torch.int64)

    class BadBuffer:
        def dispatch(self, hidden_states, **kwargs):
            event = _FakeEvent()
            handle = SimpleNamespace(psum_num_recv_tokens_per_expert=[1, 2, 3, 4])
            recv_weights = torch.empty(
                hidden_states.shape[0] * kwargs["topk_idx"].shape[1],
                dtype=torch.float32,
                device=hidden_states.device,
            )
            return hidden_states.repeat_interleave(2, 0), None, recv_weights, handle, event

    adapter = DeepEPDispatchAdapter(
        BadBuffer(),
        num_experts=4,
        num_local_experts=4,
        num_max_tokens_per_rank=16,
    )
    with pytest.raises(TypeError, match="psum_num_recv_tokens_per_expert"):
        adapter.dispatch(hidden, topk_ids, weights)
