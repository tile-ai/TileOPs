"""Tests for the communication-independent dispatched expert MLP."""

import inspect

import pytest
import torch
import torch.nn.functional as F

from tileops.ops.moe import (
    DispatchedExpertMLPFwdOp,
    ExpertBatch,
    ExpertBatchOutput,
)


def _reference(hidden, w_gate_up, w_down, sizes):
    outputs = []
    start = 0
    ffn_size = w_down.shape[-1]
    for expert, size in enumerate(sizes):
        rows = hidden[start : start + size].float()
        gate_up = rows @ w_gate_up[expert].float().t()
        act = F.silu(gate_up[:, :ffn_size]) * gate_up[:, ffn_size:]
        outputs.append(act @ w_down[expert].float().t())
        start += size
    return torch.cat(outputs).to(hidden.dtype)


@pytest.mark.smoke
def test_public_interface_has_no_routing_or_communication_inputs():
    parameters = inspect.signature(DispatchedExpertMLPFwdOp.forward).parameters
    assert tuple(parameters) == (
        "self",
        "expert_input",
        "w_gate_up",
        "w_down",
        "true_sizes",
        "true_offsets",
    )
    forbidden = {"topk_ids", "topk_weights", "rank", "world_size", "handle"}
    assert forbidden.isdisjoint(parameters)
    batch_parameters = inspect.signature(
        DispatchedExpertMLPFwdOp.forward_batch
    ).parameters
    assert tuple(batch_parameters) == ("self", "batch", "w_gate_up", "w_down")
    assert forbidden.isdisjoint(batch_parameters)


@pytest.mark.smoke
def test_expert_batch_contract_rejects_non_tight_layout():
    with pytest.raises(ValueError, match="layout='tight'"):
        ExpertBatch(
            hidden=torch.empty(4, 8),
            expert_offsets=torch.tensor([0, 2, 4], dtype=torch.int32),
            layout="aligned",
        )


@pytest.mark.smoke
def test_expert_batch_output_contract_defaults():
    hidden = torch.empty(4, 8)
    output = ExpertBatchOutput(hidden=hidden)
    assert output.hidden is hidden
    assert output.row_order_preserved
    assert not output.routing_weights_applied


@pytest.mark.smoke
def test_expert_batch_valid_rows_contract():
    hidden = torch.empty(4, 8)
    offsets = torch.tensor([0, 2, 4], dtype=torch.int32)
    with pytest.raises(ValueError, match="one element"):
        ExpertBatch(
            hidden=hidden,
            expert_offsets=offsets,
            valid_rows=torch.tensor([3, 4], dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="torch.int32"):
        ExpertBatch(
            hidden=hidden,
            expert_offsets=offsets,
            valid_rows=torch.tensor(4, dtype=torch.int64),
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("sizes", [[4, 4, 4, 4], [0, 2, 13, 1]])
@pytest.mark.smoke
def test_dispatched_expert_matches_reference_and_preserves_rows(dtype, sizes):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(0)
    num_experts = len(sizes)
    num_pairs = sum(sizes)
    hidden_size, ffn_size = 128, 96
    hidden = torch.randn(
        num_pairs, hidden_size, device="cuda", dtype=dtype
    ) * 0.1
    w_gate_up = torch.randn(
        num_experts, 2 * ffn_size, hidden_size, device="cuda", dtype=dtype
    ) * 0.02
    w_down = torch.randn(
        num_experts, hidden_size, ffn_size, device="cuda", dtype=dtype
    ) * 0.02
    true_sizes = torch.tensor(sizes, device="cuda", dtype=torch.int32)
    true_offsets = torch.tensor(
        [sum(sizes[:expert]) for expert in range(num_experts)],
        device="cuda",
        dtype=torch.int32,
    )

    op = DispatchedExpertMLPFwdOp(
        num_pairs=num_pairs,
        num_experts=num_experts,
        hidden_size=hidden_size,
        ffn_size=ffn_size,
        dtype=dtype,
    )
    output = op(hidden, w_gate_up, w_down, true_sizes, true_offsets)
    reference = _reference(hidden, w_gate_up, w_down, sizes)

    assert output.shape == hidden.shape
    assert output.dtype == dtype
    assert torch.allclose(output.float(), reference.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.smoke
def test_expert_batch_mock_dispatch_combine_applies_weights_once():
    """A mock multi-source dispatch/combine is equivalent to explicit MoE."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(1)
    tokens, hidden_size, ffn_size, num_experts, top_k = 8, 128, 96, 4, 2
    dtype = torch.bfloat16
    hidden = torch.randn(
        tokens, hidden_size, device="cuda", dtype=dtype
    ) * 0.1
    w_gate_up = torch.randn(
        num_experts,
        2 * ffn_size,
        hidden_size,
        device="cuda",
        dtype=dtype,
    ) * 0.02
    w_down = torch.randn(
        num_experts,
        hidden_size,
        ffn_size,
        device="cuda",
        dtype=dtype,
    ) * 0.02
    topk_ids = torch.tensor(
        [[0, 2], [1, 2], [2, 0], [0, 1], [2, 1], [1, 0], [0, 2], [2, 1]],
        device="cuda",
        dtype=torch.int64,
    )
    topk_weights = torch.softmax(
        torch.randn(tokens, top_k, device="cuda"), dim=-1
    )

    # Mock dispatch: pairs from two source ranks are sorted expert-major.
    source_tokens = torch.arange(tokens, device="cuda").repeat_interleave(top_k)
    source_ranks = (torch.arange(tokens, device="cuda") % 2).repeat_interleave(
        top_k
    )
    flat_experts = topk_ids.flatten()
    order = torch.argsort(flat_experts, stable=True)
    dispatched = hidden[source_tokens[order]]
    counts = torch.bincount(flat_experts, minlength=num_experts).to(torch.int32)
    expert_offsets = torch.cat(
        (
            torch.zeros(1, device="cuda", dtype=torch.int32),
            torch.cumsum(counts, dim=0, dtype=torch.int32),
        )
    )
    batch = ExpertBatch(
        hidden=dispatched,
        expert_offsets=expert_offsets,
        layout="tight",
    )
    # The reverse handle is external metadata and is never passed to TileOps.
    reverse_handle = {
        "source_tokens": source_tokens[order],
        "source_ranks": source_ranks[order],
        "weights": topk_weights.flatten()[order],
    }

    op = DispatchedExpertMLPFwdOp(
        num_pairs=tokens * top_k,
        num_experts=num_experts,
        hidden_size=hidden_size,
        ffn_size=ffn_size,
        dtype=dtype,
    )
    expert_output = op.forward_batch(batch, w_gate_up, w_down)

    sizes = counts.cpu().tolist()
    unweighted_reference = _reference(
        dispatched, w_gate_up, w_down, sizes
    )
    assert expert_output.row_order_preserved
    assert not expert_output.routing_weights_applied
    assert torch.allclose(
        expert_output.hidden.float(),
        unweighted_reference.float(),
        atol=1e-2,
        rtol=1e-2,
    )

    # Mock combine: reverse source-token mapping and apply routing weight once.
    combined = torch.zeros(
        tokens, hidden_size, device="cuda", dtype=torch.float32
    )
    combined.index_add_(
        0,
        reverse_handle["source_tokens"],
        expert_output.hidden.float()
        * reverse_handle["weights"].float().unsqueeze(1),
    )
    combined_reference = torch.zeros_like(combined)
    combined_reference.index_add_(
        0,
        reverse_handle["source_tokens"],
        unweighted_reference.float()
        * reverse_handle["weights"].float().unsqueeze(1),
    )
    assert torch.equal(
        reverse_handle["source_ranks"].unique().cpu(), torch.tensor([0, 1])
    )
    assert torch.allclose(combined, combined_reference, atol=1e-2, rtol=1e-2)


@pytest.mark.smoke
@pytest.mark.parametrize("use_fused_activation", [False, True])
@pytest.mark.parametrize("sizes", [[0, 3, 5, 2], [0, 0, 0, 0]])
def test_expert_batch_capacity_processes_only_device_valid_rows(
    use_fused_activation, sizes,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if use_fused_activation and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("fused activation requires SM90")

    torch.manual_seed(2)
    capacity, hidden_size, ffn_size, num_experts = 24, 256, 128, 4
    valid_count = sum(sizes)
    dtype = torch.bfloat16
    hidden = torch.randn(
        capacity, hidden_size, device="cuda", dtype=dtype
    ) * 0.1
    # An invalid tail must not affect valid rows.
    hidden[valid_count:].fill_(float("nan"))
    w_gate_up = torch.randn(
        num_experts,
        2 * ffn_size,
        hidden_size,
        device="cuda",
        dtype=dtype,
    ) * 0.02
    w_down = torch.randn(
        num_experts,
        hidden_size,
        ffn_size,
        device="cuda",
        dtype=dtype,
    ) * 0.02
    host_offsets = [0]
    for size in sizes:
        host_offsets.append(host_offsets[-1] + size)
    offsets = torch.tensor(
        host_offsets, device="cuda", dtype=torch.int32
    )
    valid_rows = torch.tensor(
        valid_count, device="cuda", dtype=torch.int32
    )
    batch = ExpertBatch(hidden, offsets, valid_rows=valid_rows)
    op = DispatchedExpertMLPFwdOp(
        num_pairs=capacity,
        num_experts=num_experts,
        hidden_size=hidden_size,
        ffn_size=ffn_size,
        dtype=dtype,
        use_fused_activation=use_fused_activation,
    )

    output = op.forward_batch(batch, w_gate_up, w_down)
    reference = _reference(
        hidden[:valid_count], w_gate_up, w_down, sizes
    )

    assert output.hidden.shape == (capacity, hidden_size)
    assert output.valid_rows is valid_rows
    assert torch.isfinite(output.hidden[:valid_count]).all()
    assert torch.allclose(
        output.hidden[:valid_count].float(),
        reference.float(),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.smoke
def test_expert_batch_cuda_graph_replays_different_valid_rows():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(3)
    capacity, hidden_size, ffn_size, num_experts = 24, 128, 96, 4
    dtype = torch.bfloat16
    hidden = torch.empty(
        capacity, hidden_size, device="cuda", dtype=dtype
    )
    offsets = torch.empty(
        num_experts + 1, device="cuda", dtype=torch.int32
    )
    valid_rows = torch.empty((), device="cuda", dtype=torch.int32)
    w_gate_up = torch.randn(
        num_experts,
        2 * ffn_size,
        hidden_size,
        device="cuda",
        dtype=dtype,
    ) * 0.02
    w_down = torch.randn(
        num_experts,
        hidden_size,
        ffn_size,
        device="cuda",
        dtype=dtype,
    ) * 0.02
    batch = ExpertBatch(hidden, offsets, valid_rows=valid_rows)
    op = DispatchedExpertMLPFwdOp(
        num_pairs=capacity,
        num_experts=num_experts,
        hidden_size=hidden_size,
        ffn_size=ffn_size,
        dtype=dtype,
    )

    def set_batch(sizes):
        count = sum(sizes)
        hidden[:count].copy_(
            torch.randn(
                count, hidden_size, device="cuda", dtype=dtype
            ) * 0.1
        )
        hidden[count:].fill_(float("nan"))
        host_offsets = [0]
        for size in sizes:
            host_offsets.append(host_offsets[-1] + size)
        offsets.copy_(
            torch.tensor(host_offsets, device="cuda", dtype=torch.int32)
        )
        valid_rows.fill_(count)
        return count

    first_sizes = [2, 0, 3, 2]
    first_count = set_batch(first_sizes)
    # Compile and warm allocator state before capture.
    op.forward_batch(batch, w_gate_up, w_down)
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(2):
            op.forward_batch(batch, w_gate_up, w_down)
    torch.cuda.current_stream().wait_stream(side_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op.forward_batch(batch, w_gate_up, w_down).hidden
    graph.replay()
    first_reference = _reference(
        hidden[:first_count], w_gate_up, w_down, first_sizes
    )
    assert torch.allclose(
        captured[:first_count].float(),
        first_reference.float(),
        atol=1e-2,
        rtol=1e-2,
    )

    second_sizes = [0, 4, 1, 6]
    second_count = set_batch(second_sizes)
    graph.replay()
    second_reference = _reference(
        hidden[:second_count], w_gate_up, w_down, second_sizes
    )
    assert torch.isfinite(captured[:second_count]).all()
    assert torch.allclose(
        captured[:second_count].float(),
        second_reference.float(),
        atol=1e-2,
        rtol=1e-2,
    )
