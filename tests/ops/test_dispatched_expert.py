"""Tests for the communication-independent dispatched expert MLP."""

import pytest
import torch
import torch.nn.functional as F

from tileops.kernels.grouped_gemm import GroupedGemmPersistent3WGKernel
from tileops.ops.moe import (
    DispatchedExpertMLPFwdOp,
    ExpertBatch,
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
def test_expert_batch_contract_rejects_non_tight_layout():
    with pytest.raises(ValueError, match="layout='tight'"):
        ExpertBatch(
            hidden=torch.empty(4, 8),
            expert_offsets=torch.tensor([0, 2, 4], dtype=torch.int32),
            layout="aligned",
        )


@pytest.mark.smoke
def test_expert_batch_valid_rows_is_offsets_view():
    hidden = torch.empty(4, 8)
    offsets = torch.tensor([0, 2, 4], dtype=torch.int32)
    batch = ExpertBatch(hidden=hidden, expert_offsets=offsets)

    assert batch.valid_rows.shape == (1,)
    assert batch.valid_rows.data_ptr() == offsets[-1:].data_ptr()
    offsets[-1] = 3
    assert batch.valid_rows.item() == 3


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("sizes", [[4, 4, 4, 4], [0, 2, 13, 1]])
@pytest.mark.parametrize("explicit_3wg_override", [False, True])
@pytest.mark.smoke
def test_dispatched_expert_matches_reference_and_preserves_rows(
    dtype,
    sizes,
    explicit_3wg_override,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(0)
    num_experts = len(sizes)
    num_pairs = sum(sizes)
    hidden_size, ffn_size = 128, 96
    hidden = torch.randn(num_pairs, hidden_size, device="cuda", dtype=dtype) * 0.1
    w_gate_up = (
        torch.randn(num_experts, 2 * ffn_size, hidden_size, device="cuda", dtype=dtype) * 0.02
    )
    w_down = torch.randn(num_experts, hidden_size, ffn_size, device="cuda", dtype=dtype) * 0.02
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
        kernel_map=(
            {"moe_grouped_gemm_kernel": GroupedGemmPersistent3WGKernel}
            if explicit_3wg_override
            else None
        ),
    )
    output = op(hidden, w_gate_up, w_down, true_sizes, true_offsets)
    reference = _reference(hidden, w_gate_up, w_down, sizes)

    assert output.shape == hidden.shape
    assert output.dtype == dtype
    assert torch.allclose(output.float(), reference.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.smoke
@pytest.mark.parametrize("use_fused_activation", [False, True])
@pytest.mark.parametrize("sizes", [[0, 3, 5, 2], [0, 0, 0, 0]])
def test_expert_batch_capacity_processes_only_device_valid_rows(
    use_fused_activation,
    sizes,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if use_fused_activation and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("fused activation requires SM90")

    torch.manual_seed(2)
    capacity, hidden_size, ffn_size, num_experts = 24, 256, 128, 4
    valid_count = sum(sizes)
    dtype = torch.bfloat16
    hidden = torch.randn(capacity, hidden_size, device="cuda", dtype=dtype) * 0.1
    # An invalid tail must not affect valid rows.
    hidden[valid_count:].fill_(float("nan"))
    w_gate_up = (
        torch.randn(
            num_experts,
            2 * ffn_size,
            hidden_size,
            device="cuda",
            dtype=dtype,
        )
        * 0.02
    )
    w_down = (
        torch.randn(
            num_experts,
            hidden_size,
            ffn_size,
            device="cuda",
            dtype=dtype,
        )
        * 0.02
    )
    host_offsets = [0]
    for size in sizes:
        host_offsets.append(host_offsets[-1] + size)
    offsets = torch.tensor(host_offsets, device="cuda", dtype=torch.int32)
    batch = ExpertBatch(hidden, offsets)
    op = DispatchedExpertMLPFwdOp(
        num_pairs=capacity,
        num_experts=num_experts,
        hidden_size=hidden_size,
        ffn_size=ffn_size,
        dtype=dtype,
        use_fused_activation=use_fused_activation,
    )

    output = op.forward_batch(batch, w_gate_up, w_down)
    reference = _reference(hidden[:valid_count], w_gate_up, w_down, sizes)

    assert output.hidden.shape == (capacity, hidden_size)
    assert output.valid_rows.data_ptr() == offsets[-1:].data_ptr()
    assert torch.isfinite(output.hidden[:valid_count]).all()
    assert torch.allclose(
        output.hidden[:valid_count].float(),
        reference.float(),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.smoke
@pytest.mark.parametrize("use_fused_activation", [False, True])
def test_expert_batch_cuda_graph_replays_different_valid_rows(
    use_fused_activation,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if use_fused_activation and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("fused activation requires SM90")

    torch.manual_seed(3)
    capacity, hidden_size, ffn_size, num_experts = 24, 256, 128, 4
    dtype = torch.bfloat16
    hidden = torch.empty(capacity, hidden_size, device="cuda", dtype=dtype)
    offsets = torch.empty(num_experts + 1, device="cuda", dtype=torch.int32)
    w_gate_up = (
        torch.randn(
            num_experts,
            2 * ffn_size,
            hidden_size,
            device="cuda",
            dtype=dtype,
        )
        * 0.02
    )
    w_down = (
        torch.randn(
            num_experts,
            hidden_size,
            ffn_size,
            device="cuda",
            dtype=dtype,
        )
        * 0.02
    )
    batch = ExpertBatch(hidden, offsets)
    op = DispatchedExpertMLPFwdOp(
        num_pairs=capacity,
        num_experts=num_experts,
        hidden_size=hidden_size,
        ffn_size=ffn_size,
        dtype=dtype,
        use_fused_activation=use_fused_activation,
    )

    def set_batch(sizes):
        count = sum(sizes)
        hidden[:count].copy_(torch.randn(count, hidden_size, device="cuda", dtype=dtype) * 0.1)
        hidden[count:].fill_(float("nan"))
        host_offsets = [0]
        for size in sizes:
            host_offsets.append(host_offsets[-1] + size)
        offsets.copy_(torch.tensor(host_offsets, device="cuda", dtype=torch.int32))
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
    first_reference = _reference(hidden[:first_count], w_gate_up, w_down, first_sizes)
    assert torch.allclose(
        captured[:first_count].float(),
        first_reference.float(),
        atol=1e-2,
        rtol=1e-2,
    )

    second_sizes = [0, 4, 1, 6]
    second_count = set_batch(second_sizes)
    graph.replay()
    second_reference = _reference(hidden[:second_count], w_gate_up, w_down, second_sizes)
    assert torch.isfinite(captured[:second_count]).all()
    assert torch.allclose(
        captured[:second_count].float(),
        second_reference.float(),
        atol=1e-2,
        rtol=1e-2,
    )
