import time
import pytest
import torch

from top.ops import CountAndGatherOp


def _count_and_gather_reference(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert: int,
    rank_ep: int,
    tile_m: int,
):
    num_seq, hidden_size = x.shape
    num_topk = topk_ids.shape[1]
    total_num_topk = num_seq * num_topk

    start_expert = rank_ep * num_expert
    end_expert = (rank_ep + 1) * num_expert

    seqlens = torch.zeros(num_expert, dtype=torch.int32, device=x.device)
    topk_pos = torch.full((total_num_topk,), -1, dtype=torch.int32, device=x.device)
    cu_seqlens = torch.zeros(num_expert + 1, dtype=torch.int32, device=x.device)
    tiles = torch.zeros(num_expert, dtype=torch.int32, device=x.device)

    topk_ids_flat = topk_ids.reshape(-1)
    for idx in range(total_num_topk):
        iexpert = topk_ids_flat[idx]
        if iexpert >= start_expert and iexpert < end_expert:
            seqlens[iexpert - start_expert] += 1

    for i in range(num_expert):
        cu_seqlens[i + 1] = cu_seqlens[i] + seqlens[i]
        tiles[i] = (seqlens[i] + tile_m - 1) // tile_m

    total_tokens = cu_seqlens[-1].item()
    gate_up_input = torch.zeros(total_tokens, hidden_size, dtype=x.dtype, device=x.device)

    running = torch.zeros_like(seqlens)
    for idx in range(total_num_topk):
        iexpert = topk_ids_flat[idx]
        if iexpert >= start_expert and iexpert < end_expert:
            expert_idx = iexpert - start_expert
            abs_pos = cu_seqlens[expert_idx] + running[expert_idx]
            topk_pos[idx] = abs_pos
            gate_up_input[abs_pos] = x[idx // num_topk]
            running[expert_idx] += 1

    return gate_up_input, topk_pos, seqlens, cu_seqlens, tiles


class TestCountAndGatherOp:
    """Test CountAndGatherOp with pytest."""

    @pytest.fixture
    def test_data(self):
        """Create test data for CountAndGatherOp."""
        num_seq = 16
        hidden_size = 64
        num_topk = 2
        num_expert = 4
        rank_ep = 0

        x = torch.randn(num_seq, hidden_size, device="cuda")
        topk_ids = torch.randint(0, num_expert, (num_seq, num_topk), device="cuda", dtype=torch.int32)

        return x, topk_ids, num_expert, rank_ep, num_seq, hidden_size, num_topk

    def test_basic_functionality(self, test_data):
        """Test basic functionality of CountAndGatherOp."""
        x, topk_ids, num_expert, rank_ep, num_seq, hidden_size, num_topk = test_data

        op = CountAndGatherOp(num_expert=num_expert, rank_ep=rank_ep)
        gate_up_input, topk_pos, seqlens, cu_seqlens, tiles = op.forward(x, topk_ids)
        ref_gate_up_input, ref_topk_pos, ref_seqlens, ref_cu_seqlens, ref_tiles = _count_and_gather_reference(
            x=x,
            topk_ids=topk_ids,
            num_expert=num_expert,
            rank_ep=rank_ep,
            tile_m=op.config["tile_m"],
        )

        assert gate_up_input.shape[0] == seqlens.sum().item()
        assert gate_up_input.shape[1] == hidden_size
        assert topk_pos.shape == (num_seq * num_topk,)
        assert seqlens.shape == (num_expert,)
        assert cu_seqlens.shape == (num_expert + 1,)
        assert tiles.shape == (num_expert,)

        for i in range(1, num_expert + 1):
            assert cu_seqlens[i].item() == cu_seqlens[i - 1].item() + seqlens[i - 1].item()

        for pos in topk_pos:
            assert pos.item() >= -1
            if pos.item() != -1:
                assert pos.item() < gate_up_input.shape[0]

        assert torch.equal(seqlens, ref_seqlens)
        assert torch.equal(cu_seqlens, ref_cu_seqlens)
        assert torch.equal(tiles, ref_tiles)
        assert torch.equal(topk_pos, ref_topk_pos)
        assert torch.equal(gate_up_input, ref_gate_up_input)

    def test_performance(self, test_data):
        """Test performance of CountAndGatherOp."""
        x, topk_ids, num_expert, rank_ep, num_seq, _, _ = test_data

        op = CountAndGatherOp(num_expert=num_expert, rank_ep=rank_ep)
        num_iterations = 10
        start_time = time.time()

        for _ in range(num_iterations):
            op.forward(x, topk_ids)
            torch.cuda.synchronize()

        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations
        throughput = num_seq / avg_time

        print("Performance:")
        print(f"Average time per iteration: {avg_time * 1000:.2f} ms")
        print(f"Throughput: {throughput:.2f} sequences/sec")

        assert avg_time < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
