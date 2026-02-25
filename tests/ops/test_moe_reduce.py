import pytest
import torch

from top.ops import MoeReduceOp


def _moe_reduce_reference(
    x: torch.Tensor,
    topk_pos: torch.Tensor,
    topk_scale: torch.Tensor,
    shared_output: torch.Tensor | None = None,
) -> torch.Tensor:
    num_seq, num_topk = topk_pos.shape
    num_tokens, hidden_size = x.shape

    x_fp32 = x.to(torch.float32)
    topk_scale_fp32 = topk_scale.to(torch.float32)
    y_accum = torch.zeros((num_seq, hidden_size), dtype=torch.float32, device=x.device)

    for i in range(num_topk):
        pos = topk_pos[:, i]
        scale = topk_scale_fp32[:, i].unsqueeze(1)
        mask = (pos >= 0) & (pos < num_tokens)
        valid_pos = pos[mask]
        valid_scale = scale[mask]
        valid_seq = torch.where(mask)[0]

        if valid_pos.numel() > 0:
            expert_outputs = x_fp32[valid_pos]
            weighted_outputs = expert_outputs * valid_scale
            y_accum.index_add_(0, valid_seq, weighted_outputs)

    if shared_output is not None:
        y_accum += shared_output.to(torch.float32)

    return y_accum.to(x.dtype)


@pytest.mark.parametrize(
    "num_seq, num_topk, hidden_size",
    [
        (16, 2, 64),
        (32, 1, 128),
        (64, 4, 256),
    ],
)
def test_moe_reduce_op_basic(num_seq: int, num_topk: int, hidden_size: int) -> None:
    num_tokens = num_seq * num_topk
    x = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.float16)
    topk_pos = torch.randint(0, num_tokens, (num_seq, num_topk), device="cuda", dtype=torch.int32)
    topk_scale = torch.randn(num_seq, num_topk, device="cuda", dtype=torch.float32)

    op = MoeReduceOp()
    output = op.forward(x, topk_pos, topk_scale)
    reference = _moe_reduce_reference(x, topk_pos, topk_scale)

    assert output.shape == (num_seq, hidden_size)
    assert output.dtype == x.dtype
    assert torch.allclose(output, reference, atol=1e-3, rtol=1e-3)


def test_moe_reduce_op_with_shared_output() -> None:
    num_seq = 16
    num_topk = 2
    hidden_size = 64
    num_tokens = num_seq * num_topk

    x = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.float16)
    topk_pos = torch.randint(0, num_tokens, (num_seq, num_topk), device="cuda", dtype=torch.int32)
    topk_scale = torch.randn(num_seq, num_topk, device="cuda", dtype=torch.float32)
    shared_output = torch.randn(num_seq, hidden_size, device="cuda", dtype=torch.float16)

    op = MoeReduceOp()
    output = op.forward(x, topk_pos, topk_scale, shared_output)
    reference = _moe_reduce_reference(x, topk_pos, topk_scale, shared_output)

    assert output.shape == (num_seq, hidden_size)
    assert output.dtype == x.dtype
    assert torch.allclose(output, reference, atol=1e-3, rtol=1e-3)


def test_moe_reduce_op_invalid_shape() -> None:
    num_seq = 16
    num_topk = 2
    hidden_size = 64
    num_tokens = num_seq * num_topk

    x = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.float16)
    topk_pos = torch.randint(0, num_tokens, (num_seq, num_topk), device="cuda", dtype=torch.int32)
    invalid_topk_scale = torch.randn(num_seq, num_topk + 1, device="cuda", dtype=torch.float32)

    op = MoeReduceOp()
    with pytest.raises(ValueError):
        op.forward(x, topk_pos, invalid_topk_scale)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
