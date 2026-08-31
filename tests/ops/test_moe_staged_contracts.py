"""Contract tests for the tensor-only staged local MoE boundaries."""

import pytest
import torch

from tileops.ops.moe import (
    ContiguousLayoutSpec,
    MaskedLayoutSpec,
    MoePostPermuteFwdOp,
    MoePrePermuteFwdOp,
)

pytestmark = pytest.mark.smoke


def test_layout_specs_are_static_and_explicit() -> None:
    physical = ContiguousLayoutSpec.tight_physical_psum()
    per_row = ContiguousLayoutSpec.tight_per_row()
    masked = MaskedLayoutSpec(max_m=4)

    assert physical.selection_key == "tight_physical_psum"
    assert per_row.selection_key == "tight_per_row"
    assert masked.selection_key == "masked_predicated"
    assert physical.max_m is None
    assert masked.max_m == 4
    with pytest.raises(ValueError, match="non-negative"):
        MaskedLayoutSpec(max_m=-1)


def test_pre_rejects_an_empty_local_expert_domain() -> None:
    with pytest.raises(ValueError, match="num_local_experts must be positive"):
        MoePrePermuteFwdOp(ContiguousLayoutSpec.tight_physical_psum(), num_local_experts=0)


def test_pre_output_shapes_follow_the_layout_metadata_semantics() -> None:
    hidden_shape = (4, 64)
    ids_shape = (4, 2)
    physical = MoePrePermuteFwdOp(ContiguousLayoutSpec.tight_physical_psum(), num_local_experts=3)
    per_row = MoePrePermuteFwdOp(ContiguousLayoutSpec.tight_per_row(), num_local_experts=3)
    masked = MoePrePermuteFwdOp(MaskedLayoutSpec(max_m=8), num_local_experts=3)

    assert physical._infer_output_shapes(hidden_shape, ids_shape)["layout_metadata"] == (3,)
    assert per_row._infer_output_shapes(hidden_shape, ids_shape)["layout_metadata"] == (8,)
    assert masked._infer_output_shapes(hidden_shape, ids_shape) == {
        "expert_input": (3, 8, 64),
        "layout_metadata": (3,),
        "inverse_indices": (8,),
    }


def test_tight_pre_and_post_ship_one_default_candidate() -> None:
    layout = ContiguousLayoutSpec.tight_physical_psum()
    pre = MoePrePermuteFwdOp(layout, num_local_experts=1)
    post = MoePostPermuteFwdOp(layout=layout)
    assert tuple(pre.kernel_map) == ("tight_physical_psum",)
    assert tuple(post.kernel_map) == ("tight_physical_psum",)


def test_pre_and_post_install_device_side_index_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[bool] = []

    def capture(condition: torch.Tensor, _message: str) -> None:
        observed.append(bool(condition.item()))

    monkeypatch.setattr(torch, "_assert_async", capture)
    layout = ContiguousLayoutSpec.tight_physical_psum()
    pre = MoePrePermuteFwdOp(layout, num_local_experts=2)
    x = torch.empty(1, 8, dtype=torch.bfloat16, device="cuda")
    pre.make_call(x, torch.tensor([[0, 2]], dtype=torch.int32, device="cuda"))

    post = MoePostPermuteFwdOp(layout)
    expert_output = torch.empty(2, 8, dtype=torch.bfloat16, device="cuda")
    weights = torch.ones(1, 2, dtype=torch.float32, device="cuda")
    post.make_call(
        expert_output,
        torch.tensor([0, 2], dtype=torch.int32, device="cuda"),
        weights,
    )
    assert observed == [False, False]


def test_staged_tight_pre_post_round_trip() -> None:
    """The three-tensor boundary preserves rank-grouped contributions."""
    tokens, top_k, experts, hidden = 4, 2, 4, 64
    device = torch.device("cuda")
    dtype = torch.bfloat16
    x = torch.randn(tokens, hidden, dtype=dtype, device=device)
    local_ids = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3]], dtype=torch.int32, device=device)
    weights = torch.rand(tokens, top_k, dtype=torch.float32, device=device)
    layout = ContiguousLayoutSpec.tight_physical_psum()

    pre = MoePrePermuteFwdOp(layout, num_local_experts=experts)
    expert_input, metadata, inverse = pre(x, local_ids)
    assert expert_input.shape == (tokens * top_k, hidden)
    assert metadata.shape == (experts,)
    assert inverse.shape == (tokens * top_k,)

    token_rows = torch.arange(tokens * top_k, device=device) // top_k
    torch.testing.assert_close(expert_input[inverse.long()], x[token_rows])
    post = MoePostPermuteFwdOp(layout=layout)
    output = post(expert_input, weights, inverse)
    reference = x.float() * weights.sum(dim=1, keepdim=True)
    torch.testing.assert_close(output.float(), reference, rtol=2e-2, atol=2e-2)
    assert pre.eval_roofline() == (0, 1616)
    assert post.eval_roofline() == (1024, 1600)
