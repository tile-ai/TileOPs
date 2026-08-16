"""Compile-boundary tests for the MoE ops that register one.

Two assertions per op, both from a cold instance:

1. Every ``call_function`` node in the traced graph is an operator the op
   declares in ``compile_op_names``. A kernel's own registration, or a tensor op
   left outside the boundary, fails here — either means another target could
   change the graph.
2. ``torch.compile(op, fullgraph=True)`` returns the shapes and dtypes the fake
   promised, and the same values eager returns wherever the kernel is
   reproducible. This is the evidence behind the manifest's
   ``torch_compile_fullgraph``.

A composite registers no operator of its own, so the last test asserts the other
half: the graph of ``FusedMoEExpertsNopadPersistent3WGFwdOp`` holds its leaves'
operators and nothing else. ``FusedMoeFwdOp`` is absent because the routing op it
builds has no boundary yet.
"""

import pytest
import torch

from tests.compile_contract import (
    assert_op_owns_graph_nodes,
    operator_overload,
    register_compile_contract,
    traced_call_targets,
)
from tileops.ops.moe import MoePermuteAlignFwdOp
from tileops.ops.moe.routed_expert import FusedMoEExpertsNopadPersistent3WGFwdOp
from tileops.ops.moe.routed_expert.gate_up import MoeGateUpFwdOp
from tileops.ops.moe.routed_expert.moe_grouped_gemm_nopad import MoeGroupedGemmNopadFwdOp
from tileops.ops.moe.routed_expert.permute_nopad import MoePermuteNopadFwdOp
from tileops.ops.moe.routed_expert.unpermute import MoeUnpermuteFwdOp

_NUM_EXPERTS = 4
_TOP_K = 2
_TOKENS = 4
_HIDDEN = 64


def _compile_cold(op, *inputs) -> tuple:
    """Outputs of one cold ``fullgraph=True`` compile, always as a tuple."""
    outputs = torch.compile(op, fullgraph=True)(*inputs)
    return outputs if isinstance(outputs, tuple) else (outputs,)


def _assert_same_layout(compiled: tuple, eager: tuple) -> None:
    """The compiled call produced the shapes and dtypes the fake promised."""
    for got, want in zip(compiled, eager, strict=True):
        assert got.shape == want.shape, f"{got.shape} != {want.shape}"
        assert got.dtype == want.dtype, f"{got.dtype} != {want.dtype}"


def _grouped_gemm_inputs(numel: int, num_experts: int, n: int, k: int):
    """Tight rows split evenly across experts, plus the two index arrays."""
    a = torch.randn(numel, k, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
    per_expert = numel // num_experts
    sizes = torch.full((num_experts,), per_expert, dtype=torch.int32, device="cuda")
    offsets = torch.arange(num_experts, dtype=torch.int32, device="cuda") * per_expert
    return a, b, sizes, offsets


def _permute_align_case():
    def make():
        return MoePermuteAlignFwdOp(_TOKENS, _TOP_K, _NUM_EXPERTS, block_size=4)

    topk_ids = torch.randint(
        0, _NUM_EXPERTS, (_TOKENS, _TOP_K), dtype=torch.int32, device="cuda")
    # Only the padded token count is reproducible: a slot inside an expert is claimed
    # by ``atomic_add``, so two runs order the same tokens differently.
    return make, (topk_ids,), (2,)


def _permute_nopad_case(local: int = _NUM_EXPERTS, with_map: bool = False):
    def make():
        return MoePermuteNopadFwdOp(
            num_experts=_NUM_EXPERTS, num_experts_local=local)

    hidden_states = torch.randn(
        _TOKENS, _HIDDEN, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.randint(
        0, _NUM_EXPERTS, (_TOKENS, _TOP_K), dtype=torch.int32, device="cuda")
    inputs = (hidden_states, topk_ids)
    if with_map:
        expert_map = torch.full((_NUM_EXPERTS,), -1, dtype=torch.int32, device="cuda")
        expert_map[:local] = torch.arange(local, dtype=torch.int32, device="cuda")
        inputs += (expert_map,)
    # The per-expert offsets, sizes and prefix sum are counts. The gathered rows and
    # the forward map are not: a slot inside an expert is claimed by ``atomic_add``.
    return make, inputs, (1, 2, 3)


def _gate_up_case():
    numel, ffn, k = 64, 128, 128

    def make():
        return MoeGateUpFwdOp(numel, _NUM_EXPERTS, ffn, k)

    return make, _grouped_gemm_inputs(numel, _NUM_EXPERTS, 2 * ffn, k), "all"


def _grouped_gemm_nopad_case():
    numel, n, k = 64, 128, 128

    def make():
        return MoeGroupedGemmNopadFwdOp(numel, _NUM_EXPERTS, n, k)

    return make, _grouped_gemm_inputs(numel, _NUM_EXPERTS, n, k), "all"


_LEAF_CASES = {
    "permute_align": _permute_align_case,
    "permute_nopad": _permute_nopad_case,
    # Supplying the map keeps the graph inside the same operator: the fake sizes the
    # four per-expert outputs from ``num_experts_local``, a constructor parameter, so
    # the map's contents never reach a traced region.
    "permute_nopad_with_a_map": lambda: _permute_nopad_case(
        local=_NUM_EXPERTS // 2, with_map=True),
    "gate_up": _gate_up_case,
    "grouped_gemm_nopad": _grouped_gemm_nopad_case,
}


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
@pytest.mark.parametrize("case", _LEAF_CASES.values(), ids=_LEAF_CASES)
def test_leaf_op_owns_its_graph_nodes(case) -> None:
    make, inputs, reproducible = case()

    assert_op_owns_graph_nodes(make(), *inputs)

    compiled = _compile_cold(make(), *inputs)
    eager = make()(*inputs)
    eager = tuple(eager) if isinstance(eager, tuple) else (eager,)
    _assert_same_layout(compiled, eager)
    indices = range(len(compiled)) if reproducible == "all" else reproducible
    for i in indices:
        torch.testing.assert_close(compiled[i], eager[i])


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_unpermute_owns_its_graph_nodes() -> None:
    """Both registrations, because ``out`` picks between them at call time."""
    numel = _TOKENS * _TOP_K
    mm2_pad = torch.randn(numel, _HIDDEN, dtype=torch.bfloat16, device="cuda")
    fwd_idx = torch.arange(numel, dtype=torch.int32, device="cuda")
    topk_weights = torch.rand(_TOKENS, _TOP_K, dtype=torch.float32, device="cuda")
    out = torch.empty(_TOKENS, _HIDDEN, dtype=torch.bfloat16, device="cuda")

    def make():
        return MoeUnpermuteFwdOp(_TOKENS, _TOP_K, _HIDDEN, padded_batch_sum=numel)

    assert_op_owns_graph_nodes(make(), mm2_pad, fwd_idx, topk_weights)
    assert_op_owns_graph_nodes(make(), mm2_pad, fwd_idx, topk_weights, out)
    compiled = _compile_cold(make(), mm2_pad, fwd_idx, topk_weights)
    eager = (make()(mm2_pad, fwd_idx, topk_weights),)
    _assert_same_layout(compiled, eager)
    torch.testing.assert_close(compiled[0], eager[0])

    out.zero_()
    torch.compile(make(), fullgraph=True)(mm2_pad, fwd_idx, topk_weights, out)
    # The in-place operator declares the mutation, so what the compiled call left in
    # ``out`` is what the allocating path returns.
    torch.testing.assert_close(out, make()(mm2_pad, fwd_idx, topk_weights))


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_the_experts_composite_shows_only_its_leaf_ops() -> None:
    """A composite is not the unit of replacement, so it registers nothing.

    Its graph is the leaves' operators, which is what makes the leaf the thing a
    target replaces.
    """
    num_experts, top_k, tokens, hidden, ffn = 4, 2, 4, 128, 128
    experts = FusedMoEExpertsNopadPersistent3WGFwdOp(
        num_tokens=tokens, num_experts=num_experts, num_experts_local=num_experts,
        top_k=top_k, hidden_size=hidden, ffn_size=ffn,
    )
    assert experts.compile_op_names == ()

    hidden_states = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    args = (
        torch.empty(tokens, hidden, dtype=torch.bfloat16, device="cuda"),
        hidden_states,
        torch.randn(num_experts, 2 * ffn, hidden, dtype=torch.bfloat16, device="cuda"),
        torch.randn(num_experts, hidden, ffn, dtype=torch.bfloat16, device="cuda"),
        torch.rand(tokens, top_k, dtype=torch.float32, device="cuda"),
        torch.randint(0, num_experts, (tokens, top_k), dtype=torch.int32, device="cuda"),
        hidden_states.new_empty(0),
        hidden_states.new_empty(0),
    )
    kwargs = {"num_experts": num_experts}
    owned_by_leaves = {
        operator_overload(name)
        for leaf in (experts._permute, experts._gate_up, experts._gemm_down,
                     experts._unpermute)
        for name in type(leaf).compile_op_names
    }

    calls = traced_call_targets(experts, *args, **kwargs)

    assert calls, "the traced graph called nothing"
    assert calls <= owned_by_leaves, (
        f"graph holds nodes no leaf op owns: "
        f"{sorted(str(c) for c in calls - owned_by_leaves)}")


for _op_cls in (
    MoePermuteAlignFwdOp,
    MoePermuteNopadFwdOp,
    MoeUnpermuteFwdOp,
    MoeGateUpFwdOp,
    MoeGroupedGemmNopadFwdOp,
):
    register_compile_contract(_op_cls)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
