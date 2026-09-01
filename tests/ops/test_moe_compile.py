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

import operator

import pytest
import torch

from tests.compile_contract import (
    assert_op_owns_graph_nodes,
    operator_overload,
    register_compile_contract,
    traced_call_targets,
)
from tileops.ops.moe import (
    ContiguousLayoutSpec,
    MoePermuteAlignFwdOp,
    MoePostPermuteFwdOp,
    MoePrePermuteFwdOp,
)
from tileops.ops.moe.routed_expert import FusedMoEExpertsNopadPersistent3WGFwdOp
from tileops.ops.moe.routed_expert.gate_up import MoeGateUpFwdOp
from tileops.ops.moe.routed_expert.moe_grouped_gemm_nopad import MoeGroupedGemmNopadFwdOp

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

    topk_ids = torch.randint(0, _NUM_EXPERTS, (_TOKENS, _TOP_K), dtype=torch.int32, device="cuda")
    # Only the padded token count is reproducible: a slot inside an expert is claimed
    # by ``atomic_add``, so two runs order the same tokens differently.
    return make, (topk_ids,), (2,)


def _pre_permute_case(dtype: torch.dtype = torch.bfloat16):
    def make():
        return MoePrePermuteFwdOp(
            ContiguousLayoutSpec.tight_physical_psum(),
            num_local_experts=_NUM_EXPERTS,
        )

    hidden_states = torch.randn(_TOKENS, _HIDDEN, dtype=dtype, device="cuda")
    local_expert_ids = torch.randint(
        0, _NUM_EXPERTS, (_TOKENS, _TOP_K), dtype=torch.int32, device="cuda"
    )
    # Atomic slot assignment makes expert_input and inverse_indices non-deterministic.
    return make, (hidden_states, local_expert_ids), (1,)


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
    "pre_permute": _pre_permute_case,
    "pre_permute_fp16": lambda: _pre_permute_case(torch.float16),
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
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_post_permute_owns_its_graph_nodes(dtype: torch.dtype) -> None:
    """The staged allocating and in-place registrations own their graph nodes."""
    numel = _TOKENS * _TOP_K
    expert_output = torch.randn(numel, _HIDDEN, dtype=dtype, device="cuda")
    inverse_indices = torch.arange(numel, dtype=torch.int32, device="cuda")
    topk_weights = torch.rand(_TOKENS, _TOP_K, dtype=torch.float32, device="cuda")
    out = torch.empty(_TOKENS, _HIDDEN, dtype=dtype, device="cuda")

    def make():
        return MoePostPermuteFwdOp(ContiguousLayoutSpec.tight_physical_psum())

    assert_op_owns_graph_nodes(make(), expert_output, topk_weights, inverse_indices)
    assert_op_owns_graph_nodes(make(), expert_output, topk_weights, inverse_indices, out)
    compiled = _compile_cold(make(), expert_output, topk_weights, inverse_indices)
    eager = (make()(expert_output, topk_weights, inverse_indices),)
    _assert_same_layout(compiled, eager)
    torch.testing.assert_close(compiled[0], eager[0])

    out.zero_()
    torch.compile(make(), fullgraph=True)(expert_output, topk_weights, inverse_indices, out)
    torch.testing.assert_close(out, eager[0])


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_the_experts_composite_shows_only_its_leaf_ops() -> None:
    """A composite is not the unit of replacement, so it registers nothing.

    Its graph is the leaves' operators, which is what makes the leaf the thing a
    target replaces.
    """
    num_experts, top_k, tokens, hidden, ffn = 4, 2, 4, 128, 128
    experts = FusedMoEExpertsNopadPersistent3WGFwdOp(
        num_tokens=tokens,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden,
        ffn_size=ffn,
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
    local_pipeline_leaves = (
        experts._pre_permute,
        experts._gate_up,
        experts._gemm_down,
        experts._post_permute,
    )
    owned_by_leaves = {
        operator_overload(name)
        for leaf in local_pipeline_leaves
        for name in type(leaf).compile_op_names
    }

    calls = traced_call_targets(experts, *args)

    assert calls, "the traced graph called nothing"
    # Temporary bridge from staged physical ends to the existing grouped-GEMM
    # sizes/offsets ABI. These are the only tensor operations allowed outside a leaf.
    layout_bridge = {torch.cat, operator.sub}
    assert calls <= owned_by_leaves | layout_bridge, (
        "graph holds unexpected nodes: "
        f"{sorted(str(c) for c in calls - owned_by_leaves - layout_bridge)}"
    )


for _op_cls in (
    MoePermuteAlignFwdOp,
    MoePrePermuteFwdOp,
    MoePostPermuteFwdOp,
    MoeGateUpFwdOp,
    MoeGroupedGemmNopadFwdOp,
):
    register_compile_contract(_op_cls)
