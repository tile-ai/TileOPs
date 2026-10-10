"""Compile-boundary tests for the MoE ops that register one.

Two assertions per op, both from a cold instance:

1. Every ``call_function`` node in the traced graph is an operator the op
   declares in ``compile_op_names``. A kernel's own registration, or a tensor op
   left outside the boundary, fails here — either means another target could
   change the graph.
2. ``torch.compile(op, fullgraph=True)`` returns the shapes and dtypes the fake
   promised, and the same values eager returns wherever the kernel is
   reproducible. This is the evidence behind each class's compile-boundary
   declaration.

A composite registers no operator of its own, so the last test asserts the other
half: the graph of ``FusedMoEExpertsFwdOp`` holds its leaves'
operators and nothing else.
"""

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
    FusedTopKFwdOp,
    MaskedLayoutSpec,
    MoEGroupedGemmFP8FwdOp,
    MoEGroupedGemmFwdOp,
    MoEPermuteAlignFwdOp,
    MoEPostPermuteFwdOp,
    MoEPrePermuteFwdOp,
    SharedExpertMLPFwdOp,
)
from tileops.ops.moe.fused_moe_shared_expert import FusedMoESharedExpertFwdOp
from tileops.ops.moe.routed_expert import FusedMoEExpertsFwdOp, IndexedExpertMLPFwdOp
from workloads.device import run_device


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
    """Tight rows split evenly across experts, plus the psum ends."""
    a = torch.randn(numel, k, dtype=torch.bfloat16, device=run_device())
    b = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device=run_device())
    per_expert = numel // num_experts
    ends = torch.arange(1, num_experts + 1, dtype=torch.int32, device=run_device()) * per_expert
    return a, b, ends


def _permute_align_case():
    num_experts = 4
    top_k = 2
    num_tokens = 4

    def make():
        return MoEPermuteAlignFwdOp(num_experts, block_size=4)

    topk_ids = torch.randint(
        0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=run_device()
    )
    # Only the padded token count is reproducible: a slot inside an expert is claimed
    # by ``atomic_add``, so two runs order the same tokens differently.
    return make, (topk_ids,), (2,)


def _pre_permute_case(dtype: torch.dtype = torch.bfloat16):
    num_experts = 4
    top_k = 2
    num_tokens = 4
    hidden_size = 64

    def make():
        return MoEPrePermuteFwdOp(
            ContiguousLayoutSpec.tight_physical_psum(),
            num_local_experts=num_experts,
        )

    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype, device=run_device())
    local_expert_ids = torch.randint(
        0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=run_device()
    )
    # Atomic slot assignment makes expert_input and inverse_indices non-deterministic.
    return make, (hidden_states, local_expert_ids), (1,)


def _aligned_pre_permute_case():
    num_experts = 4
    top_k = 2
    num_tokens = 4
    hidden_size = 64

    def make():
        return MoEPrePermuteFwdOp(
            ContiguousLayoutSpec.aligned_per_row(8),
            num_local_experts=num_experts,
        )

    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=run_device())
    local_expert_ids = torch.randint(
        0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=run_device()
    )
    # Per-row metadata is reproducible; atomic slot assignment is not.
    return make, (hidden_states, local_expert_ids), (1,)


def _staged_grouped_gemm_case(dtype: torch.dtype = torch.bfloat16, activation: str | None = None):
    num_experts = 4
    numel, n, k = 64, 128, 128

    def make():
        return MoEGroupedGemmFwdOp(
            ContiguousLayoutSpec.tight_physical_psum(), activation=activation
        )

    a, b, ends = _grouped_gemm_inputs(numel, num_experts, n, k)
    return make, (a.to(dtype), b.to(dtype), ends), "all"


def _staged_grouped_gemm_masked_case():
    num_experts = 4
    max_m, n, k = 32, 128, 128

    def make():
        return MoEGroupedGemmFwdOp(MaskedLayoutSpec(max_m=max_m))

    a = torch.randn(num_experts, max_m, k, dtype=torch.bfloat16, device=run_device())
    b = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device=run_device())
    masked_m = torch.tensor([32, 0, 17, 32], dtype=torch.int32, device=run_device())
    # Rows past an expert's valid count hold unspecified values.
    return make, (a, b, masked_m), ()


def _grouped_gemm_fp8_masked_case():
    num_experts = 2
    max_m, n, k = 64, 128, 128

    def make():
        return MoEGroupedGemmFP8FwdOp(MaskedLayoutSpec(max_m=max_m))

    fp8 = torch.float8_e4m3fn
    a = (torch.randn(num_experts, max_m, k, device=run_device()) * 0.25).to(fp8)
    a_scale = torch.rand(num_experts, max_m, k // 128, device=run_device()) + 0.5
    b = (torch.randn(num_experts, n, k, device=run_device()) * 0.25).to(fp8)
    b_scale = torch.rand(num_experts, n // 128, k // 128, device=run_device()) + 0.5
    masked_m = torch.tensor([64, 17], dtype=torch.int32, device=run_device())
    # Rows past an expert's valid count hold unspecified values.
    return make, (a, a_scale, b, b_scale, masked_m), ()


def _fused_topk_case(with_bias: bool = False):
    num_experts = 4
    top_k = 2
    num_tokens = 4

    def make():
        return FusedTopKFwdOp(top_k, scoring_func="sigmoid", renormalize=True)

    gating = torch.randn(num_tokens, num_experts, dtype=torch.bfloat16, device=run_device())
    bias = torch.randn(num_experts, dtype=torch.float32, device=run_device())
    return make, (gating, bias) if with_bias else (gating,), "all"


def _shared_expert_case(tokens: int = 4):
    hidden, ffn = 128, 128
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=run_device()) * 0.1
    w_gate_up = torch.randn(2 * ffn, hidden, dtype=torch.bfloat16, device=run_device()) * 0.02
    w_down = torch.randn(hidden, ffn, dtype=torch.bfloat16, device=run_device()) * 0.02
    return SharedExpertMLPFwdOp, (x, w_gate_up, w_down), "all"


_LEAF_CASES = {
    "fused_topk": _fused_topk_case,
    "fused_topk_bias": lambda: _fused_topk_case(with_bias=True),
    "staged_grouped_gemm": _staged_grouped_gemm_case,
    "staged_grouped_gemm_fp16": lambda: _staged_grouped_gemm_case(torch.float16),
    "staged_grouped_gemm_fused": lambda: _staged_grouped_gemm_case(activation="silu_and_mul"),
    "staged_grouped_gemm_masked": _staged_grouped_gemm_masked_case,
    "permute_align": _permute_align_case,
    "pre_permute": _pre_permute_case,
    "pre_permute_fp16": lambda: _pre_permute_case(torch.float16),
    "pre_permute_aligned": _aligned_pre_permute_case,
    "shared_expert_mlp": _shared_expert_case,
    "shared_expert_mlp_empty": lambda: _shared_expert_case(tokens=0),
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


@pytest.mark.sm90
@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_fp8_grouped_gemm_owns_its_graph_nodes() -> None:
    """The FP8 masked GEMM, whose only kernel is SM90's, traces cold to its own operator."""
    make, inputs, _ = _grouped_gemm_fp8_masked_case()
    assert_op_owns_graph_nodes(make(), *inputs)
    _assert_same_layout(_compile_cold(make(), *inputs), (make()(*inputs),))


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_post_permute_owns_its_graph_nodes(dtype: torch.dtype) -> None:
    """The staged allocating and in-place registrations own their graph nodes."""
    top_k = 2
    num_tokens = 4
    hidden_size = 64
    numel = num_tokens * top_k
    expert_output = torch.randn(numel, hidden_size, dtype=dtype, device=run_device())
    inverse_indices = torch.arange(numel, dtype=torch.int32, device=run_device())
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.float32, device=run_device())
    out = torch.empty(num_tokens, hidden_size, dtype=dtype, device=run_device())

    def make():
        return MoEPostPermuteFwdOp(ContiguousLayoutSpec.tight_physical_psum())

    assert_op_owns_graph_nodes(make(), expert_output, topk_weights, inverse_indices)
    assert_op_owns_graph_nodes(make(), expert_output, topk_weights, inverse_indices, out)
    compiled = _compile_cold(make(), expert_output, topk_weights, inverse_indices)
    eager = (make()(expert_output, topk_weights, inverse_indices),)
    _assert_same_layout(compiled, eager)
    torch.testing.assert_close(compiled[0], eager[0])

    out.zero_()
    torch.compile(make(), fullgraph=True)(expert_output, topk_weights, inverse_indices, out)
    torch.testing.assert_close(out, eager[0])


def _experts_args(tokens, experts_count, top_k, hidden, ffn):
    return (
        torch.empty(tokens, hidden, dtype=torch.bfloat16, device=run_device()),
        torch.randn(tokens, hidden, dtype=torch.bfloat16, device=run_device()) * 0.1,
        torch.randn(experts_count, 2 * ffn, hidden, dtype=torch.bfloat16, device=run_device())
        * 0.02,
        torch.randn(experts_count, hidden, ffn, dtype=torch.bfloat16, device=run_device()) * 0.02,
        torch.rand(tokens, top_k, dtype=torch.float32, device=run_device()),
        torch.randint(0, experts_count, (tokens, top_k), dtype=torch.int32, device=run_device()),
    )


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_the_experts_composite_shows_only_its_leaf_ops() -> None:
    """A composite is not the unit of replacement, so it registers nothing.

    Its graph is the leaves' operators, which is what makes the leaf the thing a target
    replaces. The composite builds its pre-permute stage for the call's expert count, so an
    eager call first holds it; the composite claims no cold traced contract.
    """
    experts = FusedMoEExpertsFwdOp()
    assert experts.compile_op_names == ()
    args = _experts_args(tokens=64, experts_count=4, top_k=2, hidden=128, ffn=128)
    experts(*args)
    owned_by_leaves = {
        operator_overload(name)
        for leaf in experts.kernel_delegates()
        for op in (leaf, *leaf.kernel_delegates())
        for name in type(op).compile_op_names
    }

    calls = traced_call_targets(experts, *args)

    assert calls, "the traced graph called nothing"
    assert calls <= owned_by_leaves, (
        f"graph holds unexpected nodes: {sorted(str(c) for c in calls - owned_by_leaves)}"
    )


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_the_shared_expert_composite_compiles_cold_to_its_leaves() -> None:
    """The shared expert is a sub-op, so a cold small-route call traces to the router, the
    indexed experts and the shared expert, and matches eager."""
    top_k = 2
    num_tokens = 4
    hidden, ffn, experts, shared = 128, 256, 8, 128
    args = (
        torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=run_device()) * 0.1,
        torch.randn(num_tokens, experts, dtype=torch.float32, device=run_device()),
        torch.randn(experts, 2 * ffn, hidden, dtype=torch.bfloat16, device=run_device()) * 0.02,
        torch.randn(experts, hidden, ffn, dtype=torch.bfloat16, device=run_device()) * 0.02,
        None,
        torch.randn(2 * shared, hidden, dtype=torch.bfloat16, device=run_device()) * 0.02,
        torch.randn(hidden, shared, dtype=torch.bfloat16, device=run_device()) * 0.02,
    )
    calls = traced_call_targets(FusedMoESharedExpertFwdOp(top_k), *args)
    leaves = (FusedTopKFwdOp, IndexedExpertMLPFwdOp, SharedExpertMLPFwdOp)
    owners = {operator_overload(n): cls for cls in leaves for n in cls.compile_op_names}
    assert calls <= set(owners), sorted(str(c) for c in calls - set(owners))
    assert {owners[c] for c in calls} == set(leaves)
    compiled = _compile_cold(FusedMoESharedExpertFwdOp(top_k), *args)
    for got, want in zip(compiled, FusedMoESharedExpertFwdOp(top_k)(*args), strict=True):
        torch.testing.assert_close(got, want)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_small_route_experts_compile_to_the_indexed_leaf() -> None:
    args = _experts_args(tokens=4, experts_count=8, top_k=2, hidden=128, ffn=256)
    assert traced_call_targets(FusedMoEExpertsFwdOp(), *args) == {
        operator_overload("tileops::moe_indexed_expert_mlp_fwd_writes_output")
    }


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_the_indexed_op_compiles_cold_to_its_operator() -> None:
    args = _experts_args(tokens=4, experts_count=8, top_k=2, hidden=128, ffn=256)
    assert_op_owns_graph_nodes(IndexedExpertMLPFwdOp(), *args)
    compiled_out = torch.empty_like(args[0])
    torch.compile(IndexedExpertMLPFwdOp(), fullgraph=True)(compiled_out, *args[1:])
    IndexedExpertMLPFwdOp()(*args)
    torch.testing.assert_close(compiled_out, args[0])


for _op_cls in (
    FusedTopKFwdOp,
    IndexedExpertMLPFwdOp,
    MoEPermuteAlignFwdOp,
    MoEPrePermuteFwdOp,
    MoEPostPermuteFwdOp,
    MoEGroupedGemmFwdOp,
    MoEGroupedGemmFP8FwdOp,
    SharedExpertMLPFwdOp,
):
    register_compile_contract(_op_cls)
