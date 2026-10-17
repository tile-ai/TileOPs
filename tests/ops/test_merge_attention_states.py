"""Stable state merging and the public compile contract."""

import pytest
import torch

from tests.compile_contract import assert_op_owns_graph_nodes, register_compile_contract
from tests.workload_test_base import TestBase
from tileops.attention import MergeAttentionStatesFwdOp
from workloads.attention.merge_states import MergeAttentionStatesWorkload


@pytest.mark.parametrize(
    "tokens,heads,dim,dtype,strided",
    [
        pytest.param(17, 4, 128, torch.float16, False, marks=pytest.mark.smoke),
        pytest.param(17, 4, 128, torch.bfloat16, False, marks=pytest.mark.smoke),
        pytest.param(65, 3, 128, torch.float16, True, marks=pytest.mark.full, id="strided"),
        pytest.param(17, 3, 37, torch.float16, False, marks=pytest.mark.full, id="scalar-tail"),
        pytest.param(17, 3, 96, torch.float16, False, marks=pytest.mark.full, id="vector-tail-96"),
        pytest.param(
            17, 3, 576, torch.bfloat16, False, marks=pytest.mark.full, id="vector-tail-576"
        ),
        pytest.param(17, 3, 32, torch.float16, False, marks=pytest.mark.full, id="narrow-32"),
        pytest.param(7, 3, 0, torch.bfloat16, False, marks=pytest.mark.full),
    ],
)
def test_merge_attention_states(tokens, heads, dim, dtype, strided):
    """Empty states ignore unused NaNs; extreme LSE values must remain stable."""
    w = MergeAttentionStatesWorkload(tokens, heads, dim, dtype, edge_states=True, strided=strided)
    inputs = w.gen_inputs()
    op = MergeAttentionStatesFwdOp()
    TestBase.check(w, op, *inputs)


@pytest.mark.smoke
def test_merge_attention_states_misaligned_storage():
    """Contiguous views with an unaligned storage offset remain valid inputs."""
    w = MergeAttentionStatesWorkload(4, 2, 128, torch.float16, misaligned=True)
    TestBase.check(w, MergeAttentionStatesFwdOp(), *w.gen_inputs())


@pytest.mark.smoke
def test_merge_attention_states_finite_underflow_keeps_nan():
    """An underflowed weight does not make a finite-LSE state's NaN value empty."""
    w = MergeAttentionStatesWorkload(4, 2, 128, torch.float16, finite_underflow=True)
    TestBase.check(w, MergeAttentionStatesFwdOp(), *w.gen_inputs())


@pytest.mark.smoke
@pytest.mark.cuda_only
def test_merge_attention_states_graph_replay():
    """Graph replay reads changed LSE inputs."""
    w = MergeAttentionStatesWorkload(17, 3, 128, torch.bfloat16)
    inputs = w.gen_inputs()
    op = MergeAttentionStatesFwdOp()
    TestBase.check(w, op, *inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = op(*inputs)
    inputs[1].add_(0.5)
    graph.replay()
    TestBase.check(w, op, *inputs, runs=lambda *args: output)


@pytest.mark.smoke
def test_merge_attention_states_rejects_float32_values():
    """Values outside the supported 16-bit dtype domain are rejected."""
    w = MergeAttentionStatesWorkload(4, 2, 128, torch.float32)
    with pytest.raises(ValueError, match="dtype"):
        MergeAttentionStatesFwdOp()(*w.gen_inputs())


register_compile_contract(MergeAttentionStatesFwdOp)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_merge_attention_states_compile_and_reuse():
    """Fullgraph calls preserve the Op boundary and outputs across token/head counts."""
    w = MergeAttentionStatesWorkload(3, 2, 128, torch.float16)
    inputs = w.gen_inputs()
    op = MergeAttentionStatesFwdOp()
    assert_op_owns_graph_nodes(op, *inputs)
    compiled = torch.compile(op, fullgraph=True)
    TestBase.check(w, op, *inputs, runs=compiled)
    for tokens, heads in ((19, 3), (0, 2)):
        changed = MergeAttentionStatesWorkload(tokens, heads, 128, torch.float16)
        TestBase.check(changed, op, *changed.gen_inputs(), runs=compiled)
