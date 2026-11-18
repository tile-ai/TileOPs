"""Exact n-gram ids and context state through the public operator."""

import pytest
import torch

from tests.compile_contract import assert_op_owns_graph_nodes, register_compile_contract
from tests.workload_test_base import TestBase
from tileops.sequence_modeling import NGramHashFwdOp
from workloads.numerics import compare_outputs
from workloads.sequence_modeling.ngram_hash import (
    NGramHashWorkload,
    ngram_hash_result,
)


def _check(workload, op=None, inputs=None, subject=None):
    op = NGramHashFwdOp(workload.eos_token_id) if op is None else op
    inputs = workload.gen_inputs() if inputs is None else inputs
    TestBase.check(
        workload,
        op,
        *inputs,
        runs=lambda *args: ngram_hash_result(op if subject is None else subject, *args),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(dict(lengths=[1, 1]), marks=pytest.mark.smoke, id="decode"),
        pytest.param(
            dict(lengths=[131], n=4, heads=9), marks=pytest.mark.full, id="odd-heads-orders-tail"
        ),
        pytest.param(
            dict(lengths=[0, 3, 0, 129, 0], packed=True, slots=[5, 1, 4, 0, 3], pool=7),
            marks=pytest.mark.full,
            id="packed-empty-slots",
        ),
        pytest.param(
            dict(lengths=[1, 1], n=2, heads=3, slots=[2, 0], pool=4, width=5, accepted=[5, 2]),
            marks=pytest.mark.full,
            id="bigram-rollback-bound",
        ),
        pytest.param(
            dict(
                lengths=[0, 1, 4], packed=True, slots=[3, 1, 0], pool=5, width=5, accepted=[1, 4, 2]
            ),
            marks=pytest.mark.full,
            id="packed-rollback",
        ),
        pytest.param(
            dict(lengths=[11], extremes=True), marks=pytest.mark.full, id="int64-extremes"
        ),
        pytest.param(dict(lengths=[0, 0]), marks=pytest.mark.full, id="empty-dense"),
        pytest.param(dict(lengths=[], packed=True), marks=pytest.mark.full, id="empty-batch"),
        pytest.param(dict(lengths=[3], heads=0), marks=pytest.mark.full, id="zero-heads-state"),
    ],
)
def test_ngram_hash_exact(kwargs):
    _check(NGramHashWorkload(**kwargs))


@pytest.mark.smoke
def test_ngram_hash_strided():
    """Both context axes may be strided; holes must remain untouched."""
    workload = NGramHashWorkload([3, 3], width=4, accepted=[2, 1])
    inputs = list(workload.gen_inputs())
    storage = torch.full((4, 8), -19, dtype=torch.int64, device=inputs[1].device)
    context = storage[::2, ::2]
    context.copy_(inputs[1])
    inputs[1] = context
    inputs[0] = inputs[0].T.contiguous().T
    _check(workload, inputs=inputs)
    assert bool((storage[1::2] == -19).all())
    assert bool((storage[:, 1::2] == -19).all())


@pytest.mark.smoke
def test_ngram_hash_chunked():
    """Chunk boundaries, including an EOS boundary, preserve the full-prompt ids."""
    workload = NGramHashWorkload([11, 11])
    inputs = workload.gen_inputs()
    expected = workload.ref_program(*inputs)
    op = NGramHashFwdOp(workload.eos_token_id)
    results = []
    for begin, end in [(0, 1), (1, 5), (5, 6), (6, 11)]:
        results.append(op(inputs[0][:, begin:end], *inputs[1:]))
    compare_outputs(
        (torch.cat(results, dim=1), inputs[1]), expected, workload.verification(*inputs)
    )


@pytest.mark.smoke
def test_ngram_hash_rollback():
    """A second speculative call reads each accepted prefix, including an EOS."""
    workload = NGramHashWorkload([4, 4], slots=[3, 1], pool=5, width=5, accepted=[1, 1])
    inputs = list(workload.gen_inputs())
    op = NGramHashFwdOp(workload.eos_token_id)
    _check(workload, op, inputs)
    op(*inputs)
    inputs[7].copy_(torch.tensor([2, 4], device=inputs[7].device, dtype=torch.int32))
    inputs[0].add_(7)
    _check(workload, op, inputs)


@pytest.mark.parametrize(
    "packed",
    [
        pytest.param(False, marks=pytest.mark.smoke, id="dense"),
        pytest.param(True, marks=pytest.mark.full, id="packed"),
    ],
)
def test_ngram_hash_changing_shapes(packed):
    """One op accepts growing and shrinking token batches and context pools."""
    op = None
    for batch, length in [(2, 3), (4, 1), (1, 137), (2, 3)]:
        lengths = list(range(length, length + batch)) if packed else [length] * batch
        workload = NGramHashWorkload(lengths, packed=packed)
        if op is None:
            op = NGramHashFwdOp(workload.eos_token_id)
        _check(workload, op)


register_compile_contract(NGramHashFwdOp)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_ngram_hash_compiles_cold():
    workload = NGramHashWorkload(
        [3, 1], packed=True, slots=[2, 0], pool=4, width=4, accepted=[2, 1]
    )
    op = NGramHashFwdOp(workload.eos_token_id)
    _check(workload, op, subject=torch.compile(op, fullgraph=True))


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_ngram_hash_owns_graph():
    workload = NGramHashWorkload([1])
    assert_op_owns_graph_nodes(NGramHashFwdOp(workload.eos_token_id), *workload.gen_inputs())


@pytest.mark.parametrize(
    "invalid", ["dtype", "order", "heads", "width", "prime", "slot", "accepted", "cu"]
)
@pytest.mark.smoke
def test_ngram_hash_invalid(invalid):
    workload = NGramHashWorkload(
        [1, 2], packed=True, slots=[2, 0], pool=4, width=3, accepted=[1, 1]
    )
    inputs = list(workload.gen_inputs())
    if invalid == "dtype":
        inputs[0] = inputs[0].int()
    elif invalid == "order":
        inputs[2] = inputs[2][:1].clone()
    elif invalid == "heads":
        inputs[3], inputs[4] = inputs[3][:3].clone(), inputs[4][:3].clone()
    elif invalid == "width":
        inputs[1] = inputs[1][:, :1].clone()
    elif invalid == "prime":
        inputs[3][0] = 0
    elif invalid == "slot":
        inputs[6][0] = 4
    elif invalid == "accepted":
        inputs[7][0] = 0
    else:
        inputs[5][-1] += 1
    with pytest.raises((ValueError, TypeError)):
        NGramHashFwdOp(workload.eos_token_id, validate_inputs=True)(*inputs)


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.cuda_graph(op="NGramHashFwdOp")
def test_ngram_hash_graph_metadata():
    """Graph replay reads changed tokens, slots, packed offsets and accepted counts."""
    workload = NGramHashWorkload(
        [3, 0, 2], packed=True, slots=[3, 1, 0], pool=5, width=5, accepted=[1, 2, 3]
    )
    inputs = workload.gen_inputs()
    op = NGramHashFwdOp(workload.eos_token_id)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = op(*inputs)
    inputs[0].add_(11)
    inputs[1].add_(13)
    inputs[2].sub_(2)
    inputs[3].add_(2)
    inputs[4].add_(17)
    inputs[5].copy_(torch.tensor([0, 1, 4, 5], dtype=torch.int32, device=inputs[5].device))
    inputs[6].copy_(torch.tensor([4, 0, 2], dtype=torch.int32, device=inputs[6].device))
    inputs[7].copy_(torch.tensor([2, 3, 1], dtype=torch.int32, device=inputs[7].device))
    expected = workload.ref_program(*inputs)
    graph.replay()
    compare_outputs((result, inputs[1]), expected, workload.verification(*inputs))
