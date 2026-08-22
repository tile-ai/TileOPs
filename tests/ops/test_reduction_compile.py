"""Compile-boundary contract for the reduction family.

One case per op: a cold ``torch.compile(op, fullgraph=True)`` must match eager and the
traced graph must hold nothing but that op's own operator. Cold is the whole contract — a
warm op has nothing left for dynamo to trace into.

``dim`` is spelled out in every case. Left implicit, softmax resolves it by PyTorch's rule
and logsumexp reduces everything, and either choice keeps the output shape while changing
the values, so a case that does not say which axis it means proves little.
"""

import pytest
import torch

from tests.compile_contract import assert_op_owns_graph_nodes, register_compile_contract
from tileops.ops.reduction import (
    AllFwdOp,
    AmaxFwdOp,
    AminFwdOp,
    AnyFwdOp,
    ArgmaxFwdOp,
    ArgminFwdOp,
    CountNonzeroFwdOp,
    CumprodFwdOp,
    CumsumFwdOp,
    InfNormFwdOp,
    L1NormFwdOp,
    L2NormFwdOp,
    LogSoftmaxFwdOp,
    LogSumExpFwdOp,
    MeanFwdOp,
    ProdFwdOp,
    SoftmaxFwdOp,
    StdFwdOp,
    SumFwdOp,
    VarFwdOp,
    VarMeanFwdOp,
)

_DTYPE = torch.float16
_ROWS = 8
_COLS = 256

_OP_CLASSES = (
    SumFwdOp,
    MeanFwdOp,
    AminFwdOp,
    AmaxFwdOp,
    ProdFwdOp,
    StdFwdOp,
    VarFwdOp,
    VarMeanFwdOp,
    ArgmaxFwdOp,
    ArgminFwdOp,
    AllFwdOp,
    AnyFwdOp,
    CountNonzeroFwdOp,
    L1NormFwdOp,
    L2NormFwdOp,
    InfNormFwdOp,
    SoftmaxFwdOp,
    LogSoftmaxFwdOp,
    LogSumExpFwdOp,
    CumsumFwdOp,
    CumprodFwdOp,
)

for _op_cls in _OP_CLASSES:
    register_compile_contract(_op_cls)


def _x(*shape, dtype=_DTYPE):
    return torch.randn(*shape, dtype=dtype, device="cuda")


def _cases():
    """One builder per op. Built inside the test, not at import: this module is imported on
    the CPU-only runner that enforces the compile-contract gate."""
    rows = (_ROWS, _COLS)

    def one_tensor(op_cls, *args, **kwargs):
        return lambda: (op_cls(*args, **kwargs), (_x(*rows),))

    cases = {
        # A reduced last axis, then a reduced leading one: the second is the case whose
        # rows the kernel has to permute for.
        "sum": one_tensor(SumFwdOp, dim=-1),
        "sum-leading-axis": one_tensor(SumFwdOp, dim=0),
        "sum-keepdim": one_tensor(SumFwdOp, dim=-1, keepdim=True),
        "sum-full": one_tensor(SumFwdOp, dim=None),
        "mean": one_tensor(MeanFwdOp, dim=-1),
        "amin": one_tensor(AminFwdOp, dim=-1),
        "amax": one_tensor(AmaxFwdOp, dim=-1),
        "prod": one_tensor(ProdFwdOp, dim=-1),
        "std": one_tensor(StdFwdOp, dim=-1),
        "var": one_tensor(VarFwdOp, dim=-1),
        # Two outputs, so two names in the fake and a getitem per result in the graph.
        "var-mean": one_tensor(VarMeanFwdOp, dim=-1),
        "argmax": one_tensor(ArgmaxFwdOp, dim=-1),
        "argmin": one_tensor(ArgminFwdOp, dim=-1),
        # A non-contiguous reduced axis, which is the layout argreduce strides along
        # instead of transposing.
        "argmax-strided-axis": one_tensor(ArgmaxFwdOp, dim=0),
        # bool out, not same_as(x): the fake reads the manifest, not the input.
        "all": one_tensor(AllFwdOp, dim=-1),
        # dim=[] is a no-op, and a bool input makes the cast to bool one too, so the
        # result must still be a tensor of its own.
        "all-empty-dim-bool": lambda: (
            AllFwdOp(dim=[]),
            (torch.randint(2, (_ROWS, _COLS), dtype=torch.bool, device="cuda"),),
        ),
        "any": one_tensor(AnyFwdOp, dim=-1),
        # int64 out.
        "count-nonzero": one_tensor(CountNonzeroFwdOp, dim=-1),
        "l1-norm": one_tensor(L1NormFwdOp, dim=-1),
        "l2-norm": one_tensor(L2NormFwdOp, dim=-1),
        "inf-norm": one_tensor(InfNormFwdOp, dim=-1),
        "softmax": one_tensor(SoftmaxFwdOp, dim=-1),
        # A same-shape result over a non-last axis comes back through a permute, so its
        # strides have to match what the fake promised.
        "softmax-leading-axis": one_tensor(SoftmaxFwdOp, dim=0),
        "log-softmax": one_tensor(LogSoftmaxFwdOp, dim=-1),
        "logsumexp": one_tensor(LogSumExpFwdOp, dim=-1),
        "cumsum": one_tensor(CumsumFwdOp, dim=-1),
        "cumsum-leading-axis": one_tensor(CumsumFwdOp, dim=0),
        "cumprod": one_tensor(CumprodFwdOp, dim=-1),
    }
    return [pytest.param(builder, id=name) for name, builder in cases.items()]


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
@pytest.mark.parametrize("build_case", _cases())
def test_a_cold_op_traces_fullgraph_and_matches_eager(build_case) -> None:
    """Cold is the whole contract: a warm op has nothing left for dynamo to trace into."""
    op, inputs = build_case()

    compiled = torch.compile(op, fullgraph=True)(*inputs)

    torch.testing.assert_close(compiled, op(*inputs))


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
@pytest.mark.parametrize("build_case", _cases())
def test_the_traced_graph_holds_only_this_ops_operator(build_case) -> None:
    """The node is the op's, so replacing the kernel cannot change the graph."""
    op, inputs = build_case()

    assert_op_owns_graph_nodes(op, *inputs)
