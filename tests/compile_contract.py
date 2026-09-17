"""Contract-coverage registry for torch.compile fullgraph evidence.

Op classes exercised cold with ``torch.compile(op, fullgraph=True)`` by the
curated compile tests are registered here at evidence-module import time —
parametrized case tables register their ``op_cls`` entries directly, direct
tests call :func:`register_compile_contract` next to the test they back.
:func:`compile_contract_ops` aggregates the registered evidence set the
manifest's ``torch_compile_fullgraph`` declarations must mirror.

Exploratory or regression compile tests that do not back the fullgraph
contract must not register here.
"""

import importlib
import operator

import torch

# Modules whose import populates the registry. Add a module here when it
# gains contract-backing compile tests.
_EVIDENCE_MODULES = (
    "tests.ops.test_compile_boundary",
    "tests.ops.test_convolution",
    "tests.ops.test_elementwise_compile",
    "tests.ops.test_moe_compile",
    "tests.ops.test_norm_compile",
    "tests.ops.test_pool",
    "tests.ops.test_reduction_compile",
    "tests.ops.test_rms_norm",
)

_registered: set[str] = set()


def register_compile_contract(op_cls: type) -> None:
    """Register ``op_cls`` as fullgraph compile-contract evidence.

    Call at module import, adjacent to the compile test that backs the
    promise. Side-effect only.

    Raises:
        AssertionError: The op names no operators, so nothing can be asserted about
            the graph it traces to.
    """
    assert op_cls.compile_op_names, (
        f"{op_cls.__name__} names no operators in compile_op_names, so a traced graph "
        f"cannot be checked against them"
    )
    _registered.add(op_cls.__name__)


def operator_overload(name: str):
    """``"tileops::foo"`` as the overload object a graph node carries."""
    namespace, _, opname = name.partition("::")
    return getattr(getattr(torch.ops, namespace), opname).default


def traced_call_targets(op, *inputs, **kwargs) -> set:
    """Compile *op* with ``fullgraph=True`` and return the operators its graph calls.

    ``operator.getitem`` is left out: it is how a multi-output operator's results are
    unpacked, carries no computation, and no target supplies it.
    """
    traced: list[list[object]] = []

    def capture(gm, example_inputs):
        traced.append(
            [
                node.target
                for node in gm.graph.nodes
                if node.op == "call_function" and node.target is not operator.getitem
            ]
        )
        return gm.forward

    torch.compile(op, backend=capture, fullgraph=True)(*inputs, **kwargs)

    (calls,) = traced
    return set(calls)


def assert_op_owns_graph_nodes(op, *inputs, **kwargs) -> None:
    """Compile *op* once and assert the graph holds nothing but its own operators.

    A kernel's own registration, or a tensor op left outside the boundary, would show up
    here — either means the node identity is not the op's alone, so another target could
    change the graph.
    """
    declared = {operator_overload(name) for name in type(op).compile_op_names}
    assert declared, f"{type(op).__name__} declares no compile_op_names"

    calls = traced_call_targets(op, *inputs, **kwargs)
    assert calls, "the traced graph called nothing"
    assert calls <= declared, (
        f"graph holds nodes this op does not own: "
        f"{sorted(str(c) for c in calls - declared)}; "
        f"declared: {sorted(str(d) for d in declared)}"
    )


def assert_same_result(compiled, eager, *, exact: bool) -> None:
    """Hold a compiled call to an eager one, or to its shapes and dtypes alone.

    *exact* is false where equality is not a property the kernel has; the caller names
    which cases those are.
    """
    compiled = compiled if isinstance(compiled, tuple) else (compiled,)
    eager = eager if isinstance(eager, tuple) else (eager,)
    for got, want in zip(compiled, eager, strict=True):
        assert got.shape == want.shape and got.dtype == want.dtype
        if exact:
            torch.testing.assert_close(got, want)


def assert_fake_matches_eager(op, *inputs, **kwargs) -> None:
    """Hold the fake's output metadata to what the op actually returns.

    The fake is all the compiler learns about this op's node, and a single-node graph that
    returns directly never forces its answer to be checked against anything.
    """
    import torch._subclasses.fake_tensor as fake_tensor

    real = op(*inputs, **kwargs)
    real = real if isinstance(real, tuple) else (real,)

    mode = fake_tensor.FakeTensorMode()
    with mode:
        faked = op(
            *(None if t is None else mode.from_tensor(t) for t in inputs),
            **{k: (None if v is None else mode.from_tensor(v)) for k, v in kwargs.items()},
        )
    faked = faked if isinstance(faked, tuple) else (faked,)

    assert len(faked) == len(real), (
        f"{type(op).__name__}: the fake returns {len(faked)} tensors, the op {len(real)}"
    )
    for i, (claimed, actual) in enumerate(zip(faked, real, strict=True)):
        assert claimed.dtype == actual.dtype, (
            f"{type(op).__name__} output {i}: the fake reports {claimed.dtype}, the op "
            f"returns {actual.dtype}"
        )
        assert tuple(claimed.shape) == tuple(actual.shape), (
            f"{type(op).__name__} output {i}: the fake reports shape "
            f"{tuple(claimed.shape)}, the op returns {tuple(actual.shape)}"
        )


def compile_contract_ops() -> frozenset[str]:
    """Aggregate registered evidence from all evidence modules.

    Evidence modules are imported lazily here (not at module top) so they
    can import this module without recursion.
    """
    for module in _EVIDENCE_MODULES:
        importlib.import_module(module)
    return frozenset(_registered)
