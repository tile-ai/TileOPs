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

import torch

# Modules whose import populates the registry. Add a module here when it
# gains contract-backing compile tests.
_EVIDENCE_MODULES = (
    "tests.ops.test_elementwise_compile",
    "tests.ops.test_pool",
    "tests.ops.test_rms_norm",
    "tests.test_compile",
)

_registered: set[str] = set()


def register_compile_contract(op_cls: type) -> None:
    """Register ``op_cls`` as fullgraph compile-contract evidence.

    Call at module import, adjacent to the compile test that backs the
    promise. Side-effect only.
    """
    _registered.add(op_cls.__name__)


def assert_op_owns_graph_nodes(op, *inputs) -> None:
    """Compile *op* once and assert the graph holds nothing but its own operators.

    What the graph contains is what has to stay the same when another target serves this
    op. The op says which operators are its own through ``compile_op_names``; anything
    else in the graph — a kernel's own registration, or a tensor op left outside the
    boundary — means the node identity is not the op's alone.

    Args:
        op: The op instance, already constructed.
        *inputs: Arguments to call it with.

    Raises:
        AssertionError: The op names no operators, or the graph holds a node that is not
            one of them.
    """
    declared = {name.replace("::", ".") + ".default" for name in op.compile_op_names}
    assert declared, f"{type(op).__name__} declares no compile_op_names"

    traced: list[list[str]] = []

    def capture(gm, example_inputs):
        traced.append([str(node.target) for node in gm.graph.nodes
                       if node.op == "call_function"])
        return gm.forward

    torch.compile(op, backend=capture, fullgraph=True)(*inputs)

    (calls,) = traced
    assert calls, "the traced graph called nothing"
    assert set(calls) <= declared, (
        f"graph holds nodes this op does not own: {sorted(set(calls) - declared)}; "
        f"declared: {sorted(declared)}")


def compile_contract_ops() -> frozenset[str]:
    """Aggregate registered evidence from all evidence modules.

    Evidence modules are imported lazily here (not at module top) so they
    can import this module without recursion.
    """
    for module in _EVIDENCE_MODULES:
        importlib.import_module(module)
    return frozenset(_registered)
