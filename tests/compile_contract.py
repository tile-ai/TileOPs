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

# Modules whose import populates the registry. Add a module here when it
# gains contract-backing compile tests.
_EVIDENCE_MODULES = (
    "tests.ops.test_elementwise_compile",
    "tests.ops.test_pool",
    "tests.test_compile",
)

_registered: set[str] = set()


def register_compile_contract(op_cls: type) -> None:
    """Register ``op_cls`` as fullgraph compile-contract evidence.

    Call at module import, adjacent to the compile test that backs the
    promise. Side-effect only.
    """
    _registered.add(op_cls.__name__)


def compile_contract_ops() -> frozenset[str]:
    """Aggregate registered evidence from all evidence modules.

    Evidence modules are imported lazily here (not at module top) so they
    can import this module without recursion.
    """
    for module in _EVIDENCE_MODULES:
        importlib.import_module(module)
    return frozenset(_registered)
