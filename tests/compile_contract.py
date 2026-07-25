"""Contract-coverage registry for torch.compile fullgraph evidence.

Op classes exercised cold with ``torch.compile(op, fullgraph=True)`` by the
curated compile tests are registered here at evidence-module import time —
parametrized case tables register their ``op_cls`` entries directly, direct
tests call :func:`compile_contract` next to the test they back. The
aggregate ``COMPILE_CONTRACT_OPS`` is the registered evidence set the
manifest's ``torch_compile_fullgraph`` declarations must mirror.

Exploratory or regression compile tests that do not back the fullgraph
contract must not register here.
"""

import importlib

# Modules whose import populates the registry. Add a module here when it
# gains contract-backing compile tests.
_EVIDENCE_MODULES = (
    "tests.ops.test_elementwise_compile",
    "tests.test_compile",
)

_registered: set[str] = set()


def compile_contract(op_cls: type) -> type:
    """Register ``op_cls`` as fullgraph compile-contract evidence.

    Call at module import, adjacent to the compile test that backs the
    promise. Returns ``op_cls`` unchanged so call sites stay expression-
    friendly.
    """
    _registered.add(op_cls.__name__)
    return op_cls


def compile_contract_ops() -> frozenset[str]:
    """Aggregate registered evidence from all evidence modules."""
    for module in _EVIDENCE_MODULES:
        importlib.import_module(module)
    return frozenset(_registered)


def __getattr__(name: str):
    # COMPILE_CONTRACT_OPS is computed lazily so importing this module from
    # an evidence module does not recurse into the evidence imports.
    if name == "COMPILE_CONTRACT_OPS":
        return compile_contract_ops()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
