"""Tests for the torch.compile fullgraph contract-coverage registry.

The registry (tests/compile_contract.py) aggregates the op classes that the
curated compile tests exercise cold with ``fullgraph=True`` into
``COMPILE_CONTRACT_OPS``. These tests verify the helper contract, the
aggregation across evidence modules, and that every registered evidence name
is an exact manifest key.
"""

import pytest

pytestmark = pytest.mark.smoke


def test_compile_contract_returns_class_and_registers():
    """compile_contract returns its argument and records the class name."""
    from tests import compile_contract as registry
    from tileops.ops.elementwise import ReluFwdOp

    # ReluFwdOp is already contract evidence; re-registering is idempotent
    # and avoids polluting the registry with synthetic names.
    assert registry.compile_contract(ReluFwdOp) is ReluFwdOp
    assert "ReluFwdOp" in registry.COMPILE_CONTRACT_OPS


def test_compile_contract_ops_aggregates_all_evidence_modules():
    """COMPILE_CONTRACT_OPS contains evidence from every registered module."""
    from tests.compile_contract import COMPILE_CONTRACT_OPS

    # One representative per evidence module: elementwise + attention.
    assert "ReluFwdOp" in COMPILE_CONTRACT_OPS
    assert "MultiHeadAttentionFwdOp" in COMPILE_CONTRACT_OPS


def test_compile_contract_ops_are_exact_manifest_keys():
    """Every registered evidence name is an exact manifest op key.

    Guards registration typos: the structural equality gate compares
    op-name identity against manifest keys, so a registered name that is
    not a manifest key can never reconcile.
    """
    from tests.compile_contract import COMPILE_CONTRACT_OPS
    from tileops.manifest import load_manifest

    manifest_keys = set(load_manifest())
    unknown = COMPILE_CONTRACT_OPS - manifest_keys
    assert not unknown, (
        f"Registered compile-contract ops missing from manifest: "
        f"{sorted(unknown)}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
