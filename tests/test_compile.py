# This test validates the compatibility of TileOps operators with torch.compile().
# Check: https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html

import pytest
import torch

from tests.compile_contract import register_compile_contract
from tests.ops.attention.test_mha import MhaFwdTest
from tests.test_base import FixtureBase
from tileops.ops import MultiHeadAttentionFwdOp

register_compile_contract(MultiHeadAttentionFwdOp)


class MhaCompileFixture(FixtureBase):
    PARAMS = [
        ("B, S, H, D, causal, dtype", [
            (8, 1024, 32, 128, False, torch.float16),
            (4, 512, 16, 64, True, torch.bfloat16),
        ]),
    ]


@pytest.mark.full
@pytest.mark.usefixtures("isolated_dynamo")
@MhaCompileFixture
def test_mha_kernel_compile(B: int, S: int, H: int, D: int, causal: bool, dtype: torch.dtype):
    test = MhaFwdTest(B, H, S, D, causal, dtype)
    op = MultiHeadAttentionFwdOp(B, H, S, D, causal, dtype)
    compiled_op = torch.compile(op, fullgraph=True)
    inputs = test.gen_inputs()
    test.check(compiled_op, *inputs, atol=5e-3, rtol=1e-5)
    print('Successfully validate the compatibility with torch.compile().')


@pytest.mark.smoke
class TestCompileContractRegistry:
    """Structural guards for the torch.compile contract-coverage registry.

    The registry (tests/compile_contract.py) aggregates the op classes the
    curated compile tests exercise cold with ``fullgraph=True`` via
    ``compile_contract_ops()``. The strict declaration/evidence equality
    gate lands here once the manifest carries ``torch_compile_fullgraph``
    declarations.
    """

    def test_register_compile_contract_records_class_name(self):
        """register_compile_contract records the class name (side effect only)."""
        from tests import compile_contract as registry
        from tileops.ops.elementwise import ReluFwdOp

        # ReluFwdOp is already contract evidence; re-registering is
        # idempotent and avoids polluting the registry with synthetic names.
        assert registry.register_compile_contract(ReluFwdOp) is None
        assert "ReluFwdOp" in registry.compile_contract_ops()

    def test_compile_contract_ops_aggregates_all_evidence_modules(self):
        """compile_contract_ops() covers evidence from every registered module."""
        from tests.compile_contract import compile_contract_ops

        ops = compile_contract_ops()
        assert isinstance(ops, frozenset)
        assert ops
        # One representative per evidence module: elementwise + attention.
        assert "ReluFwdOp" in ops
        assert "MultiHeadAttentionFwdOp" in ops
        # Stable across calls: the registry only grows at import time.
        assert compile_contract_ops() == ops

    def test_compile_contract_ops_are_exact_manifest_keys(self):
        """Every registered evidence name is an exact manifest op key.

        Guards registration typos: the structural equality gate compares
        op-name identity against manifest keys, so a registered name that
        is not a manifest key can never reconcile.
        """
        from tests.compile_contract import compile_contract_ops
        from tileops.manifest import load_manifest

        manifest_keys = set(load_manifest())
        unknown = compile_contract_ops() - manifest_keys
        assert not unknown, (
            f"Registered compile-contract ops missing from manifest: "
            f"{sorted(unknown)}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
