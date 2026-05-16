"""Critical-path check for manifest-driven ``_validate_dtypes`` codegen.

Guards that every ``status: implemented`` op in the live manifest has a
non-stub ``_validate_dtypes`` installed by the ``__init_subclass__``
hook. Per-op dtype-rejection behavior lives in each op's own test file.
"""

import pytest

from tileops.ops.op_base import Op

pytestmark = pytest.mark.smoke


class TestRealManifestParity:
    def test_no_stubs_for_implemented_ops(self):
        from tileops.manifest import load_manifest
        ops = load_manifest()
        import tileops.ops  # noqa: F401
        stubs = []
        for op_name, entry in ops.items():
            if entry.get("status") != "implemented":
                continue
            cls = _find_op_class(op_name)
            if cls is None:
                continue
            if cls._validate_dtypes is Op._validate_dtypes:
                stubs.append(op_name)
        assert stubs == [], (
            f"{len(stubs)} implemented ops still have the base "
            f"_validate_dtypes stub: {stubs[:5]}..."
        )


def _find_op_class(op_name):
    seen = set()
    stack = list(Op.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        if cls.__name__ == op_name:
            return cls
        stack.extend(cls.__subclasses__())
    return None
