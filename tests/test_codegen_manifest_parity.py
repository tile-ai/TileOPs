"""The generated validator takes the call's arguments, workspaces included.

``_validate_dtypes`` is synthesised from the manifest, so a change to how the
call's argument list is derived reaches the wheel without moving a single
validator diagnostic. One synthetic entry pins the shape.
"""

import inspect
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _manifest_facts as F  # noqa: E402

from tileops.manifest import forward_signature  # noqa: E402
from tileops.ops._dtype_codegen import synthesize_validate_dtypes  # noqa: E402

_ENTRY = {
    "status": "implemented",
    "signature": {
        "inputs": {"x": {"dtype": "float16 | bfloat16"}},
        "outputs": {"y": {"dtype": "same_as(x)"}},
        "dtype_combos": [{"x": "float16"}, {"x": "bfloat16"}],
    },
    "resources": {"workspaces": [{"name": "ws", "dtype": "float16 | bfloat16"}]},
}


def test_generated_validator_takes_the_workspace():
    """Omitting it would reject every call that passes one."""
    fn = synthesize_validate_dtypes("Op", forward_signature(_ENTRY))
    assert [p for p in inspect.signature(fn).parameters if p != "self"] == ["x", "ws"]


def test_a_combo_row_carries_no_workspace_column():
    """A row states what a caller may pass; a workspace's dtype is strategy."""
    assert F.build("Op", _ENTRY).combo_columns == ("x",)
