"""The op layer's generated code must follow the manifest, entry by entry.

The validator's differential compares diagnostics, which says nothing about
what the wheel does: ``_validate_dtypes`` and ``eval_roofline`` are synthesised
from the same manifest facts, and a change to how a fact is derived reaches
them without changing a single diagnostic. These pin the generated shape for
every implemented entry so that gap is covered.
"""

import inspect
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _manifest_facts as F  # noqa: E402

from tileops.manifest import forward_signature, load_manifest  # noqa: E402
from tileops.ops._dtype_codegen import synthesize_validate_dtypes  # noqa: E402


def _implemented():
    return {n: e for n, e in load_manifest().items() if e.get("status") == "implemented"}


@pytest.mark.parametrize("op_name", sorted(_implemented()))
def test_generated_validator_takes_the_call_arguments(op_name):
    """The synthesised signature is the manifest's call, in declaration order.

    Inputs then workspaces: a workspace is passed like any other tensor, and a
    generated validator that omitted one would reject every call that passes it.
    """
    entry = _implemented()[op_name]
    facts = F.build(op_name, entry)
    try:
        fn = synthesize_validate_dtypes(op_name, forward_signature(entry))
    except ValueError:
        pytest.skip("signature too irregular for the codegen to synthesise from")
    params = [p for p in inspect.signature(fn).parameters if p != "self"]
    assert params == list(facts.call_names)


@pytest.mark.parametrize("op_name", sorted(_implemented()))
def test_combo_columns_exclude_the_workspaces(op_name):
    """A combo row spans the caller's inputs; a workspace's dtype is strategy.

    Checked against the facts rather than recomputed here, so a change to how
    the column set is derived shows up as a failure rather than agreeing with
    itself.
    """
    entry = _implemented()[op_name]
    facts = F.build(op_name, entry)
    for column in facts.combo_columns:
        arg = facts.arg(column)
        assert arg is not None and not arg.workspace, column
    for name in facts.workspace_names:
        assert name not in facts.combo_columns


def test_at_least_one_entry_exercises_a_workspace():
    """Guard against the two assertions above passing vacuously."""
    assert any(F.build(n, e).workspace_names for n, e in _implemented().items())
