"""Guard the public op surface: `tileops.<family>.<Op>`, two levels, nothing deeper.

The family modules re-export from `tileops.ops`, so nothing here checks behaviour —
only that the surface a caller imports from is complete, matches the manifest, and
stays free of torch until an op is actually reached for.
"""

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import tileops
from tileops.manifest import load_manifest

SRC = Path(tileops.__file__).parent
# Read off the tree, so the tests are driven by what ships rather than by the
# list in `tileops/__init__.py` — one test compares the two.
FAMILIES = sorted(p.stem for p in SRC.glob("*.py") if not p.stem.startswith("_"))

# Public names the manifest does not declare. An addition here needs a reason, so
# the public surface cannot grow past the spec unnoticed.
NOT_IN_MANIFEST = {
    # Abstract bases a caller subclasses to add an op of their own. The API
    # reference documents these as template base classes.
    "UnaryOp",
    "BinaryOp",
    "FusedGatedOp",
    "CumulativeOp",
    # Marked `UnmanifestedOp` in the tree, and public before the family modules
    # existed. Each needs either a manifest entry or a deprecation; neither is this
    # change's business, and dropping them silently would break callers. The API
    # reference documents all but `MeanPoolingForwardOp`, which is in-tree only
    # until it has a manifest entry.
    "DeltaNetOp",
    "GatedDeltaNetOp",
    "GroupedQueryAttentionPrefillVarlenFwdOp",
    "MeanPoolingForwardOp",
    "NSACmpFwdVarlenOp",
    "NSAFwdVarlenOp",
    "NSATopkVarlenOp",
}


def _family_module(family: str):
    return importlib.import_module(f"tileops.{family}")


def _imported_names(path: Path) -> set[str]:
    """Every name a module binds through an import."""
    return {
        alias.asname or alias.name
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }


@pytest.mark.smoke
def test_families_match_the_tree():
    assert sorted(tileops._FAMILIES) == FAMILIES
    assert len(set(tileops._FAMILIES)) == len(tileops._FAMILIES)


@pytest.mark.smoke
def test_family_order_follows_the_ops_aggregate():
    """`_FAMILIES` is ordered simple to composite, and the docs nav follows it. The
    order `tileops.ops.__all__` groups its names in is the same statement, so a
    reordering of either that forgets the other fails here."""
    import tileops.ops as ops

    owner = {
        name: family for family in tileops._FAMILIES for name in _family_module(family).__all__
    }
    seen: list[str] = []
    for name in ops.__all__:
        family = owner.get(name)
        if family is not None and family not in seen:
            seen.append(family)
    assert seen == [f for f in tileops._FAMILIES if f in seen]


@pytest.mark.smoke
@pytest.mark.parametrize("family", FAMILIES)
def test_family_all_covers_every_import(family: str):
    mod = _family_module(family)
    assert _imported_names(Path(mod.__file__)) == set(mod.__all__)
    assert len(mod.__all__) == len(set(mod.__all__))
    assert [name for name in mod.__all__ if not hasattr(mod, name)] == []


@pytest.mark.smoke
@pytest.mark.parametrize("family", FAMILIES)
def test_family_reexports_the_same_objects(family: str):
    mod = _family_module(family)
    impl = importlib.import_module(f"tileops.ops.{family}")
    assert all(getattr(mod, name) is getattr(impl, name) for name in mod.__all__)


@pytest.mark.smoke
def test_each_public_name_has_exactly_one_family():
    owners: dict[str, list[str]] = {}
    for family in FAMILIES:
        for name in _family_module(family).__all__:
            owners.setdefault(name, []).append(family)
    assert {name: fams for name, fams in owners.items() if len(fams) > 1} == {}


@pytest.mark.smoke
def test_public_surface_is_the_manifest_plus_the_listed_exceptions():
    public = {name for family in FAMILIES for name in _family_module(family).__all__}
    assert public == set(load_manifest()) | NOT_IN_MANIFEST


@pytest.mark.smoke
def test_families_are_reachable_as_attributes():
    assert set(FAMILIES) <= set(tileops.__all__)
    assert set(FAMILIES) <= set(dir(tileops))
    assert tileops.elementwise.HardtanhFwdOp is not None


@pytest.mark.smoke
def test_importing_tileops_does_not_import_torch():
    """The manifest tooling reads YAML where torch is not installed, so resolving a
    family — not importing the package — has to be what pulls torch in. Checked in a
    subprocess because this one has torch imported already."""
    code = "import sys, tileops; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


@pytest.mark.smoke
def test_ops_aggregate_is_internally_consistent():
    """`tileops.ops` is the implementation path, not the public one. Its list is not
    required to cover the manifest; every name in it must resolve."""
    import tileops.ops as ops

    assert _imported_names(Path(ops.__file__)) == set(ops.__all__)
    assert len(ops.__all__) == len(set(ops.__all__))
    assert [name for name in ops.__all__ if not hasattr(ops, name)] == []
