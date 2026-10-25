"""Guard the public op surface: `tileops.<family>.<Op>`, two levels, nothing deeper.

The family modules re-export from `tileops._ops`, so nothing here checks behaviour —
only that the surface a caller imports from is complete, matches the manifest, and
stays free of torch until an op is actually reached for.
"""

import ast
import dataclasses
import importlib
from pathlib import Path

import pytest

import tileops
from tests.fresh_interpreter import probe
from tileops.manifest import load_manifest

SRC = Path(tileops.__file__).parent
# Off the tree, not off `_FAMILIES` — one test compares the two.
FAMILIES = sorted(p.stem for p in SRC.glob("*.py") if not p.stem.startswith("_"))


def _family_module(family: str):
    return importlib.import_module(f"tileops.{family}")


def _imported_names(path: Path) -> set[str]:
    """Every name a module binds from its own package.

    A lazy package root binds its names through ``__getattr__`` and lists them under
    ``if TYPE_CHECKING:`` for type checkers, so that block counts as an import.
    """
    body = ast.parse(path.read_text()).body
    guarded = [
        inner
        for node in body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "TYPE_CHECKING"
        for inner in node.body
    ]
    return {
        alias.asname or alias.name
        for node in body + guarded
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("tileops.")
        for alias in node.names
    }


@pytest.mark.smoke
def test_families_match_the_tree():
    assert sorted(tileops._FAMILIES) == FAMILIES
    assert len(set(tileops._FAMILIES)) == len(tileops._FAMILIES)


@pytest.mark.smoke
@pytest.mark.parametrize("family", FAMILIES)
def test_family_all_covers_every_import(family: str):
    mod = _family_module(family)
    assert _imported_names(Path(mod.__file__)) == set(mod.__all__)
    assert len(mod.__all__) == len(set(mod.__all__))
    assert [name for name in mod.__all__ if not hasattr(mod, name)] == []


@pytest.mark.smoke
@pytest.mark.parametrize("family", FAMILIES)
def test_family_defines_nothing(family: str):
    """A family module re-exports; it never defines. A family can draw on more than one
    implementation module, so this checks where each object came from rather than
    assuming `tileops._ops.<family>` holds all of them."""
    mod = _family_module(family)
    foreign = {
        name: getattr(mod, name).__module__
        for name in mod.__all__
        if not getattr(mod, name).__module__.startswith("tileops._ops.")
    }
    assert foreign == {}


@pytest.mark.smoke
def test_each_public_name_has_exactly_one_family():
    owners: dict[str, list[str]] = {}
    for family in FAMILIES:
        for name in _family_module(family).__all__:
            owners.setdefault(name, []).append(family)
    assert {name: fams for name, fams in owners.items() if len(fams) > 1} == {}


def _public_objects() -> dict[str, object]:
    return {
        name: getattr(_family_module(family), name)
        for family in FAMILIES
        for name in _family_module(family).__all__
    }


@pytest.mark.smoke
def test_public_surface_is_the_manifest():
    """An op reachable as `tileops.<family>.<Op>` has a manifest entry, and every
    implemented entry is reachable; a spec-only entry may have no class yet. Abstract
    bases are not public at all."""
    from tileops._ops.op_base import Op

    public = {
        name
        for name, obj in _public_objects().items()
        if isinstance(obj, type) and issubclass(obj, Op)
    }
    manifest = load_manifest()
    assert public <= set(manifest)
    assert {name for name, entry in manifest.items() if entry["status"] == "implemented"} <= public


@pytest.mark.smoke
def test_every_other_public_name_is_a_frozen_value_type():
    """Such as `tileops.moe.MaskedLayoutSpec`; an op that holds one never sees it change."""
    from tileops._ops.op_base import Op

    assert [
        name
        for name, obj in _public_objects().items()
        if not (isinstance(obj, type) and issubclass(obj, Op))
        and not (dataclasses.is_dataclass(obj) and obj.__dataclass_params__.frozen)
    ] == []


@pytest.mark.smoke
def test_families_are_listed():
    assert set(FAMILIES) <= set(tileops.__all__)
    assert set(FAMILIES) <= set(dir(tileops))


@pytest.mark.smoke
def test_a_family_resolves_from_a_bare_import():
    """`import tileops` then `tileops.<family>` has to work through `__getattr__`.

    In a subprocess because the other tests in this file have already imported every
    family module, which binds the attribute on the parent package: an in-process
    assertion would then hold whether `__getattr__` works or not.
    """
    assert probe("import tileops", "tileops.elementwise.HardtanhFwdOp.__name__") == "HardtanhFwdOp"


@pytest.mark.smoke
def test_importing_tileops_does_not_import_torch():
    """The manifest tooling reads YAML where torch is not installed, so resolving a
    family — not importing the package — has to be what pulls torch in. Checked in a
    subprocess because this one has torch imported already."""
    assert probe("import sys, tileops", "'torch' in sys.modules") == "False"


@pytest.mark.smoke
def test_the_kernels_root_lists_every_name_it_resolves():
    """The lazy root states each name twice: in `_LAZY`, which resolves it, and under
    `if TYPE_CHECKING:`, which is all a type checker sees. One without the other is a
    name that resolves but cannot be checked, or the reverse."""
    mod = importlib.import_module("tileops._kernels")
    listed = _imported_names(Path(mod.__file__))
    assert listed == set(mod._LAZY) == set(mod.__all__)


@pytest.mark.smoke
def test_the_op_base_class_does_not_import_the_compiler():
    """`Op` and `Kernel` describe ops; only a concrete kernel compiles one. Reaching the
    base classes must not import TileLang, which it does the moment either package root
    re-exports its subtree eagerly."""
    assert probe("import sys, tileops._ops.op_base", "'tilelang' in sys.modules") == "False"
