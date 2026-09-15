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
        if isinstance(node, ast.ImportFrom) and node.level > 0
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
    reordering of either that forgets the other fails here.

    Grouping means one contiguous run per family, not merely a first appearance in the
    right order: `elementwise, dropout, elementwise` would satisfy the latter.
    """
    import tileops.ops as ops

    owner = {
        name: family for family in tileops._FAMILIES for name in _family_module(family).__all__
    }
    runs: list[str] = []
    for name in ops.__all__:
        family = owner.get(name)
        if family is not None and (not runs or runs[-1] != family):
            runs.append(family)
    assert len(runs) == len(set(runs)), f"a family appears in more than one run: {runs}"
    assert runs == [f for f in tileops._FAMILIES if f in runs]


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
    assuming `tileops.ops.<family>` holds all of them."""
    mod = _family_module(family)
    foreign = {
        name: getattr(mod, name).__module__
        for name in mod.__all__
        if not getattr(mod, name).__module__.startswith("tileops.ops.")
    }
    assert foreign == {}


@pytest.mark.smoke
def test_each_public_name_has_exactly_one_family():
    owners: dict[str, list[str]] = {}
    for family in FAMILIES:
        for name in _family_module(family).__all__:
            owners.setdefault(name, []).append(family)
    assert {name: fams for name, fams in owners.items() if len(fams) > 1} == {}


@pytest.mark.smoke
def test_public_surface_is_the_manifest():
    """An op reachable as `tileops.<family>.<Op>` has a manifest entry, and every entry
    is reachable. Abstract bases are not public at all."""
    public = {name for family in FAMILIES for name in _family_module(family).__all__}
    assert public == set(load_manifest())


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
    code = "import tileops; print(tileops.elementwise.HardtanhFwdOp.__name__)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "HardtanhFwdOp"


@pytest.mark.smoke
def test_importing_tileops_does_not_import_torch():
    """The manifest tooling reads YAML where torch is not installed, so resolving a
    family — not importing the package — has to be what pulls torch in. Checked in a
    subprocess because this one has torch imported already."""
    code = "import sys, tileops; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


@pytest.mark.smoke
def test_a_package_root_lists_every_name_it_resolves():
    """A lazy root states each name twice: in `_LAZY`, which resolves it, and under
    `if TYPE_CHECKING:`, which is all a type checker sees. One without the other is a
    name that resolves but cannot be checked, or the reverse."""
    for root in ("tileops.kernels", "tileops.ops"):
        mod = importlib.import_module(root)
        listed = _imported_names(Path(mod.__file__))
        assert listed == set(mod._LAZY) == set(mod.__all__), root


@pytest.mark.smoke
def test_the_op_base_class_does_not_import_the_compiler():
    """`Op` and `Kernel` describe ops; only a concrete kernel compiles one. Reaching the
    base classes must not import TileLang, which it does the moment either package root
    re-exports its subtree eagerly."""
    code = "import sys, tileops.ops.op_base; print('tilelang' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


@pytest.mark.smoke
def test_ops_aggregate_is_internally_consistent():
    """`tileops.ops` is the implementation path, not the public one. Its list is not
    required to cover the manifest; every name in it must resolve."""
    import tileops.ops as ops

    assert len(ops.__all__) == len(set(ops.__all__))
    assert [name for name in ops.__all__ if not hasattr(ops, name)] == []
