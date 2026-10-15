"""Unit tests for ``scripts/lint/module_constant_lint.py``.

One case per form the spelling rule has to tell apart: a value written out in the source,
which is a constant and is spelled ``UPPER_SNAKE``, against everything that only looks like
one. Then the reader rule: who reads a constant decides whether it is one.
"""

import subprocess
from pathlib import Path

import module_constant_lint as lint
import pytest


def _findings(tmp_path: Path, source: str) -> list[str]:
    path = tmp_path / "probe.py"
    path.write_text(source, encoding="utf-8")
    return lint.findings(path)


FLAGGED = [
    pytest.param("block_size = 128\n", id="literal"),
    pytest.param("offset = -1\n", id="negative-literal"),
    pytest.param("_pc = {tilelang.PassConfigKey.X: True}\n", id="dict-keyed-by-a-dotted-name"),
    pytest.param('_cf = ["-O3", "-DNDEBUG"]\n', id="list-of-literals"),
    pytest.param("if TYPE_CHECKING:\n    block_size = 128\n", id="inside-if"),
    pytest.param("try:\n    header_words = 2\nexcept Exception:\n    pass\n", id="inside-try"),
    pytest.param("[block_m, block_n] = [64, 128]\n", id="destructured-list"),
    pytest.param('name = "tileops." + "params"\n', id="concatenated-literals"),
    pytest.param("for _ in (1,):\n    size = 128\n", id="inside-for"),
    pytest.param('attribute = f"tileops.params"\n', id="f-string-without-a-field"),
    pytest.param("block_sizes = (*(64, 128), 256)\n", id="unpacked-literal-sequence"),
    pytest.param("__pass_configs = {tilelang.PassConfigKey.X: True}\n", id="leading-dunder-only"),
    pytest.param("eps: float = 1e-6\n", id="annotated"),
]

ACCEPTED = [
    pytest.param("BLOCK_SIZE = 128\n", id="upper-snake"),
    pytest.param("__all__ = ['a']\n", id="dunder"),
    pytest.param("logger = logging.getLogger(__name__)\n", id="call"),
    pytest.param("device = torch.empty(0).device\n", id="attribute-of-a-call"),
    pytest.param("Tensor: TypeAlias = torch.Tensor\n", id="type-alias"),
    pytest.param("Tensor: typing.TypeAlias = torch.Tensor\n", id="qualified-type-alias"),
    pytest.param("counter = 0\ncounter += 1\n", id="augmented-assignment"),
    pytest.param("block_size = 128\nblock_size = 64\n", id="rebound"),
    pytest.param("size = 128\nsize, other = load_sizes()\n", id="rebound-by-destructuring"),
    pytest.param("size = 128\nfor size in range(3):\n    pass\n", id="rebound-by-a-loop"),
    pytest.param("flag = False\n\n\ndef flag():\n    pass\n", id="shadowed-by-a-def"),
    pytest.param("size = 128\nfrom x import size\n", id="shadowed-by-an-import"),
    pytest.param(
        "flag = False\n\n\ndef set_it():\n    global flag\n    flag = True\n", id="global"
    ),
    pytest.param("sizes = [x for x in (1, 2)]\n", id="comprehension"),
    pytest.param('name = f"{prefix}.params"\n', id="interpolated-f-string"),
    pytest.param("size = 128 if fast else 64\n", id="conditional"),
    pytest.param("class Holder:\n    field = 1\n", id="class-attribute"),
    pytest.param("def f():\n    local_size = 128\n", id="function-local"),
]


@pytest.mark.smoke
@pytest.mark.parametrize("source", FLAGGED)
def test_a_constant_that_is_not_upper_snake_is_reported(tmp_path: Path, source: str) -> None:
    assert _findings(tmp_path, source), source


@pytest.mark.smoke
@pytest.mark.parametrize("source", ACCEPTED)
def test_what_is_not_a_lowercase_constant_is_left_alone(tmp_path: Path, source: str) -> None:
    assert _findings(tmp_path, source) == [], source


@pytest.mark.smoke
def test_a_destructured_assignment_reports_every_name_it_binds(tmp_path: Path) -> None:
    found = _findings(tmp_path, "block_m, block_n = (64, 128)\n")

    assert sorted(f.split("module constant ")[1].split(" ")[0] for f in found) == [
        "block_m",
        "block_n",
    ]


@pytest.mark.smoke
def test_the_message_names_the_file_the_line_and_the_constant(tmp_path: Path) -> None:
    (found,) = _findings(tmp_path, "BLOCK_SIZE = 1\nblock_n = 128\n")

    assert found.endswith("probe.py:2: module constant block_n is not UPPER_SNAKE")


@pytest.mark.smoke
def test_the_tree_is_clean() -> None:
    assert lint.main([]) == 0


def _reader_findings(tmp_path: Path, source: str, added: set[int] | None = None) -> list[str]:
    path = tmp_path / "probe.py"
    path.write_text(source, encoding="utf-8")
    return lint.reader_findings(path, added, root=tmp_path)


ONE_READER = [
    pytest.param("LIMIT = 4\n\n\ndef f():\n    return LIMIT\n", id="public"),
    pytest.param("_LIMIT = 4\n\n\ndef f():\n    return _LIMIT\n", id="private"),
    pytest.param("LIMIT = 4\n\n\ndef f():\n    return LIMIT + LIMIT\n", id="read-twice"),
    pytest.param(
        "LIMIT = 4\n\n\nclass f:\n    def a(self):\n        return LIMIT\n\n"
        "    def b(self):\n        return LIMIT\n",
        id="two-methods-of-one-class",
    ),
]


@pytest.mark.smoke
@pytest.mark.parametrize("source", ONE_READER)
def test_a_constant_with_one_reader_is_reported(tmp_path: Path, source: str) -> None:
    (found,) = _reader_findings(tmp_path, source)

    assert "has one reader, f — make it a local of f" in found


@pytest.mark.smoke
def test_a_constant_with_no_reader_is_reported(tmp_path: Path) -> None:
    (found,) = _reader_findings(tmp_path, "LIMIT = 4\n")

    assert found.endswith("probe.py:1: module constant LIMIT has no reader — delete it")


SHADOWS = [
    pytest.param("def g(LIMIT):\n    return LIMIT\n", id="parameter"),
    pytest.param("def g():\n    LIMIT = 8\n    return LIMIT\n", id="assignment"),
    pytest.param("def g():\n    for LIMIT in (8,):\n        return LIMIT\n", id="for-target"),
    pytest.param(
        "def g(manager):\n    with manager() as LIMIT:\n        return LIMIT\n", id="with-target"
    ),
    pytest.param("def g():\n    return [LIMIT for LIMIT in (8,)]\n", id="comprehension-target"),
    pytest.param("def g():\n    from other import LIMIT\n    return LIMIT\n", id="import"),
    pytest.param(
        "def g():\n    LIMIT = 8\n    def nested():\n        return LIMIT\n    return nested()\n",
        id="nested-closure",
    ),
]


@pytest.mark.smoke
@pytest.mark.parametrize("shadow", SHADOWS)
def test_a_shadowed_name_does_not_count_as_a_reader(tmp_path: Path, shadow: str) -> None:
    source = "LIMIT = 4\n\n\ndef f():\n    return LIMIT\n\n\n" + shadow

    (found,) = _reader_findings(tmp_path, source)

    assert "has one reader, f" in found


@pytest.mark.smoke
def test_a_comprehension_iterable_can_still_read_the_module_constant(tmp_path: Path) -> None:
    source = (
        "# f and g share the domain.\nLIMIT = (4,)\n\n\n"
        "def f():\n    return LIMIT\n\n\n"
        "def g():\n    return [LIMIT for LIMIT in LIMIT]\n"
    )

    assert _reader_findings(tmp_path, source) == []


# Two functions reading LIMIT, appended to every probe of the comment rule.
_SHARED = "\n\ndef f():\n    return LIMIT\n\n\ndef g():\n    return LIMIT\n"


@pytest.mark.smoke
def test_a_shared_constant_without_a_comment_is_reported(tmp_path: Path) -> None:
    (found,) = _reader_findings(tmp_path, "LIMIT = 4" + _SHARED)

    assert "module constant LIMIT is shared by 2 readers and no comment" in found


EXPLAINED = [
    pytest.param("# f and g must agree on it.\nLIMIT = 4", id="comment-above"),
    pytest.param("LIMIT = 4  # f and g must agree on it.", id="comment-beside"),
    pytest.param("# f and g agree on both.\nOTHER = 1\nLIMIT = 4", id="comment-above-the-run"),
]


@pytest.mark.smoke
@pytest.mark.parametrize("head", EXPLAINED)
def test_a_shared_constant_with_a_comment_is_left_alone(tmp_path: Path, head: str) -> None:
    found = _reader_findings(tmp_path, head + _SHARED + "\n\nOTHER_READER = OTHER\n")

    assert [f for f in found if " LIMIT " in f] == []


@pytest.mark.smoke
def test_an_exported_constant_is_left_alone(tmp_path: Path) -> None:
    assert _reader_findings(tmp_path, '__all__ = ["LIMIT"]\nLIMIT = 4\n') == []


@pytest.mark.smoke
def test_only_a_constant_bound_on_an_added_line_is_judged(tmp_path: Path) -> None:
    assert _reader_findings(tmp_path, "LIMIT = 4\nOTHER = 2\n", added={2}) == [
        "probe.py:2: module constant OTHER has no reader — delete it"
    ]


def _package(tmp_path: Path, other: str, name: str = "b.py") -> Path:
    """A git repository holding ``src/pkg/a.py``, whose ``f`` reads LIMIT, and *other*."""
    package = tmp_path / "src" / "pkg"
    package.mkdir(parents=True)
    (package / "a.py").write_text("LIMIT = 4\n\n\ndef f():\n    return LIMIT\n")
    (package / name).write_text(other)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    return package / "a.py"


OTHER_MODULE_READERS = [
    pytest.param("from pkg.a import LIMIT\n\n\ndef g():\n    return LIMIT\n", id="imported-name"),
    pytest.param("import pkg.a\n\n\ndef g():\n    return pkg.a.LIMIT\n", id="module-attribute"),
    pytest.param("from pkg import a\n\n\ndef g():\n    return a.LIMIT\n", id="imported-module"),
]


@pytest.mark.smoke
@pytest.mark.parametrize("other", OTHER_MODULE_READERS)
def test_a_reader_in_another_module_counts(tmp_path: Path, other: str) -> None:
    (found,) = lint.reader_findings(_package(tmp_path, other), None, root=tmp_path)

    assert "module constant LIMIT is shared by 2 readers" in found


@pytest.mark.smoke
def test_a_same_named_constant_of_another_module_does_not_count(tmp_path: Path) -> None:
    other = "LIMIT = 8\n\n\ndef g():\n    return LIMIT\n"

    (found,) = lint.reader_findings(_package(tmp_path, other), None, root=tmp_path)

    assert "has one reader, f" in found


@pytest.mark.smoke
def test_a_constant_another_module_exports_is_left_alone(tmp_path: Path) -> None:
    init = 'from pkg.a import LIMIT\n\n__all__ = ["LIMIT"]\n'

    assert lint.reader_findings(_package(tmp_path, init, "__init__.py"), None, root=tmp_path) == []
