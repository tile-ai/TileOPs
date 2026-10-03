"""Unit tests for ``scripts/lint/module_constant_lint.py``.

One case per form the rule has to tell apart: a value written out in the source, which is a
constant and is spelled ``UPPER_SNAKE``, against everything that only looks like one.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LINT_SCRIPT = REPO_ROOT / "scripts" / "lint" / "module_constant_lint.py"

_spec = importlib.util.spec_from_file_location("module_constant_lint", LINT_SCRIPT)
lint = importlib.util.module_from_spec(_spec)
sys.modules["module_constant_lint"] = lint
_spec.loader.exec_module(lint)


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
