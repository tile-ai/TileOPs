"""Unit tests for ``scripts/lint/csrc_names_lint.py``.

One case per way a csrc header can define a name outside ``namespace tileops`` or spell it
with a prefix the namespace replaces, against the layouts a conforming header uses.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LINT_SCRIPT = REPO_ROOT / "scripts" / "lint" / "csrc_names_lint.py"

_spec = importlib.util.spec_from_file_location("csrc_names_lint", LINT_SCRIPT)
lint = importlib.util.module_from_spec(_spec)
sys.modules["csrc_names_lint"] = lint
_spec.loader.exec_module(lint)


def _findings(tmp_path: Path, source: str) -> list[str]:
    path = tmp_path / "probe.h"
    path.write_text(source, encoding="utf-8")
    return lint.findings(path)


FLAGGED = [
    pytest.param(
        "namespace tileops {\nnamespace tl {\ninline int f() { return 0; }\n}\n}\n",
        id="namespace-tl-nested",
    ),
    pytest.param(
        "namespace tileops {\ninline int tileops_f() { return 0; }\n}\n", id="tileops-prefix"
    ),
    pytest.param("namespace tileops {\ninline int __tl_f() { return 0; }\n}\n", id="tl-prefix"),
    pytest.param("#define HELPER(N) N\nnamespace tileops {}\n", id="unprefixed-macro"),
    pytest.param("namespace tileops {}\nconstexpr int k = 1;\n", id="file-scope-variable"),
    pytest.param(
        "#define TILEOPS_HELPER inline int tileops_f() { return 0; }\nnamespace tileops {}\n",
        id="prefixed-name-in-macro-body",
    ),
]

ACCEPTED = [
    pytest.param(
        "namespace tileops {\nnamespace detail {\nstruct S { int x; };\n}\n}\n",
        id="nested-namespace",
    ),
    pytest.param(
        "#define TILEOPS_HELPER(N) \\\n  inline int f##N() { return N; }\n"
        "namespace tileops {\nTILEOPS_HELPER(1)\n}\n#undef TILEOPS_HELPER\n",
        id="prefixed-macro",
    ),
    pytest.param(
        "namespace tileops {\n// tileops_old_name and namespace tl in a comment\n"
        'inline const char* f() { return "tileops_x"; }\n}\n',
        id="comment-and-string",
    ),
]


@pytest.mark.smoke
@pytest.mark.parametrize("source", FLAGGED)
def test_a_name_outside_the_style_is_reported(tmp_path: Path, source: str) -> None:
    assert _findings(tmp_path, source), source


@pytest.mark.smoke
@pytest.mark.parametrize("source", ACCEPTED)
def test_a_conforming_header_is_left_alone(tmp_path: Path, source: str) -> None:
    assert _findings(tmp_path, source) == [], source


@pytest.mark.smoke
def test_the_message_names_the_file_and_the_line(tmp_path: Path) -> None:
    (found,) = _findings(tmp_path, "namespace tileops {}\n\nconstexpr int k = 1;\n")

    assert found.endswith("probe.h:3: declaration outside namespace tileops")


@pytest.mark.smoke
def test_the_tree_is_clean() -> None:
    assert lint.main([]) == 0
