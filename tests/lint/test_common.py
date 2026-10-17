"""Unit tests for ``scripts/lint/common.py`` and the allow marker every prose lint reads."""

from pathlib import Path

import common
import pytest
from comment_lines import allowed_lines

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("x = 1  # lint: narration allow: compatibility\n", id="marker"),
        pytest.param("x = 1  # LINT: narration ALLOW: compatibility\n", id="any-case"),
    ],
)
def test_a_marker_with_a_reason_exempts_its_line(source: str) -> None:
    assert allowed_lines(source, "narration") == {1}


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("x = 1  # lint: narration allow:\n", id="no-reason"),
        pytest.param("x = 1  # lint: code-ref allow: compatibility\n", id="another-rule"),
        pytest.param('x = "lint: narration allow: compatibility"\n', id="in-a-string"),
    ],
)
def test_a_marker_without_a_reason_for_another_rule_or_in_code_exempts_nothing(
    source: str,
) -> None:
    assert allowed_lines(source, "narration") == set()


def test_a_marker_exempts_only_its_own_line() -> None:
    assert allowed_lines("# lint: narration allow: compatibility\nx = 1\n", "narration") == {1}


def test_named_files_are_kept_by_suffix() -> None:
    named = ["a.py", "b.md"]

    assert common.files(named, ("src",), (".py",)) == [Path("a.py")]
    assert common.files(named, ("src",), None) == [Path("a.py"), Path("b.md")]


def test_no_named_file_scans_the_default_trees() -> None:
    found = common.files([], ("scripts/lint",), (".py",))

    assert common.REPO_ROOT / "scripts" / "lint" / "common.py" in found
    assert not any("__pycache__" in p.parts for p in found)


def test_the_report_prints_each_finding_and_exits_1_on_any(capsys) -> None:
    assert common.report([]) == 0
    assert common.report(["a.py:1: one", "a.py:2: two"]) == 1
    assert capsys.readouterr().out == "a.py:1: one\na.py:2: two\n"
