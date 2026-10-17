"""Unit tests for ``scripts/lint/file_set_lint.py``."""

from pathlib import Path

import file_set_lint as lint
import pytest

pytestmark = pytest.mark.smoke

FLAGGED = [
    pytest.param("docs/w4a16-m1-research.md", id="research-note"),
    pytest.param("docs/design/gemm-template.zh.md", id="translated-note"),
    pytest.param("notes/plan.md", id="notes"),
    pytest.param("profile_run.log", id="log"),
    pytest.param("benchmarks/kernel.sass", id="sass"),
    pytest.param("benchmarks/kernel.cu", id="cuda-outside-csrc"),
    pytest.param("tests/ops/test_gemm_scratch.py", id="scratch-test"),
    pytest.param("tests/ops/test_gemm_phase2.py", id="phase-test"),
    pytest.param("benchmark_runs/run.json", id="benchmark-runs"),
    pytest.param("codex_file/prompt.txt", id="codex-file"),
]

ACCEPTED = [
    pytest.param("src/tileops/csrc/gemm.cu", id="cuda-in-csrc"),
    pytest.param("docs/design/testing.md", id="design-doc"),
    pytest.param("tests/ops/test_gemm.py", id="test"),
    pytest.param(next(iter(lint.ALLOWED)), id="allowlisted"),
]


@pytest.mark.parametrize("path", FLAGGED)
def test_a_work_artifact_is_reported(tmp_path: Path, path: str) -> None:
    assert lint.findings(path, tmp_path), path


@pytest.mark.parametrize("path", ACCEPTED)
def test_a_file_that_ships_is_left_alone(tmp_path: Path, path: str) -> None:
    assert lint.findings(path, tmp_path) == [], path


def test_a_file_carrying_the_allow_comment_is_left_alone(tmp_path: Path) -> None:
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "plan.md").write_text("<!-- lint: file-set allow: the index -->\n")

    assert lint.findings("notes/plan.md", tmp_path) == []


def test_a_file_carrying_a_bare_allow_comment_is_reported(tmp_path: Path) -> None:
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "plan.md").write_text("<!-- lint: file-set allow: -->\n")

    assert lint.findings("notes/plan.md", tmp_path)


def test_the_marker_outside_a_comment_is_reported(tmp_path: Path) -> None:
    (tmp_path / "run.log").write_text("lint: file-set allow: quoted in the log\n")

    assert lint.findings("run.log", tmp_path)


def test_the_message_names_the_allow_comment(tmp_path: Path) -> None:
    (found,) = lint.findings("notes/plan.md", tmp_path)

    assert found.endswith('or a "lint: file-set allow: <reason>" comment in the file')


_NOTE = "docs/w4a16-m1-research.md"


def _research_note(tmp_path: Path, text: str) -> list[str]:
    (tmp_path / "docs").mkdir()
    (tmp_path / _NOTE).write_text(text)
    return lint.findings(_NOTE, tmp_path)


FENCED_ONLY = [
    pytest.param("# Note\n\n```python\n# lint: file-set allow: fine\n```\n", id="backtick-fence"),
    pytest.param("~~~\n<!-- lint: file-set allow: fine -->\n~~~\n", id="tilde-fence"),
    pytest.param(
        "````md\n```\n<!-- lint: file-set allow: fine -->\n```\n````\n",
        id="shorter-fence-inside-a-longer-one",
    ),
    pytest.param(
        "```\n<!-- lint: file-set allow: fine -->\n~~~\n```\n", id="other-fence-char-inside"
    ),
    pytest.param("```\ntext\n```\n```\n<!-- lint: file-set allow: fine -->\n", id="unclosed"),
    pytest.param(
        "- step\n\n    ```\n    <!-- lint: file-set allow: fine -->\n    ```\n",
        id="fence-in-a-list-item",
    ),
]


@pytest.mark.parametrize("text", FENCED_ONLY)
def test_a_note_whose_only_allow_comment_is_fenced_is_reported(tmp_path: Path, text: str) -> None:
    assert _research_note(tmp_path, text), text


UNFENCED = [
    pytest.param(
        "<!-- lint: file-set allow: the index -->\n```\nx = 1\n```\n", id="before-a-fence"
    ),
    pytest.param("```\nx = 1\n```\n<!-- lint: file-set allow: the index -->\n", id="after-a-fence"),
    pytest.param(
        "Inline `` ``` `` code.\n<!-- lint: file-set allow: the index -->\n",
        id="backticks-mid-line",
    ),
]


@pytest.mark.parametrize("text", UNFENCED)
def test_a_note_with_an_allow_comment_outside_a_fence_is_left_alone(
    tmp_path: Path, text: str
) -> None:
    assert _research_note(tmp_path, text) == [], text


def test_a_fence_in_a_file_that_is_not_markdown_does_not_hide_the_comment(tmp_path: Path) -> None:
    (tmp_path / "run.log").write_text("```\n# lint: file-set allow: fixture for the parser\n```\n")

    assert lint.findings("run.log", tmp_path) == []


def test_every_allowlisted_path_is_in_the_tree() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    assert [p for p in lint.ALLOWED if not (repo_root / p).is_file()] == []
