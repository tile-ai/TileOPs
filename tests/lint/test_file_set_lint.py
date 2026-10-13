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


def test_every_allowlisted_path_is_in_the_tree() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    assert [p for p in lint.ALLOWED if not (repo_root / p).is_file()] == []
