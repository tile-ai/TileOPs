"""Unit tests for ``scripts/lint/lint_diff.py``, on a throwaway git repository."""

import subprocess
from pathlib import Path
from unittest import mock

import lint_diff
import pytest

pytestmark = pytest.mark.smoke


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository whose ref ``base`` holds ``kept.py`` with three lines."""
    _git(tmp_path, "init", "-q")
    (tmp_path / "kept.py").write_text("a = 1\nb = 2\nc = 3\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "base")
    _git(tmp_path, "branch", "base")
    return tmp_path


def test_a_changed_file_adds_the_lines_it_changed(repo: Path) -> None:
    (repo / "kept.py").write_text("a = 1\nb = 20\nc = 3\nd = 4\n")

    assert lint_diff.added_lines(["kept.py"], "base", repo) == {"kept.py": {2, 4}}


def test_an_unchanged_file_adds_nothing(repo: Path) -> None:
    assert lint_diff.added_lines(["kept.py"], "base", repo) == {"kept.py": set()}


@pytest.mark.parametrize("staged", [True, False], ids=["staged", "untracked"])
def test_a_file_the_base_lacks_adds_every_line(repo: Path, staged: bool) -> None:
    (repo / "new.py").write_text("x = 1\n")
    if staged:
        _git(repo, "add", "new.py")

    assert lint_diff.added_lines(["new.py"], "base", repo) == {"new.py": None}


def test_a_changed_line_that_reads_like_a_diff_header_is_content(repo: Path) -> None:
    # The second line diffs as `+++ b/other.py`; the hunk after it still belongs here.
    (repo / "kept.py").write_text("a = 1\n++ b/other.py\nb = 2\nc = 3\nd = 4\n")

    assert lint_diff.added_lines(["kept.py"], "base", repo) == {"kept.py": {2, 5}}


def test_a_renamed_file_adds_only_the_lines_it_changed(repo: Path) -> None:
    _git(repo, "mv", "kept.py", "moved.py")
    (repo / "moved.py").write_text("a = 1\nb = 2\nc = 3\nd = 4\n")
    _git(repo, "add", "moved.py")

    assert lint_diff.added_lines(["moved.py"], "base", repo) == {"moved.py": {4}}


@pytest.mark.parametrize("name", ['quote".py', "back\\slash.py", "tab\tname.py", "line\nname.py"])
def test_a_git_quoted_path_is_decoded(repo: Path, name: str) -> None:
    path = repo / name
    path.write_text("a = 1\n")
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", "quoted base")
    _git(repo, "branch", "-f", "base", "HEAD")
    path.write_text("a = 2\n")

    assert lint_diff.added_lines([name], "base", repo) == {name: {1}}


def test_a_git_quoted_rename_is_decoded(repo: Path) -> None:
    moved = 'moved"\\\t.py'
    _git(repo, "mv", "kept.py", moved)
    (repo / moved).write_text("a = 1\nb = 2\nc = 3\nd = 4\n")
    _git(repo, "add", moved)

    assert lint_diff.added_lines([moved], "base", repo) == {moved: {4}}


@pytest.mark.parametrize(
    "url", ["https://github.com/tile-ai/TileOPs.git", "git@github.com:tile-ai/TileOPs.git"]
)
def test_auto_base_finds_the_official_remote_not_a_personal_origin(repo: Path, url: str) -> None:
    _git(repo, "remote", "add", "origin", "git@github.com:someone/TileOPs.git")
    _git(repo, "remote", "add", "company", url)
    base = subprocess.run(
        ["git", "rev-parse", "base"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    _git(repo, "update-ref", "refs/remotes/company/main", base)

    with mock.patch.dict(lint_diff.os.environ, {}, clear=True):
        assert lint_diff.resolve_base("auto", repo) == "company/main"


def test_auto_base_honors_the_environment(repo: Path) -> None:
    with mock.patch.dict(lint_diff.os.environ, {lint_diff.BASE_ENV: "base"}, clear=True):
        assert lint_diff.resolve_base("auto", repo) == "base"


def test_auto_base_without_a_fetched_upstream_returns_none(repo: Path) -> None:
    _git(repo, "remote", "add", "origin", "https://github.com/someone/TileOPs.git")

    with mock.patch.dict(lint_diff.os.environ, {}, clear=True):
        assert lint_diff.resolve_base("auto", repo) is None


def test_auto_base_with_an_unfetched_official_remote_returns_none(repo: Path) -> None:
    _git(repo, "remote", "add", "upstream", "https://github.com/tile-ai/TileOPs.git")

    with mock.patch.dict(lint_diff.os.environ, {}, clear=True):
        assert lint_diff.resolve_base("auto", repo) is None


def test_an_invalid_environment_base_is_an_error(repo: Path) -> None:
    with (
        mock.patch.dict(lint_diff.os.environ, {lint_diff.BASE_ENV: "missing"}, clear=True),
        pytest.raises(lint_diff.NoMergeBaseError, match="no merge base of missing and HEAD"),
    ):
        lint_diff.resolve_base("auto", repo)


def test_a_base_without_a_merge_base_raises(repo: Path) -> None:
    with pytest.raises(lint_diff.NoMergeBaseError, match="no merge base of missing and HEAD"):
        lint_diff.added_lines(["kept.py"], "missing", repo)
