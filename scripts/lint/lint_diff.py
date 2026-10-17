"""What a branch adds, compared with the merge base of a base ref and HEAD.

The diff-aware lints in this directory import it. A file is compared as it stands in the
working tree, which under pre-commit is the staged content, with the same path at the merge
base: a line counts as added when the diff of the two shows it with a ``+``. A path the
merge base does not hold adds every line, whether git tracks it yet or not. A rename is
followed, so a moved file adds only the lines it changed on the way.

An explicit base that does not resolve raises `NoMergeBaseError`. Hooks use the special
base ``auto``: it finds a fetched remote for ``tile-ai/TileOPs`` and skips locally when a
clone has none, while CI's upstream checkout remains authoritative.
"""

import ast
import os
import re
import subprocess
from pathlib import Path

_HUNK = re.compile(r"^@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_UPSTREAM_URL = re.compile(r"(?:^|[@/:])github\.com[/:]tile-ai/TileOPs(?:\.git)?/?$", re.IGNORECASE)

# Every diff-aware hook names the same environment override in its skip guidance.
BASE_ENV = "TILEOPS_LINT_BASE"


class NoMergeBaseError(RuntimeError):
    """The base ref has no merge base with HEAD in this clone."""


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "core.quotepath=false", *args], cwd=root, capture_output=True, text=True
    )
    if done.returncode != 0:
        raise NoMergeBaseError(done.stderr.strip() or f"git {' '.join(args)} failed")
    return done.stdout


def merge_base(base: str, root: Path) -> str:
    """The merge base of *base* and HEAD in the repository at *root*."""
    try:
        return _git(root, "merge-base", base, "HEAD").strip()
    except NoMergeBaseError as exc:
        raise NoMergeBaseError(
            f"no merge base of {base} and HEAD ({exc}); fetch {base} or pass --base"
        ) from None


def resolve_base(base: str, root: Path) -> str | None:
    """Resolve *base*, discovering the TileOPs upstream when it is ``auto``.

    An explicit ref, including one supplied through ``TILEOPS_LINT_BASE``, is strict.
    Automatic discovery only considers remotes whose URL names ``tile-ai/TileOPs`` and
    returns None when none has a fetched ``main`` ref. Local hooks may then skip while CI,
    whose checkout has the upstream remote, remains authoritative.
    """
    if base != "auto":
        merge_base(base, root)
        return base
    configured = os.environ.get(BASE_ENV)
    if configured:
        merge_base(configured, root)
        return configured
    try:
        remotes = _git(root, "remote").splitlines()
    except NoMergeBaseError:
        return None
    for remote in remotes:
        try:
            urls = _git(root, "remote", "get-url", "--all", remote).splitlines()
        except NoMergeBaseError:
            continue
        if not any(_UPSTREAM_URL.search(url) for url in urls):
            continue
        candidate = f"{remote}/main"
        try:
            merge_base(candidate, root)
        except NoMergeBaseError:
            continue
        return candidate
    return None


def _unquote_path(path: str) -> str:
    """Decode a path from a git patch header, including C-style quoted paths."""
    if not path.startswith('"'):
        return path
    try:
        value = ast.literal_eval(path)
    except (SyntaxError, ValueError):
        return path
    return value if isinstance(value, str) else path


def files_at(commit: str, root: Path) -> set[str]:
    """Every path *commit* holds, relative to *root*."""
    return {
        path
        for path in _git(root, "ls-tree", "-r", "-z", "--name-only", commit).split("\0")
        if path
    }


def relative(path: Path | str, root: Path) -> str:
    """*path* relative to *root*, in the form git prints it."""
    path = Path(path)
    if path.is_absolute():
        path = path.resolve().relative_to(root.resolve())
    return path.as_posix()


def added_lines(paths: list[Path | str], base: str, root: Path) -> dict[str, set[int] | None]:
    """Map each of *paths*, relative to *root*, to the line numbers it adds.

    None stands for every line: the path is absent from the merge base.
    """
    commit = merge_base(base, root)
    held = files_at(commit, root)
    hunks: dict[str, set[int]] = {}
    renamed = set()
    diff = _git(root, "diff", "-U0", "-M", "--no-color", "--no-ext-diff", commit).splitlines()
    current, body = None, 0
    for line in diff:
        if body:
            # A hunk line, whatever it starts with: `+++ x` here is content, not a header.
            # The no-newline marker follows a counted line without being one.
            body -= not line.startswith("\\")
            continue
        match = _HUNK.match(line)
        if match and current is not None:
            removed, start, count = (int(g) if g else 1 for g in match.groups())
            body = removed + count
            hunks[current].update(range(start, start + count))
        elif line.startswith("diff --git "):
            current = None
        elif line.startswith("rename to "):
            current = _unquote_path(line.removeprefix("rename to "))
            renamed.add(current)
            hunks.setdefault(current, set())
        elif line.startswith("+++ "):
            target = _unquote_path(line.removeprefix("+++ "))
            current = None if target == "/dev/null" else target.removeprefix("b/")
            if current is not None:
                hunks.setdefault(current, set())
    out: dict[str, set[int] | None] = {}
    for path in paths:
        rel = relative(path, root)
        out[rel] = hunks.get(rel, set()) if rel in held or rel in renamed else None
    return out
