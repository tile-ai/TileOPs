"""What a branch adds, compared with the merge base of a base ref and HEAD.

The diff-aware lints in this directory import it. A file is compared as it stands in the
working tree, which under pre-commit is the staged content, with the same path at the merge
base: a line counts as added when the diff of the two shows it with a ``+``. A path the
merge base does not hold adds every line, whether git tracks it yet or not. A rename is
followed, so a moved file adds only the lines it changed on the way.

A base that does not resolve raises `NoMergeBaseError`; the lints report it and fail, since
a rule over added lines cannot hold for lines it never saw.
"""

import re
import subprocess
from pathlib import Path

_HUNK = re.compile(r"^@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


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


def files_at(commit: str, root: Path) -> set[str]:
    """Every path *commit* holds, relative to *root*."""
    return set(_git(root, "ls-tree", "-r", "--name-only", commit).splitlines())


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
            current = line.removeprefix("rename to ").strip('"')
            renamed.add(current)
            hunks.setdefault(current, set())
        elif line.startswith("+++ "):
            target = line.removeprefix("+++ ")
            current = None if target == "/dev/null" else target.strip('"').removeprefix("b/")
            if current is not None:
                hunks.setdefault(current, set())
    out: dict[str, set[int] | None] = {}
    for path in paths:
        rel = relative(path, root)
        out[rel] = hunks.get(rel, set()) if rel in held or rel in renamed else None
    return out
