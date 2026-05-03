"""Unit tests for the review-tileops convergence rules.

Covers two deterministic mechanism-level rules added to the
review-tileops loop (issue #1161):

* Rule 1 — same-SHA APPROVE skip (``round-pre.sh``): if a prior round
  in the same loop run already produced ``codex_event == "APPROVE"`` on
  the current HEAD sha, the loop must skip the codex invocation and
  emit a marker ``round-NN.json`` that points back at the prior round.
* Rule 2 — same-path 3-strike monitor (``round-post.sh``): track per-
  path consecutive-blocker streaks across rounds; when any path's
  counter hits >= 3, ensure the PR carries the ``agent-stuck`` GitHub
  label. The hook MUST NOT mutate any blocker, comment, PR title, or
  thread.

These tests stay hermetic: they invoke the bash scripts directly via
``subprocess`` against synthetic ``RUN_DIR`` trees and stub the ``gh``
CLI with a recording shell script.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / ".claude" / "skills" / "review-tileops"
ROUND_PRE = SKILL_DIR / "round-pre.sh"
ROUND_POST = SKILL_DIR / "round-post.sh"


def _write_round(rounds_dir: Path, n: int, sha: str, event: str, blockers: int = 0) -> Path:
    rounds_dir.mkdir(parents=True, exist_ok=True)
    p = rounds_dir / f"round-{n:02d}.json"
    p.write_text(
        json.dumps(
            {
                "round": n,
                "finished_at": "2026-01-01T00:00:00Z",
                "head_sha_before": sha,
                "head_sha_after": sha,
                "codex_event": event,
                "blockers_after": blockers,
            }
        )
    )
    return p


def _run(script: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    full_env = os.environ.copy()
    full_env.update(env)
    return subprocess.run(
        ["bash", str(script)],
        env=full_env,
        capture_output=True,
        text=True,
        check=False,
    )


def _make_gh_stub(tmp_path: Path) -> tuple[Path, Path]:
    """Create a recording stub for the gh CLI.

    Returns (path-to-stub, path-to-log). The stub appends each call's
    argv (one per line, tab-separated) to the log file and exits 0.
    """
    log = tmp_path / "gh.log"
    stub = tmp_path / "gh-stub.sh"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" >> {log!s}\n"
        "exit 0\n"
    )
    stub.chmod(0o755)
    return stub, log


# ---------------------------------------------------------------------------
# Rule 1 — round-pre.sh
# ---------------------------------------------------------------------------


def test_rule1_skip_on_prior_approve(tmp_path: Path) -> None:
    """Prior APPROVE on same SHA → round-pre emits 'skip' and writes a marker."""
    run_dir = tmp_path / "review"
    rounds = run_dir / "rounds"
    sha = "a" * 40
    _write_round(rounds, 1, sha, "APPROVE", blockers=0)

    result = _run(ROUND_PRE, {"RUN_DIR": str(run_dir), "NEXT_ROUND": "2", "HEAD_SHA": sha})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "skip"

    marker = rounds / "round-02.json"
    assert marker.exists()
    payload = json.loads(marker.read_text())
    assert payload["codex_event"] == "APPROVE"
    assert payload["head_sha_after"] == sha
    assert payload["approve_reused_from"] == 1
    assert payload["skipped_codex"] is True


def test_rule1_no_skip_when_sha_differs(tmp_path: Path) -> None:
    """Prior APPROVE on a different SHA must NOT trigger skip."""
    run_dir = tmp_path / "review"
    rounds = run_dir / "rounds"
    _write_round(rounds, 1, "a" * 40, "APPROVE", blockers=0)

    new_sha = "b" * 40
    result = _run(
        ROUND_PRE, {"RUN_DIR": str(run_dir), "NEXT_ROUND": "2", "HEAD_SHA": new_sha}
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "proceed"
    # No marker written — loop must run codex.
    assert not (rounds / "round-02.json").exists()


def test_rule1_no_skip_on_request_changes_same_sha(tmp_path: Path) -> None:
    """A prior REQUEST_CHANGES on the same SHA must not be reused as APPROVE."""
    run_dir = tmp_path / "review"
    rounds = run_dir / "rounds"
    sha = "c" * 40
    _write_round(rounds, 1, sha, "REQUEST_CHANGES", blockers=2)

    result = _run(ROUND_PRE, {"RUN_DIR": str(run_dir), "NEXT_ROUND": "2", "HEAD_SHA": sha})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "proceed"


def test_rule1_handles_missing_rounds_dir(tmp_path: Path) -> None:
    """First round (no rounds/ yet) is a clean 'proceed', not an error."""
    run_dir = tmp_path / "review"
    run_dir.mkdir()
    result = _run(
        ROUND_PRE, {"RUN_DIR": str(run_dir), "NEXT_ROUND": "1", "HEAD_SHA": "d" * 40}
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "proceed"


# ---------------------------------------------------------------------------
# Rule 2 — round-post.sh
# ---------------------------------------------------------------------------


def _write_comments(path: Path, paths: list[str]) -> bytes:
    """Write a synthetic new-review-comments.json with one blocker per path."""
    payload = [
        {"id": 1000 + i, "path": p, "severity": "blocker", "body": "x"}
        for i, p in enumerate(paths)
    ]
    raw = json.dumps(payload).encode()
    path.write_bytes(raw)
    return raw


def _post_env(
    run_dir: Path, round_no: int, comments: Path, gh_bin: Path | str = "none"
) -> dict[str, str]:
    return {
        "RUN_DIR": str(run_dir),
        "ROUND": str(round_no),
        "COMMENTS_JSON": str(comments),
        "REPO": "tile-ai/TileOPs",
        "PR": "1234",
        "GH_BIN": str(gh_bin),
    }


def test_rule2_label_after_three_strikes(tmp_path: Path) -> None:
    """Three consecutive rounds with a blocker on the same path → label applied."""
    run_dir = tmp_path / "review"
    run_dir.mkdir()
    stub, log = _make_gh_stub(tmp_path)

    target = "tileops/manifest/elementwise_binary.yaml"
    for r in (1, 2, 3):
        comments = run_dir / f"round-{r:02d}.new-review-comments.json"
        _write_comments(comments, [target])
        result = _run(ROUND_POST, _post_env(run_dir, r, comments, gh_bin=stub))
        assert result.returncode == 0, result.stderr

    history = json.loads((run_dir / "region-history.json").read_text())
    assert history["counters"][target] == 3
    triggered_events = [e for e in history["events"] if e["path"] == target]
    assert triggered_events, "expected at least one triggered event entry"
    assert triggered_events[-1]["round"] == 3

    log_lines = log.read_text().splitlines()
    # Only round 3 should attempt label ops; rounds 1 and 2 are below threshold.
    assert any("pr edit 1234" in line and "agent-stuck" in line for line in log_lines)
    assert any("label create agent-stuck" in line for line in log_lines)


def test_rule2_counter_resets_on_break(tmp_path: Path) -> None:
    """A round with no blocker on the path resets that path's counter."""
    run_dir = tmp_path / "review"
    run_dir.mkdir()
    target = "a/b.py"

    # Rounds 1 & 2: blocker on target.
    for r in (1, 2):
        comments = run_dir / f"round-{r:02d}.new-review-comments.json"
        _write_comments(comments, [target])
        result = _run(ROUND_POST, _post_env(run_dir, r, comments))
        assert result.returncode == 0, result.stderr

    history = json.loads((run_dir / "region-history.json").read_text())
    assert history["counters"][target] == 2

    # Round 3: blocker on a different path; target absent → counter resets.
    comments3 = run_dir / "round-03.new-review-comments.json"
    _write_comments(comments3, ["other/path.py"])
    result = _run(ROUND_POST, _post_env(run_dir, 3, comments3))
    assert result.returncode == 0, result.stderr

    history = json.loads((run_dir / "region-history.json").read_text())
    assert target not in history["counters"], "counter for absent path must reset"
    assert history["counters"]["other/path.py"] == 1

    # Round 4: blocker on target again → counter starts at 1, not 3.
    comments4 = run_dir / "round-04.new-review-comments.json"
    _write_comments(comments4, [target])
    result = _run(ROUND_POST, _post_env(run_dir, 4, comments4))
    assert result.returncode == 0, result.stderr

    history = json.loads((run_dir / "region-history.json").read_text())
    assert history["counters"][target] == 1


def test_rule2_no_blocker_mutation(tmp_path: Path) -> None:
    """round-post.sh must not modify the new-review-comments.json file."""
    run_dir = tmp_path / "review"
    run_dir.mkdir()
    stub, _log = _make_gh_stub(tmp_path)
    target = "x.py"

    # Drive three rounds so the threshold fires (which is exactly when a
    # buggy implementation might be tempted to "downgrade" or rewrite a
    # blocker). Compare bytes of the comments file before and after each
    # call.
    for r in (1, 2, 3):
        comments = run_dir / f"round-{r:02d}.new-review-comments.json"
        original = _write_comments(comments, [target])
        before = comments.read_bytes()
        assert before == original  # sanity

        result = _run(ROUND_POST, _post_env(run_dir, r, comments, gh_bin=stub))
        assert result.returncode == 0, result.stderr

        after = comments.read_bytes()
        assert after == before, f"round-post mutated comments file at round {r}"


def test_rule2_works_with_missing_region_history(tmp_path: Path) -> None:
    """Missing region-history.json is treated as empty state (no crash)."""
    run_dir = tmp_path / "review"
    run_dir.mkdir()
    assert not (run_dir / "region-history.json").exists()

    comments = run_dir / "round-01.new-review-comments.json"
    _write_comments(comments, ["only/once.py"])
    result = _run(ROUND_POST, _post_env(run_dir, 1, comments))
    assert result.returncode == 0, result.stderr

    history_path = run_dir / "region-history.json"
    assert history_path.exists()
    history = json.loads(history_path.read_text())
    assert history["counters"] == {"only/once.py": 1}
    assert history["events"] == []


def test_rule2_handles_missing_comments_file(tmp_path: Path) -> None:
    """A missing or empty comments file must not crash the post hook."""
    run_dir = tmp_path / "review"
    run_dir.mkdir()
    missing = run_dir / "round-01.new-review-comments.json"  # never created

    result = _run(ROUND_POST, _post_env(run_dir, 1, missing))
    assert result.returncode == 0, result.stderr

    history = json.loads((run_dir / "region-history.json").read_text())
    assert history["counters"] == {}


def test_rule2_no_label_below_threshold(tmp_path: Path) -> None:
    """Below-threshold rounds must not invoke gh label / pr edit."""
    run_dir = tmp_path / "review"
    run_dir.mkdir()
    stub, log = _make_gh_stub(tmp_path)

    for r in (1, 2):
        comments = run_dir / f"round-{r:02d}.new-review-comments.json"
        _write_comments(comments, ["one.py"])
        result = _run(ROUND_POST, _post_env(run_dir, r, comments, gh_bin=stub))
        assert result.returncode == 0, result.stderr

    assert not log.exists() or log.read_text() == "", (
        "gh must not be invoked before the 3rd consecutive blocker round"
    )
