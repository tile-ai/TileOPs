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
LOOP_SH = SKILL_DIR / "loop.sh"


def _write_round(
    rounds_dir: Path,
    n: int,
    sha: str,
    event: str,
    blockers: int = 0,
    last_human_comment_id: int | None = None,
) -> Path:
    rounds_dir.mkdir(parents=True, exist_ok=True)
    p = rounds_dir / f"round-{n:02d}.json"
    payload: dict[str, object] = {
        "round": n,
        "finished_at": "2026-01-01T00:00:00Z",
        "head_sha_before": sha,
        "head_sha_after": sha,
        "codex_event": event,
        "blockers_after": blockers,
    }
    if last_human_comment_id is not None:
        payload["last_human_comment_id"] = last_human_comment_id
    p.write_text(json.dumps(payload))
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


def test_rule1_no_skip_when_new_human_comment_on_same_sha(tmp_path: Path) -> None:
    """Bug-1 regression: same SHA + new human comment must NOT skip codex.

    If a human posts a fresh comment after the prior APPROVE on the same
    HEAD sha, the loop is contractually obligated to ingest it. Rule 1
    must therefore bypass the SHA-only reuse and force codex to run.
    """
    run_dir = tmp_path / "review"
    rounds = run_dir / "rounds"
    sha = "e" * 40
    # Prior APPROVE was issued when the latest human-comment id was 100.
    _write_round(rounds, 1, sha, "APPROVE", blockers=0, last_human_comment_id=100)

    # A new human comment with id 101 has since landed.
    result = _run(
        ROUND_PRE,
        {
            "RUN_DIR": str(run_dir),
            "NEXT_ROUND": "2",
            "HEAD_SHA": sha,
            "LATEST_HUMAN_ID": "101",
        },
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "proceed", (
        "new human comment on the approved SHA must force codex to re-review"
    )
    # No marker — the loop must run codex.
    assert not (rounds / "round-02.json").exists()


def test_rule1_skip_when_human_watermark_unchanged(tmp_path: Path) -> None:
    """Same SHA + same human-comment watermark → skip is still safe."""
    run_dir = tmp_path / "review"
    rounds = run_dir / "rounds"
    sha = "f" * 40
    _write_round(rounds, 1, sha, "APPROVE", blockers=0, last_human_comment_id=100)

    result = _run(
        ROUND_PRE,
        {
            "RUN_DIR": str(run_dir),
            "NEXT_ROUND": "2",
            "HEAD_SHA": sha,
            "LATEST_HUMAN_ID": "100",
        },
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "skip"
    marker = rounds / "round-02.json"
    assert marker.exists()
    payload = json.loads(marker.read_text())
    # The marker must carry the watermark forward so a subsequent reuse
    # decision can compare against the *current* loop state, not the
    # original approving round.
    assert payload["last_human_comment_id"] == 100


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


def test_rule2_event_fires_only_on_threshold_transition(tmp_path: Path) -> None:
    """Event must append exactly once per streak (round 3 only, not 4 or 5)."""
    run_dir = tmp_path / "review"
    run_dir.mkdir()
    stub, log = _make_gh_stub(tmp_path)

    target = "tileops/manifest/elementwise_binary.yaml"
    # Five consecutive rounds with the same blocker. Counter goes 1→2→3→4→5;
    # only the 2→3 transition (round 3) should emit an events[] entry, and
    # the agent-stuck label should be added at most once across the streak.
    for r in (1, 2, 3, 4, 5):
        comments = run_dir / f"round-{r:02d}.new-review-comments.json"
        _write_comments(comments, [target])
        result = _run(ROUND_POST, _post_env(run_dir, r, comments, gh_bin=stub))
        assert result.returncode == 0, result.stderr

    history = json.loads((run_dir / "region-history.json").read_text())
    assert history["counters"][target] == 5

    triggered_events = [e for e in history["events"] if e["path"] == target]
    assert len(triggered_events) == 1, (
        f"event must fire only on threshold transition, got {triggered_events}"
    )
    assert triggered_events[0]["round"] == 3

    # Label application is idempotent on the gh side, but we should see the
    # add-label call invoked at most once for this streak (i.e. only on
    # round 3 when TRIGGERED_PATHS becomes non-empty for the first time).
    log_lines = log.read_text().splitlines()
    add_label_lines = [
        line for line in log_lines if "pr edit 1234" in line and "agent-stuck" in line
    ]
    assert len(add_label_lines) == 1, (
        f"expected exactly one add-label invocation, got {add_label_lines}"
    )


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


# ---------------------------------------------------------------------------
# Bug-2 regression: loop wires the post-codex reviewer-authored artifact
# (codex-blockers.json) into round-post.sh, NOT the pre-codex
# new-review-comments.json that explicitly excludes the reviewer login.
# ---------------------------------------------------------------------------


def test_loop_passes_codex_blockers_to_round_post(tmp_path: Path) -> None:
    """Bug-2 regression: round-post.sh must be fed codex-blockers.json.

    The pre-codex ``round-NN.new-review-comments.json`` artifact filters
    OUT REVIEWER_LOGIN, so it never contains this round's codex
    blockers. Rule 2 (per-path 3-strike monitor) is contractually
    supposed to track codex-authored blocker paths, so loop.sh must
    invoke round-post.sh against ``round-NN.codex-blockers.json``,
    which is gathered AFTER codex finishes and filtered to
    REVIEWER_LOGIN.
    """
    src = LOOP_SH.read_text()
    # The Rule 2 invocation block must hand round-post.sh the post-codex
    # artifact, not the pre-codex one.
    assert "COMMENTS_JSON=\"$CODEX_BLOCKERS_JSON\"" in src, (
        "loop.sh must pass the post-codex codex-blockers artifact to round-post.sh"
    )
    # And the artifact itself must be derived from the reviewer-login
    # filter on /pulls/<pr>/comments AFTER codex runs.
    assert "CODEX_BLOCKERS_JSON=\"$SNAP.codex-blockers.json\"" in src
    assert ".user.login==\\\"$REVIEWER_LOGIN\\\"" in src, (
        "codex-blockers artifact must filter to comments authored by REVIEWER_LOGIN"
    )
    # The pre-codex artifact (which excludes REVIEWER_LOGIN) must NOT be
    # the file passed to round-post.sh.
    assert "COMMENTS_JSON=\"$SNAP.new-review-comments.json\"" not in src, (
        "loop.sh must not feed round-post.sh the pre-codex new-review-comments.json "
        "(it explicitly filters OUT the reviewer login)"
    )


def test_round_post_counts_codex_authored_blockers(tmp_path: Path) -> None:
    """Bug-2 regression: when fed a reviewer-authored blocker file, the
    monitor tracks those paths. When fed an empty file (the shape that
    new-review-comments.json would have for a PR with no human chatter),
    no codex paths are tracked — illustrating that wiring the wrong file
    silently disables Rule 2.
    """
    run_dir = tmp_path / "review"
    run_dir.mkdir()
    stub, _log = _make_gh_stub(tmp_path)

    target = "tileops/manifest/elementwise_binary.yaml"

    # Round 1: post-codex artifact contains the codex blocker.
    codex_blockers = run_dir / "round-01.codex-blockers.json"
    codex_blockers.write_text(
        json.dumps(
            [
                {
                    "id": 5001,
                    "user": "tileops-reviewer-bot",
                    "path": target,
                    "severity": "blocker",
                    "body": "x",
                }
            ]
        )
    )
    result = _run(ROUND_POST, _post_env(run_dir, 1, codex_blockers, gh_bin=stub))
    assert result.returncode == 0, result.stderr
    history = json.loads((run_dir / "region-history.json").read_text())
    assert history["counters"].get(target) == 1, (
        "codex-authored blocker on a path must increment its counter"
    )

    # Round 2: simulate the buggy wiring — feed the pre-codex
    # new-review-comments.json artifact, which excludes REVIEWER_LOGIN
    # and is therefore empty for a PR with no human chatter. The bug
    # would manifest as Rule 2 silently never tracking codex blockers
    # at all. This assertion documents that behavior so a future
    # regression can be diagnosed quickly.
    pre_codex = run_dir / "round-02.new-review-comments.json"
    pre_codex.write_text("[]")
    result = _run(ROUND_POST, _post_env(run_dir, 2, pre_codex, gh_bin=stub))
    assert result.returncode == 0, result.stderr
    history = json.loads((run_dir / "region-history.json").read_text())
    assert target not in history["counters"], (
        "feeding round-post the pre-codex (empty) artifact must NOT carry "
        "the codex-blocker counter forward — this is exactly the bug-2 "
        "failure mode the loop wiring must avoid"
    )


# ---------------------------------------------------------------------------
# Edge-case hardening for round-pre.sh and round-post.sh (review-round 3
# follow-up on PR #1168). Both hooks must degrade safely on malformed
# inputs rather than crash the loop or silently mis-track state.
# ---------------------------------------------------------------------------


def test_rule1_proceeds_on_non_numeric_blockers_after(tmp_path: Path) -> None:
    """A corrupt prior-round file with non-numeric ``blockers_after``
    must not crash marker generation.

    Defensive contract: round-pre.sh reads ``.blockers_after`` and
    ``.round`` from a prior round file and passes them through
    ``jq --argjson``, which rejects non-numeric input. A garbage value
    (e.g. ``"many"``) must be coerced to a safe default so the marker
    still writes and the loop can legitimately reuse the prior APPROVE.
    """
    run_dir = tmp_path / "review"
    rounds = run_dir / "rounds"
    rounds.mkdir(parents=True)
    sha = "deadbeef" * 5
    # Hand-craft a prior round whose blockers_after is a string and
    # whose round number is null — both would normally break --argjson.
    (rounds / "round-01.json").write_text(
        json.dumps(
            {
                "round": None,
                "finished_at": "2026-01-01T00:00:00Z",
                "head_sha_before": sha,
                "head_sha_after": sha,
                "codex_event": "APPROVE",
                "blockers_after": "many",
            }
        )
    )

    result = _run(
        ROUND_PRE,
        {"RUN_DIR": str(run_dir), "NEXT_ROUND": "2", "HEAD_SHA": sha},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "skip", result.stdout
    marker = rounds / "round-02.json"
    assert marker.exists(), "marker must still be written despite garbage prior values"
    payload = json.loads(marker.read_text())
    assert payload["blockers_after"] == 0
    assert payload["approve_reused_from"] == 0


def test_rule1_proceeds_when_next_round_non_numeric(tmp_path: Path) -> None:
    """Caller-supplied NEXT_ROUND that is not numeric must force a
    real codex re-run (``proceed``), never a silent ``skip`` without
    a marker."""
    run_dir = tmp_path / "review"
    rounds = run_dir / "rounds"
    sha = "cafebabe" * 5
    _write_round(rounds, 1, sha, "APPROVE")

    result = _run(
        ROUND_PRE,
        {"RUN_DIR": str(run_dir), "NEXT_ROUND": "abc", "HEAD_SHA": sha},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "proceed", result.stdout
    # No marker may be written when proceeding.
    assert not (rounds / "round-abc.json").exists()


def test_rule2_object_shaped_artifact_treated_as_empty(tmp_path: Path) -> None:
    """A non-array (object-shaped) comments artifact must NOT be
    iterated with ``.[]``.

    ``jq empty`` accepts any valid JSON, so a stray ``{path: "x"}``
    payload (e.g. an error envelope from a failed gh call) would pass
    the early validity check and then ``.[]`` on it would yield the
    object's *values* — counting them as blocker paths. The hook must
    require ``type == "array"`` before iterating; non-array input
    contributes no blocker paths this round.
    """
    run_dir = tmp_path / "review"
    run_dir.mkdir()

    # Object-shaped artifact: a single comment as a top-level object,
    # not wrapped in an array.
    comments = run_dir / "round-01.codex-blockers.json"
    comments.write_text(
        json.dumps(
            {
                "id": 9001,
                "path": "should/not/count.py",
                "severity": "blocker",
                "body": "x",
            }
        )
    )
    result = _run(ROUND_POST, _post_env(run_dir, 1, comments))
    assert result.returncode == 0, result.stderr

    history = json.loads((run_dir / "region-history.json").read_text())
    assert history["counters"] == {}, (
        "object-shaped artifact must contribute zero blocker paths; got "
        f"{history['counters']}"
    )
