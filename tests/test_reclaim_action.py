"""Unit tests for .github/actions/reclaim-runner-disk/reclaim_cache.sh.

Covers the sentinel-repair + atomic-trim primitives that protect caches
whose consumers assume "directory exists => contents complete" (the
tilelang autotuner cache in particular), plus the gpu-smoke
trust-routing contract, whose failure mode has no CI signal.

Required cases:
  - half_dead       : atomic subdir with files but no sentinel is removed
                      on a single invocation.
  - atomic_stale    : atomic subdir whose newest file is older than
                      cache-age-days is removed whole-directory.
  - atomic_fresh    : fresh atomic subdirs are preserved.
  - invariant       : atomic roots never have their *individual files*
                      trimmed, even when file-level trim runs.

Runs on every PR (smoke tier), so does not depend on a self-hosted
runner or the Tilelang runtime. Must stay fast and hermetic.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parent.parent
# Script is colocated with the composite action so the gpu-smoke
# `.trusted/.github/actions` sparse-checkout picks it up; see action.yml.
RECLAIM_SCRIPT = REPO_ROOT / ".github" / "actions" / "reclaim-runner-disk" / "reclaim_cache.sh"


def _age_path(path: Path, *, days: float) -> None:
    """Backdate mtime+atime of every file under ``path`` by ``days`` days."""
    past = time.time() - days * 86400
    if path.is_file():
        os.utime(path, (past, past))
        return
    for entry in path.rglob("*"):
        try:
            os.utime(entry, (past, past), follow_symlinks=False)
        except (FileNotFoundError, PermissionError):
            continue
    os.utime(path, (past, past))


def _run(subcommand: str, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    cmd = ["bash", str(RECLAIM_SCRIPT), subcommand, *args]
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    result = subprocess.run(cmd, capture_output=True, text=True, env=merged_env, check=False)
    assert result.returncode == 0, (
        f"{cmd!r} exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


def _make_autotuner_subdir(
    root: Path, name: str, *, with_sentinel: bool, sentinel: str = "best_config.json"
) -> Path:
    subdir = root / name
    subdir.mkdir(parents=True, exist_ok=True)
    (subdir / "kernel.cu").write_text("// cached kernel\n")
    (subdir / "kernel.so").write_bytes(b"\x7fELF")
    if with_sentinel:
        (subdir / sentinel).write_text('{"block": [128, 64]}\n')
    return subdir


# sentinel-repair


def test_sentinel_repair_removes_half_dead_subdir(tmp_path: Path) -> None:
    """half_dead case: subdir missing best_config.json must be removed."""
    root = tmp_path / "autotuner"
    half_dead = _make_autotuner_subdir(root, "halfdead_sig", with_sentinel=False)
    healthy = _make_autotuner_subdir(root, "healthy_sig", with_sentinel=True)

    _run("sentinel-repair", str(root))

    assert not half_dead.exists(), "half-dead subdir should have been removed"
    assert healthy.exists(), "healthy subdir with sentinel must be preserved"
    assert (healthy / "best_config.json").exists()


def test_sentinel_repair_is_idempotent(tmp_path: Path) -> None:
    """Running sentinel-repair twice on a clean tree must not regress."""
    root = tmp_path / "autotuner"
    healthy = _make_autotuner_subdir(root, "healthy_sig", with_sentinel=True)

    _run("sentinel-repair", str(root))
    _run("sentinel-repair", str(root))

    assert healthy.exists()
    assert (healthy / "best_config.json").exists()


def test_all_subcommands_tolerate_missing_root(tmp_path: Path) -> None:
    """A non-existent cache root must be a no-op, not an error, for every
    subcommand."""
    missing = tmp_path / "does-not-exist"
    _run("sentinel-repair", str(missing))  # asserts rc==0
    _run("atomic-trim", "7", str(missing))
    _run("trim-files", "7", str(missing))


def test_sentinel_repair_honours_custom_sentinel_filename(tmp_path: Path) -> None:
    """The sentinel filename is overridable via $SENTINEL_FILENAME (used in tests)."""
    root = tmp_path / "autotuner"
    subdir = _make_autotuner_subdir(root, "sig", with_sentinel=False, sentinel="SENTINEL")
    (subdir / "SENTINEL").write_text("ok\n")

    _run("sentinel-repair", str(root), env={"SENTINEL_FILENAME": "SENTINEL"})

    assert subdir.exists(), "subdir with the custom sentinel must be preserved"


# atomic-trim


def test_atomic_trim_removes_stale_subdir_whole(tmp_path: Path) -> None:
    """atomic_stale case: subdir whose newest file is older than the
    cutoff is removed as a whole unit."""
    root = tmp_path / "autotuner"
    stale = _make_autotuner_subdir(root, "stale_sig", with_sentinel=True)
    _age_path(stale, days=30)

    _run("atomic-trim", "7", str(root))

    assert not stale.exists(), "stale atomic subdir must be removed whole-directory"


def test_atomic_trim_preserves_fresh_subdir(tmp_path: Path) -> None:
    """atomic_fresh case: subdirs within the age window are kept intact."""
    root = tmp_path / "autotuner"
    fresh = _make_autotuner_subdir(root, "fresh_sig", with_sentinel=True)

    _run("atomic-trim", "7", str(root))

    assert fresh.exists()
    # All files preserved, not just the directory.
    assert (fresh / "best_config.json").exists()
    assert (fresh / "kernel.cu").exists()
    assert (fresh / "kernel.so").exists()


def test_atomic_trim_never_trims_individual_files(tmp_path: Path) -> None:
    """invariant: even when *some* files inside an atomic subdir are old,
    atomic-trim must not delete individual files — the unit of deletion
    is the whole subdirectory. The subdir is kept whenever *any* entry
    inside it is within the age window."""
    root = tmp_path / "autotuner"
    subdir = _make_autotuner_subdir(root, "mixed_sig", with_sentinel=True)
    # Age just the kernel files, leave best_config.json fresh.
    _age_path(subdir / "kernel.cu", days=30)
    _age_path(subdir / "kernel.so", days=30)

    _run("atomic-trim", "7", str(root))

    assert subdir.exists(), "subdir with at least one fresh file must survive"
    assert (subdir / "kernel.cu").exists(), (
        "individual files inside atomic root must never be trimmed"
    )
    assert (subdir / "kernel.so").exists()
    assert (subdir / "best_config.json").exists()


def test_atomic_trim_uses_file_mtime_not_dir_mtime(tmp_path: Path) -> None:
    """Directory mtime must not make a stale subdir look fresh.

    A cache restore/extract can bump the subdir's own mtime to "now" while
    every regular file inside keeps its original (old) timestamp. atomic-trim
    must decide staleness from the newest FILE mtime in the subtree, not the
    directory mtime, otherwise age-based reclaim is defeated.
    """
    root = tmp_path / "autotuner"
    root.mkdir()
    stale = root / "deadbeef"
    stale.mkdir()
    (stale / "best_config.json").write_text("{}")
    (stale / "kernel.so").write_text("x")

    # Backdate only the files; deliberately leave the subdir mtime at "now".
    past = time.time() - 30 * 86400
    for entry in stale.iterdir():
        os.utime(entry, (past, past))
    os.utime(stale, (time.time(), time.time()))

    _run("atomic-trim", "7", str(root))

    assert not stale.exists(), (
        "atomic-trim regressed: stale subdir kept alive by dir mtime. "
        "Newest-mtime logic must restrict to -type f."
    )


def test_atomic_trim_handles_empty_root(tmp_path: Path) -> None:
    root = tmp_path / "autotuner"
    root.mkdir()
    _run("atomic-trim", "7", str(root))
    assert root.exists(), "the cache root itself is never removed"


# trim-files (non-atomic roots)


def test_trim_files_removes_old_files_but_leaves_atomic_roots_alone(
    tmp_path: Path,
) -> None:
    """invariant: file-level trim is only applied to the roots passed
    in. Callers must keep atomic roots out of the trim-files list —
    which this test reinforces by exercising a non-atomic root and
    asserting that a neighbouring autotuner root (*not* passed in) is
    untouched even if its files are ancient."""
    triton_root = tmp_path / "triton-cache"
    triton_root.mkdir()
    old_file = triton_root / "old.bin"
    old_file.write_bytes(b"\x00")
    _age_path(old_file, days=30)
    fresh_file = triton_root / "fresh.bin"
    fresh_file.write_bytes(b"\x01")

    # A *separate* autotuner root the action would NOT pass to trim-files.
    autotuner_root = tmp_path / "autotuner"
    subdir = _make_autotuner_subdir(autotuner_root, "sig", with_sentinel=True)
    _age_path(subdir, days=30)

    _run("trim-files", "7", str(triton_root))

    # Non-atomic root: old files pruned, fresh files kept.
    assert not old_file.exists()
    assert fresh_file.exists()
    # Atomic root untouched — trim-files must NOT be called on it.
    assert subdir.exists()
    assert (subdir / "best_config.json").exists()
    assert (subdir / "kernel.cu").exists()


# gpu-smoke security trust routing

GPU_SMOKE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "gpu-smoke.yml"


def test_security_policy_routes_trust_by_collaborator_permission() -> None:
    """is_fork must derive from the PR author's collaborator permission
    (write/maintain/admin -> trusted), fail closed to the fork pool, and
    drive both runs-on and the trusted-action ref."""
    import yaml

    wf = yaml.safe_load(GPU_SMOKE_WORKFLOW.read_text())
    policy_job = wf["jobs"]["security-policy"]
    run_steps = [s for s in policy_job["steps"] if "run" in s and s.get("id") == "policy"]
    assert run_steps, "expected a 'policy' step in security-policy job"
    step = run_steps[0]
    script = step["run"]
    env = step["env"]

    assert "AUTHOR_ASSOC" not in env, "author_association must not drive trust"
    assert "AUTHOR_ASSOC" not in script
    assert "collaborators/${PR_AUTHOR}/permission" in script
    assert "admin|maintain|write" in script
    assert 'is_fork="false"' in script  # trusted branch
    assert 'is_fork="true"' in script  # fail-closed / external branch
    assert "PR_AUTHOR" in env, "PR_AUTHOR must be plumbed via env:"
    assert "pull_request.user.login" in env["PR_AUTHOR"]

    gpu_job = wf["jobs"]["gpu-smoke"]
    assert "needs.security-policy.outputs.is_fork" in str(gpu_job["runs-on"])
    ref_step = next(
        s for s in gpu_job["steps"] if (s.get("name") or "").startswith("Checkout trusted actions")
    )
    assert "needs.security-policy.outputs.is_fork" in str(ref_step["with"]["ref"])


# gpu-smoke targeted test selection


def _policy_script() -> str:
    import yaml

    wf = yaml.safe_load(GPU_SMOKE_WORKFLOW.read_text())
    policy_job = wf["jobs"]["security-policy"]
    step = next(s for s in policy_job["steps"] if s.get("id") == "policy")
    return step["run"]


def _targeted_arm_patterns(script: str) -> str:
    """The `case` arm that turns a changed test file into a targeted pytest target."""
    import re

    arms = re.findall(r"^\s*(tests/ops[^)\n]*)\)\s*$", script, re.M)
    assert len(arms) == 1, f"expected one tests/ops case arm, found {arms}"
    return arms[0]


def _case_matches(patterns: str, path: str) -> bool:
    proc = subprocess.run(
        ["bash", "-c", f'case "$1" in {patterns}) exit 0 ;; *) exit 1 ;; esac', "_", path],
        check=False,
    )
    return proc.returncode == 0


def test_every_op_test_file_reaches_the_targeted_arm() -> None:
    """A file the arm misses falls through to the catch-all and runs the whole suite,
    and the catch-all's reason reads like policy rather than a miss."""
    patterns = _targeted_arm_patterns(_policy_script())
    op_tests = sorted(p for p in (REPO_ROOT / "tests" / "ops").rglob("test_*.py"))
    assert op_tests, "expected test files under tests/ops"

    unmatched = [
        str(p.relative_to(REPO_ROOT))
        for p in op_tests
        if not _case_matches(patterns, str(p.relative_to(REPO_ROOT)))
    ]
    assert not unmatched, (
        f"case arm '{patterns}' does not select these op tests, so changing one of them "
        f"runs the whole suite instead: {unmatched}"
    )


def test_fork_fast_path_accepts_the_same_op_tests() -> None:
    """Otherwise a fork PR touching a nested op test is classified "outside the
    fast-path policy" rather than as the test-only change it is."""
    import re

    script = _policy_script()
    # The arm itself, not the whole script: a substring search also matches the
    # pattern quoted in a comment, and would pass over an arm that dropped it. The
    # `;;` requirement is what keeps a comment shaped like an arm from matching.
    fork_arms = [
        a
        for a in re.findall(
            r"^[ \t]*([^\s)#][^\s)]*ISSUE_TEMPLATE[^)\n]*)\)\n(?:.*\n)*?[ \t]*;;\s*$",
            script,
            re.M,
        )
        if "tests/ops" in a
    ]
    assert len(fork_arms) == 1, f"expected one fork fast-path arm, found {fork_arms}"
    listed = fork_arms[0].split("|")

    for pattern in _targeted_arm_patterns(script).split("|"):
        assert pattern in listed, (
            f"fork fast-path arm is missing '{pattern}'; the two arms must describe "
            "the same set of op test files"
        )


# preflight manifest gate

PREFLIGHT_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "preflight.yml"


def test_manifest_gate_covers_every_manifest_file() -> None:
    """`validate-manifest` is the only job that runs the validator and is gated on
    this arm, so a file the arm misses leaves no failing check to read — just an
    absent one."""
    import re

    import yaml

    wf = yaml.safe_load(PREFLIGHT_WORKFLOW.read_text())
    step = next(
        s for s in wf["jobs"]["detect-changes"]["steps"] if "manifest=false" in (s.get("run") or "")
    )
    arms = re.findall(r"^\s*(src/tileops/manifest[^)\n]*)\)\s*$", step["run"], re.M)
    assert len(arms) == 1, f"expected one manifest case arm, found {arms}"

    tracked = subprocess.run(
        ["git", "ls-files", "src/tileops/manifest"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert tracked, "expected tracked files under src/tileops/manifest"

    ungated = [f for f in tracked if not _case_matches(arms[0], f)]
    assert not ungated, (
        f"case arm '{arms[0]}' does not gate these manifest files, so changing one of "
        f"them skips validate-manifest: {ungated}"
    )


# runner maintenance vs nightly

MAINTENANCE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "runner-maintenance.yml"
NIGHTLY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "nightly.yml"


def test_reclaim_defers_to_an_in_flight_nightly() -> None:
    """A concurrent trim deletes work nightly is doing: `sentinel-repair` cannot tell
    an autotune run mid-write from a crashed one. Nightly may run to its declared
    ceiling, so the separation has to be a guard, and it has to fail closed."""
    import yaml

    maint = yaml.safe_load(MAINTENANCE_WORKFLOW.read_text())
    nightly = yaml.safe_load(NIGHTLY_WORKFLOW.read_text())
    # `on` is the YAML 1.1 boolean `true` once parsed.
    assert "schedule" not in (nightly.get(True) or nightly["on"]), (
        "a nightly cron would have to be separated from the reclaim cron again"
    )

    guard = next(s for s in maint["jobs"]["nightly-guard"]["steps"] if s.get("id") == "check")
    reclaim = maint["jobs"]["reclaim-disk"]
    assert "nightly-guard" in str(reclaim["needs"])
    assert "nightly_active" in reclaim["if"]
    # The manual dispatch is the escape hatch for a disk-full runner, so it runs
    # anyway — but not silently.
    assert "github.event_name != 'schedule'" in reclaim["if"]
    assert "::warning::manual dispatch overrides the nightly guard" in guard["run"]

    assert 'nightly_active=true" >> "$GITHUB_OUTPUT"' in guard["run"], (
        "guard must be able to report an active nightly"
    )
    assert '-z "$active"' in guard["run"], "guard must fail closed when the query fails"


# gpu-smoke required-check evaluation


def _required_check_loop() -> str:
    """The loop that turns preflight check-runs into pending / failed / satisfied."""
    script = _policy_script_of("ci-prereq", "check")
    start = script.index('for check_name in "${required_checks[@]}"; do')
    # YAML strips the block scalar's common indentation, so the loop's own
    # indentation is whatever survives, not the column it occupies in the file.
    indent = script[:start].rpartition("\n")[2]
    end = script.index(f"\n{indent}done", start)
    return script[start:end]


def _policy_script_of(job: str, step_id: str) -> str:
    import yaml

    wf = yaml.safe_load(GPU_SMOKE_WORKFLOW.read_text())
    step = next(s for s in wf["jobs"][job]["steps"] if s.get("id") == step_id)
    return step["run"]


def _evaluate_checks(runs: list[tuple[str, str, str, int]]) -> tuple[str, str]:
    """Run the workflow's own loop over synthetic check-runs."""
    import json

    checks_json = json.dumps(
        {"check_runs": [{"name": n, "status": s, "conclusion": c, "id": i} for n, s, c, i in runs]}
    )
    # Every verdict in the loop comes out of jq, so a jq that is missing or errors
    # yields empty strings — and empty reads as "no success, not completed", a
    # plausible-looking pending rather than a failure. set -e plus the guard below
    # turn that into an error instead of a wrong answer.
    # check_ever_succeeded is the workflow's one call out to the API; stubbing it
    # against the same synthetic runs keeps the loop text under test verbatim.
    harness = f"""
    set -euo pipefail
    all_runs={json.dumps(checks_json)}
    latest_checks=$(echo "$all_runs" | jq '.check_runs | group_by(.name) | map(max_by(.id))')
    [[ -n "$latest_checks" ]]
    check_ever_succeeded() {{
      local hits
      hits=$(echo "$all_runs" | jq -r --arg name "$1" \
        '[.check_runs[] | select(.name == $name and .conclusion == "success")] | length')
      [[ -n "$hits" && "$hits" -gt 0 ]]
    }}
    required_checks=(pre-commit gitleaks actionlint)
    pending="false"
    failed=""
    {_required_check_loop()}
    done
    echo "$pending $failed"
    """
    out = subprocess.run(
        ["bash", "-c", harness], capture_output=True, text=True, check=True
    ).stdout.split()
    return out[0], (out[1] if len(out) > 1 else "")


@pytest.mark.skipif(shutil.which("jq") is None, reason="executes the workflow's jq calls")
def test_a_skipped_check_does_not_outrank_an_earlier_success() -> None:
    """Blocking on a skipped run burned the 900s deadline and skipped GPU smoke with
    "Timed out" in the log, so a PR could reach ready with GPU smoke never having
    run. A skip with no prior success must still fail closed."""
    ok = [("pre-commit", "completed", "success", 10), ("gitleaks", "completed", "success", 11)]
    third = ("actionlint", "completed", "success", 12)

    skips = [
        ("pre-commit", "completed", "skipped", 20),
        ("gitleaks", "completed", "skipped", 21),
        ("actionlint", "completed", "skipped", 22),
    ]
    assert _evaluate_checks([*ok, third, *skips]) == ("false", "")
    assert _evaluate_checks(skips) == ("true", "")

    assert _evaluate_checks(ok) == ("true", "")  # actionlint absent
    assert _evaluate_checks([("pre-commit", "in_progress", "", 20), ok[1], third]) == ("true", "")
    assert _evaluate_checks([("pre-commit", "completed", "failure", 20), ok[1], third]) == (
        "false",
        "pre-commit",
    )
    # A newer failure is the current state of a required check; an older success on
    # the same SHA must not talk the gate into spending GPU time.
    assert _evaluate_checks([*ok, third, ("pre-commit", "completed", "failure", 20)]) == (
        "false",
        "pre-commit",
    )
