"""Contract tests for the runner-disk reclaim script and the CI workflows.

``.github/actions/reclaim-runner-disk/reclaim_cache.sh`` must never leave a cache
directory whose consumer reads "directory exists" as "contents complete" (the TileLang
autotuner cache) half-trimmed. The workflow tests pin what the workflows select, which
actionlint cannot see; preflight runs this file without torch, so it needs only pytest,
PyYAML and jq.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest
import yaml

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
    """A subdir missing best_config.json is removed."""
    root = tmp_path / "autotuner"
    half_dead = _make_autotuner_subdir(root, "halfdead_sig", with_sentinel=False)
    healthy = _make_autotuner_subdir(root, "healthy_sig", with_sentinel=True)

    _run("sentinel-repair", str(root))

    assert not half_dead.exists(), "half-dead subdir should have been removed"
    assert healthy.exists(), "healthy subdir with sentinel must be preserved"
    assert (healthy / "best_config.json").exists()


def test_all_subcommands_tolerate_missing_root(tmp_path: Path) -> None:
    """A non-existent cache root is a no-op, not an error, for every subcommand."""
    missing = tmp_path / "does-not-exist"
    _run("sentinel-repair", str(missing))
    _run("atomic-trim", "7", str(missing))
    _run("trim-files", "7", str(missing))


def test_sentinel_repair_honours_custom_sentinel_filename(tmp_path: Path) -> None:
    """$SENTINEL_FILENAME overrides the sentinel filename."""
    root = tmp_path / "autotuner"
    subdir = _make_autotuner_subdir(root, "sig", with_sentinel=False, sentinel="SENTINEL")
    (subdir / "SENTINEL").write_text("ok\n")

    _run("sentinel-repair", str(root), env={"SENTINEL_FILENAME": "SENTINEL"})

    assert subdir.exists(), "subdir with the custom sentinel must be preserved"


# atomic-trim


def test_atomic_trim_removes_stale_subdir_whole(tmp_path: Path) -> None:
    """A subdir whose newest file is older than the cutoff is removed as a whole."""
    root = tmp_path / "autotuner"
    stale = _make_autotuner_subdir(root, "stale_sig", with_sentinel=True)
    _age_path(stale, days=30)

    _run("atomic-trim", "7", str(root))

    assert not stale.exists(), "stale atomic subdir must be removed whole-directory"


def test_atomic_trim_never_trims_individual_files(tmp_path: Path) -> None:
    """The unit of deletion is the whole subdirectory: a subdir holding any file within
    the age window is kept with every file, old ones included."""
    root = tmp_path / "autotuner"
    subdir = _make_autotuner_subdir(root, "mixed_sig", with_sentinel=True)
    # Age just the kernel files, leave best_config.json fresh.
    _age_path(subdir / "kernel.cu", days=30)
    _age_path(subdir / "kernel.so", days=30)

    _run("atomic-trim", "7", str(root))

    assert (subdir / "kernel.cu").exists()
    assert (subdir / "kernel.so").exists()
    assert (subdir / "best_config.json").exists()


def test_atomic_trim_uses_file_mtime_not_dir_mtime(tmp_path: Path) -> None:
    """Directory mtime does not make a stale subdir look fresh: a cache restore sets the
    subdir's own mtime to now while every file inside keeps its old one."""
    root = tmp_path / "autotuner"
    root.mkdir()
    stale = root / "deadbeef"
    stale.mkdir()
    (stale / "best_config.json").write_text("{}")
    (stale / "kernel.so").write_text("x")

    past = time.time() - 30 * 86400
    for entry in stale.iterdir():
        os.utime(entry, (past, past))
    os.utime(stale, (time.time(), time.time()))

    _run("atomic-trim", "7", str(root))

    assert not stale.exists()


def test_atomic_trim_handles_empty_root(tmp_path: Path) -> None:
    root = tmp_path / "autotuner"
    root.mkdir()
    _run("atomic-trim", "7", str(root))
    assert root.exists(), "the cache root itself is never removed"


# trim-files (non-atomic roots)


def test_trim_files_removes_old_files_but_leaves_atomic_roots_alone(
    tmp_path: Path,
) -> None:
    """File-level trim touches only the roots passed in: an autotuner root left out of
    the list keeps its ancient files."""
    triton_root = tmp_path / "triton-cache"
    triton_root.mkdir()
    old_file = triton_root / "old.bin"
    old_file.write_bytes(b"\x00")
    _age_path(old_file, days=30)
    fresh_file = triton_root / "fresh.bin"
    fresh_file.write_bytes(b"\x01")

    autotuner_root = tmp_path / "autotuner"
    subdir = _make_autotuner_subdir(autotuner_root, "sig", with_sentinel=True)
    _age_path(subdir, days=30)

    _run("trim-files", "7", str(triton_root))

    assert not old_file.exists()
    assert fresh_file.exists()
    assert subdir.exists()
    assert (subdir / "best_config.json").exists()
    assert (subdir / "kernel.cu").exists()


# gpu-smoke security trust routing

GPU_SMOKE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "gpu-smoke.yml"


def test_security_policy_routes_trust_by_collaborator_permission() -> None:
    """is_fork derives from the PR author's collaborator permission
    (write/maintain/admin -> trusted), fails closed to the fork pool, and
    drives both runs-on and the trusted-action ref."""
    wf = yaml.safe_load(GPU_SMOKE_WORKFLOW.read_text())
    step = _step(wf, "security-policy", "policy")
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


def _step(workflow: dict, job: str, step_id: str) -> dict:
    return next(s for s in workflow["jobs"][job]["steps"] if s.get("id") == step_id)


def _policy_script(job: str = "security-policy", step_id: str = "policy") -> str:
    return _step(yaml.safe_load(GPU_SMOKE_WORKFLOW.read_text()), job, step_id)["run"]


def _targeted_arm_patterns(script: str) -> str:
    """The `case` arm that turns a changed test file into a targeted pytest target."""
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


def test_every_native_source_forces_gpu_smoke() -> None:
    """Kernels compile `src/tileops/csrc` in at build time; a file the arm misses
    reads as a non-Python change and skips GPU smoke."""
    arms = re.findall(r"^\s*(src/tileops/csrc[^)\n]*)\)\s*$", _policy_script(), re.M)
    assert len(arms) == 1, f"expected one src/tileops/csrc case arm, found {arms}"

    # Read off the tree, not `git ls-files`: a test that needs a repository
    # cannot run from an installed package or an exported source archive, and
    # the question is which files exist, which the directory answers.
    root = REPO_ROOT / "src" / "tileops" / "csrc"
    sources = sorted(str(p.relative_to(REPO_ROOT)) for p in root.rglob("*") if p.is_file())
    assert sources, "expected native sources under src/tileops/csrc"

    ungated = [f for f in sources if not _case_matches(arms[0], f)]
    assert not ungated, f"case arm '{arms[0]}' misses these native sources: {ungated}"


# preflight manifest validator

PREFLIGHT_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "preflight.yml"


def test_the_manifest_validator_runs_on_every_pr() -> None:
    """`validate-manifest` is the only run of the validator over the real manifest, and the
    validator imports every op family and reads every benchmark file, so no path filter can
    say which PRs leave it unaffected."""
    job = yaml.safe_load(PREFLIGHT_WORKFLOW.read_text())["jobs"]["validate-manifest"]
    needs = job.get("needs") or []
    assert "detect-changes" not in ([needs] if isinstance(needs, str) else needs)
    assert "detect-changes" not in job.get("if", "")


# runner maintenance vs nightly

MAINTENANCE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "runner-maintenance.yml"
NIGHTLY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "nightly.yml"


def test_reclaim_defers_to_an_in_flight_nightly() -> None:
    """A concurrent trim deletes work nightly is doing: `sentinel-repair` cannot tell
    an autotune run mid-write from a crashed one. Nightly may run to its declared
    ceiling, so the separation has to be a guard, and it has to fail closed."""
    maint = yaml.safe_load(MAINTENANCE_WORKFLOW.read_text())
    nightly = yaml.safe_load(NIGHTLY_WORKFLOW.read_text())
    # `on` is the YAML 1.1 boolean `true` once parsed.
    assert "schedule" not in (nightly.get(True) or nightly["on"]), (
        "a nightly cron needs its own separation from the reclaim cron"
    )

    guard = _step(maint, "nightly-guard", "check")
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
    script = _policy_script("ci-prereq", "check")
    start = script.index('for check_name in "${required_checks[@]}"; do')
    # YAML strips the block scalar's common indentation, so the loop's own
    # indentation is whatever survives, not the column it occupies in the file.
    indent = script[:start].rpartition("\n")[2]
    end = script.index(f"\n{indent}done", start)
    return script[start:end]


def _evaluate_checks(runs: list[tuple[str, str, str, int]]) -> tuple[str, str]:
    """Run the workflow's own loop over synthetic check-runs."""
    checks_json = json.dumps(
        {"check_runs": [{"name": n, "status": s, "conclusion": c, "id": i} for n, s, c, i in runs]}
    )
    # Every verdict in the loop comes out of jq, so a jq that is missing or errors
    # yields empty strings — and empty reads as "no success, not completed", a
    # plausible-looking pending rather than a failure. set -e plus the guard below
    # turn that into an error instead of a wrong answer.
    # check_ever_succeeded is the workflow's one call out to the API; stubbing it
    # against the same synthetic runs keeps the loop text under test verbatim.
    script = f"""
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
        ["bash", "-c", script], capture_output=True, text=True, check=True
    ).stdout.split()
    return out[0], (out[1] if len(out) > 1 else "")


@pytest.mark.skipif(shutil.which("jq") is None, reason="executes the workflow's jq calls")
def test_a_skipped_check_does_not_outrank_an_earlier_success() -> None:
    """A skipped run after a success leaves the check satisfied rather than pending until
    the deadline; a skip with no earlier success stays pending."""
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
