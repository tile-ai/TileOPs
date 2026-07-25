"""Structural tests for the gpu-smoke / runner-maintenance CI wiring.

These tests do NOT spin up a self-hosted runner — they parse the workflow
YAML and assert the contract:

* The ``security-policy`` job derives ``is_fork`` from the PR author's
  collaborator permission (write/maintain/admin -> trusted), failing closed
  to the fork pool on any lookup failure.
* The ``gpu-smoke`` job passes ``skip-atomic-age-trim: "true"`` when
  invoking the ``reclaim-runner-disk`` composite action.
* The daily ``runner-maintenance.yml`` job does NOT pass
  ``skip-atomic-age-trim``, so the full destructive trim still runs there.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parent.parent
GPU_SMOKE = REPO_ROOT / ".github" / "workflows" / "gpu-smoke.yml"
RUNNER_MAINT = REPO_ROOT / ".github" / "workflows" / "runner-maintenance.yml"
RECLAIM_ACTION = REPO_ROOT / ".github" / "actions" / "reclaim-runner-disk" / "action.yml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _find_step(steps: list[dict], *, uses_contains: str) -> dict:
    for step in steps:
        if uses_contains in (step.get("uses") or ""):
            return step
    raise AssertionError(f"no step with uses containing {uses_contains!r}")


# ---------------------------------------------------------------------------
# Member-fork demotion
# ---------------------------------------------------------------------------


def test_security_policy_routes_trust_by_collaborator_permission() -> None:
    """The security-policy step must derive is_fork from the PR author's collaborator
    permission (write/maintain/admin -> trusted), NOT author_association, and fail closed
    to the fork pool on any lookup failure. The same is_fork drives runs-on and the
    trusted-action ref selection."""
    wf = _load(GPU_SMOKE)
    policy_job = wf["jobs"]["security-policy"]
    run_steps = [s for s in policy_job["steps"] if "run" in s and s.get("id") == "policy"]
    assert run_steps, "expected a 'policy' step in security-policy job"
    step = run_steps[0]
    script = step["run"]
    env = step["env"]

    # Trust is keyed off the collaborator-permission endpoint, not author_association.
    assert "AUTHOR_ASSOC" not in env, "author_association must no longer drive trust"
    assert "AUTHOR_ASSOC" not in script
    assert "collaborators/${PR_AUTHOR}/permission" in script, (
        "is_fork must be derived from the collaborator-permission endpoint"
    )
    # Only write/maintain/admin are trusted; the catch-all fails closed to the fork pool.
    assert "admin|maintain|write" in script
    assert 'is_fork="false"' in script  # trusted branch
    assert 'is_fork="true"' in script  # fail-closed / external branch
    # PR_AUTHOR must be plumbed from the PR author login.
    assert "PR_AUTHOR" in env, "PR_AUTHOR must be plumbed via env:"
    assert "pull_request.user.login" in env["PR_AUTHOR"]

    # runs-on and the trusted-action ref must both consume the same is_fork output.
    gpu_job = wf["jobs"]["gpu-smoke"]
    assert "needs.security-policy.outputs.is_fork" in str(gpu_job["runs-on"])
    ref_step = next(
        s for s in gpu_job["steps"] if (s.get("name") or "").startswith("Checkout trusted actions")
    )
    assert "needs.security-policy.outputs.is_fork" in str(ref_step["with"]["ref"])


# ---------------------------------------------------------------------------
# Diff-scoped full-tier PR gate
# ---------------------------------------------------------------------------


def _policy_step(wf: dict) -> dict:
    policy_job = wf["jobs"]["security-policy"]
    return next(s for s in policy_job["steps"] if s.get("id") == "policy")


def _run_tests_step(wf: dict) -> dict:
    steps = wf["jobs"]["gpu-smoke"]["steps"]
    return next(s for s in steps if s.get("name") == "Run tests")


def test_security_policy_exports_full_tier_targets() -> None:
    """security-policy must compute the changed test files (any tests/**/test_*.py
    in the PR diff) and export them as the full_tier_targets job output so the
    gpu-smoke job can run them under `-m "smoke or full"`."""
    wf = _load(GPU_SMOKE)
    outputs = wf["jobs"]["security-policy"]["outputs"]
    assert "full_tier_targets" in outputs
    assert "steps.policy.outputs.full_tier_targets" in outputs["full_tier_targets"]

    script = _policy_step(wf)["run"]
    # Matches test files at any depth under tests/, by basename prefix, so
    # helper modules (e.g. *_test_utils.py) are not selected.
    assert "tests/*/test_*.py|tests/test_*.py" in script
    assert "full_tier_targets=" in script


def test_security_policy_sparse_checkout_covers_tests_tree() -> None:
    """The policy job existence-checks changed test files before selecting them;
    the sparse checkout must therefore cover the whole tests/ tree, not just
    tests/ops."""
    wf = _load(GPU_SMOKE)
    policy_job = wf["jobs"]["security-policy"]
    checkout = next(s for s in policy_job["steps"] if "checkout" in (s.get("uses") or ""))
    sparse = checkout["with"]["sparse-checkout"]
    assert "tests" in sparse.split()
    assert "tests/ops" not in sparse.split()


def test_pr_gate_runs_full_tier_on_changed_test_files() -> None:
    """A PR touching a test file must execute that file's full-tier cases:
    targeted scope promotes the single pass to `-m "smoke or full"`; full-smoke
    scope runs a dedicated diff-scoped pass and excludes those files from the
    smoke pass."""
    wf = _load(GPU_SMOKE)
    step = _run_tests_step(wf)
    env = step["env"]
    assert "FULL_TIER_TARGETS" in env
    assert "needs.security-policy.outputs.full_tier_targets" in env["FULL_TIER_TARGETS"]

    script = step["run"]
    # Diff-scoped pass over the changed test files.
    assert 'pytest -q "${FULL_TARGETS[@]}" -m "smoke or full"' in script
    assert "--junit-xml=gpu_smoke_full_results.xml" in script
    # Changed files are excluded from the smoke pass (no double execution).
    assert '--ignore=${full_target}' in script
    # Targeted scope (targets == changed test files) promotes in place.
    assert '"$TEST_SCOPE" == "targeted"' in script


def test_pr_gate_residual_gap_documented() -> None:
    """The accepted residual gap (cross-file cross-tier interactions still only
    caught on push-to-main) must be documented in the workflow itself."""
    script = _run_tests_step(_load(GPU_SMOKE))["run"]
    assert "Residual gap" in script


def test_push_tier_selection_unchanged() -> None:
    """Push-to-main keeps `-m "smoke or full"`; only pull_request downgrades the
    baseline pass to smoke."""
    script = _run_tests_step(_load(GPU_SMOKE))["run"]
    assert 'TEST_MARK_EXPR="smoke or full"' in script
    assert 'TEST_MARK_EXPR="smoke"' in script


# ---------------------------------------------------------------------------
# Diff-scoped full-tier PR gate: runtime behaviour
#
# These tests EXECUTE the workflow's "Run tests" shell block with a stubbed
# python3 on PATH (logging every invocation), so they verify what actually
# runs — not just what the YAML text contains.
# ---------------------------------------------------------------------------

_PY3_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$CALL_LOG"
cat > /dev/null
case "$*" in
  *gpu_smoke_full_results.xml*) exit "${FULL_PASS_EXIT:-0}" ;;
esac
exit 0
"""


def _simulate_run_tests(
    tmp_path: Path,
    *,
    event: str,
    scope: str,
    pytest_targets: str = "tests",
    full_targets: str = "",
    full_pass_exit: str = "0",
) -> tuple[subprocess.CompletedProcess, list[str], str]:
    """Run the gpu-smoke "Run tests" script with python3 stubbed out.

    Returns (completed process, logged python3 invocations, GITHUB_ENV text).
    """
    script = _run_tests_step(_load(GPU_SMOKE))["run"]
    workdir = tmp_path / "work"
    workdir.mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "python3"
    stub.write_text(_PY3_STUB)
    stub.chmod(0o755)
    call_log = tmp_path / "calls.log"
    github_env = tmp_path / "github_env"
    github_env.touch()
    env = {
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "HOME": os.environ.get("HOME", str(tmp_path)),
        "CALL_LOG": str(call_log),
        "GITHUB_ENV": str(github_env),
        "FULL_PASS_EXIT": full_pass_exit,
        "EVENT_NAME": event,
        "TEST_SCOPE": scope,
        "PYTEST_TARGETS": pytest_targets,
        "FULL_TIER_TARGETS": full_targets,
    }
    proc = subprocess.run(
        ["bash", "-c", script],
        cwd=workdir,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )
    calls = call_log.read_text().splitlines() if call_log.exists() else []
    pytest_calls = [c for c in calls if "-m pytest" in c]
    return proc, pytest_calls, github_env.read_text()


def test_run_tests_executes_full_tier_for_changed_test_file(tmp_path: Path) -> None:
    """PR + full-smoke scope + a changed test file: the script must actually run
    a diff-scoped `-m "smoke or full"` pass on that file, then exclude it from
    the smoke pass. The baseline pass stays smoke (and is reported as such)."""
    proc, pytest_calls, github_env = _simulate_run_tests(
        tmp_path,
        event="pull_request",
        scope="full-smoke",
        full_targets="tests/test_changed.py",
    )
    assert proc.returncode == 0, proc.stderr
    full_passes = [c for c in pytest_calls if "gpu_smoke_full_results.xml" in c]
    assert len(full_passes) == 1
    assert "tests/test_changed.py" in full_passes[0]
    assert '-m smoke or full' in full_passes[0]
    baseline = [c for c in pytest_calls if "gpu_smoke_results.xml" in c]
    assert len(baseline) == 1
    assert "--ignore=tests/test_changed.py" in baseline[0]
    assert "smoke or full" not in baseline[0]
    assert "EXECUTED_MARK_EXPR=smoke\n" in github_env


def test_run_tests_targeted_scope_promotes_single_pass(tmp_path: Path) -> None:
    """PR + targeted scope (targets == the changed test files): one pass only,
    promoted to `-m "smoke or full"`, and the report label follows it."""
    proc, pytest_calls, github_env = _simulate_run_tests(
        tmp_path,
        event="pull_request",
        scope="targeted",
        pytest_targets="tests/test_changed.py",
        full_targets="tests/test_changed.py",
    )
    assert proc.returncode == 0, proc.stderr
    assert len(pytest_calls) == 1
    assert "-m smoke or full" in pytest_calls[0]
    assert "gpu_smoke_full_results.xml" not in pytest_calls[0]
    assert "EXECUTED_MARK_EXPR=smoke or full\n" in github_env


def test_run_tests_push_tier_runtime_unchanged(tmp_path: Path) -> None:
    """Push events run one `-m "smoke or full"` pass; the diff-scoped PR
    machinery must not trigger even if full_tier_targets leaked non-empty."""
    proc, pytest_calls, _ = _simulate_run_tests(
        tmp_path,
        event="push",
        scope="full-smoke",
        full_targets="tests/test_changed.py",
    )
    assert proc.returncode == 0, proc.stderr
    assert len(pytest_calls) == 1
    assert "-m smoke or full" in pytest_calls[0]
    assert "gpu_smoke_full_results.xml" not in pytest_calls[0]


def test_run_tests_diff_pass_tolerates_nothing_collected(tmp_path: Path) -> None:
    """pytest exit 5 (nothing collected) from the diff-scoped pass — e.g. the
    changed file holds only nightly cases — must not fail the gate; the smoke
    pass still runs."""
    proc, pytest_calls, _ = _simulate_run_tests(
        tmp_path,
        event="pull_request",
        scope="full-smoke",
        full_targets="tests/test_changed.py",
        full_pass_exit="5",
    )
    assert proc.returncode == 0, proc.stderr
    assert any("gpu_smoke_results.xml" in c for c in pytest_calls)


_GH_STUB = """#!/usr/bin/env bash
printf 'gh %s\\n' "$*" >> "$CALL_LOG"
case "$*" in
  */files*) printf '%s\\n' ${CHANGED_FILES} ;;
  */permission*) echo "write" ;;
esac
exit 0
"""


def test_policy_script_selects_changed_full_tier_files(tmp_path: Path) -> None:
    """Execute the actual security-policy shell block for a simulated PR diff:
    changed test files that exist at head must land in full_tier_targets;
    helper modules and deleted files must not; a shared tileops/ path keeps
    scope=full-smoke."""
    script = _policy_step(_load(GPU_SMOKE))["run"]
    workdir = tmp_path / "work"
    for rel in (
        "tests/ops/test_full_case.py",
        "tests/ops/attention_test_utils.py",
        "tests/conftest.py",
    ):
        path = workdir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    gh_stub = bindir / "gh"
    gh_stub.write_text(_GH_STUB)
    gh_stub.chmod(0o755)
    call_log = tmp_path / "calls.log"
    github_output = tmp_path / "github_output"
    github_output.touch()
    changed = (
        "tileops/foo.py tests/ops/test_full_case.py tests/ops/test_deleted.py "
        "tests/ops/attention_test_utils.py tests/conftest.py"
    )
    env = {
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "HOME": os.environ.get("HOME", str(tmp_path)),
        "CALL_LOG": str(call_log),
        "GITHUB_OUTPUT": str(github_output),
        "CHANGED_FILES": changed,
        "GH_TOKEN": "stub",
        "EVENT_NAME": "pull_request",
        "PR_NUMBER": "1",
        "HEAD_REPO": "owner/repo",
        "BASE_REPO": "owner/repo",
        "PR_AUTHOR": "dev",
    }
    proc = subprocess.run(
        ["bash", "-c", script],
        cwd=workdir,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    outputs = dict(
        line.split("=", 1) for line in github_output.read_text().splitlines() if "=" in line
    )
    assert outputs["scope"] == "full-smoke"
    assert outputs["skip_gpu_smoke"] == "false"
    assert outputs["full_tier_targets"] == "tests/ops/test_full_case.py"


def _simulate_report(
    tmp_path: Path, *, event: str, executed_mark: str | None
) -> list[str]:
    """Run the report step script with python3 stubbed; return its invocations."""
    wf = _load(GPU_SMOKE)
    steps = wf["jobs"]["gpu-smoke"]["steps"]
    report = next(s for s in steps if s.get("name") == "Generate gpu-smoke report")
    workdir = tmp_path / "report_work"
    workdir.mkdir()
    (workdir / "gpu_smoke_results.xml").touch()
    bindir = tmp_path / "report_bin"
    bindir.mkdir()
    stub = bindir / "python3"
    stub.write_text(_PY3_STUB)
    stub.chmod(0o755)
    call_log = tmp_path / "report_calls.log"
    env = {
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "HOME": os.environ.get("HOME", str(tmp_path)),
        "CALL_LOG": str(call_log),
        "EVENT_NAME": event,
    }
    if executed_mark is not None:
        env["EXECUTED_MARK_EXPR"] = executed_mark
    proc = subprocess.run(
        ["bash", "-c", report["run"]],
        cwd=workdir,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return call_log.read_text().splitlines() if call_log.exists() else []


def test_report_target_derived_from_executed_marker(tmp_path: Path) -> None:
    """The report step must label the tier from the marker the Run tests step
    actually executed (EXECUTED_MARK_EXPR via GITHUB_ENV), not re-derive it
    from the event name alone — a targeted PR promoted to "smoke or full"
    would otherwise be reported as plain smoke."""
    calls = _simulate_report(tmp_path, event="pull_request", executed_mark="smoke or full")
    report_calls = [c for c in calls if "gpu_smoke_report.py" in c]
    assert len(report_calls) == 1
    assert "--target smoke, full --output" in report_calls[0]


def test_report_target_falls_back_to_event_default(tmp_path: Path) -> None:
    """If Run tests died before resolving the tier, the report label falls back
    to the event default (smoke for PRs)."""
    calls = _simulate_report(tmp_path, event="pull_request", executed_mark=None)
    report_calls = [c for c in calls if "gpu_smoke_report.py" in c]
    assert len(report_calls) == 1
    assert "--target smoke --output" in report_calls[0]


# ---------------------------------------------------------------------------
# gpu-smoke opts out of atomic age-trim
# ---------------------------------------------------------------------------


def test_gpu_smoke_invokes_reclaim_with_skip_atomic_age_trim() -> None:
    wf = _load(GPU_SMOKE)
    steps = wf["jobs"]["gpu-smoke"]["steps"]
    reclaim_step = _find_step(steps, uses_contains="reclaim-runner-disk")
    with_ = reclaim_step.get("with") or {}
    assert str(with_.get("skip-atomic-age-trim")).lower() == "true", (
        "gpu-smoke must pass skip-atomic-age-trim: true so the daily "
        "maintenance job is the only place autotuner subdirs get evicted."
    )


# ---------------------------------------------------------------------------
# Daily maintenance preserves full-trim behaviour
# ---------------------------------------------------------------------------


def test_runner_maintenance_still_runs_full_atomic_trim() -> None:
    wf = _load(RUNNER_MAINT)
    steps = wf["jobs"]["reclaim-disk"]["steps"]
    reclaim_step = _find_step(steps, uses_contains="reclaim-runner-disk")
    with_ = reclaim_step.get("with") or {}
    # Absent OR explicitly "false". Presence of "true" would regress this
    # daily-maintenance contract.
    value = with_.get("skip-atomic-age-trim", "false")
    assert str(value).lower() == "false", (
        "runner-maintenance.yml must not opt out of the atomic age-trim "
        "pass — that daily job is what ultimately reclaims stale autotuner "
        "entries."
    )
    # And force-reclaim must still be on so it actually runs every day.
    assert str(with_.get("force-reclaim")).lower() == "true"


# ---------------------------------------------------------------------------
# Composite action: surface + semantics
# ---------------------------------------------------------------------------


def test_reclaim_action_declares_skip_atomic_age_trim_input() -> None:
    action = _load(RECLAIM_ACTION)
    inputs = action["inputs"]
    assert "skip-atomic-age-trim" in inputs, (
        "The skip-atomic-age-trim input must exist on the composite action "
        "so callers can opt out of the destructive trim."
    )
    assert str(inputs["skip-atomic-age-trim"]["default"]).lower() == "false", (
        "Default must be false so existing callers (runner-maintenance.yml) "
        "keep their full-trim behaviour without changes."
    )


def test_reclaim_action_emits_opt_out_log_line() -> None:
    """When the opt-out is active, operators need a grep-able log line to
    confirm the destructive path was skipped. The AC explicitly names this
    string ('Skipping atomic age-trim (opted out)')."""
    text = RECLAIM_ACTION.read_text()
    assert "Skipping atomic age-trim (opted out)" in text
