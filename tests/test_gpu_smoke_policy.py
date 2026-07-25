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
