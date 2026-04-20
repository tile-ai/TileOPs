"""Structural tests for the gpu-smoke / runner-maintenance CI wiring.

These tests do NOT spin up a self-hosted runner — they parse the workflow
YAML and assert the contract documented in the acceptance criteria:

* AC-3: the ``security-policy`` job downgrades ``is_fork`` to ``false`` for
  OWNER/MEMBER/COLLABORATOR authors on head != base PRs.
* AC-4: the ``gpu-smoke`` job passes ``skip-atomic-age-trim: "true"`` when
  invoking the ``reclaim-runner-disk`` composite action.
* AC-5: the daily ``runner-maintenance.yml`` job does NOT pass
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
# AC-3: member-fork demotion
# ---------------------------------------------------------------------------


def test_security_policy_handles_member_fork_as_trusted() -> None:
    """AC-3: PRs from OWNER/MEMBER/COLLABORATOR authors on a fork repo
    must be treated as trusted (is_fork=false) by the security-policy
    step so the gpu-smoke job uses the /home/ci-runner trusted runtime
    layout."""
    wf = _load(GPU_SMOKE)
    policy_job = wf["jobs"]["security-policy"]
    run_steps = [s for s in policy_job["steps"] if "run" in s and s.get("id") == "policy"]
    assert run_steps, "expected a 'policy' step in security-policy job"
    script = run_steps[0]["run"]

    # The promotion logic must name all three trusted author_association
    # values, and must key off AUTHOR_ASSOC in the head != base branch.
    assert "AUTHOR_ASSOC" in script, "AUTHOR_ASSOC env must be referenced"
    for assoc in ("OWNER", "MEMBER", "COLLABORATOR"):
        assert assoc in script, f"author_association {assoc!r} must be handled"
    # Must actually set is_fork=false in the trusted branch.
    assert 'is_fork="false"' in script
    # And AUTHOR_ASSOC must be sourced from github.event.pull_request.author_association.
    env = run_steps[0]["env"]
    assert "AUTHOR_ASSOC" in env, "AUTHOR_ASSOC must be plumbed via env:"
    assert "author_association" in env["AUTHOR_ASSOC"]


# ---------------------------------------------------------------------------
# AC-4: gpu-smoke opts out of atomic age-trim
# ---------------------------------------------------------------------------


def test_gpu_smoke_invokes_reclaim_with_skip_atomic_age_trim() -> None:
    wf = _load(GPU_SMOKE)
    steps = wf["jobs"]["gpu-smoke"]["steps"]
    reclaim_step = _find_step(steps, uses_contains="reclaim-runner-disk")
    with_ = reclaim_step.get("with") or {}
    assert str(with_.get("skip-atomic-age-trim")).lower() == "true", (
        "AC-4: gpu-smoke must pass skip-atomic-age-trim: true so the daily "
        "maintenance job is the only place autotuner subdirs get evicted."
    )


# ---------------------------------------------------------------------------
# AC-5: daily maintenance preserves full-trim behaviour
# ---------------------------------------------------------------------------


def test_runner_maintenance_still_runs_full_atomic_trim() -> None:
    wf = _load(RUNNER_MAINT)
    steps = wf["jobs"]["reclaim-disk"]["steps"]
    reclaim_step = _find_step(steps, uses_contains="reclaim-runner-disk")
    with_ = reclaim_step.get("with") or {}
    # Absent OR explicitly "false". Presence of "true" would regress AC-5.
    value = with_.get("skip-atomic-age-trim", "false")
    assert str(value).lower() == "false", (
        "AC-5: runner-maintenance.yml must not opt out of the atomic age-trim "
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


def test_fork_divergent_pyproject_forces_fresh_install() -> None:
    """Fork PRs must not reuse the trusted venv when the fork's workspace
    pyproject.toml diverges from the trusted base-repo pyproject.toml.

    The trusted venv is keyed by the base-repo pyproject hash. If a fork
    edits `[project].dependencies` (or any other hashed section) in its own
    workspace pyproject, copying the trusted venv would install a package
    set that silently disagrees with the fork's `.[dev]`. The Resolve
    runtime state step must re-run the (trusted) hash script against the
    fork's workspace pyproject and gate FORK_CAN_COPY_TRUSTED_VENV on the
    two hashes being equal.
    """
    wf = _load(GPU_SMOKE)
    steps = wf["jobs"]["gpu-smoke"]["steps"]
    resolve_step = next(
        (s for s in steps if s.get("name") == "Resolve runtime state"), None
    )
    assert resolve_step is not None, "Resolve runtime state step must exist"
    script = resolve_step["run"]

    # Must run the trusted hash script against the workspace pyproject.
    assert "python .trusted/scripts/ci_venv_hash.py ./pyproject.toml" in script, (
        "Workspace pyproject must be hashed via the trusted script so the "
        "fork cannot substitute its own ci_venv_hash.py."
    )
    # Must gate the copy fast-path on hashes matching.
    assert "WORKSPACE_PYPROJECT_HASH" in script
    assert 'WORKSPACE_PYPROJECT_HASH}" == "${PYPROJECT_HASH}' in script, (
        "FORK_CAN_COPY_TRUSTED_VENV must require workspace hash == trusted hash."
    )


def test_reclaim_action_emits_opt_out_log_line() -> None:
    """When the opt-out is active, operators need a grep-able log line to
    confirm the destructive path was skipped. The AC explicitly names this
    string ('Skipping atomic age-trim (opted out)')."""
    text = RECLAIM_ACTION.read_text()
    assert "Skipping atomic age-trim (opted out)" in text
