"""Unit tests for .github/actions/reclaim-runner-disk/reclaim_cache.sh.

Covers the sentinel-repair + atomic-trim primitives that protect caches
whose consumers assume "directory exists => contents complete" (the
tilelang autotuner cache in particular). See issue #989.

Required cases per milestone plan:
  - half_dead       : atomic subdir with files but no sentinel is removed
                      on a single invocation (AC-2, AC-3).
  - atomic_stale    : atomic subdir whose newest file is older than
                      cache-age-days is removed whole-directory.
  - atomic_fresh    : fresh atomic subdirs are preserved.
  - invariant       : atomic roots never have their *individual files*
                      trimmed, even when file-level trim runs (AC-4).

Runs on every PR (smoke tier), so does not depend on a self-hosted
runner or the Tilelang runtime. Must stay fast and hermetic.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parent.parent
# Script is colocated with the composite action so the gpu-smoke
# `.trusted/.github/actions` sparse-checkout picks it up; see action.yml.
RECLAIM_SCRIPT = REPO_ROOT / ".github" / "actions" / "reclaim-runner-disk" / "reclaim_cache.sh"


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


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
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=merged_env, check=False
    )
    assert result.returncode == 0, (
        f"{cmd!r} exited {result.returncode}\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
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


# ---------------------------------------------------------------------------
# sentinel-repair
# ---------------------------------------------------------------------------


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


def test_sentinel_repair_tolerates_missing_root(tmp_path: Path) -> None:
    """A non-existent cache root must be a no-op, not an error."""
    missing = tmp_path / "does-not-exist"
    _run("sentinel-repair", str(missing))  # asserts rc==0


def test_sentinel_repair_honours_custom_sentinel_filename(tmp_path: Path) -> None:
    """The sentinel filename is overridable via $SENTINEL_FILENAME (used in tests)."""
    root = tmp_path / "autotuner"
    subdir = _make_autotuner_subdir(root, "sig", with_sentinel=False, sentinel="SENTINEL")
    (subdir / "SENTINEL").write_text("ok\n")

    _run("sentinel-repair", str(root), env={"SENTINEL_FILENAME": "SENTINEL"})

    assert subdir.exists(), "subdir with the custom sentinel must be preserved"


def test_no_half_dead_after_reclaim(tmp_path: Path) -> None:
    """AC-2: after a full reclaim sequence no autotuner subdir is left
    in the half-dead state (exists but missing best_config.json)."""
    root = tmp_path / "autotuner"
    _make_autotuner_subdir(root, "halfdead_a", with_sentinel=False)
    _make_autotuner_subdir(root, "halfdead_b", with_sentinel=False)
    _make_autotuner_subdir(root, "healthy", with_sentinel=True)

    # Full reclaim order: sentinel-repair → atomic-trim.
    _run("sentinel-repair", str(root))
    _run("atomic-trim", "7", str(root))

    for subdir in root.iterdir():
        if subdir.is_dir():
            assert (subdir / "best_config.json").exists(), (
                f"{subdir} is half-dead after reclaim"
            )


# ---------------------------------------------------------------------------
# atomic-trim
# ---------------------------------------------------------------------------


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
    assert (subdir / "kernel.cu").exists(), "individual files inside atomic root must never be trimmed"
    assert (subdir / "kernel.so").exists()
    assert (subdir / "best_config.json").exists()


def test_atomic_trim_tolerates_missing_root(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    _run("atomic-trim", "7", str(missing))


def test_atomic_trim_handles_empty_root(tmp_path: Path) -> None:
    root = tmp_path / "autotuner"
    root.mkdir()
    _run("atomic-trim", "7", str(root))
    assert root.exists(), "the cache root itself is never removed"


# ---------------------------------------------------------------------------
# trim-files (non-atomic roots, legacy behaviour)
# ---------------------------------------------------------------------------


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


def test_trim_files_tolerates_missing_root(tmp_path: Path) -> None:
    _run("trim-files", "7", str(tmp_path / "nope"))
