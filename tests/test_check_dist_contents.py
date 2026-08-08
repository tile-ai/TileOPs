"""Unit tests for ``scripts/ci/check_dist_contents.py``.

Each case assembles a synthetic repo root plus wheel/sdist archives in a tmp
directory, then runs the checker against them — never against the live repo
or a real build.
"""

import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK_SCRIPT = REPO_ROOT / "scripts" / "ci" / "check_dist_contents.py"

MANIFEST_YAMLS = ["attention.yaml", "gemm.yaml"]
HEADER = "tileops/kernels/moe/_atomic_helper.h"


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    manifest_dir = repo / "tileops" / "manifest"
    manifest_dir.mkdir(parents=True)
    for name in MANIFEST_YAMLS:
        (manifest_dir / name).write_text("ops: {}\n")
    header = repo / HEADER
    header.parent.mkdir(parents=True)
    header.write_text("// helper\n")
    (repo / "LICENSE").write_text("MIT\n")
    (repo / "README.md").write_text("# tileops\n")
    return repo


def make_wheel(dist_dir: Path, entries: list[str]) -> Path:
    dist_dir.mkdir(parents=True, exist_ok=True)
    wheel = dist_dir / "tileops-0.1.dev1+gabc1234-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        for entry in entries:
            zf.writestr(entry, "content\n")
    return wheel


def make_sdist(dist_dir: Path, entries: list[str]) -> Path:
    import io

    dist_dir.mkdir(parents=True, exist_ok=True)
    sdist = dist_dir / "tileops-0.1.dev1+gabc1234.tar.gz"
    with tarfile.open(sdist, "w:gz") as tf:
        for entry in entries:
            data = b"content\n"
            info = tarfile.TarInfo(f"tileops-0.1.dev1+gabc1234/{entry}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return sdist


WHEEL_OK = [f"tileops/manifest/{n}" for n in MANIFEST_YAMLS] + [HEADER, "tileops/__init__.py"]
SDIST_OK = (
    [f"tileops/manifest/{n}" for n in MANIFEST_YAMLS]
    + [HEADER, "LICENSE", "README.md", "pyproject.toml"]
)


def run_check(repo: Path, dist_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(CHECK_SCRIPT),
            "--repo-root",
            str(repo),
            "--dist-dir",
            str(dist_dir),
        ],
        capture_output=True,
        text=True,
    )


def test_complete_dist_passes(tmp_path):
    repo = make_repo(tmp_path)
    dist = tmp_path / "dist"
    make_wheel(dist, WHEEL_OK)
    make_sdist(dist, SDIST_OK)
    result = run_check(repo, dist)
    assert result.returncode == 0, result.stdout + result.stderr


def test_wheel_missing_manifest_yaml_fails(tmp_path):
    repo = make_repo(tmp_path)
    dist = tmp_path / "dist"
    incomplete = [e for e in WHEEL_OK if not e.endswith("gemm.yaml")]
    make_wheel(dist, incomplete)
    make_sdist(dist, SDIST_OK)
    result = run_check(repo, dist)
    assert result.returncode == 1
    assert "gemm.yaml" in result.stdout


def test_wheel_missing_header_fails(tmp_path):
    repo = make_repo(tmp_path)
    dist = tmp_path / "dist"
    make_wheel(dist, [e for e in WHEEL_OK if e != HEADER])
    make_sdist(dist, SDIST_OK)
    result = run_check(repo, dist)
    assert result.returncode == 1
    assert "_atomic_helper.h" in result.stdout


def test_sdist_missing_license_fails(tmp_path):
    repo = make_repo(tmp_path)
    dist = tmp_path / "dist"
    make_wheel(dist, WHEEL_OK)
    make_sdist(dist, [e for e in SDIST_OK if e != "LICENSE"])
    result = run_check(repo, dist)
    assert result.returncode == 1
    assert "LICENSE" in result.stdout


def test_sdist_with_pruned_directory_fails(tmp_path):
    repo = make_repo(tmp_path)
    dist = tmp_path / "dist"
    make_wheel(dist, WHEEL_OK)
    make_sdist(dist, SDIST_OK + ["tests/test_leak.py"])
    result = run_check(repo, dist)
    assert result.returncode == 1
    assert "tests/" in result.stdout


def test_missing_archives_fail(tmp_path):
    repo = make_repo(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    result = run_check(repo, dist)
    assert result.returncode == 1
