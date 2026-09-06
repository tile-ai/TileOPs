"""Unit tests for ``scripts/ci/check_dist_contents.py``.

Each case assembles a synthetic repo root plus wheel/sdist archives in a tmp
directory, then runs the checker against them — never against the live repo
or a real build. The fixture repo carries one resource of each kind the
package ships at runtime: a nested-package header, a hardware perf profile,
and manifest YAMLs.
"""

import io
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK_SCRIPT = REPO_ROOT / "scripts" / "ci" / "check_dist_contents.py"

# As they appear in the wheel; `_in_src` gives the repo and sdist form.
MANIFEST_YAMLS = ["tileops/manifest/attention.yaml", "tileops/manifest/gemm.yaml"]
NESTED_HEADER = "tileops/kernels/attention/_fp8_gqa_helper.h"
PERF_PROFILE = "tileops/perf/profiles/h200.yaml"

# Every tracked non-.py file the fixture package ships.
RESOURCES = [*MANIFEST_YAMLS, NESTED_HEADER, PERF_PROFILE]
SOURCES = ["tileops/__init__.py", "tileops/perf/profile.py"]


def _in_src(entries: list[str]) -> list[str]:
    return [f"src/{entry}" for entry in entries]


WHEEL_OK = [*RESOURCES, *SOURCES]
SDIST_OK = [
    *_in_src(RESOURCES),
    *_in_src(SOURCES),
    "LICENSE",
    "README.md",
    "CONTRIBUTING.md",
    "pyproject.toml",
]


def make_repo(tmp_path: Path) -> Path:
    """Write a source tree holding every fixture resource. Not a git repo."""
    repo = tmp_path / "repo"
    for rel in _in_src(RESOURCES + SOURCES):
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("content\n")
    (repo / "LICENSE").write_text("MIT\n")
    (repo / "README.md").write_text("# tileops\n")
    return repo


def make_wheel(dist_dir: Path, entries: list[str]) -> Path:
    dist_dir.mkdir(parents=True, exist_ok=True)
    wheel = dist_dir / "tileops-0.0.1.dev1+gabc1234-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        for entry in entries:
            zf.writestr(entry, "content\n")
    return wheel


def make_sdist(dist_dir: Path, entries: list[str]) -> Path:
    dist_dir.mkdir(parents=True, exist_ok=True)
    sdist = dist_dir / "tileops-0.0.1.dev1+gabc1234.tar.gz"
    with tarfile.open(sdist, "w:gz") as tf:
        for entry in entries:
            data = b"content\n"
            info = tarfile.TarInfo(f"tileops-0.0.1.dev1+gabc1234/{entry}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return sdist


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


def build_dist(tmp_path: Path, wheel_entries, sdist_entries) -> tuple[Path, Path]:
    repo = make_repo(tmp_path)
    dist = tmp_path / "dist"
    make_wheel(dist, wheel_entries)
    make_sdist(dist, sdist_entries)
    return repo, dist


def test_complete_dist_passes(tmp_path):
    repo, dist = build_dist(tmp_path, WHEEL_OK, SDIST_OK)
    result = run_check(repo, dist)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "dropped",
    [NESTED_HEADER, PERF_PROFILE, MANIFEST_YAMLS[1]],
    ids=["nested-package-header", "perf-profile", "manifest-yaml"],
)
def test_wheel_missing_any_resource_fails(tmp_path, dropped):
    """Every tracked non-.py resource is required, whatever its subpackage."""
    repo, dist = build_dist(tmp_path, [e for e in WHEEL_OK if e != dropped], SDIST_OK)
    result = run_check(repo, dist)
    assert result.returncode == 1
    assert dropped in result.stdout


def test_all_missing_resources_are_reported(tmp_path):
    """A wheel short several resources names each one, not just the first."""
    dropped = [NESTED_HEADER, PERF_PROFILE]
    repo, dist = build_dist(tmp_path, [e for e in WHEEL_OK if e not in dropped], SDIST_OK)
    result = run_check(repo, dist)
    assert result.returncode == 1
    for entry in dropped:
        assert entry in result.stdout


def test_expected_set_comes_from_git_when_available(tmp_path):
    """With git present the expectation follows the index, not the working tree."""
    repo, dist = build_dist(tmp_path, WHEEL_OK, SDIST_OK)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", *_in_src(RESOURCES), *_in_src(SOURCES)], cwd=repo, check=True)
    # Untracked, so it is not part of the shipped resource set.
    scratch = repo / "src" / "tileops" / "perf" / "profiles" / "scratch.yaml"
    scratch.write_text("draft\n")

    result = run_check(repo, dist)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "git" in result.stdout


def test_sdist_missing_license_fails(tmp_path):
    repo, dist = build_dist(tmp_path, WHEEL_OK, [e for e in SDIST_OK if e != "LICENSE"])
    result = run_check(repo, dist)
    assert result.returncode == 1
    assert "LICENSE" in result.stdout


def test_sdist_without_contributing_passes(tmp_path):
    """CONTRIBUTING.md may ship, but the release does not depend on it existing."""
    kept = [e for e in SDIST_OK if e != "CONTRIBUTING.md"]
    repo, dist = build_dist(tmp_path, WHEEL_OK, kept)
    result = run_check(repo, dist)
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize(
    "leaked, named",
    [("tests/test_leak.py", "tests"), ("constraints.txt", "constraints.txt")],
    ids=["tree", "root-file"],
)
def test_sdist_with_extra_top_level_fails(tmp_path, leaked, named):
    """Anything the sdist does not need is named, not tolerated."""
    repo, dist = build_dist(tmp_path, WHEEL_OK, SDIST_OK + [leaked])
    result = run_check(repo, dist)
    assert result.returncode == 1
    assert f"unexpected top-level {named}" in result.stdout


@pytest.mark.parametrize(
    "leaked, named",
    [("scripts/leaked.py", "scripts"), ("other-1.0.dist-info/METADATA", "other-1.0.dist-info")],
    ids=["tooling-tree", "foreign-dist-info"],
)
def test_wheel_with_extra_top_level_fails(tmp_path, leaked, named):
    """Only the package and its own metadata may install into site-packages.

    Under the previous flat layout every top-level directory was a package
    candidate, so tooling trees reached the wheel and became importable names.
    """
    repo, dist = build_dist(tmp_path, WHEEL_OK + [leaked], SDIST_OK)
    result = run_check(repo, dist)
    assert result.returncode == 1
    assert f"unexpected top-level {named}" in result.stdout


def test_missing_archives_fail(tmp_path):
    repo = make_repo(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    result = run_check(repo, dist)
    assert result.returncode == 1
