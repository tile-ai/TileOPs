"""Unit tests for ``scripts/ci_venv_hash.py``.

The CI trusted-venv cache key must:

* stay stable across changes to tooling/style sections (``[tool.pytest]``,
  ``[tool.ruff]``, ``[tool.codespell]``) so fork PRs can reuse a warm venv
  even when someone tightens a lint rule;
* change when the actual dependency surface changes
  (``[project.dependencies]``, ``[project.optional-dependencies]``,
  ``[build-system].requires``);
* preserve the 16-char hex format required by the ``tileops_ci_venv_*``
  path convention, even on the parse-error fallback path.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "ci_venv_hash.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("ci_venv_hash", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CI_VENV_HASH = _load_module()
HEX16 = re.compile(r"^[0-9a-f]{16}$")


BASE_PYPROJECT = """\
[project]
name = "tileops"
version = "0.0.1"
dependencies = [
    "torch>=2.1.0",
    "tilelang==0.1.9",
]
requires-python = ">=3.10"

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff==0.14.13"]

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
markers = ["smoke", "full"]

[tool.ruff]
line-length = 100

[tool.codespell]
ignore-words-list = "nd, te"
"""


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "pyproject.toml"
    p.write_text(text)
    return p


def test_hash_is_16_char_hex(tmp_path: Path) -> None:
    h = CI_VENV_HASH.compute(_write(tmp_path, BASE_PYPROJECT))
    assert HEX16.match(h), h


def test_hash_stable_when_tool_pytest_changes(tmp_path: Path) -> None:
    """AC-1: mutating ``[tool.pytest]`` must NOT change the hash."""
    base = CI_VENV_HASH.compute(_write(tmp_path, BASE_PYPROJECT))
    mutated = BASE_PYPROJECT.replace(
        'markers = ["smoke", "full"]',
        'markers = ["smoke", "full", "nightly"]\naddopts = "-q"',
    )
    new = CI_VENV_HASH.compute(_write(tmp_path, mutated))
    assert new == base, "pytest-only config change must not invalidate the venv"


def test_hash_stable_when_tool_ruff_changes(tmp_path: Path) -> None:
    """AC-1: mutating ``[tool.ruff]`` must NOT change the hash."""
    base = CI_VENV_HASH.compute(_write(tmp_path, BASE_PYPROJECT))
    mutated = BASE_PYPROJECT.replace("line-length = 100", "line-length = 120")
    new = CI_VENV_HASH.compute(_write(tmp_path, mutated))
    assert new == base, "ruff-only config change must not invalidate the venv"


def test_hash_changes_when_project_dependencies_change(tmp_path: Path) -> None:
    """AC-2: adding a runtime dep MUST change the hash."""
    base = CI_VENV_HASH.compute(_write(tmp_path, BASE_PYPROJECT))
    mutated = BASE_PYPROJECT.replace(
        '"tilelang==0.1.9",',
        '"tilelang==0.1.9",\n    "einops",',
    )
    new = CI_VENV_HASH.compute(_write(tmp_path, mutated))
    assert new != base


def test_hash_changes_when_build_system_requires_change(tmp_path: Path) -> None:
    """AC-2: bumping ``[build-system].requires`` MUST change the hash."""
    base = CI_VENV_HASH.compute(_write(tmp_path, BASE_PYPROJECT))
    mutated = BASE_PYPROJECT.replace(
        'requires = ["setuptools>=68", "wheel"]',
        'requires = ["setuptools>=70", "wheel"]',
    )
    new = CI_VENV_HASH.compute(_write(tmp_path, mutated))
    assert new != base


def test_hash_changes_when_optional_dependencies_change(tmp_path: Path) -> None:
    base = CI_VENV_HASH.compute(_write(tmp_path, BASE_PYPROJECT))
    mutated = BASE_PYPROJECT.replace(
        'dev = ["pytest>=8.0", "ruff==0.14.13"]',
        'dev = ["pytest>=8.0", "ruff==0.14.13", "pytest-xdist"]',
    )
    new = CI_VENV_HASH.compute(_write(tmp_path, mutated))
    assert new != base


def test_fallback_on_parse_error_still_emits_16_hex(tmp_path: Path) -> None:
    """Fallback contract: malformed TOML still produces a 16-char hex hash."""
    p = _write(tmp_path, "this is [not valid == toml\n")
    h = CI_VENV_HASH.compute(p)
    assert HEX16.match(h), h


def test_fallback_matches_full_file_hash_shape(tmp_path: Path) -> None:
    """The fallback path is a pure function of the file bytes, so identical
    malformed files must hash identically."""
    malformed = "nope = = =\nstill bad\n"
    a = CI_VENV_HASH.compute(_write(tmp_path, malformed))
    p2 = tmp_path / "other.toml"
    p2.write_text(malformed)
    b = CI_VENV_HASH.compute(p2)
    assert a == b


def test_hash_stable_when_dependencies_reordered(tmp_path: Path) -> None:
    """Pure reordering of ``[project].dependencies`` must NOT change the hash.

    ``pip install`` is order-insensitive, so swapping two dep entries installs
    the same wheels and must share a venv cache key.
    """
    base = CI_VENV_HASH.compute(_write(tmp_path, BASE_PYPROJECT))
    reordered = BASE_PYPROJECT.replace(
        '    "torch>=2.1.0",\n    "tilelang==0.1.9",',
        '    "tilelang==0.1.9",\n    "torch>=2.1.0",',
    )
    new = CI_VENV_HASH.compute(_write(tmp_path, reordered))
    assert new == base, "reordering runtime deps must not invalidate the venv"


def test_hash_stable_when_optional_dependencies_reordered(tmp_path: Path) -> None:
    """Pure reordering inside a ``[project.optional-dependencies]`` extra
    list must NOT change the hash."""
    base = CI_VENV_HASH.compute(_write(tmp_path, BASE_PYPROJECT))
    reordered = BASE_PYPROJECT.replace(
        'dev = ["pytest>=8.0", "ruff==0.14.13"]',
        'dev = ["ruff==0.14.13", "pytest>=8.0"]',
    )
    new = CI_VENV_HASH.compute(_write(tmp_path, reordered))
    assert new == base, "reordering dev extras must not invalidate the venv"


def test_hash_stable_when_build_system_requires_reordered(tmp_path: Path) -> None:
    """Pure reordering of ``[build-system].requires`` must NOT change the hash."""
    base = CI_VENV_HASH.compute(_write(tmp_path, BASE_PYPROJECT))
    reordered = BASE_PYPROJECT.replace(
        'requires = ["setuptools>=68", "wheel"]',
        'requires = ["wheel", "setuptools>=68"]',
    )
    new = CI_VENV_HASH.compute(_write(tmp_path, reordered))
    assert new == base, "reordering build requires must not invalidate the venv"


def test_hash_stable_when_project_version_changes(tmp_path: Path) -> None:
    """Metadata-only edits to ``[project].version`` must NOT change the hash."""
    base = CI_VENV_HASH.compute(_write(tmp_path, BASE_PYPROJECT))
    mutated = BASE_PYPROJECT.replace('version = "0.0.1"', 'version = "0.0.2"')
    new = CI_VENV_HASH.compute(_write(tmp_path, mutated))
    assert new == base, "version-only bump must not invalidate the venv"


def test_hash_stable_when_project_description_changes(tmp_path: Path) -> None:
    """Adding/changing ``[project].description`` must NOT change the hash."""
    base = CI_VENV_HASH.compute(_write(tmp_path, BASE_PYPROJECT))
    mutated = BASE_PYPROJECT.replace(
        'version = "0.0.1"',
        'version = "0.0.1"\ndescription = "A different blurb"',
    )
    new = CI_VENV_HASH.compute(_write(tmp_path, mutated))
    assert new == base, "description-only edit must not invalidate the venv"
