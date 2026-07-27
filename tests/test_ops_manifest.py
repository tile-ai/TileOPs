"""Tests for the programmatic manifest API (tileops.manifest).

Schema policies for manifest entries are owned by scripts/validate_manifest.py
(see tests/test_validate_manifest.py); this file covers only the package's
load/merge surface.
"""

from pathlib import Path

import pytest

from tileops.manifest import load_manifest, load_workloads, manifest_files

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = REPO_ROOT / "tileops" / "manifest"


class TestManifestStructure:
    """Manifest package exists and contributes at least one family file."""

    def test_manifest_dir_exists(self):
        assert MANIFEST_DIR.is_dir()

    def test_manifest_has_family_files(self):
        files = manifest_files()
        assert len(files) >= 1
        assert all(p.name.endswith(".yaml") for p in files)

    def test_manifest_loads(self):
        ops = load_manifest()
        assert isinstance(ops, dict)
        assert ops


class TestManifestAPI:
    """Load helpers accept only canonical PascalCase op keys."""

    def test_load_workloads_returns_list(self):
        workloads = load_workloads("RMSNormFwdOp")
        assert isinstance(workloads, list)
        assert len(workloads) >= 1
        assert "x_shape" in workloads[0]

    def test_load_workloads_unknown_op_raises(self):
        with pytest.raises(KeyError, match="NonexistentOp"):
            load_workloads("NonexistentOp")

    def test_load_workloads_snake_case_raises(self):
        """Legacy snake_case names are not resolved."""
        with pytest.raises(KeyError, match="rmsnorm_fwd"):
            load_workloads("rmsnorm_fwd")

    def test_manifest_does_not_expose_roofline_evaluator(self):
        import tileops.manifest as manifest

        for name in (
            "_safe_eval",
            "eval_roofline",
            "has_roofline_vars",
            "resolve_roofline_vars",
        ):
            assert not hasattr(manifest, name)
