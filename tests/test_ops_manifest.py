"""Tests for the programmatic manifest API (tileops.manifest).

Schema policies for manifest entries are owned by scripts/validate_manifest.py
(see tests/test_validate_manifest.py); this file covers only the package's
load/merge surface.
"""

from pathlib import Path

import pytest
import yaml

from tileops.manifest import load_manifest, load_workloads, manifest_files

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = REPO_ROOT / "src" / "tileops" / "manifest" / "spec"


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
        assert "label" in workloads[0]

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


@pytest.fixture
def manifest_from(monkeypatch):
    """Point `load_manifest` at the given spec files, uncached."""
    import tileops.manifest as manifest

    def use(*paths: Path) -> None:
        monkeypatch.setattr(manifest, "manifest_files", lambda: list(paths))
        load_manifest.cache_clear()

    yield use
    load_manifest.cache_clear()


def test_a_missing_entry_is_none_and_a_broken_manifest_raises(manifest_from, tmp_path):
    from tileops.manifest import _try_load_entry

    good = tmp_path / "probe.yaml"
    good.write_text("ProbeFwdOp:\n  family: probe\n")
    manifest_from(good)
    assert _try_load_entry("ProbeFwdOp") == {"family": "probe"}
    assert _try_load_entry("MissingFwdOp") is None

    broken = tmp_path / "broken.yaml"
    broken.write_text("ProbeFwdOp: [unclosed\n")
    manifest_from(broken)
    with pytest.raises(yaml.YAMLError, match="broken.yaml"):
        _try_load_entry("MissingFwdOp")


@pytest.mark.parametrize(
    ("text", "line", "key"),
    [
        ("ProbeFwdOp:\n  family: probe\nProbeFwdOp:\n  family: probe\n", 3, "ProbeFwdOp"),
        ("ProbeFwdOp:\n  family: probe\n  signature:\n    a: 1\n    a: 2\n", 5, "a"),
    ],
)
def test_a_duplicate_key_at_any_level_raises(manifest_from, tmp_path, text, line, key):
    path = tmp_path / "probe.yaml"
    path.write_text(text)
    manifest_from(path)
    with pytest.raises(ValueError, match=rf"probe\.yaml:{line}: duplicate key '{key}'"):
        load_manifest()


def test_family_file_named_after_another_family_raises():
    from tileops.manifest import _check_family_file

    with pytest.raises(ValueError, match="FFTC2CFwdOp.*family 'fft'"):
        _check_family_file("sequence_modeling.yaml", {"FFTC2CFwdOp": {"family": "fft"}})


def test_every_implemented_roofline_func_is_an_exported_formula():
    """An implemented entry's ``roofline.func`` is a formula ``formulas.__all__`` exports.

    A spec-only entry may name a formula not written yet, as the validator allows.
    """
    import tileops._perf.formulas as formulas

    named = {
        entry["roofline"]["func"].removeprefix("tileops._perf.formulas.")
        for entry in load_manifest().values()
        if entry.get("status") == "implemented" and "func" in (entry.get("roofline") or {})
    }
    assert named - set(formulas.__all__) == set()
    assert all(callable(getattr(formulas, name)) for name in named)
