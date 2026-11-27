"""A malformed entry is reported, never raised on.

Every malformed entry shape runs through every level: a check that assumes a mapping and
gets a string must not abort the run and hide the entry's other problems.
"""

import importlib.util
import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATOR = REPO_ROOT / "scripts" / "validate_manifest.py"

# The shapes a broken entry takes; the real manifest exercises none of them.
MALFORMED: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "empty_entry",
        {"EmptyOp": {}},
    ),
    (
        "missing_required_top",
        {"BadTopOp": {"family": "elementwise", "ref_api": "none"}},
    ),
    (
        "missing_signature_halves",
        {
            "BadSigOp": {
                "family": "elementwise",
                "ref_api": "none",
                "status": "spec-only",
                "signature": {},
                "workloads": [],
                "roofline": {"flops": "1"},
            }
        },
    ),
    (
        "wrong_types",
        {
            "BadTypeOp": {
                "family": 1,
                "ref_api": None,
                "status": "nonsense",
                "signature": "not a mapping",
                "workloads": "not a list",
                "roofline": [],
            }
        },
    ),
    (
        "unknown_keys",
        {
            "UnknownKeyOp": {
                "family": "elementwise",
                "ref_api": "none",
                "status": "spec-only",
                "made_up_field": True,
                "signature": {
                    "inputs": {"x": {"dtype": "float16", "shape": "[N]"}},
                    "outputs": {"y": {"dtype": "float16", "shape": "[N]"}},
                    "also_made_up": 1,
                },
                "workloads": [{"N": 4, "label": "n4"}],
                "roofline": {"flops": "1"},
            }
        },
    ),
    (
        "non_string_keys",
        {
            "BadKeyOp": {
                "family": "elementwise",
                "ref_api": "none",
                "status": "spec-only",
                "signature": {
                    "inputs": {"x": {"dtype": "float16", "shape": "[N]"}},
                    "outputs": {"y": {"dtype": "float16", "shape": "[N]"}},
                    # A key that is not a string: ordering the leftovers by the
                    # key itself would raise rather than report.
                    7: "junk",
                },
                "workloads": [{"N": 4, "label": "n4"}],
                "roofline": {"flops": "1"},
            }
        },
    ),
    (
        "scalar_entry",
        {"ScalarOp": 5},
    ),
    (
        "non_string_dtype",
        {
            "BadDtypeOp": {
                "family": "elementwise",
                "ref_api": "none",
                # Implemented, so the class parity check runs too.
                "status": "implemented",
                "signature": {
                    "inputs": {"x": {"dtype": "float16", "shape": "[N]"}},
                    "outputs": {"y": {"dtype": {"junk": [1]}, "shape": "[N]"}},
                },
                "workloads": [{"N": 4, "label": "n4"}],
                "roofline": {"flops": "1"},
            }
        },
    ),
    (
        "implemented_without_family",
        {
            "NoFamilyFwdOp": {
                # Implemented and otherwise well formed, so the class parity check, which
                # imports the class by family, runs too.
                "status": "implemented",
                "signature": {
                    "forall": {"N": "Dim"},
                    "inputs": {"x": {"dtype": "float16", "shape": "[N]"}},
                    "outputs": {"y": {"dtype": "float16", "shape": "[N]"}},
                },
                "workloads": [{"N": 4, "label": "n4"}],
                "roofline": {"flops": "1"},
            }
        },
    ),
    (
        "composite_with_a_malformed_stage",
        {
            "BadCompositeOp": {
                "family": "elementwise",
                "ref_api": "none",
                "status": "spec-only",
                "signature": {
                    "inputs": {"x": {"dtype": "float16", "shape": "[N]"}},
                    "outputs": {"y": {"dtype": "float16", "shape": "[N]"}},
                },
                "composition": {"kind": "composite", "stages": [{"name": 3, "op": ["k"]}]},
                "workloads": [{"N": 4, "label": "n4"}],
                "roofline": {"flops": "1"},
            }
        },
    ),
)

LEVELS = (None, "schema", "signature", "bench")


@pytest.fixture(scope="module")
def validator():
    spec = importlib.util.spec_from_file_location("validate_manifest_malformed", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def written():
    """Each fixture on disk, at a path that does not vary between runs."""
    with tempfile.TemporaryDirectory() as tmp:
        yield {name: _write(Path(tmp) / f"{name}.yaml", mapping) for name, mapping in MALFORMED}


def _write(path: Path, mapping) -> Path:
    path.write_text(yaml.safe_dump(mapping, sort_keys=True))
    return path


@pytest.mark.parametrize("fixture_name", [name for name, _ in MALFORMED])
@pytest.mark.parametrize("level", LEVELS, ids=[lv or "all" for lv in LEVELS])
def test_reports_rather_than_raises(validator, written, fixture_name, level):
    """Per level, because each runs a different check that could assume a mapping."""
    levels = frozenset({level}) if level else None
    errors, warnings = validator.validate_manifest(
        manifest_path=written[fixture_name], levels=levels
    )
    assert isinstance(errors, list) and isinstance(warnings, list)


def test_a_broken_field_does_not_hide_the_others(validator, written):
    """An entry with a bad roofline and a bad signature reports both."""
    errors, _ = validator.validate_manifest(manifest_path=written["wrong_types"])
    assert len(errors) > 1, errors
