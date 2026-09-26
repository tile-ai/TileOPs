"""A malformed entry is reported, never raised on.

An entry can hold anything a YAML file can hold. When a check assumes a mapping
and gets a string, the run dies and the entry's other problems go unreported —
which is the opposite of what a validator is for. These drive every malformed
shape through every level and both parity modes.
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

#: The shapes a broken entry takes. The well-formed manifest exercises none of
#: them: the code paths reporting missing or mistyped fields only run on an
#: entry that has them.
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
                "roofline": {"flops": "1", "bytes": "1"},
                "source": {},
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
                "source": 3,
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
                    "inputs": {"x": {"dtype": "float16"}},
                    "outputs": {"y": {"dtype": "same_as(x)"}},
                    "also_made_up": 1,
                },
                "workloads": [{"x_shape": [4], "dtypes": ["float16"]}],
                "roofline": {"flops": "1", "bytes": "1"},
                "source": {"kernel": "k.py", "op": "o.py", "test": "t.py", "bench": "b.py"},
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
                    "inputs": {"x": {"dtype": "float16"}},
                    "outputs": {"y": {"dtype": "same_as(x)"}},
                    # A key that is not a string: ordering the leftovers by the
                    # key itself would raise rather than report.
                    7: "junk",
                },
                "workloads": [{"x_shape": [4], "dtypes": ["float16"]}],
                "roofline": {"flops": "1", "bytes": "1"},
                "source": {"kernel": "k.py", "op": "o.py", "test": "t.py", "bench": "b.py"},
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
                # Implemented, so the dtype level runs: its parser splits the
                # declaration, and a mapping there raised rather than reported.
                "status": "implemented",
                "signature": {
                    "inputs": {"x": {"dtype": "float16"}},
                    "outputs": {"y": {"dtype": {"junk": [1]}}},
                    "shape_rules": ["y.shape == x.shape"],
                },
                "workloads": [
                    {"x_shape": [4], "dtypes": ["float16"]},
                    {"x_shape": [8], "dtypes": ["float16"]},
                ],
                "roofline": {"flops": "1", "bytes": "1"},
                "source": {"kernel": "k.py", "op": "o.py", "test": "t.py", "bench": "b.py"},
            }
        },
    ),
    (
        "composite_without_roofline_composition",
        {
            "BadCompositeOp": {
                "family": "elementwise",
                "ref_api": "none",
                "status": "spec-only",
                "signature": {
                    "inputs": {"x": {"dtype": "float16"}},
                    "outputs": {"y": {"dtype": "same_as(x)"}},
                },
                "composition": {"kind": "composite", "stages": [{"name": "s", "kernel": "k"}]},
                "resources": {"workspaces": [{"name": "x", "dtype": "float16"}]},
                "workloads": [{"x_shape": [4], "dtypes": ["float16"]}],
                "roofline": {"flops": "1", "bytes": "1"},
                "source": {"kernel": "k.py", "op": "o.py", "test": "t.py", "bench": "b.py"},
            }
        },
    ),
)

#: Not every level: the three checks that assumed a mapping ran under these,
#: and running the rest is a cartesian product of one property.
LEVELS = (None, "schema", "signature", "shape", "dtype", "bench")


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
    """Per level, because each ran a different check that assumed a mapping.

    Parity mode is not a dimension here: it routes findings, it does not change
    whether producing them raises.
    """
    levels = frozenset({level}) if level else None
    errors, warnings = validator.validate_manifest(
        manifest_path=written[fixture_name], levels=levels
    )
    assert isinstance(errors, list) and isinstance(warnings, list)


def test_a_broken_field_does_not_hide_the_others(validator, written):
    """One unreadable field must not stop the rest of the entry being read.

    The parser accumulates for this reason: an entry with a bad source and a
    bad signature should say so about both.
    """
    errors, _ = validator.validate_manifest(manifest_path=written["wrong_types"])
    assert len(errors) > 1, errors
