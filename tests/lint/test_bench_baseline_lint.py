"""Unit tests for ``scripts/lint/bench_baseline_lint.py``."""

import bench_baseline_lint as lint
import pytest

pytestmark = pytest.mark.smoke

# Concatenated so this file does not carry the marker it tests.
MARKER = "FIXME" + "(staged-rollout)"

UNBASELINED = [
    pytest.param("def test_a(r):\n    r.compare({'tileops': f})\n", id="dict-display"),
    pytest.param(
        "def test_a(r):\n    impls = {'tileops': f}\n    impls['tileops-nopad'] = g\n"
        "    r.compare(impls)\n",
        id="named-dict",
    ),
    pytest.param("def test_a(r):\n    r.record(t, tag='tileops')\n", id="record-only-ours"),
]

BASELINED = [
    pytest.param("def test_a(r):\n    r.compare({'tileops': f, 'torch': g})\n", id="with-torch"),
    pytest.param(
        "def test_a(r):\n    r.record(t, tag='tileops')\n    r.record(u, tag='torch')\n",
        id="record-with-torch",
    ),
    # A dict this cannot name the keys of is decided elsewhere.
    pytest.param("def test_a(r):\n    r.compare({**base, 'tileops': f})\n", id="unpacked"),
    pytest.param("def test_a(r):\n    r.compare(build_impls())\n", id="helper-built"),
    pytest.param(
        f"def test_a(r):\n    # {MARKER}: no public baseline exists yet\n"
        "    r.compare({'tileops': f})\n",
        id="marked",
    ),
]


@pytest.mark.parametrize("source", UNBASELINED)
def test_a_comparison_with_only_ours_is_reported(source: str) -> None:
    assert lint.unbaselined_comparisons(source), source


@pytest.mark.parametrize("source", BASELINED)
def test_a_comparison_with_a_baseline_or_a_marker_is_left_alone(source: str) -> None:
    assert lint.unbaselined_comparisons(source) == [], source
