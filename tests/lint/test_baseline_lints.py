"""The baseline lints: a bench file imports no baseline library itself, and compares against one."""

import baseline_import_lint
import bench_baseline_lint
import pytest

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize(
    "source",
    [
        "import flashinfer\n",
        "from fla.ops.gla import chunk_gla\n",
        "def f():\n    try:\n        from vllm import _custom_ops\n    except ImportError:\n        pass\n",
    ],
    ids=["module", "from-submodule", "guarded-in-function"],
)
def test_an_import_of_a_baseline_library_is_flagged(source: str) -> None:
    assert len(baseline_import_lint.findings(source)) == 1


def test_the_harness_tools_and_the_resolvers_pass() -> None:
    source = (
        "import torch\nimport triton\n"
        "from benchmarks.baselines import optional_baseline\n"
        'chunk_gla = optional_baseline("fla.ops.gla", "fla", "chunk_gla")\n'
    )
    assert baseline_import_lint.findings(source) == []


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
    assert bench_baseline_lint.unbaselined_comparisons(source), source


@pytest.mark.parametrize("source", BASELINED)
def test_a_comparison_with_a_baseline_or_a_marker_is_left_alone(source: str) -> None:
    assert bench_baseline_lint.unbaselined_comparisons(source) == [], source
