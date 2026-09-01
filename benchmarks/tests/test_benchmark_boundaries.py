"""Structural gates for the benchmark stage's boundary.

Coverage is deliberately literal: these scan `benchmarks/ops` for a static
`tests` import and for a definition named exactly `gen_inputs`. They do not prove input construction is absent — a
module-level draw helper, or one injected into a workload as a callable, would
pass. Both forms existed here and were removed rather than gated for, because
naming every way to build a tensor is a losing game.

See docs/design/trust-model.md §Benchmark.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIRS = ("benchmarks/ops",)


def _benchmark_files() -> list[Path]:
    return [
        path
        for rel in BENCHMARK_DIRS
        for path in sorted((REPO_ROOT / rel).rglob("*.py"))
        if path.name != "__init__.py"
    ]


def _scan(finder) -> dict[str, list[str]]:
    offenders = {}
    for path in _benchmark_files():
        hits = finder(ast.parse(path.read_text(), filename=str(path)))
        if hits:
            offenders[str(path.relative_to(REPO_ROOT))] = hits
    return offenders


def _imports_tests(tree: ast.AST) -> list[str]:
    def is_tests(name: str) -> bool:
        return name == "tests" or name.startswith("tests.")

    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and is_tests(node.module or ""):
            hits.append(f"from {node.module} (line {node.lineno})")
        elif isinstance(node, ast.Import):
            hits += [
                f"import {a.name} (line {node.lineno})" for a in node.names if is_tests(a.name)
            ]
    return hits


def _defines_gen_inputs(tree: ast.AST) -> list[str]:
    return [
        f"line {n.lineno}"
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "gen_inputs"
    ]


@pytest.mark.smoke
def test_benchmarks_do_not_import_tests_package() -> None:
    """A benchmark that imports tests/ breaks on any test-side refactor, and no
    PR gate runs the nightly benchmarks that would catch it."""
    assert _scan(_imports_tests) == {}


@pytest.mark.smoke
def test_benchmarks_do_not_author_gen_inputs() -> None:
    """Import the op's workload from workloads/; if it has none, add it there."""
    assert _scan(_defines_gen_inputs) == {}


# A benchmark takes (flops, bytes) from its op — docs/design/roofline.md §4.2. These
# measure something the manifest does not model, so the arithmetic has nowhere else to
# live. Each entry goes when its subject gains a manifest entry to take a roofline from.
_ROOFLINE_OF_ITS_OWN = {
    "FusedGatedBenchmark": "times a kernel strategy rather than an op, and publishes nothing",
    "SharedFusedMoEBenchmark": "SharedFusedMoE has no manifest entry: its first output is "
    "None when no shared expert is configured, and outputs cannot say that",
}


def _writes_its_own_roofline(tree: ast.AST) -> list[str]:
    return [
        f"{node.name}.{fn.name} (line {fn.lineno})"
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name not in _ROOFLINE_OF_ITS_OWN
        for fn in node.body
        if isinstance(fn, ast.FunctionDef) and fn.name in ("calculate_flops", "calculate_memory")
    ]


@pytest.mark.smoke
def test_benchmarks_take_their_roofline_from_the_op() -> None:
    """Two sources for one op's FLOPs are two numbers that can disagree, and the
    manifest is the one every other consumer reads. Subclass ``ManifestBenchmark``;
    where the op has no roofline to take, name the class above and say why."""
    assert _scan(_writes_its_own_roofline) == {}
