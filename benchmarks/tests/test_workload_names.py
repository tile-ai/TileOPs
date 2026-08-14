"""Every benchmark case must be named.

A case id is the workload's name wherever it is read later: the nightly report,
the published Benchmarks page, the perf history key. A case left to pytest's
positional naming collects as `shape0-dtype0`, which names nothing.

Source is parsed, not imported: this suite runs without a GPU or the baseline
libraries the bench files import. Only literal case lists are visible that way;
cases a file builds in a comprehension are covered by review.
"""

import ast
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parents[1] / "ops"
REPO_ROOT = Path(__file__).resolve().parents[2]

# Files whose cases predate the rule, with how many unnamed cases each still
# has. Never raise a number: name the cases instead. A file missing from the
# table is held to zero.
GRANDFATHERED = {
    "benchmarks/ops/bench_binary_elementwise.py": 47,
    "benchmarks/ops/bench_deltanet.py": 13,
    "benchmarks/ops/bench_gated_deltanet.py": 13,
    "benchmarks/ops/bench_gla_chunkwise.py": 10,
    "benchmarks/ops/bench_moe_fused_topk.py": 8,
    "benchmarks/ops/bench_moe_shared_fused_moe.py": 5,
}


def _case_lists(tree: ast.Module):
    """The value lists pytest will turn into cases.

    Two forms carry them: the second argument of a `parametrize(...)` call, and
    the `("a, b", [...])` entry a fixture base expands into one.
    """
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "parametrize"
                and len(node.args) > 1
                and isinstance(node.args[1], ast.List)):
            yield node.args[1]
        elif (isinstance(node, ast.Tuple) and len(node.elts) == 2
                and isinstance(node.elts[0], ast.Constant)
                and isinstance(node.elts[0].value, str)
                and isinstance(node.elts[1], ast.List)):
            yield node.elts[1]


def _unnamed_cases(tree: ast.Module) -> list[int]:
    """Line numbers of parametrize cases that carry no id.

    Two shapes count: `pytest.param(...)` without `id=`, and a bare tuple, which
    cannot carry one at all.
    """
    lines = []
    for case_list in _case_lists(tree):
        for element in case_list.elts:
            if isinstance(element, ast.Call):
                if (isinstance(element.func, ast.Attribute)
                        and element.func.attr == "param"
                        and not any(kw.arg == "id" for kw in element.keywords)):
                    lines.append(element.lineno)
            elif isinstance(element, (ast.Tuple, ast.List)):
                lines.append(element.lineno)
    return lines


def test_every_benchmark_case_is_named():
    offenders = {}
    for path in sorted(BENCH_DIR.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        found = _unnamed_cases(ast.parse(path.read_text(encoding="utf-8")))
        allowed = GRANDFATHERED.get(rel, 0)
        if len(found) > allowed:
            offenders[rel] = (len(found), allowed, found[:5])

    assert not offenders, "\n".join(
        f"{rel}: {n} unnamed parametrize cases, {allowed} grandfathered "
        f"(first at line{'s' if len(lines) > 1 else ''} "
        f"{', '.join(map(str, lines))}). Give each case an id naming the "
        f"scenario it stands for."
        for rel, (n, allowed, lines) in sorted(offenders.items())
    )


def test_grandfathered_files_exist():
    """A stale entry hides a file that was renamed away from the rule."""
    missing = [rel for rel in GRANDFATHERED if not (REPO_ROOT / rel).exists()]
    assert not missing, f"grandfathered files no longer present: {missing}"
