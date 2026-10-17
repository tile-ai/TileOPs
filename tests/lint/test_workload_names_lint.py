"""Unit tests for ``scripts/lint/workload_names_lint.py``."""

import pytest
import workload_names_lint as lint

pytestmark = pytest.mark.smoke

UNNAMED = [
    pytest.param("pytest.param(1, 2)\n", id="param-without-id"),
    pytest.param("@pytest.mark.parametrize('a, b', [(1, 2)])\ndef f(a, b): pass\n", id="tuple"),
    pytest.param("CASES = ('a, b', [(1, 2)])\n", id="fixture-base-tuple"),
]

NAMED = [
    pytest.param("pytest.param(1, 2, id='small')\n", id="param-with-id"),
    pytest.param("@pytest.mark.parametrize('a', [1, 2])\ndef f(a): pass\n", id="scalar-values"),
    # A tuple outside a parametrize value list is a value, not a case.
    pytest.param("SHAPE = (1, 2)\n", id="plain-tuple"),
]


@pytest.mark.parametrize("source", UNNAMED)
def test_a_case_without_an_id_is_reported(source: str) -> None:
    assert lint.unnamed_cases(source), source


@pytest.mark.parametrize("source", NAMED)
def test_a_named_case_or_a_plain_value_is_left_alone(source: str) -> None:
    assert lint.unnamed_cases(source) == [], source


def test_each_unnamed_case_is_its_own_finding(tmp_path, monkeypatch) -> None:
    bench = tmp_path / "benchmarks" / "ops"
    bench.mkdir(parents=True)
    path = bench / "bench_probe.py"
    path.write_text("pytest.param(1)\npytest.param(2)\n", encoding="utf-8")
    monkeypatch.setattr(lint, "relative", lambda p: p.relative_to(tmp_path).as_posix())

    assert [f.split(": ", 1)[0] for f in lint.findings(path)] == [
        "benchmarks/ops/bench_probe.py:1",
        "benchmarks/ops/bench_probe.py:2",
    ]


def test_a_file_outside_the_op_benchmarks_is_left_alone(tmp_path) -> None:
    path = tmp_path / "bench_probe.py"
    path.write_text("pytest.param(1)\n", encoding="utf-8")

    assert lint.findings(path) == []
