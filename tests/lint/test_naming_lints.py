"""The naming lints: manifest entry names, kernel interface names and benchmark case ids."""

from pathlib import Path

import interface_names_lint
import op_naming_lint
import pytest
import workload_names_lint

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize(
    "ops, expected",
    [
        pytest.param({"MoeFwdOp": {}}, "M1 MoeFwdOp: spell this abbreviation MoE", id="m1"),
        pytest.param({"TopkSelectorFwdOp": {}}, "M2 TopkSelectorFwdOp: 'Selector'", id="m2"),
        pytest.param(
            {
                "SoftmaxFwdOp": {"ref_api": "torch.softmax"},
                "SoftmaxOnlineFwdOp": {"ref_api": "torch.softmax"},
            },
            "M3 SoftmaxFwdOp: torch.softmax has no default side",
            id="m3",
        ),
    ],
)
def test_an_entry_name_a_rule_reaches_is_reported(ops: dict, expected: str) -> None:
    assert any(expected in f for f in op_naming_lint.findings(ops)), op_naming_lint.findings(ops)


def test_a_finding_names_the_line_its_entry_starts_on() -> None:
    (found,) = op_naming_lint.findings({"MlpFwdOp": {}}, {"MlpFwdOp": "spec/mlp.yaml:7"})

    assert found.startswith("spec/mlp.yaml:7: M1 MlpFwdOp:")


def test_an_allowed_or_defaulted_name_is_left_alone() -> None:
    ops = {
        **{name: {} for name in op_naming_lint.AGENTIVE_EXCEPTIONS},
        "MaxPool2dFwdOp": {"ref_api": "torch.nn.functional.max_pool2d"},
        "MaxPool2dIndicesFwdOp": {"ref_api": "torch.nn.functional.max_pool2d"},
    }

    assert op_naming_lint.findings(ops) == []


def test_every_manifest_entry_has_a_location() -> None:
    assert set(op_naming_lint.load_manifest()) <= set(op_naming_lint.entry_locations())


def _findings(tmp_path: Path, source: str) -> list[str]:
    path = tmp_path / "probe.py"
    path.write_text(source, encoding="utf-8")
    return interface_names_lint.findings(path)


@pytest.mark.parametrize("name", ["BatchNormFwdTrainInterface", "GEMMInterface"])
def test_a_name_without_the_direction_last_is_reported(tmp_path: Path, name: str) -> None:
    (found,) = _findings(tmp_path, f"class {name}(KernelInterface):\n    pass\n")

    assert f"kernel interface {name} is not" in found


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("class BatchNormTrainFwdInterface(KernelInterface): pass\n", id="named"),
        pytest.param("class GEMMBwdInterface(base.KernelInterface): pass\n", id="attribute-base"),
        # Only a direct subclass is named by the rule.
        pytest.param("class GEMMHelper(GEMMFwdInterface): pass\n", id="indirect"),
    ],
)
def test_a_conforming_or_indirect_class_is_left_alone(tmp_path: Path, source: str) -> None:
    assert _findings(tmp_path, source) == []


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
    assert workload_names_lint.unnamed_cases(source), source


@pytest.mark.parametrize("source", NAMED)
def test_a_named_case_or_a_plain_value_is_left_alone(source: str) -> None:
    assert workload_names_lint.unnamed_cases(source) == [], source


def test_each_unnamed_case_is_its_own_finding(tmp_path, monkeypatch) -> None:
    bench = tmp_path / "benchmarks" / "ops"
    bench.mkdir(parents=True)
    path = bench / "bench_probe.py"
    path.write_text("pytest.param(1)\npytest.param(2)\n", encoding="utf-8")
    monkeypatch.setattr(
        workload_names_lint, "relative", lambda p: p.relative_to(tmp_path).as_posix()
    )

    assert [f.split(": ", 1)[0] for f in workload_names_lint.findings(path)] == [
        "benchmarks/ops/bench_probe.py:1",
        "benchmarks/ops/bench_probe.py:2",
    ]


def test_a_file_outside_the_op_benchmarks_is_left_alone(tmp_path) -> None:
    path = tmp_path / "bench_probe.py"
    path.write_text("pytest.param(1)\n", encoding="utf-8")

    assert workload_names_lint.findings(path) == []
