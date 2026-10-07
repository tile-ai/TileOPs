"""Tests for scripts/validate_manifest.py (docs/design/manifest.md § Validation).

The signature, workload and roofline checks the validator renders are tested where they are
implemented (tests/test_manifest_signature.py, tests/test_manifest_workload.py). This file
covers what the script itself owns: the entry schema, `composition`, the class parity check,
the benchmark contract and the CLI scoping. The run over the real manifest is the preflight
`validate-manifest` job.
"""

import sys
import textwrap
import types
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATOR_SCRIPT = REPO_ROOT / "scripts" / "validate_manifest.py"


@pytest.fixture(scope="module")
def validator():
    """Import validate_manifest as a module (it lives in scripts/, not a package)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("validate_manifest", VALIDATOR_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _entry(**extra) -> dict:
    """A well-formed spec-only entry."""
    cases = yaml.safe_load((REPO_ROOT / "tests" / "manifest_cases.yaml").read_text())
    base = cases["entries"]["SiluAndMulFwdOp"]
    return {
        "family": "elementwise",
        "status": "spec-only",
        **base,
        "roofline": {"flops": "6 * M * N"},
        **extra,
    }


def _write_manifest(tmp_path: Path, ops: dict) -> Path:
    path = tmp_path / "ops_manifest.yaml"
    path.write_text(yaml.safe_dump(ops))
    return path


class TestSchema:
    """Top-level fields and the key format."""

    def test_a_well_formed_entry_passes(self, validator, tmp_path):
        path = _write_manifest(tmp_path, {"ProbeFwdOp": _entry()})
        errors, _ = validator.validate_manifest(manifest_path=path, repo_root=tmp_path)
        assert errors == [], errors

    def test_missing_unknown_and_mistyped_fields_are_reported(self, validator):
        entry = _entry(status="done", surprise=1, workloads={})
        del entry["roofline"]
        errors = validator._schema_errors("ProbeFwdOp", entry, {"ProbeFwdOp"})
        assert errors == [
            "[schema] ProbeFwdOp: missing required field 'roofline'",
            "[schema] ProbeFwdOp: 'workloads' must be a list",
            "[schema] ProbeFwdOp: unknown field 'surprise'",
            "[schema] ProbeFwdOp: status must be 'implemented' or 'spec-only'",
        ], errors

    def test_a_non_mapping_entry_is_reported(self, validator, tmp_path):
        path = _write_manifest(tmp_path, {"ProbeFwdOp": ["not", "an", "entry"]})
        errors, _ = validator.validate_manifest(manifest_path=path, repo_root=tmp_path)
        assert errors == ["[schema] ProbeFwdOp: entry must be a mapping"], errors

    def test_variant_words_precede_the_direction_suffix(self, validator):
        errors = validator._key_format_errors("GroupNormFwdOpNoAffine", ())
        assert any("NoAffine" in e and "precede" in e for e in errors), errors

    def test_the_direction_suffix_is_required_beside_a_sibling(self, validator):
        errors = validator._key_format_errors("SoftmaxOp", {"SoftmaxOp", "SoftmaxFwdOp"})
        assert any("direction suffix" in e and "SoftmaxFwdOp" in e for e in errors), errors


def test_schema_does_not_import_reference_packages(validator, monkeypatch, tmp_path):
    ref = "uninstalled_baseline.ops.forward"
    path = _write_manifest(tmp_path, {"ProbeFwdOp": _entry(ref_api=ref)})
    original = validator.importlib.import_module

    def import_module(name):
        assert not name.startswith("uninstalled_baseline"), name
        return original(name)

    monkeypatch.setattr(validator.importlib, "import_module", import_module)
    errors, _ = validator.validate_manifest(manifest_path=path, repo_root=tmp_path)
    assert errors == [], errors
    assert validator._ref_api_errors("ProbeFwdOp", "not a.path")


@pytest.mark.parametrize(
    "ref, diagnostic",
    [
        ("torch.Tensor.add", None),
        ("torch.nn.functional.missing_reference", "does not resolve"),
        ("uninstalled_baseline.ops.forward", "no prefix"),
        ("broken_baseline.forward", "missing_transitive_dependency"),
    ],
)
def test_refs_resolve_and_report_failures(validator, monkeypatch, tmp_path, ref, diagnostic):
    original = validator.importlib.import_module

    def import_module(name):
        if name == "broken_baseline":
            raise ModuleNotFoundError("missing_transitive_dependency", name="dependency")
        return original(name)

    monkeypatch.setattr(validator.importlib, "import_module", import_module)
    path = _write_manifest(tmp_path, {"ProbeFwdOp": {"ref_api": ref}})
    errors, _ = validator.validate_manifest(manifest_path=path, levels=frozenset({"refs"}))
    if diagnostic is None:
        assert errors == [], errors
    else:
        assert len(errors) == 1 and errors[0].startswith("[refs] ProbeFwdOp:"), errors
        assert diagnostic in errors[0], errors


class TestComposition:
    """`composition`: uniquely named stages, each naming an entry or a `kernel_types` key."""

    @staticmethod
    def _errors(validator, stages, kind="composite"):
        composition = {"kind": kind, "stages": stages}
        return validator._composition_errors("ProbeFwdOp", composition, {"AFwdOp"})

    def test_a_valid_composition_passes(self, validator):
        stages = [{"name": "own", "kernel": "own"}, {"name": "a", "op": "AFwdOp", "optional": True}]
        assert self._errors(validator, stages) == []

    def test_each_malformed_stage_is_reported(self, validator):
        stages = [
            {"name": "a", "op": "MissingFwdOp"},
            {"name": "a", "kernel": "own", "op": "AFwdOp"},
            {"name": "b", "kernel": "own", "optional": "yes", "variants": []},
        ]
        assert self._errors(validator, stages, kind="pipeline") == [
            "[schema] ProbeFwdOp: composition.kind must be one of ['composite'], got 'pipeline'",
            "[schema] ProbeFwdOp: composition.stages[0].op 'MissingFwdOp' is not a manifest entry",
            "[schema] ProbeFwdOp: composition.stages[1].name 'a' is declared twice",
            "[schema] ProbeFwdOp: composition.stages[1] must have exactly one of 'op' or 'kernel'",
            "[schema] ProbeFwdOp: composition.stages[2] has unknown keys ['variants']",
            "[schema] ProbeFwdOp: composition.stages[2].optional must be a bool",
        ]


def test_an_entry_is_held_to_its_constructor_and_forward(validator, monkeypatch):
    """`__init__` takes `signature.params` then the policy suffix; `forward` the inputs."""

    class ProbeFwdOp:
        def __init__(self, dim, *, kernel_map=None, target=None, tune=False, surprise=None):
            pass

        def forward(self, x=123, y=None):
            pass

    monkeypatch.setitem(sys.modules, "tileops.probe", types.SimpleNamespace(ProbeFwdOp=ProbeFwdOp))
    entry = {
        "family": "probe",
        "signature": {
            "params": {"dim": {"type": "int"}},
            "inputs": {"x": {"dtype": "T", "shape": "[M]"}, "y": {"optional": True}},
        },
    }
    assert validator._parity_errors("ProbeFwdOp", entry) == [
        "[signature] ProbeFwdOp: __init__ must end its policy parameters with *, target=None, kernel_map=None, tune=False",
        "[signature] ProbeFwdOp: __init__ parameter 'surprise' is not a signature or execution-policy parameter",
        "[signature] ProbeFwdOp: forward 'x' must have no default",
    ]


def test_a_composition_is_held_to_the_class_declarations(validator, monkeypatch):
    """`op` stages are `delegate_types` and `kernel` stages `kernel_types`, each in order."""

    class ProbeFwdOp:
        delegate_types = {"first": type("AFwdOp", (), {}), "second": type("BFwdOp", (), {})}
        kernel_types = {"own": object}

        def __init__(self, *, target=None, kernel_map=None, tune=False):
            pass

        def forward(self):
            pass

    monkeypatch.setitem(sys.modules, "tileops.probe", types.SimpleNamespace(ProbeFwdOp=ProbeFwdOp))
    entry = {"family": "probe", "signature": {}}
    stages = [
        {"name": "own", "kernel": "own"},
        {"name": "first", "op": "AFwdOp"},
        {"name": "second", "op": "BFwdOp", "optional": True},
    ]
    entry["composition"] = {"kind": "composite", "stages": stages}
    assert validator._parity_errors("ProbeFwdOp", entry) == []
    entry["composition"]["stages"] = [stages[2], stages[1]]
    assert validator._parity_errors("ProbeFwdOp", entry) == [
        "[signature] ProbeFwdOp: composition op stages [('second', 'BFwdOp'), ('first', 'AFwdOp')] "
        "are not delegate_types [('first', 'AFwdOp'), ('second', 'BFwdOp')]",
        "[signature] ProbeFwdOp: composition kernel stages [] are not kernel_types ['own']",
    ]


class TestBench:
    """A bench file takes its cases from the manifest and its roofline off the op.

    Which op the file benchmarks is a run-time fact, checked against a benchmark run by
    ``scripts/check_bench_coverage.py``, so no case here names an op.
    """

    @pytest.mark.parametrize(
        "text, expected",
        [
            (
                """\
                from benchmarks import api as bench
                for case in bench.cases(OP):
                    bench.Runner(OP(), case).compare({})
                """,
                [],
            ),
            (
                """\
                from benchmarks.api import Runner, cases
                for case in cases(OP):
                    Runner(OP(), case)
                """,
                [],
            ),
            (
                """\
                from benchmarks import api as bench
                cases = bench.cases(OP)
                """,
                ["Runner"],
            ),
            (
                """\
                from benchmarks import api as bench
                shapes = [(1024, 4096)]
                """,
                ["cases", "Runner"],
            ),
            ("def broken(\n", ["syntax error"]),
        ],
    )
    def test_the_contract_is_read_off_the_source(self, validator, tmp_path, text, expected):
        bench_file = tmp_path / "bench_probe.py"
        bench_file.write_text(textwrap.dedent(text))
        errors = validator.check_benchmark("probe", str(bench_file), REPO_ROOT)
        assert len(errors) == len(expected), errors
        for substring, error in zip(expected, errors, strict=True):
            assert substring in error, errors


class TestCheckOp:
    """--check-op scopes the entry checks to one op."""

    def test_an_unknown_op_is_reported(self, validator, tmp_path):
        path = _write_manifest(tmp_path, {"ProbeFwdOp": _entry()})
        errors, _ = validator.validate_manifest(
            manifest_path=path, repo_root=tmp_path, check_op="MissingFwdOp"
        )
        assert errors == ["--check-op: op 'MissingFwdOp' not found in manifest"], errors

    def test_other_ops_are_not_checked(self, validator, tmp_path):
        path = _write_manifest(
            tmp_path, {"ProbeFwdOp": _entry(), "BrokenFwdOp": _entry(status="done")}
        )
        errors, _ = validator.validate_manifest(
            manifest_path=path, repo_root=tmp_path, check_op="ProbeFwdOp"
        )
        assert errors == [], errors

    def test_a_non_mapping_manifest_root_is_reported(self, validator, tmp_path):
        path = tmp_path / "ops_manifest.yaml"
        path.write_text(yaml.safe_dump(["ProbeFwdOp"]))
        errors, _ = validator.validate_manifest(manifest_path=path, repo_root=tmp_path)
        assert any("top-level mapping" in e for e in errors), errors


class TestCompileContractRegistry:
    """Enforcement point for the compile contract.

    Must stay in this file: the always-on ``compile-contract-gate`` preflight job runs pytest
    on this file on a CPU runner.
    """

    def test_declarations_match_registered_evidence(self):
        """The implemented classes declaring a compile boundary are exactly the registered
        compile tests; a broken registration or a typo'd op name surfaces as a set diff."""
        from tests.compile_contract import compile_contract_ops
        from tileops.manifest import load_manifest
        from tileops.manifest.registry import op_class

        declared = {
            name
            for name, entry in load_manifest().items()
            if entry.get("status") == "implemented" and op_class(name, entry).compile_boundary
        }
        registered = compile_contract_ops()
        assert declared == registered, (
            f"evidence without declaration: {sorted(registered - declared)}; "
            f"declaration without evidence: {sorted(declared - registered)}"
        )
