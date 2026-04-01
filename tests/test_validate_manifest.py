"""Tests for scripts/validate_manifest.py.

Verifies that the manifest validator correctly implements L0-L4 checks.
Uses synthetic manifest data to test individual check functions,
plus an integration test against the real ops_manifest.yaml.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATOR_SCRIPT = REPO_ROOT / "scripts" / "validate_manifest.py"


# ---------------------------------------------------------------------------
# Import the validator module dynamically (it lives in scripts/, not a package)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def validator():
    """Import validate_manifest as a module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("validate_manifest", VALIDATOR_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# L0: YAML schema validation
# ---------------------------------------------------------------------------

class TestL0Schema:
    """L0 checks that required fields exist and have correct types."""

    def test_valid_entry_passes(self, validator):
        entry = {
            "family": "norm",
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "params": {},
                "shape_rules": ["y.shape == x.shape"],
            },
            "workloads": [{"x_shape": [1, 4096], "dtypes": ["float16"]}],
            "roofline": {"flops": "2 * M * N", "bytes": "M * N * 2"},
            "source": {
                "kernel": "tileops/kernels/norm/rms_norm.py",
                "op": "tileops/ops/norm/rms_norm.py",
                "test": "tests/ops/test_rms_norm.py",
                "bench": "benchmarks/ops/bench_rms_norm.py",
            },
        }
        errors = validator.check_l0("test_op", entry)
        assert errors == [], f"Unexpected L0 errors: {errors}"

    def test_missing_family_fails(self, validator):
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "params": {},
            },
            "workloads": [{"x_shape": [1, 4096], "dtypes": ["float16"]}],
            "roofline": {"flops": "2 * M", "bytes": "M * 2"},
            "source": {
                "kernel": "k.py", "op": "o.py",
                "test": "t.py", "bench": "b.py",
            },
        }
        errors = validator.check_l0("test_op", entry)
        assert any("family" in e for e in errors)

    def test_missing_signature_fields_fails(self, validator):
        entry = {
            "family": "norm",
            "signature": {"inputs": {"x": {"dtype": "float16"}}},
            "workloads": [{"x_shape": [1, 4096], "dtypes": ["float16"]}],
            "roofline": {"flops": "2 * M", "bytes": "M * 2"},
            "source": {
                "kernel": "k.py", "op": "o.py",
                "test": "t.py", "bench": "b.py",
            },
        }
        errors = validator.check_l0("test_op", entry)
        assert any("outputs" in e for e in errors)

    def test_roofline_needs_flops_and_bytes_or_func(self, validator):
        entry = {
            "family": "norm",
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "float16"}},
                "params": {},
            },
            "workloads": [{"x_shape": [1, 4096], "dtypes": ["float16"]}],
            "roofline": {"flops": "2 * M"},  # missing bytes
            "source": {
                "kernel": "k.py", "op": "o.py",
                "test": "t.py", "bench": "b.py",
            },
        }
        errors = validator.check_l0("test_op", entry)
        assert any("roofline" in e.lower() or "bytes" in e.lower() for e in errors)

    def test_params_as_list_fails(self, validator):
        """signature.params as a YAML list must produce L0 error, not crash."""
        entry = {
            "family": "norm",
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "params": ["training", "epsilon"],  # list instead of mapping
                "shape_rules": ["y.shape == x.shape"],
            },
            "workloads": [{"x_shape": [1, 4096], "dtypes": ["float16"]}],
            "roofline": {"flops": "2 * M * N", "bytes": "M * N * 2"},
            "source": {
                "kernel": "k.py", "op": "o.py",
                "test": "t.py", "bench": "b.py",
            },
        }
        errors = validator.check_l0("test_op", entry)
        assert any("params" in e and "L0" in e for e in errors), (
            f"Expected L0 error about params being non-dict, got: {errors}"
        )

    def test_tensor_missing_dtype_fails(self, validator):
        entry = {
            "family": "norm",
            "signature": {
                "inputs": {"x": {}},  # no dtype
                "outputs": {"y": {"dtype": "float16"}},
                "params": {},
            },
            "workloads": [{"x_shape": [1, 4096], "dtypes": ["float16"]}],
            "roofline": {"flops": "2 * M", "bytes": "M * 2"},
            "source": {
                "kernel": "k.py", "op": "o.py",
                "test": "t.py", "bench": "b.py",
            },
        }
        errors = validator.check_l0("test_op", entry)
        assert any("dtype" in e for e in errors)


# ---------------------------------------------------------------------------
# L1: Signature consistency
# ---------------------------------------------------------------------------

class TestL1Signature:
    """L1 checks that Op.forward() params match manifest inputs."""

    def test_matching_signature_passes(self, validator):
        manifest_inputs = {"x": {"dtype": "float16"}, "weight": {"dtype": "same_as(x)"}}
        manifest_params = {}
        forward_params = ["x", "weight"]
        errors = validator.check_l1_signature(
            "test_op", manifest_inputs, manifest_params, forward_params,
        )
        assert errors == []

    def test_missing_forward_param_fails(self, validator):
        manifest_inputs = {"x": {"dtype": "float16"}, "weight": {"dtype": "same_as(x)"}}
        manifest_params = {}
        forward_params = ["x"]  # missing 'weight'
        errors = validator.check_l1_signature(
            "test_op", manifest_inputs, manifest_params, forward_params,
        )
        assert any("weight" in e for e in errors)

    def test_extra_forward_param_fails(self, validator):
        manifest_inputs = {"x": {"dtype": "float16"}}
        manifest_params = {}
        forward_params = ["x", "extra"]
        errors = validator.check_l1_signature(
            "test_op", manifest_inputs, manifest_params, forward_params,
        )
        assert any("extra" in e for e in errors)

    def test_malformed_params_does_not_crash(self, validator):
        """L1 must return errors, not crash, when params is not a dict."""
        manifest_inputs = {"x": {"dtype": "float16"}}
        manifest_params = ["training"]  # list, not dict
        forward_params = ["x", "training"]
        errors = validator.check_l1_signature(
            "test_op", manifest_inputs, manifest_params, forward_params,
        )
        assert any("L1" in e and "params" in e.lower() for e in errors)

    def test_params_in_forward_accepted(self, validator):
        """Manifest params that appear as forward() args are valid."""
        manifest_inputs = {
            "x": {"dtype": "float16"},
            "weight": {"dtype": "float32"},
        }
        manifest_params = {"training": {"type": "bool", "default": True}}
        forward_params = ["x", "weight", "training"]
        errors = validator.check_l1_signature(
            "test_op", manifest_inputs, manifest_params, forward_params,
        )
        assert errors == []

    def test_import_error_skips_l1_with_warning(self, validator, monkeypatch):
        """L1 gracefully skips when Op module cannot be imported (missing deps)."""
        # Simulate _resolve_op_class returning an import error
        monkeypatch.setattr(
            validator,
            "_resolve_op_class",
            lambda op_file, op_name: validator._ResolveResult(import_error=True),
        )
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "params": {},
            },
            "source": {"op": "tileops/ops/norm/rms_norm.py"},
        }
        warnings: list[str] = []
        errors = validator.check_l1("test_op", entry, warnings=warnings)
        assert errors == [], f"Expected no errors when import fails, got: {errors}"
        assert len(warnings) == 1
        assert "skipped" in warnings[0]
        assert "missing dependencies" in warnings[0]

    def test_resolve_failure_without_import_error_is_error(self, validator, monkeypatch):
        """L1 reports an error when Op class is not found (not an import issue)."""
        monkeypatch.setattr(
            validator,
            "_resolve_op_class",
            lambda op_file, op_name: validator._ResolveResult(),
        )
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "params": {},
            },
            "source": {"op": "tileops/ops/norm/rms_norm.py"},
        }
        warnings: list[str] = []
        errors = validator.check_l1("test_op", entry, warnings=warnings)
        assert len(errors) == 1
        assert "could not resolve" in errors[0]
        assert warnings == []


# ---------------------------------------------------------------------------
# L3: Dtype conformance
# ---------------------------------------------------------------------------

class TestL3Dtype:
    """L3 checks that dtype strings are valid torch dtype names."""

    def test_valid_signature_dtypes_pass(self, validator):
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16"}, "w": {"dtype": "bfloat16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
            },
            "workloads": [{"dtypes": ["float16", "bfloat16"]}],
        }
        errors = validator.check_l3("test_op", entry)
        assert errors == []

    def test_invalid_workload_dtype_fails(self, validator):
        """Workloads with unrecognized dtype must produce L3 error."""
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
            },
            "workloads": [{"dtypes": ["not_a_dtype"]}],
        }
        errors = validator.check_l3("test_op", entry)
        assert any("not_a_dtype" in e and "L3" in e for e in errors), (
            f"Expected L3 error for invalid workload dtype, got: {errors}"
        )

    def test_multiple_invalid_workload_dtypes(self, validator):
        """Multiple invalid workload dtypes each produce their own error."""
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
            },
            "workloads": [
                {"dtypes": ["float16", "bogus1"]},
                {"dtypes": ["bogus2"]},
            ],
        }
        errors = validator.check_l3("test_op", entry)
        assert any("bogus1" in e for e in errors)
        assert any("bogus2" in e for e in errors)

    def test_valid_workload_dtype_passes(self, validator):
        """Valid workload dtypes do not produce errors."""
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
            },
            "workloads": [{"dtypes": ["float16", "bfloat16", "float32"]}],
        }
        errors = validator.check_l3("test_op", entry)
        assert errors == []


# ---------------------------------------------------------------------------
# L4: Benchmark file uses manifest workloads
# ---------------------------------------------------------------------------

class TestL4Benchmark:
    """L4 checks that bench files import from tileops.manifest."""

    def test_bench_with_load_workloads_passes(self, validator, tmp_path):
        bench_file = tmp_path / "bench_test.py"
        bench_file.write_text(
            "from tileops.manifest import load_workloads, eval_roofline\n"
            "workloads = load_workloads('test_op')\n"
            "eval_roofline(op, workload)\n"
        )
        errors, warnings = validator.check_l4_benchmark("test_op", str(bench_file), REPO_ROOT)
        assert errors == [] and warnings == []

    def test_bench_with_both_patterns_passes(self, validator, tmp_path):
        """Bench using both load_workloads and eval_roofline passes L4."""
        bench_file = tmp_path / "bench_test.py"
        bench_file.write_text(
            "from tileops.manifest import load_workloads, eval_roofline\n"
            "workloads = load_workloads('test_op')\n"
            "eval_roofline(op, workload)\n"
        )
        errors, warnings = validator.check_l4_benchmark("test_op", str(bench_file), REPO_ROOT)
        assert errors == [] and warnings == []

    def test_bench_with_load_workloads_only_warns(self, validator, tmp_path):
        """Bench using load_workloads but not eval_roofline warns on L4."""
        bench_file = tmp_path / "bench_test.py"
        bench_file.write_text(
            "from tileops.manifest import load_workloads\n"
            "workloads = load_workloads('test_op')\n"
        )
        errors, warnings = validator.check_l4_benchmark("test_op", str(bench_file), REPO_ROOT)
        assert errors == []
        assert any("eval_roofline" in w for w in warnings), (
            f"Expected L4 warning about missing eval_roofline, got: {warnings}"
        )

    def test_bench_without_load_workloads_warns(self, validator, tmp_path):
        bench_file = tmp_path / "bench_test.py"
        bench_file.write_text(
            "import pytest\n"
            "shapes = [(1024, 4096)]\n"
        )
        errors, warnings = validator.check_l4_benchmark("test_op", str(bench_file), REPO_ROOT)
        assert errors == []
        assert len(warnings) > 0
        assert any("load_workloads" in w for w in warnings)

    def test_comments_only_do_not_satisfy_l4(self, validator, tmp_path):
        """Substring in comments must NOT pass L4 (AST-based check)."""
        bench_file = tmp_path / "bench_test.py"
        bench_file.write_text(textwrap.dedent("""\
            # load_workloads
            # eval_roofline
            shapes = [(1024, 4096)]
        """))
        errors, warnings = validator.check_l4_benchmark("test_op", str(bench_file), REPO_ROOT)
        assert any("load_workloads" in w for w in warnings), (
            f"Comments-only should warn on L4, got: {warnings}"
        )
        assert any("eval_roofline" in w for w in warnings)

    def test_import_without_call_warns_l4(self, validator, tmp_path):
        """Importing but never calling load_workloads/eval_roofline warns on L4."""
        bench_file = tmp_path / "bench_test.py"
        bench_file.write_text(textwrap.dedent("""\
            from tileops.manifest import load_workloads, eval_roofline
            shapes = [(1024, 4096)]
        """))
        errors, warnings = validator.check_l4_benchmark("test_op", str(bench_file), REPO_ROOT)
        assert errors == []
        assert any("load_workloads" in w for w in warnings), (
            f"Import-only (no call) should warn on L4, got: {warnings}"
        )

    def test_call_without_import_warns_l4(self, validator, tmp_path):
        """Calling load_workloads without importing it warns on L4."""
        bench_file = tmp_path / "bench_test.py"
        bench_file.write_text(textwrap.dedent("""\
            workloads = load_workloads('test_op')
            eval_roofline(op, workload)
        """))
        errors, warnings = validator.check_l4_benchmark("test_op", str(bench_file), REPO_ROOT)
        assert errors == []
        assert any("load_workloads" in w for w in warnings), (
            f"Call-without-import should warn on L4, got: {warnings}"
        )

    def test_attribute_style_only_warns_l4(self, validator, tmp_path):
        """manifest.load_workloads() without bare call warns on L4.

        The import check requires ``from tileops.manifest import name``, and
        usage check requires a bare ``name(...)`` call. Attribute-style calls
        like ``manifest.load_workloads(...)`` could be on any object and are
        not accepted.
        """
        bench_file = tmp_path / "bench_test.py"
        bench_file.write_text(textwrap.dedent("""\
            from tileops import manifest
            from tileops.manifest import load_workloads, eval_roofline
            workloads = manifest.load_workloads('test_op')
            manifest.eval_roofline(op, workload)
        """))
        errors, warnings = validator.check_l4_benchmark("test_op", str(bench_file), REPO_ROOT)
        assert errors == []
        assert any("load_workloads" in w for w in warnings), (
            f"Attribute-only usage should warn on L4, got: {warnings}"
        )

    def test_spoofed_attribute_call_warns_l4(self, validator, tmp_path):
        """fake.load_workloads('wrong_op') must warn on L4."""
        bench_file = tmp_path / "bench_test.py"
        bench_file.write_text(textwrap.dedent("""\
            from tileops.manifest import load_workloads, eval_roofline
            workloads = fake.load_workloads('wrong_op')
            fake.eval_roofline('wrong_op')
        """))
        errors, warnings = validator.check_l4_benchmark("test_op", str(bench_file), REPO_ROOT)
        assert errors == []
        assert any("load_workloads" in w for w in warnings), (
            f"Spoofed attribute call should warn on L4, got: {warnings}"
        )
        assert any("eval_roofline" in w for w in warnings), (
            f"Spoofed attribute call should warn on L4 for eval_roofline, got: {warnings}"
        )

    def test_import_from_wrong_module_warns_l4(self, validator, tmp_path):
        """Importing load_workloads from a non-tileops.manifest module warns on L4."""
        bench_file = tmp_path / "bench_test.py"
        bench_file.write_text(textwrap.dedent("""\
            from fake_module import load_workloads, eval_roofline
            workloads = load_workloads('test_op')
            eval_roofline(op, workload)
        """))
        errors, warnings = validator.check_l4_benchmark("test_op", str(bench_file), REPO_ROOT)
        assert errors == []
        assert any("load_workloads" in w for w in warnings), (
            f"Import from wrong module should warn on L4, got: {warnings}"
        )
        assert any("eval_roofline" in w for w in warnings), (
            f"Import from wrong module should warn on L4 for eval_roofline, got: {warnings}"
        )

    def test_syntax_error_in_bench_file_fails_l4(self, validator, tmp_path):
        """A bench file with syntax errors produces an L4 error."""
        bench_file = tmp_path / "bench_test.py"
        bench_file.write_text("def broken(\n")
        errors, warnings = validator.check_l4_benchmark("test_op", str(bench_file), REPO_ROOT)
        assert any("syntax error" in e for e in errors)


# ---------------------------------------------------------------------------
# --levels flag (T009: CI level selection)
# ---------------------------------------------------------------------------

class TestLevelsFlag:
    """Verify that --levels controls which validation levels run."""

    def test_parse_levels_returns_none_when_absent(self, validator):
        assert validator._parse_levels(["script.py"]) is None

    def test_parse_levels_with_flag(self, validator):
        result = validator._parse_levels(["script.py", "--levels", "L0,L2,L3"])
        assert result == frozenset({"L0", "L2", "L3"})

    def test_parse_levels_with_equals(self, validator):
        result = validator._parse_levels(["script.py", "--levels=L0,L4"])
        assert result == frozenset({"L0", "L4"})

    def test_parse_levels_case_insensitive(self, validator):
        result = validator._parse_levels(["script.py", "--levels", "l0,l2"])
        assert result == frozenset({"L0", "L2"})

    def test_validate_manifest_skips_l1_when_not_in_levels(self, validator, tmp_path):
        """When levels excludes L1, no L1 checks run and no warnings appear."""
        # Create a minimal manifest that would trigger L1 import-error warnings
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text(textwrap.dedent("""\
            ops:
              test_op:
                family: test
                signature:
                  inputs:
                    x: {dtype: float16}
                  outputs:
                    y: {dtype: "same_as(x)"}
                  params: {}
                  shape_rules: ["y.shape == x.shape"]
                workloads:
                  - {x_shape: [1, 4096], dtypes: [float16]}
                roofline: {flops: "2 * M", bytes: "M * 2"}
                source:
                  kernel: nonexistent/kernel.py
                  op: nonexistent/op.py
                  test: nonexistent/test.py
                  bench: nonexistent/bench.py
        """))
        # L0,L2,L3 only — no L1, no L4
        errors, warnings = validator.validate_manifest(
            manifest_path=manifest,
            repo_root=tmp_path,
            levels=frozenset({"L0", "L2", "L3"}),
        )
        # No L1 warnings should be emitted
        assert not any("[L1]" in w for w in warnings), (
            f"L1 should not run when excluded from levels, got warnings: {warnings}"
        )

    def test_validate_manifest_skips_l4_when_not_in_levels(self, validator, tmp_path):
        """When levels excludes L4, missing bench file does not cause L4 error."""
        manifest = tmp_path / "manifest.yaml"
        # Create op source file so it is not spec-only
        op_dir = tmp_path / "ops"
        op_dir.mkdir()
        (op_dir / "op.py").write_text("class TestOp:\n    pass\n")
        manifest.write_text(textwrap.dedent("""\
            ops:
              test_op:
                family: test
                signature:
                  inputs:
                    x: {dtype: float16}
                  outputs:
                    y: {dtype: "same_as(x)"}
                  params: {}
                workloads:
                  - {x_shape: [1, 4096], dtypes: [float16]}
                roofline: {flops: "2 * M", bytes: "M * 2"}
                source:
                  kernel: ops/kernel.py
                  op: ops/op.py
                  test: ops/test.py
                  bench: nonexistent/bench.py
        """))
        errors, warnings = validator.validate_manifest(
            manifest_path=manifest,
            repo_root=tmp_path,
            levels=frozenset({"L0"}),
        )
        assert not any("[L4]" in e for e in errors), (
            f"L4 should not run when excluded, got: {errors}"
        )


# ---------------------------------------------------------------------------
# Integration: validate_manifest.py passes on the real codebase
# ---------------------------------------------------------------------------

class TestIntegration:
    """Run the actual validator script and verify it passes."""

    def test_validator_passes_on_current_codebase(self):
        result = subprocess.run(
            [sys.executable, str(VALIDATOR_SCRIPT)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, (
            f"Validator failed with return code {result.returncode}.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    def test_validator_with_levels_flag_passes(self):
        """Preflight-style invocation with --levels L0,L2,L3,L4 passes."""
        result = subprocess.run(
            [sys.executable, str(VALIDATOR_SCRIPT), "--levels", "L0,L2,L3,L4"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, (
            f"Validator with --levels failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
