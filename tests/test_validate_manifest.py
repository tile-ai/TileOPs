"""Tests for scripts/validate_manifest.py.

Verifies that the manifest validator correctly implements schema/signature/shape/dtype/bench checks.
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
# schema: YAML structure validation
# ---------------------------------------------------------------------------

def _make_entry(*, inputs=None, outputs=None, params=None, dtype_combos=None,
                 source_kernel="k.py", **extra):
    """Build a minimal valid manifest entry for testing, with overrides."""
    sig = {
        "inputs": inputs or {"x": {"dtype": "float16"}},
        "outputs": outputs or {"y": {"dtype": "same_as(x)"}},
    }
    if params is not None:
        sig["params"] = params
    if dtype_combos is not None:
        sig["dtype_combos"] = dtype_combos
    entry = {
        "family": "test",
        "signature": sig,
        "workloads": [{"x_shape": [1, 4096], "dtypes": ["float16"]}],
        "roofline": {"flops": "2 * M", "bytes": "M * 2"},
        "source": {
            "kernel": source_kernel, "op": "o.py",
            "test": "t.py", "bench": "b.py",
        },
    }
    entry.update(extra)
    return entry


class TestSchema:
    """schema checks that required fields exist and have correct types."""

    def test_non_dict_entry_fails(self, validator):
        """Non-dict entry must return schema error, not crash."""
        errors = validator.check_l0("bad_op", 123)
        assert any("must be a mapping" in e for e in errors)

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
        assert errors == [], f"Unexpected schema errors: {errors}"

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
        """signature.params as a YAML list must produce schema error, not crash."""
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
        assert any("params" in e and "schema" in e for e in errors), (
            f"Expected schema error about params being non-dict, got: {errors}"
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

    def test_layout_valid_passes(self, validator):
        """Tensor with valid layout field passes schema check (R19)."""
        entry = _make_entry(
            inputs={"x": {"dtype": "float16", "shape": "[N, H, W, C]",
                          "layout": "channels_last"}},
        )
        errors = validator.check_l0("test_op", entry)
        assert errors == [], f"Unexpected schema errors: {errors}"

    def test_layout_invalid_fails(self, validator):
        """Tensor with unrecognized layout value fails schema check (R19)."""
        entry = _make_entry(
            inputs={"x": {"dtype": "float16", "layout": "nchw"}},
        )
        errors = validator.check_l0("test_op", entry)
        assert any("layout" in e and "nchw" in e for e in errors)

    def test_params_missing_type_fails(self, validator):
        """Param entry without 'type' field fails schema check."""
        entry = _make_entry(params={"eps": {"default": 1e-6}})
        errors = validator.check_l0("test_op", entry)
        assert any("params.eps" in e and "type" in e for e in errors)

    def test_params_with_type_passes(self, validator):
        """Param entry with 'type' field passes schema check."""
        entry = _make_entry(
            params={"eps": {"type": "float", "default": 1e-6}},
        )
        errors = validator.check_l0("test_op", entry)
        assert errors == [], f"Unexpected schema errors: {errors}"

    def test_dtype_combos_valid_passes(self, validator):
        """Valid dtype_combos list passes schema check (R4)."""
        entry = _make_entry(
            inputs={"x": {"dtype": "float16"}, "w": {"dtype": "float16"}},
            dtype_combos=[{"x": "float16", "w": "float16"}],
        )
        errors = validator.check_l0("test_op", entry)
        assert errors == [], f"Unexpected schema errors: {errors}"

    def test_dtype_combos_bad_key_fails(self, validator):
        """dtype_combos referencing unknown tensor name fails (R4)."""
        entry = _make_entry(
            dtype_combos=[{"x": "float16", "nonexistent": "bfloat16"}],
        )
        errors = validator.check_l0("test_op", entry)
        assert any("nonexistent" in e and "dtype_combos" in e for e in errors)

    def test_source_kernel_list_passes(self, validator):
        """source.kernel as a list of strings passes schema check."""
        entry = _make_entry(
            source_kernel=["k1.py", "k2.py"],
        )
        errors = validator.check_l0("test_op", entry)
        assert errors == [], f"Unexpected schema errors: {errors}"

    def test_source_kernel_int_fails(self, validator):
        """source.kernel as non-string/non-list fails schema check."""
        entry = _make_entry(source_kernel=42)
        errors = validator.check_l0("test_op", entry)
        assert any("source.kernel" in e for e in errors)


# ---------------------------------------------------------------------------
# variant_of: cross-entry consistency (R16-R18)
# ---------------------------------------------------------------------------

class TestVariantOf:
    """variant_of checks cross-entry consistency."""

    def test_valid_variant_passes(self, validator):
        """Variant pointing to existing primary with shared source passes."""
        ops = {
            "moe_fused_moe": _make_entry(),
            "moe_fused_moe_cb": {
                **_make_entry(),
                "variant_of": "moe_fused_moe",
            },
        }
        errors = validator.check_variant_of_consistency(ops)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_variant_target_missing_fails(self, validator):
        """variant_of pointing to nonexistent entry fails (R16)."""
        ops = {
            "moe_fused_moe_cb": {
                **_make_entry(),
                "variant_of": "nonexistent",
            },
        }
        errors = validator.check_variant_of_consistency(ops)
        assert any("nonexistent" in e and "does not exist" in e for e in errors)

    def test_malformed_entry_does_not_crash(self, validator):
        """Non-dict entry must not crash variant_of check."""
        ops = {"bad": 123, "ok": _make_entry()}
        errors = validator.check_variant_of_consistency(ops)
        assert errors == []

    def test_variant_chaining_fails(self, validator):
        """Chained variant_of fails (R17)."""
        ops = {
            "primary": _make_entry(),
            "variant_a": {**_make_entry(), "variant_of": "primary"},
            "variant_b": {**_make_entry(), "variant_of": "variant_a"},
        }
        errors = validator.check_variant_of_consistency(ops)
        assert any("chaining" in e.lower() for e in errors)

    def test_variant_mismatched_kernel_fails(self, validator):
        """Variant with different source.kernel fails (R18)."""
        ops = {
            "primary": _make_entry(source_kernel="shared.py"),
            "variant": {
                **_make_entry(source_kernel="different.py"),
                "variant_of": "primary",
            },
        }
        errors = validator.check_variant_of_consistency(ops)
        assert any("source.kernel" in e and "R18" in e for e in errors)

    def test_variant_mismatched_op_fails(self, validator):
        """Variant with different source.op fails (R18)."""
        primary = _make_entry()
        variant = _make_entry()
        variant["source"]["op"] = "different_op.py"
        variant["variant_of"] = "primary"
        ops = {"primary": primary, "variant": variant}
        errors = validator.check_variant_of_consistency(ops)
        assert any("source.op" in e and "R18" in e for e in errors)


# ---------------------------------------------------------------------------
# signature: Op.forward() consistency
# ---------------------------------------------------------------------------

class TestSignature:
    """signature checks that Op.forward() params match manifest inputs."""

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
        assert any("do not match" in e for e in errors)

    def test_extra_forward_param_fails(self, validator):
        manifest_inputs = {"x": {"dtype": "float16"}}
        manifest_params = {}
        forward_params = ["x", "extra"]
        errors = validator.check_l1_signature(
            "test_op", manifest_inputs, manifest_params, forward_params,
        )
        assert any("do not match" in e for e in errors)

    def test_malformed_params_does_not_crash(self, validator):
        """signature check must return errors, not crash, when params is not a dict."""
        manifest_inputs = {"x": {"dtype": "float16"}}
        manifest_params = ["training"]  # list, not dict
        forward_params = ["x", "training"]
        errors = validator.check_l1_signature(
            "test_op", manifest_inputs, manifest_params, forward_params,
        )
        assert any("signature" in e and "params" in e.lower() for e in errors)

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

    def test_params_in_init_accepted(self, validator):
        """Manifest params that appear only in __init__() are valid."""
        manifest_inputs = {"x": {"dtype": "float16"}}
        manifest_params = {"eps": {"type": "float", "default": 1e-6}}
        forward_params = ["x"]
        init_params = ["M", "N", "dtype", "eps"]
        errors = validator.check_l1_signature(
            "test_op", manifest_inputs, manifest_params, forward_params,
            init_params=init_params,
        )
        assert errors == []

    def test_params_missing_from_both_init_and_forward_fails(self, validator):
        """Manifest params not in __init__() or forward() must fail."""
        manifest_inputs = {"x": {"dtype": "float16"}}
        manifest_params = {"dim": {"type": "int", "default": -1}}
        forward_params = ["x"]
        init_params = ["M", "N", "dtype", "eps"]
        errors = validator.check_l1_signature(
            "test_op", manifest_inputs, manifest_params, forward_params,
            init_params=init_params,
        )
        assert any("dim" in e for e in errors), (
            f"Expected error about 'dim' missing from init+forward, got: {errors}"
        )

    def test_params_split_across_init_and_forward_accepted(self, validator):
        """Some params in __init__, others in forward() — all valid."""
        manifest_inputs = {"x": {"dtype": "float16"}}
        manifest_params = {
            "eps": {"type": "float", "default": 1e-6},
            "training": {"type": "bool", "default": True},
        }
        forward_params = ["x", "training"]
        init_params = ["N", "dtype", "eps"]
        errors = validator.check_l1_signature(
            "test_op", manifest_inputs, manifest_params, forward_params,
            init_params=init_params,
        )
        assert errors == []

    def test_no_init_params_falls_back_to_forward_only(self, validator):
        """When init_params is None, only forward() is checked (backward compat)."""
        manifest_inputs = {"x": {"dtype": "float16"}}
        manifest_params = {"eps": {"type": "float", "default": 1e-6}}
        forward_params = ["x"]
        # No init_params — eps is in neither, should fail
        errors = validator.check_l1_signature(
            "test_op", manifest_inputs, manifest_params, forward_params,
        )
        assert any("eps" in e for e in errors), (
            f"Expected error about 'eps' not found, got: {errors}"
        )


# ---------------------------------------------------------------------------
# dtype: dtype string conformance
# ---------------------------------------------------------------------------

class TestDtype:
    """dtype checks that dtype strings are valid torch dtype names."""

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
        """Workloads with unrecognized dtype must produce dtype error."""
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
            },
            "workloads": [{"dtypes": ["not_a_dtype"]}],
        }
        errors = validator.check_l3("test_op", entry)
        assert any("not_a_dtype" in e and "dtype" in e for e in errors), (
            f"Expected dtype error for invalid workload dtype, got: {errors}"
        )

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
# bench: benchmark uses manifest workloads
# ---------------------------------------------------------------------------

class TestBench:
    """bench checks that bench files import from tileops.manifest."""

    def test_bench_with_load_workloads_passes(self, validator, tmp_path):
        bench_file = tmp_path / "bench_test.py"
        bench_file.write_text(
            "from tileops.manifest import load_workloads, eval_roofline\n"
            "workloads = load_workloads('test_op')\n"
            "eval_roofline('test_op')\n"
        )
        errors = validator.check_l4_benchmark("test_op", str(bench_file), REPO_ROOT)
        assert errors == []

    def test_bench_with_load_workloads_only_fails(self, validator, tmp_path):
        """Bench using load_workloads but not eval_roofline fails bench validation."""
        bench_file = tmp_path / "bench_test.py"
        bench_file.write_text(
            "from tileops.manifest import load_workloads\n"
            "workloads = load_workloads('test_op')\n"
        )
        errors = validator.check_l4_benchmark("test_op", str(bench_file), REPO_ROOT)
        assert any("eval_roofline" in e for e in errors), (
            f"Expected bench error about missing eval_roofline, got: {errors}"
        )

    def test_bench_without_load_workloads_fails(self, validator, tmp_path):
        bench_file = tmp_path / "bench_test.py"
        bench_file.write_text(
            "import pytest\n"
            "shapes = [(1024, 4096)]\n"
        )
        errors = validator.check_l4_benchmark("test_op", str(bench_file), REPO_ROOT)
        assert any("load_workloads" in e for e in errors)

    def test_wrong_op_name_fails_l4(self, validator, tmp_path):
        """Calling manifest helpers with a different op name must fail."""
        bench_file = tmp_path / "bench_test.py"
        bench_file.write_text(textwrap.dedent("""\
            from tileops.manifest import load_workloads, eval_roofline
            workloads = load_workloads('wrong_op')
            eval_roofline('wrong_op')
        """))
        errors = validator.check_l4_benchmark("test_op", str(bench_file), REPO_ROOT)
        assert any("load_workloads" in e for e in errors)
        assert any("eval_roofline" in e for e in errors)

    def test_syntax_error_in_bench_file_fails_l4(self, validator, tmp_path):
        """A bench file with syntax errors produces an bench error."""
        bench_file = tmp_path / "bench_test.py"
        bench_file.write_text("def broken(\n")
        errors = validator.check_l4_benchmark("test_op", str(bench_file), REPO_ROOT)
        assert any("syntax error" in e for e in errors)


# ---------------------------------------------------------------------------
# --check-op: force all levels on a specific op, ignoring status
# ---------------------------------------------------------------------------

class TestCheckOp:
    """--check-op forces all validation levels on a named op, ignoring spec-only."""

    def test_spec_only_op_with_check_op_runs_all_levels(self, validator, tmp_path):
        """When check_op matches a spec-only op, L1-L4 checks run (not skipped)."""
        # Create a minimal bench file that will fail L4 (no load_workloads)
        bench_file = tmp_path / "bench_test.py"
        bench_file.write_text("import pytest\n")

        entry = _make_entry(status="spec-only")
        entry["source"]["bench"] = str(bench_file)
        entry["source"]["bench_manifest_driven"] = True

        # Build a temp manifest and call validate_manifest directly.
        manifest_file = tmp_path / "ops_manifest.yaml"
        import yaml
        manifest_file.write_text(yaml.safe_dump({"ops": {"my_op": entry}}))

        # Without check_op: spec-only op skips L1-L4
        errors_no_flag, warnings_no_flag = validator.validate_manifest(
            manifest_path=manifest_file,
            repo_root=tmp_path,
        )
        # Should have no bench errors (spec-only skips L4)
        bench_errors_no_flag = [e for e in errors_no_flag if "[bench]" in e]
        assert bench_errors_no_flag == [], (
            f"Spec-only op should skip bench check without --check-op: {bench_errors_no_flag}"
        )

        # With check_op="my_op": forces all levels despite spec-only
        errors_flag, warnings_flag = validator.validate_manifest(
            manifest_path=manifest_file,
            repo_root=tmp_path,
            check_op="my_op",
        )
        # Should now have bench errors (L4 ran)
        bench_errors_flag = [e for e in errors_flag if "[bench]" in e]
        assert len(bench_errors_flag) > 0, (
            "With --check-op, spec-only op should run bench check"
        )

    def test_spec_only_op_without_check_op_still_skipped(self, validator, tmp_path):
        """Default behavior unchanged: spec-only ops skip L1-L4."""
        entry = _make_entry(status="spec-only")

        manifest_file = tmp_path / "ops_manifest.yaml"
        import yaml
        manifest_file.write_text(yaml.safe_dump({"ops": {"my_op": entry}}))

        errors, warnings = validator.validate_manifest(
            manifest_path=manifest_file,
            repo_root=tmp_path,
        )
        # No signature/shape/dtype/bench errors for spec-only
        non_schema = [e for e in errors if "[schema]" not in e]
        assert non_schema == [], (
            f"Spec-only op should only have schema errors (if any), got: {non_schema}"
        )

    def test_check_op_nonexistent_op_reports_error(self, validator, tmp_path):
        """--check-op with a name not in manifest reports an error."""
        entry = _make_entry()

        manifest_file = tmp_path / "ops_manifest.yaml"
        import yaml
        manifest_file.write_text(yaml.safe_dump({"ops": {"my_op": entry}}))

        errors, warnings = validator.validate_manifest(
            manifest_path=manifest_file,
            repo_root=tmp_path,
            check_op="nonexistent_op",
        )
        assert any("nonexistent_op" in e and "not found" in e for e in errors), (
            f"Expected error about 'nonexistent_op' not found in manifest, got: {errors}"
        )

    def test_check_op_scopes_to_single_op(self, validator, tmp_path):
        """--check-op validates only the named op; unrelated ops are not processed."""
        import yaml

        # target_op: spec-only, has a real bench file -> L4 will run and find errors
        bench_file = tmp_path / "bench_target.py"
        bench_file.write_text("import pytest\n")
        target_entry = _make_entry(status="spec-only")
        target_entry["source"]["bench"] = str(bench_file)
        target_entry["source"]["bench_manifest_driven"] = True

        # other_op: implemented (not spec-only), points to a nonexistent kernel
        # If validated, L1 would fail trying to import a missing module.
        other_entry = _make_entry(source_kernel="nonexistent_impl.py")

        manifest_file = tmp_path / "ops_manifest.yaml"
        manifest_file.write_text(yaml.safe_dump(
            {"ops": {"target_op": target_entry, "other_op": other_entry}},
        ))

        # With check_op="target_op": only target_op is validated.
        # other_op must be completely skipped -- no import errors from its
        # missing kernel.
        errors, _ = validator.validate_manifest(
            manifest_path=manifest_file,
            repo_root=tmp_path,
            check_op="target_op",
        )
        other_errors = [e for e in errors if "other_op" in e]
        assert other_errors == [], (
            f"--check-op should not validate unrelated ops, but got: {other_errors}"
        )
        # target_op should have been validated (bench errors expected)
        target_errors = [e for e in errors if "target_op" in e]
        assert len(target_errors) > 0, (
            "target_op should have validation errors from forced L4 check"
        )

    def test_check_op_ignores_unrelated_variant_of_errors(self, validator, tmp_path):
        """--check-op must not report variant_of errors from unrelated ops.

        Regression: check_variant_of_consistency ran across the full manifest
        before the check_op filter, causing --check-op to fail on unrelated
        ops with invalid variant_of references.
        """
        import yaml

        target_entry = _make_entry()
        # other_op has an invalid variant_of pointing to a nonexistent primary
        other_entry = _make_entry()
        other_entry["variant_of"] = "nonexistent_primary"

        manifest_file = tmp_path / "ops_manifest.yaml"
        manifest_file.write_text(yaml.safe_dump(
            {"ops": {"target_op": target_entry, "other_op": other_entry}},
        ))

        # With check_op="target_op": the invalid variant_of on other_op
        # must not appear in errors.
        errors, _ = validator.validate_manifest(
            manifest_path=manifest_file,
            repo_root=tmp_path,
            check_op="target_op",
        )
        variant_errors = [e for e in errors if "variant_of" in e]
        assert variant_errors == [], (
            f"--check-op should not report variant_of errors from unrelated ops: "
            f"{variant_errors}"
        )

        # Without check_op: the variant_of error IS reported.
        errors_all, _ = validator.validate_manifest(
            manifest_path=manifest_file,
            repo_root=tmp_path,
            check_op=None,
        )
        variant_errors_all = [e for e in errors_all if "variant_of" in e]
        assert len(variant_errors_all) > 0, (
            "Without --check-op, invalid variant_of should be reported"
        )

    def test_check_op_validates_variant_family(self, validator, tmp_path):
        """--check-op on a primary also validates its immediate variants.

        Regression: an agent could modify a variant to break R17/R18 rules,
        and --check-op <primary> would still pass because variants were
        excluded from the validation scope.
        """
        import yaml

        primary = _make_entry(source_kernel="shared_kernel.py")
        # Variant shares source with primary (valid)
        valid_variant = _make_entry(source_kernel="shared_kernel.py")
        valid_variant["variant_of"] = "primary_op"

        # Broken variant: different source.kernel violates R18
        broken_variant = _make_entry(source_kernel="different_kernel.py")
        broken_variant["variant_of"] = "primary_op"

        manifest_file = tmp_path / "ops_manifest.yaml"
        manifest_file.write_text(yaml.safe_dump({"ops": {
            "primary_op": primary,
            "good_variant": valid_variant,
            "bad_variant": broken_variant,
        }}))

        # check_op="primary_op" must catch the R18 violation on bad_variant
        errors, _ = validator.validate_manifest(
            manifest_path=manifest_file,
            repo_root=tmp_path,
            check_op="primary_op",
        )
        r18_errors = [e for e in errors if "bad_variant" in e and "R18" in e]
        assert len(r18_errors) > 0, (
            f"--check-op on primary must catch R18 violation in variant, "
            f"got errors: {errors}"
        )

        # good_variant should NOT have R18 errors
        good_r18 = [e for e in errors if "good_variant" in e and "R18" in e]
        assert good_r18 == [], (
            f"good_variant should pass R18, got: {good_r18}"
        )

    def test_check_op_variant_family_runs_schema_on_variants(self, validator, tmp_path):
        """--check-op on primary runs per-op schema checks on variants too."""
        import yaml

        primary = _make_entry(source_kernel="shared.py")
        # Variant with broken source (missing required fields)
        broken_variant = {
            "family": "test",
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
            },
            "workloads": [{"x_shape": [1, 4096], "dtypes": ["float16"]}],
            "roofline": {"flops": "2 * M", "bytes": "M * 2"},
            "source": {
                "kernel": "shared.py",
                # missing "op", "test", "bench" fields
            },
            "variant_of": "primary_op",
        }

        manifest_file = tmp_path / "ops_manifest.yaml"
        manifest_file.write_text(yaml.safe_dump({"ops": {
            "primary_op": primary,
            "broken_var": broken_variant,
        }}))

        errors, _ = validator.validate_manifest(
            manifest_path=manifest_file,
            repo_root=tmp_path,
            check_op="primary_op",
        )
        schema_errors = [e for e in errors if "broken_var" in e and "source" in e]
        assert len(schema_errors) > 0, (
            f"--check-op on primary must run schema checks on variants, "
            f"got errors: {errors}"
        )

    def test_check_op_cli_parsing(self, validator):
        """_parse_check_op extracts the op name from argv."""
        assert validator._parse_check_op(["--check-op", "softmax_fwd"]) == "softmax_fwd"
        assert validator._parse_check_op(["--check-op=softmax_fwd"]) == "softmax_fwd"
        assert validator._parse_check_op(["--verbose"]) is None
        assert validator._parse_check_op([]) is None

    def test_check_op_rejects_missing_value(self, validator):
        """_parse_check_op exits with status 2 when no op name is given."""
        with pytest.raises(SystemExit, match="2"):
            validator._parse_check_op(["--check-op"])

    def test_check_op_rejects_flag_as_value(self, validator):
        """_parse_check_op exits with status 2 when value looks like a flag."""
        with pytest.raises(SystemExit, match="2"):
            validator._parse_check_op(["--check-op", "--verbose"])

    def test_check_op_rejects_empty_equals_value(self, validator):
        """_parse_check_op exits with status 2 for --check-op= (empty value)."""
        with pytest.raises(SystemExit, match="2"):
            validator._parse_check_op(["--check-op="])


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
