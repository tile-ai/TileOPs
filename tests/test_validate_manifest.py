"""Tests for scripts/validate_manifest.py.

Verifies that the manifest validator correctly implements schema/signature/shape/dtype/bench checks.
Uses synthetic manifest data to test individual check functions,
plus an integration test against the real ops manifest (tileops/manifest/).
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
                 source_kernel="k.py", status="spec-only", kernel_map=None,
                 **extra):
    """Build a minimal valid manifest entry for testing, with overrides.

    Use ``status=None`` to explicitly omit the status field (for testing
    that the validator rejects entries without status).
    ``kernel_map`` is placed under ``source`` per the manifest spec.
    """
    sig = {
        "inputs": inputs or {"x": {"dtype": "float16"}},
        "outputs": outputs or {"y": {"dtype": "same_as(x)"}},
    }
    if params is not None:
        sig["params"] = params
    if dtype_combos is not None:
        sig["dtype_combos"] = dtype_combos
    source = {
        "kernel": source_kernel, "op": "o.py",
        "test": "t.py", "bench": "b.py",
    }
    if kernel_map is not None:
        source["kernel_map"] = kernel_map
    entry = {
        "family": "test",
        "ref_api": "none",
        "signature": sig,
        "workloads": [{"x_shape": [1, 4096], "dtypes": ["float16"]}],
        "roofline": {"flops": "2 * M", "bytes": "M * 2"},
        "source": source,
    }
    if status is not None:
        entry["status"] = status
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
            "family": "normalization",
            "ref_api": "torch.nn.functional.rms_norm",
            "status": "implemented",
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
                "kernel_map": {"fwd": "RMSNormFwdKernel"},
                "op": "tileops/ops/norm/rms_norm.py",
                "test": "tests/ops/test_rms_norm.py",
                "bench": "benchmarks/ops/bench_rms_norm.py",
            },
        }
        errors = validator.check_l0("test_op", entry)
        assert errors == [], f"Unexpected schema errors: {errors}"

    def test_missing_ref_api_fails(self, validator):
        entry = _make_entry()
        del entry["ref_api"]
        errors = validator.check_l0("test_op", entry)
        assert any("ref_api" in e for e in errors)

    def test_ref_api_non_string_fails(self, validator):
        entry = _make_entry(ref_api=123)
        errors = validator.check_l0("test_op", entry)
        assert any("ref_api" in e and "string" in e for e in errors)

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
            "family": "normalization",
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
            "family": "normalization",
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
            "family": "normalization",
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
            "family": "normalization",
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

    def test_static_dims_dict_passes(self, validator):
        """Valid static_dims mapping passes schema check (R20)."""
        entry = _make_entry()
        entry["signature"]["static_dims"] = {"N": "x.shape[-1]"}
        errors = validator.check_l0("test_op", entry)
        assert errors == [], f"Unexpected schema errors: {errors}"

    def test_static_dims_list_fails(self, validator):
        """Non-dict static_dims (e.g. list) is rejected at L0 (R20)."""
        entry = _make_entry()
        entry["signature"]["static_dims"] = ["N"]  # list, not mapping
        errors = validator.check_l0("test_op", entry)
        assert any(
            "static_dims" in e and "must be a mapping" in e for e in errors
        ), f"Expected static_dims mapping error, got: {errors}"

    def test_static_dims_non_string_value_fails(self, validator):
        """static_dims entries must map to string expressions (R20)."""
        entry = _make_entry()
        entry["signature"]["static_dims"] = {"N": {"from": "x.shape[-1]"}}
        errors = validator.check_l0("test_op", entry)
        assert any(
            "static_dims.N" in e and "string expression" in e for e in errors
        ), f"Expected static_dims value-type error, got: {errors}"

    def test_static_dims_single_axis_integer_passes(self, validator):
        """Single-axis reference with integer literal passes R20 parser."""
        entry = _make_entry()
        entry["signature"]["static_dims"] = {"N": "x.shape[-1]"}
        errors = validator.check_l0("test_op", entry)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_static_dims_single_axis_param_passes(self, validator):
        """Axis reference via a declared param passes R20 parser."""
        entry = _make_entry(params={"dim": {"type": "int", "default": -1}})
        entry["signature"]["static_dims"] = {"N": "x.shape[dim]"}
        errors = validator.check_l0("test_op", entry)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_static_dims_multi_axis_product_fails(self, validator):
        """Multi-axis product forms are rejected at L0 (R20 single-axis rule)."""
        entry = _make_entry(params={"dim": {"type": "int | None", "default": -1}})
        entry["signature"]["static_dims"] = {
            "N": "product(x.shape[i] for i in range(x.ndim))"
        }
        errors = validator.check_l0("test_op", entry)
        assert any(
            "static_dims.N" in e and "single-axis reference" in e for e in errors
        ), f"Expected single-axis rule rejection, got: {errors}"

    def test_static_dims_unknown_tensor_fails(self, validator):
        """Referenced tensor must be in signature.inputs."""
        entry = _make_entry()  # inputs = {'x': ...}
        entry["signature"]["static_dims"] = {"N": "weight.shape[0]"}
        errors = validator.check_l0("test_op", entry)
        assert any(
            "static_dims.N" in e and "'weight'" in e and "inputs" in e
            for e in errors
        ), f"Expected unknown-tensor error, got: {errors}"

    def test_static_dims_unknown_param_axis_fails(self, validator):
        """Non-int axis reference must be a declared param name."""
        entry = _make_entry()  # no params
        entry["signature"]["static_dims"] = {"N": "x.shape[dim]"}
        errors = validator.check_l0("test_op", entry)
        assert any(
            "static_dims.N" in e and "'dim'" in e and "param" in e
            for e in errors
        ), f"Expected unknown-param error, got: {errors}"

    def test_init_dims_deprecated_key_fails(self, validator):
        """Deprecated `init_dims` key must be flagged with a migration error (R20)."""
        entry = _make_entry()
        entry["signature"]["init_dims"] = {"N": {"from": "x.shape[-1]"}}
        errors = validator.check_l0("test_op", entry)
        assert any(
            "init_dims" in e and "deprecated" in e and "static_dims" in e
            for e in errors
        ), f"Expected init_dims deprecation error, got: {errors}"

    def test_static_dims_multi_input_non_primary_tensor_passes(self, validator):
        """Expression may reference any tensor in signature.inputs
        (LinearFwdOp-style: out_features binds to weight.shape[0])."""
        entry = _make_entry(
            inputs={
                "input": {"dtype": "float16"},
                "weight": {"dtype": "float16"},
            },
        )
        entry["signature"]["static_dims"] = {
            "in_features": "input.shape[-1]",
            "out_features": "weight.shape[0]",
        }
        errors = validator.check_l0("test_op", entry)
        assert errors == [], f"Unexpected errors: {errors}"

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

    def test_missing_status_fails(self, validator):
        """Entry without status field must produce a schema error."""
        entry = _make_entry(status=None)
        assert "status" not in entry
        errors = validator.check_l0("test_op", entry)
        assert any("status" in e for e in errors), (
            f"Expected schema error about missing status, got: {errors}"
        )

    def test_status_implemented_passes(self, validator):
        """Entry with status: implemented and kernel_map passes schema check."""
        entry = _make_entry(status="implemented", kernel_map={"fwd": "FwdKernel"})
        errors = validator.check_l0("test_op", entry)
        assert errors == [], f"Unexpected schema errors: {errors}"

    def test_status_spec_only_passes(self, validator):
        """Entry with status: spec-only passes schema check."""
        entry = _make_entry(status="spec-only")
        errors = validator.check_l0("test_op", entry)
        assert errors == [], f"Unexpected schema errors: {errors}"

    def test_status_non_string_rejected(self, validator):
        """Non-string status (e.g. integer) must produce a schema error."""
        entry = _make_entry(status="placeholder")
        entry["status"] = 123
        errors = validator.check_l0("test_op", entry)
        assert any("status" in e and "string" in e for e in errors), (
            f"Expected schema error about non-string status, got: {errors}"
        )

    def test_kernel_map_valid_passes(self, validator):
        """status: implemented with valid kernel_map dict passes."""
        entry = _make_entry(status="implemented", kernel_map={"fwd": "FwdKernel"})
        errors = validator.check_l0("test_op", entry)
        assert errors == [], f"Unexpected schema errors: {errors}"

    def test_kernel_map_missing_when_implemented_warns(self, validator):
        """status: implemented without kernel_map produces a warning, not error."""
        entry = _make_entry(status="implemented")
        entry["source"].pop("kernel_map", None)
        assert "kernel_map" not in entry["source"]
        warnings = []
        errors = validator.check_l0("test_op", entry, warnings=warnings)
        assert not any("kernel_map" in e for e in errors), (
            f"Missing kernel_map should be a warning, not an error: {errors}"
        )
        assert any("kernel_map" in w for w in warnings), (
            f"Expected warning about missing kernel_map, got warnings: {warnings}"
        )

    def test_kernel_map_not_required_when_spec_only(self, validator):
        """status: spec-only without kernel_map must NOT produce a kernel_map error."""
        entry = _make_entry(status="spec-only")
        errors = validator.check_l0("test_op", entry)
        kernel_map_errors = [e for e in errors if "kernel_map" in e]
        assert kernel_map_errors == [], (
            f"spec-only op should not need kernel_map, got: {kernel_map_errors}"
        )

    def test_kernel_map_non_dict_fails(self, validator):
        """kernel_map that is not a dict must fail."""
        entry = _make_entry(status="implemented", kernel_map="not_a_dict")
        errors = validator.check_l0("test_op", entry)
        assert any("kernel_map" in e and "mapping" in e for e in errors)

    def test_kernel_map_non_string_key_fails(self, validator):
        """kernel_map with non-string values must fail."""
        entry = _make_entry(status="implemented", kernel_map={"fwd": 123})
        errors = validator.check_l0("test_op", entry)
        assert any("kernel_map" in e for e in errors)

    def test_kernel_map_empty_dict_passes(self, validator):
        """status: implemented with empty kernel_map dict passes (valid dict of str:str)."""
        entry = _make_entry(status="implemented", kernel_map={})
        errors = validator.check_l0("test_op", entry)
        assert errors == [], f"Unexpected schema errors: {errors}"


# ---------------------------------------------------------------------------
# variant_of: cross-entry consistency (R16)
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
        """Chained variant_of fails (R16 single-level)."""
        ops = {
            "primary": _make_entry(),
            "variant_a": {**_make_entry(), "variant_of": "primary"},
            "variant_b": {**_make_entry(), "variant_of": "variant_a"},
        }
        errors = validator.check_variant_of_consistency(ops)
        assert any("chaining" in e.lower() for e in errors)

    def test_variant_mismatched_kernel_fails(self, validator):
        """Variant with different source.kernel fails (R16)."""
        ops = {
            "primary": _make_entry(source_kernel="shared.py"),
            "variant": {
                **_make_entry(source_kernel="different.py"),
                "variant_of": "primary",
            },
        }
        errors = validator.check_variant_of_consistency(ops)
        assert any("source.kernel" in e and "R16" in e for e in errors)

    def test_variant_mismatched_op_fails(self, validator):
        """Variant with different source.op fails (R16)."""
        primary = _make_entry()
        variant = _make_entry()
        variant["source"]["op"] = "different_op.py"
        variant["variant_of"] = "primary"
        ops = {"primary": primary, "variant": variant}
        errors = validator.check_variant_of_consistency(ops)
        assert any("source.op" in e and "R16" in e for e in errors)


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

    def test_static_dims_key_in_init_accepted(self, validator):
        """static_dims key that appears in __init__() is valid (R20)."""
        manifest_inputs = {"x": {"dtype": "float16"}}
        manifest_params = {"dim": {"type": "int", "default": -1}}
        manifest_static_dims = {"N": "x.shape[dim]"}
        forward_params = ["x"]
        init_params = ["N", "dtype", "dim"]
        errors = validator.check_l1_signature(
            "test_op", manifest_inputs, manifest_params, forward_params,
            init_params=init_params,
            manifest_static_dims=manifest_static_dims,
        )
        assert errors == []

    def test_static_dims_key_missing_from_init_fails(self, validator):
        """static_dims key not in __init__() must fail (R20)."""
        manifest_inputs = {"x": {"dtype": "float16"}}
        manifest_params = {"dim": {"type": "int", "default": -1}}
        manifest_static_dims = {"N": "x.shape[dim]"}
        forward_params = ["x"]
        init_params = ["dtype", "dim"]  # N missing
        errors = validator.check_l1_signature(
            "test_op", manifest_inputs, manifest_params, forward_params,
            init_params=init_params,
            manifest_static_dims=manifest_static_dims,
        )
        assert any("static_dims" in e and "'N'" in e for e in errors), (
            f"Expected error about static_dims 'N' missing from __init__, got: {errors}"
        )

    def test_static_dims_absent_ignored(self, validator):
        """When manifest has no static_dims, no related error is raised."""
        manifest_inputs = {"x": {"dtype": "float16"}}
        manifest_params = {}
        forward_params = ["x"]
        init_params = ["dtype"]
        errors = validator.check_l1_signature(
            "test_op", manifest_inputs, manifest_params, forward_params,
            init_params=init_params,
            manifest_static_dims=None,
        )
        assert errors == []

    def test_static_dims_non_dict_fails(self, validator):
        """static_dims must be a mapping; non-dict values are reported."""
        manifest_inputs = {"x": {"dtype": "float16"}}
        manifest_params = {}
        forward_params = ["x"]
        init_params = ["dtype"]
        errors = validator.check_l1_signature(
            "test_op", manifest_inputs, manifest_params, forward_params,
            init_params=init_params,
            manifest_static_dims=["N"],  # list, not dict
        )
        assert any("static_dims" in e for e in errors)


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

    def test_dtype_combos_same_as_identity_pass(self, validator):
        """dtype_combos with matching dtypes for same_as-bound tensors pass."""
        entry = {
            "signature": {
                "inputs": {
                    "x": {"dtype": "float16 | bfloat16"},
                    "w": {"dtype": "same_as(x)"},
                },
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "dtype_combos": [
                    {"x": "float16", "w": "float16"},
                    {"x": "bfloat16", "w": "bfloat16"},
                ],
            },
            "workloads": [{"dtypes": ["float16"]}],
        }
        errors = validator.check_l3("test_op", entry)
        assert errors == []

    def test_dtype_combos_same_as_mismatch_fails(self, validator):
        """dtype_combos with different dtypes for same_as-bound tensors must fail."""
        entry = {
            "signature": {
                "inputs": {
                    "x": {"dtype": "float16 | bfloat16"},
                    "w": {"dtype": "same_as(x)"},
                },
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "dtype_combos": [
                    {"x": "float16", "w": "bfloat16"},
                ],
            },
            "workloads": [{"dtypes": ["float16"]}],
        }
        errors = validator.check_l3("test_op", entry)
        assert any("same_as" in e and "identity" in e for e in errors), (
            f"Expected identity constraint error, got: {errors}"
        )

    def test_dtype_combos_same_as_multi_binding_mismatch_fails(self, validator):
        """dtype_combos with multiple same_as(q) bindings where one mismatches must fail."""
        entry = {
            "signature": {
                "inputs": {
                    "q": {"dtype": "float16 | bfloat16"},
                    "k": {"dtype": "same_as(q)"},
                    "v": {"dtype": "same_as(q)"},
                },
                "outputs": {"o": {"dtype": "same_as(q)"}},
                "dtype_combos": [
                    {"q": "float16", "k": "float16", "v": "bfloat16"},
                ],
            },
            "workloads": [{"dtypes": ["float16"]}],
        }
        errors = validator.check_l3("test_op", entry)
        assert any("same_as" in e and "identity" in e for e in errors), (
            f"Expected identity constraint error, got: {errors}"
        )

    def test_dtype_combos_same_as_partial_combo_fails(self, validator):
        """dtype_combos with same_as-bound tensor but missing reference must fail."""
        entry = {
            "signature": {
                "inputs": {
                    "x": {"dtype": "float16 | bfloat16"},
                    "w": {"dtype": "same_as(x)"},
                },
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "dtype_combos": [
                    {"w": "float16"},  # x missing — cannot verify identity
                ],
            },
            "workloads": [{"dtypes": ["float16"]}],
        }
        errors = validator.check_l3("test_op", entry)
        assert any("without its reference" in e for e in errors), (
            f"Expected partial-combo error, got: {errors}"
        )

    def test_dtype_combos_invalid_value_is_hard_l3_error(self, validator):
        """Un-migrated op with invalid dtype_combos value still produces
        a hard L3 error in ``check_l3`` — does not depend on
        ``_validate_dtypes`` override.
        """
        entry = {
            "status": "implemented",
            "signature": {
                "inputs": {"x": {"dtype": "float16 | bfloat16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "dtype_combos": [
                    {"x": "not_a_real_dtype"},
                ],
            },
            "workloads": [{"dtypes": ["float16"]}],
        }
        errors = validator.check_l3("test_op", entry)
        assert any(
            "not_a_real_dtype" in e and "dtype_combos" in e for e in errors
        ), f"Expected hard L3 error for invalid combo value, got: {errors}"

    def test_resolve_dtype_options_forward_reference(self, validator):
        """``_resolve_tensor_dtype_options`` resolves ``same_as(y)`` even
        when ``y`` is declared later than ``x``. Declaration order must
        not affect resolution (R3 is an identity constraint, not an
        ordering rule).
        """
        sig = {
            "inputs": {
                "x": {"dtype": "same_as(y)"},
                "y": {"dtype": "float16 | bfloat16"},
            },
            "outputs": {"z": {"dtype": "same_as(y)"}},
        }
        resolved = validator._resolve_tensor_dtype_options(sig)
        assert resolved is not None, "Forward same_as reference must resolve"
        assert resolved["x"] == ["float16", "bfloat16"]
        assert resolved["y"] == ["float16", "bfloat16"]
        assert resolved["z"] == ["float16", "bfloat16"]

    def test_resolve_dtype_options_same_as_cycle_fails(self, validator):
        """A pure ``same_as`` cycle (``x: same_as(y)``, ``y: same_as(x)``)
        has no concrete dtype grounding — resolver returns None.
        """
        sig = {
            "inputs": {
                "x": {"dtype": "same_as(y)"},
                "y": {"dtype": "same_as(x)"},
            },
            "outputs": {},
        }
        resolved = validator._resolve_tensor_dtype_options(sig)
        assert resolved is None, (
            "same_as cycle without concrete dtype must not resolve"
        )


# ---------------------------------------------------------------------------
# L2 extension: _infer_output_shapes parity with shape_rules
# ---------------------------------------------------------------------------



def _make_op_cls_with_infer(infer_fn, *, name="FakeOp"):
    """Build a minimal Op subclass whose ``_infer_output_shapes`` is *infer_fn*.

    Uses the real :class:`tileops.ops.op_base.Op` so ``_class_overrides_method``
    correctly treats the method as an override.
    """
    from tileops.ops.op_base import Op

    attrs = {
        "_infer_output_shapes": infer_fn,
        "forward": lambda self, *a, **kw: None,
        "default_kernel_map": property(lambda self: {}),
    }
    return type(name, (Op,), attrs)


class TestInferShapeParity:
    """L2 extension: ``_infer_output_shapes`` output must satisfy shape_rules."""

    def test_no_override_skipped(self, validator):
        """Ops without a ``_infer_output_shapes`` override produce no errors.

        The check still surfaces the missing manifest-derived method as a
        warning (see ``test_no_override_emits_missing_method_warning``);
        this test pins the no-hard-error behaviour.
        """
        from tileops.ops.op_base import Op

        class BareOp(Op):
            def forward(self):  # noqa: D401
                return None

            @property
            def default_kernel_map(self):
                return {}

        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "shape_rules": ["y.shape == x.shape"],
            },
        }
        assert validator.check_l2_infer_parity("BareOp", entry, BareOp) == []

    def test_no_override_emits_missing_method_warning(self, validator):
        """F002 regression: missing override must not pass silently.

        Implemented ops whose class does not yet define
        ``_infer_output_shapes`` must surface a warning naming the gap.
        Silently skipping every such op would make the parity check
        vacuous on the current manifest (no op currently overrides this
        method).
        """
        from tileops.ops.op_base import Op

        class BareOp(Op):
            def forward(self):
                return None

            @property
            def default_kernel_map(self):
                return {}

        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "shape_rules": ["y.shape == x.shape"],
            },
        }
        warnings: list[str] = []
        errors = validator.check_l2_infer_parity(
            "BareOp", entry, BareOp, warnings=warnings,
        )
        assert errors == []
        assert any(
            "does not override _infer_output_shapes" in w for w in warnings
        ), (
            f"Expected missing-method warning when implemented op lacks "
            f"the codegen-derived method, got: {warnings}"
        )

    def test_no_override_opt_out_suppresses_warning(self, validator):
        """F002: parity_opt_out: [shape_parity] suppresses the warning.

        Documented GPU-only ops can opt out of shape parity via a
        manifest flag.
        """
        from tileops.ops.op_base import Op

        class BareOp(Op):
            def forward(self):
                return None

            @property
            def default_kernel_map(self):
                return {}

        entry = {
            "parity_opt_out": ["shape_parity"],
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "shape_rules": ["y.shape == x.shape"],
            },
        }
        warnings: list[str] = []
        errors = validator.check_l2_infer_parity(
            "BareOp", entry, BareOp, warnings=warnings,
        )
        assert errors == []
        assert not any(
            "does not override _infer_output_shapes" in w for w in warnings
        ), warnings

    def test_symbolic_dim_rule_detects_wrong_output(self, validator):
        """F001 regression: rules like ``o.shape == (B, S, H, D)`` must fire.

        The MultiHeadAttentionFwdOp-style manifest rule references
        symbolic dimension names (B, S, H, D) declared only via literal
        ``tensor.shape == (...)`` rules. Without binding those names into
        the evaluation context, the rule raised NameError and the
        validator silently downgraded it to a warning — meaning a
        genuinely wrong ``_infer_output_shapes`` (e.g. returning a 1-D
        shape instead of a 4-D shape) could pass parity.
        """
        def infer(self, q_shape, k_shape, v_shape):
            # Wrong: returns a 1-D shape instead of a 4-D shape.
            return {"o": (999,)}

        cls = _make_op_cls_with_infer(infer, name="BadMHA")
        entry = {
            "signature": {
                "inputs": {
                    "q": {"dtype": "float16"},
                    "k": {"dtype": "float16"},
                    "v": {"dtype": "float16"},
                },
                "outputs": {"o": {"dtype": "float16"}},
                "shape_rules": [
                    "q.shape == (B, S, H, D)",
                    "k.shape == (B, S, H, D)",
                    "v.shape == (B, S, H, D)",
                    "o.shape == (B, S, H, D)",
                ],
            },
        }
        warnings: list[str] = []
        errors = validator.check_l2_infer_parity(
            "BadMHA", entry, cls, warnings=warnings,
        )
        assert any(
            "_infer_output_shapes output violates" in e and "o.shape" in e
            for e in errors
        ), (
            f"Expected symbolic-dim rule to evaluate and flag mismatch, "
            f"got errors={errors} warnings={warnings}"
        )
        assert not any(
            "could not be evaluated" in w for w in warnings
        ), (
            f"Symbolic dim names must be bound into ctx, not reported as "
            f"NameError: {warnings}"
        )

    def test_no_cls_skipped(self, validator):
        entry = {"signature": {"shape_rules": ["y.shape == x.shape"]}}
        assert validator.check_l2_infer_parity("Foo", entry, None) == []

    def test_correct_infer_passes(self, validator):
        def infer(self, x_shape):
            return {"y": x_shape}

        cls = _make_op_cls_with_infer(infer)
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "shape_rules": ["y.shape == x.shape"],
            },
        }
        assert validator.check_l2_infer_parity("FakeOp", entry, cls) == []

    def test_incorrect_infer_fails(self, validator):
        """AC-2: parity error when _infer_output_shapes disagrees with shape_rules."""
        def infer(self, x_shape):
            # Wrong: drops a dim.
            return {"y": x_shape[:-1]}

        cls = _make_op_cls_with_infer(infer)
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "shape_rules": ["y.shape == x.shape"],
            },
        }
        errors = validator.check_l2_infer_parity("FakeOp", entry, cls)
        assert any("_infer_output_shapes output violates" in e for e in errors), (
            f"Expected parity error, got: {errors}"
        )

    def test_missing_output_fails(self, validator):
        def infer(self, x_shape):
            return {}  # missing y

        cls = _make_op_cls_with_infer(infer)
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "shape_rules": ["y.shape == x.shape"],
            },
        }
        errors = validator.check_l2_infer_parity("FakeOp", entry, cls)
        assert any("missing output" in e for e in errors), errors

    def test_signature_mismatch_reports(self, validator):
        def infer(self, a_shape):
            return {"y": a_shape}

        cls = _make_op_cls_with_infer(infer)
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "shape_rules": ["y.shape == x.shape"],
            },
        }
        errors = validator.check_l2_infer_parity("FakeOp", entry, cls)
        assert any("signature does not match" in e for e in errors), errors

    def test_tuple_literal_rule_rank(self, validator):
        """tensor.shape == (A, B) rules inform the mock input rank."""
        seen_rank: list[int] = []

        def infer(self, x_shape):
            seen_rank.append(len(x_shape))
            return {"y": x_shape}

        cls = _make_op_cls_with_infer(infer)
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "shape_rules": [
                    "x.shape == (B, S, H, D)",
                    "y.shape == x.shape",
                ],
            },
        }
        assert validator.check_l2_infer_parity("FakeOp", entry, cls) == []
        assert seen_rank == [4], seen_rank

    def test_r11_style_rule_uses_len_helper(self, validator):
        """R11 / R11a rules that call ``len`` must be evaluable.

        Regression: previously the eval context stripped ``__builtins__`` so
        every rule using ``len`` / ``isinstance`` / ``set`` raised NameError
        and was silently downgraded to a warning, causing real parity
        mismatches to slip through.
        """
        # Wrong: reduction op declares dim=-1, keepdim=False so y should drop
        # a rank, but _infer_output_shapes returns x_shape verbatim.
        def infer(self, x_shape):
            return {"y": x_shape}

        cls = _make_op_cls_with_infer(infer)
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "params": {
                    "dim": {"default": -1},
                    "keepdim": {"default": False},
                },
                "shape_rules": [
                    "y.ndim == x.ndim - len({dim % x.ndim})",
                ],
            },
        }
        errors = validator.check_l2_infer_parity("FakeOp", entry, cls)
        assert any("_infer_output_shapes output violates" in e for e in errors), (
            f"Expected R11-style rule to evaluate and flag mismatch, got: {errors}"
        )

    def test_r11a_comprehension_rule_evaluates(self, validator):
        """R11a rules using generator / set comprehensions must be evaluable.

        Regression: ``eval(rule, globals, locals)`` scopes comprehension
        bodies against globals only, so passing ctx names only as locals
        made every comprehension-shaped rule raise ``NameError`` and get
        silently downgraded to a warning. An infer-shape mismatch could
        pass because no rule ever evaluated to False.
        """
        # Wrong infer: keeps input rank even though dim reduction + keepdim=False
        # must drop a rank.
        def infer(self, x_shape):
            return {"y": x_shape}

        cls = _make_op_cls_with_infer(infer)
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "params": {
                    "dim": {"default": [-1]},
                    "keepdim": {"default": False},
                },
                "shape_rules": [
                    # Generator expression inside all(...): comprehension scope.
                    "all(d % x.ndim in range(x.ndim) for d in dim)",
                    # Set comprehension: also its own scope.
                    "len({d % x.ndim for d in dim}) == len(dim)",
                    # Actual parity rule we expect to catch the mismatch:
                    # y.ndim must shrink by len(dim) when keepdim is False.
                    "y.ndim == x.ndim - len(dim)",
                ],
            },
        }
        warnings: list[str] = []
        errors = validator.check_l2_infer_parity(
            "FakeOp", entry, cls, warnings=warnings,
        )
        # No rule should be skipped with a NameError from a comprehension.
        assert not any(
            "could not be evaluated" in w for w in warnings
        ), f"Comprehension rule skipped via warning: {warnings}"
        # The parity mismatch on y.ndim must surface as an error.
        assert any(
            "_infer_output_shapes output violates" in e and "y.ndim" in e
            for e in errors
        ), f"Expected ndim parity mismatch, got: {errors}"

    def test_input_only_precondition_not_blamed_on_infer(self, validator):
        """Mock-input precondition violations must not become parity errors.

        When ``shape_rules`` encode an input-only precondition (e.g.
        ``weight.shape == (x.shape[dim],)``) that the synthesised mock
        inputs happen to violate, a correct ``_infer_output_shapes`` must
        not be blamed. The rule must be reported as skipped via warning
        rather than an error.
        """
        def infer(self, x_shape, weight_shape):
            # Correct: y has the same shape as x.
            return {"y": x_shape}

        cls = _make_op_cls_with_infer(infer)
        entry = {
            "signature": {
                "inputs": {
                    "x": {"dtype": "float16"},
                    "weight": {"dtype": "float16"},
                },
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "params": {"dim": {"default": -1}},
                "shape_rules": [
                    # Input-only precondition the mock (4,4) vs (4,4)
                    # arrangement would naively satisfy; use a form that
                    # the mock synthesis will not satisfy to trigger the
                    # precondition-violation path.
                    "weight.shape == (x.shape[dim] + 1,)",
                    "y.shape == x.shape",
                ],
            },
        }
        warnings: list[str] = []
        errors = validator.check_l2_infer_parity(
            "FakeOp", entry, cls, warnings=warnings,
        )
        assert not any(
            "_infer_output_shapes output violates" in e for e in errors
        ), (
            "Correct _infer_output_shapes should not be blamed for an "
            f"input-only precondition violation on mock inputs: {errors}"
        )
        assert any(
            "input-only precondition" in w for w in warnings
        ), f"Expected precondition-skip warning, got: {warnings}"

    def test_mock_input_shapes_cross_tensor_dims_distinct(self, validator):
        """Distinct symbolic dims across rules must get distinct mock sizes.

        Regression for a latent correctness bug where a per-rule index
        caused different symbolic dims to collide (e.g. ``A`` in rule
        ``x.shape == (A, B)`` and ``C`` in rule ``y.shape == (C, D)``
        both got assigned ``_MOCK_DIM_SIZE + 0``). A downstream rule
        comparing ``x.shape[0]`` to ``y.shape[0]`` would then get a
        spurious True / False depending on direction.
        """
        sig = {
            "inputs": {"x": {}, "y": {}},
            "shape_rules": [
                "x.shape == (A, B)",
                "y.shape == (C, D)",
            ],
        }
        result = validator._mock_input_shapes(sig)
        assert result is not None
        shapes, dim_sizes = result
        # Four distinct symbolic dims → four distinct mock sizes.
        assert len({dim_sizes[k] for k in ("A", "B", "C", "D")}) == 4
        # Corollary: the mock shapes of x and y disagree on the first
        # dim, so a ``x.shape[0] == y.shape[0]`` rule would correctly
        # evaluate False on mock inputs.
        assert tuple(shapes["x"])[0] != tuple(shapes["y"])[0]

    def test_eval_shape_rule_rejects_dunder_attr(self, validator):
        """Shape-rule evaluator must reject dunder attribute access.

        Defense-in-depth: manifest content is trusted (PR-gated), but
        a classic sandbox-escape expression such as
        ``().__class__.__mro__[1].__subclasses__()`` would bypass the
        restricted builtins. The evaluator runs an AST filter that
        rejects any attribute whose name starts or ends with ``__``.
        """
        ok, reason = validator._eval_shape_rule(
            "().__class__ is None", {},
        )
        assert ok is False
        assert reason is not None
        assert "dunder attribute access not permitted" in reason

    def test_body_typeerror_not_reported_as_signature_mismatch(self, validator):
        """TypeError raised inside _infer_output_shapes body must not be
        misreported as a signature mismatch.

        The signature is pre-bound via ``inspect.signature().bind`` so a
        TypeError from the body is distinguished from a signature
        mismatch. The body-raise is surfaced as a hard L2 parity error
        (policy tightened in PR #1005: previously a warning, which let
        genuine bugs silently pass).
        """
        def infer(self, x_shape):
            # Signature matches; the body itself raises TypeError.
            raise TypeError("simulated implementation bug")

        cls = _make_op_cls_with_infer(infer)
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "shape_rules": ["y.shape == x.shape"],
            },
        }
        warnings: list[str] = []
        errors = validator.check_l2_infer_parity(
            "FakeOp", entry, cls, warnings=warnings,
        )
        assert not any(
            "signature does not match manifest inputs" in e for e in errors
        ), (
            "Body TypeError must not be misreported as signature mismatch; "
            f"errors={errors}"
        )
        assert any(
            "raised TypeError" in e for e in errors
        ), (
            "Body TypeError must surface as a hard L2 parity error; "
            f"errors={errors}"
        )

    def test_body_raise_opt_out_downgrades_to_warning(self, validator):
        """``parity_opt_out: [shape_parity]`` downgrades a body-raise to
        a warning for documented GPU-only ops.
        """
        def infer(self, x_shape):
            raise RuntimeError("needs GPU-only state")

        cls = _make_op_cls_with_infer(infer)
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "shape_rules": ["y.shape == x.shape"],
            },
            "parity_opt_out": ["shape_parity"],
        }
        warnings: list[str] = []
        errors = validator.check_l2_infer_parity(
            "FakeOp", entry, cls, warnings=warnings,
        )
        assert errors == []
        assert any(
            "parity skipped (opt-out)" in w and "RuntimeError" in w
            for w in warnings
        )

    def test_body_runtime_error_is_hard_l2_error(self, validator):
        """Finding #2 regression: a body raising ``RuntimeError('not ready')``
        must become a hard L2 parity error unless the entry opts out.
        """
        def infer(self, x_shape):
            raise RuntimeError("not ready")

        cls = _make_op_cls_with_infer(infer)
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "shape_rules": ["y.shape == x.shape"],
            },
        }
        warnings: list[str] = []
        errors = validator.check_l2_infer_parity(
            "FakeOp", entry, cls, warnings=warnings,
        )
        assert any(
            "raised RuntimeError" in e and "not ready" in e for e in errors
        ), f"expected hard L2 error, got errors={errors} warnings={warnings}"

    def test_declared_output_shape_catches_wrong_infer(self, validator):
        """Op with shape-declared-only outputs (no shape_rules) whose
        ``_infer_output_shapes`` returns the wrong shape must produce a
        parity error.

        Regression: previously ``check_l2_infer_parity`` short-circuited
        on empty ``shape_rules``, so a manifest that specified output
        shape only via ``signature.outputs[*].shape`` (e.g. conv ops'
        ``"[N, C_out, L_out]"``) could not catch a broken
        ``_infer_output_shapes``. The validator now also compares
        inferred outputs against declared output shape fields.
        """
        def infer(self, x_shape, w_shape):
            # Wrong: returns x_shape verbatim instead of
            # ``[N, C_out, L_out]`` implied by the declared output shape.
            return {"y": tuple(x_shape)}

        cls = _make_op_cls_with_infer(infer)
        entry = {
            "signature": {
                "inputs": {
                    "x": {"dtype": "float16", "shape": "[N, C_in, L_in]"},
                    "w": {
                        "dtype": "float16",
                        "shape": "[C_out, C_in, kW]",
                    },
                },
                "outputs": {
                    "y": {
                        "dtype": "float16",
                        "shape": "[N, C_out, L_out]",
                    },
                },
                # No shape_rules; declared shape fields alone must drive
                # the parity check.
            },
        }
        warnings: list[str] = []
        errors = validator.check_l2_infer_parity(
            "FakeOp", entry, cls, warnings=warnings,
        )
        assert any(
            "disagrees with declared" in e for e in errors
        ), (
            "Wrong _infer_output_shapes against declared output shape "
            f"must surface as a parity error; errors={errors}"
        )

    def test_infer_reads_self_attr_uses_cls_new(self, validator):
        """``_infer_output_shapes`` that reads an instance attribute set
        outside manifest params must not falsely skip.

        Regression: when the mock ``self`` was a
        :class:`types.SimpleNamespace`, the call raised AttributeError
        and the parity check silently skipped. After switching to
        ``cls.__new__(cls)`` + setattr for manifest params, the mock
        ``self`` carries class-defined helpers (here a class attribute),
        and the parity check proceeds end-to-end.
        """
        from tileops.ops.op_base import Op

        class SelfAttrOp(Op):
            # Class attribute accessible via ``self.some_attr`` even when
            # ``__init__`` was not run.
            some_attr = 7

            def forward(self, x):
                return None

            @property
            def default_kernel_map(self):
                return {}

            def _infer_output_shapes(self, x_shape):
                # Read an attribute that must be reachable through self.
                _ = self.some_attr
                return {"y": tuple(x_shape)}

        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "shape_rules": ["y.shape == x.shape"],
            },
        }
        warnings: list[str] = []
        errors = validator.check_l2_infer_parity(
            "SelfAttrOp", entry, SelfAttrOp, warnings=warnings,
        )
        assert errors == [], f"Expected no errors, got: {errors}"
        assert not any(
            "parity skipped" in w and "AttributeError" in w
            for w in warnings
        ), (
            "self.<class_attr> lookup must not cause a parity skip; "
            f"warnings={warnings}"
        )

    def test_infer_reads_static_dim_attr_populated(self, validator):
        """``_infer_output_shapes`` reading ``self.<static_dim>`` must
        exercise parity, not skip with AttributeError.

        Regression (review thread 1): ``_build_mock_self`` previously
        installed only ``signature.params`` defaults, so a generated
        ``_infer_output_shapes`` that consults ``self.N`` (a
        ``static_dims`` key) raised AttributeError and the check was
        silently skipped. With static_dims now resolved against mock
        inputs and installed on mock_self, the method runs and its
        output is compared end-to-end.
        """
        from tileops.ops.op_base import Op

        class StaticDimOp(Op):

            def forward(self, x):
                return None

            @property
            def default_kernel_map(self):
                return {}

            def _infer_output_shapes(self, x_shape):
                # Reads a static_dims attribute: N = x.shape[-1]
                return {"y": (self.N, self.N)}

        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16", "shape": "[B, N]"}},
                "outputs": {"y": {"dtype": "float16", "shape": "[N, N]"}},
                "static_dims": {"N": "x.shape[-1]"},
            },
        }
        warnings: list[str] = []
        errors = validator.check_l2_infer_parity(
            "StaticDimOp", entry, StaticDimOp, warnings=warnings,
        )
        assert errors == [], f"Expected no errors, got: {errors}"
        assert not any(
            "AttributeError" in w for w in warnings
        ), f"static_dims lookup must not AttributeError-skip; warnings={warnings}"

    def test_conv_like_output_only_symbol_not_blamed(self, validator):
        """Conv-like op with output-only ``L_out`` derived by shape_rules
        must pass parity when ``_infer_output_shapes`` is correct.

        Regression (review thread 2): previously the declared
        output-shape comparison pulled an arbitrary concrete size for
        ``L_out`` from ``dim_sizes`` and flagged the correct
        ``_infer_output_shapes`` as a mismatch. Output-only symbols must
        only be checked for rank + per-symbol consistency across
        outputs.
        """
        def infer(self, x_shape, w_shape):
            # x: [N, C_in, L_in]; w: [C_out, C_in, kW]
            # y: [N, C_out, L_in - kW + 1]
            return {"y": (x_shape[0], w_shape[0], x_shape[2] - w_shape[2] + 1)}

        cls = _make_op_cls_with_infer(infer, name="ConvLikeOp")
        entry = {
            "signature": {
                "inputs": {
                    "x": {"dtype": "float16", "shape": "[N, C_in, L_in]"},
                    "w": {"dtype": "float16", "shape": "[C_out, C_in, kW]"},
                },
                "outputs": {
                    "y": {"dtype": "float16", "shape": "[N, C_out, L_out]"},
                },
                "shape_rules": ["L_out == L_in - kW + 1"],
            },
        }
        warnings: list[str] = []
        errors = validator.check_l2_infer_parity(
            "ConvLikeOp", entry, cls, warnings=warnings,
        )
        assert errors == [], (
            f"Correct conv-like infer must not be flagged for output-only "
            f"symbol L_out; errors={errors}"
        )

    def test_conv_like_wrong_output_only_value_reported(self, validator):
        """Wrong output-only symbol value must be flagged as a parity
        error via the shape_rules defining that symbol.

        Regression: previously ``check_l2_infer_parity`` classified the
        rule ``L_out == L_in - kW + 1`` as an input-only precondition
        (the rule does not mention any output tensor name), so an
        incorrect ``_infer_output_shapes`` that returned a bogus
        ``L_out`` silently slipped past: the rule failed in both the
        full and input-only contexts and was skipped. Output-only
        symbols appearing in declared output shapes now trigger
        mentions_output classification and are rebound from the
        inferred result before rule evaluation.
        """
        def infer(self, x_shape, w_shape):
            # Deliberately wrong output-only L_out value (999).
            return {"y": (x_shape[0], w_shape[0], 999)}

        cls = _make_op_cls_with_infer(infer, name="ConvLikeWrongOutOnlyOp")
        entry = {
            "signature": {
                "inputs": {
                    "x": {"dtype": "float16", "shape": "[N, C_in, L_in]"},
                    "w": {"dtype": "float16", "shape": "[C_out, C_in, kW]"},
                },
                "outputs": {
                    "y": {"dtype": "float16", "shape": "[N, C_out, L_out]"},
                },
                "shape_rules": ["L_out == L_in - kW + 1"],
            },
        }
        errors = validator.check_l2_infer_parity(
            "ConvLikeWrongOutOnlyOp", entry, cls,
        )
        assert any(
            "L_out == L_in - kW + 1" in e and "violates shape_rules" in e
            for e in errors
        ), (
            "Wrong L_out value must produce a shape_rules parity error; "
            f"errors={errors}"
        )

    def test_conv_like_wrong_rank_still_caught(self, validator):
        """Rank disagreement against declared output shape is still an
        error, even for an op with an output-only ``L_out`` symbol.

        Pins the positive side of the thread 2 fix: loosening the
        output-only value check must not weaken the rank check.
        """
        def infer(self, x_shape, w_shape):
            # Wrong rank: drops the spatial dim entirely.
            return {"y": (x_shape[0], w_shape[0])}

        cls = _make_op_cls_with_infer(infer, name="ConvLikeBadOp")
        entry = {
            "signature": {
                "inputs": {
                    "x": {"dtype": "float16", "shape": "[N, C_in, L_in]"},
                    "w": {"dtype": "float16", "shape": "[C_out, C_in, kW]"},
                },
                "outputs": {
                    "y": {"dtype": "float16", "shape": "[N, C_out, L_out]"},
                },
                "shape_rules": ["L_out == L_in - kW + 1"],
            },
        }
        errors = validator.check_l2_infer_parity(
            "ConvLikeBadOp", entry, cls,
        )
        assert any(
            "rank" in e and "disagrees" in e for e in errors
        ), f"Expected rank error; got: {errors}"

    def test_conv_like_output_only_inconsistent_across_outputs(self, validator):
        """Output-only symbol reused across multiple outputs must be
        consistent; otherwise the parity check flags the disagreement.
        """
        def infer(self, x_shape):
            # Two outputs that both claim ``L_out`` but produce different
            # concrete sizes — this is an internal inconsistency even
            # though L_out is output-only.
            return {
                "y1": (x_shape[0], x_shape[1] - 1),
                "y2": (x_shape[0], x_shape[1] - 2),
            }

        cls = _make_op_cls_with_infer(infer, name="InconsistentOutOnlyOp")
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16", "shape": "[N, L_in]"}},
                "outputs": {
                    "y1": {"dtype": "float16", "shape": "[N, L_out]"},
                    "y2": {"dtype": "float16", "shape": "[N, L_out]"},
                },
                "shape_rules": ["L_out == L_in - 1"],
            },
        }
        errors = validator.check_l2_infer_parity(
            "InconsistentOutOnlyOp", entry, cls,
        )
        assert any(
            "output-only symbol" in e and "L_out" in e for e in errors
        ), f"Expected output-only consistency error; got: {errors}"


class TestDtypeOptionsHelper:
    """Unit tests for ``_dtype_options_for_tensor`` unresolved-ref contract."""

    def test_pure_same_as_unresolved_returns_none(self, validator):
        """same_as(ref) with ref missing from ``resolved`` returns None.

        Regression: previously returned [] which silently disabled
        downstream dtype parity (``_resolve_tensor_dtype_options`` only
        bails on None, and an empty list yields an empty Cartesian
        product).
        """
        out = validator._dtype_options_for_tensor(
            "y", "same_as(x)", resolved={},
        )
        assert out is None, (
            f"Pure same_as(unresolved) must return None, got {out!r}"
        )

    def test_mixed_token_unresolved_returns_none(self, validator):
        """``same_as(ref) | float32`` with unresolved ref returns None."""
        out = validator._dtype_options_for_tensor(
            "y", "same_as(x) | float32", resolved={},
        )
        assert out is None, (
            f"Mixed same_as(unresolved) must return None, got {out!r}"
        )

    def test_pure_same_as_resolved_inherits_options(self, validator):
        out = validator._dtype_options_for_tensor(
            "y", "same_as(x)", resolved={"x": ["float16", "bfloat16"]},
        )
        assert out == ["float16", "bfloat16"]


# ---------------------------------------------------------------------------
# shape_rules broadcasting helpers (broadcast_shapes / is_broadcastable_to)
# ---------------------------------------------------------------------------


class TestShapeRuleBroadcastBuiltins:
    """Unit tests for the broadcasting helpers exposed in shape_rules eval.

    These mirror PyTorch's ``torch.broadcast_shapes`` semantics and the
    unidirectional ``a is broadcastable to b`` predicate, but are pure
    Python so the validator does not need ``torch`` to evaluate L1
    shape_rules expressions.
    """

    # -- broadcast_shapes -------------------------------------------------

    def test_broadcast_shapes_helper_registered(self, validator):
        """``broadcast_shapes`` must be exposed in ``_SHAPE_RULE_BUILTINS``."""
        assert "broadcast_shapes" in validator._SHAPE_RULE_BUILTINS

    def test_broadcast_shapes_identical_shapes(self, validator):
        fn = validator._SHAPE_RULE_BUILTINS["broadcast_shapes"]
        assert fn((2, 3), (2, 3)) == (2, 3)

    def test_broadcast_shapes_scalar_with_tensor(self, validator):
        fn = validator._SHAPE_RULE_BUILTINS["broadcast_shapes"]
        assert fn((), (4, 5)) == (4, 5)
        assert fn((4, 5), ()) == (4, 5)

    def test_broadcast_shapes_size_one_expands(self, validator):
        fn = validator._SHAPE_RULE_BUILTINS["broadcast_shapes"]
        assert fn((1, 3), (2, 1)) == (2, 3)

    def test_broadcast_shapes_rank_promotion(self, validator):
        fn = validator._SHAPE_RULE_BUILTINS["broadcast_shapes"]
        assert fn((3,), (2, 4, 3)) == (2, 4, 3)

    def test_broadcast_shapes_three_or_more_args(self, validator):
        fn = validator._SHAPE_RULE_BUILTINS["broadcast_shapes"]
        assert fn((1, 3), (2, 1), (1, 1)) == (2, 3)

    def test_broadcast_shapes_no_args_returns_empty(self, validator):
        fn = validator._SHAPE_RULE_BUILTINS["broadcast_shapes"]
        assert fn() == ()

    def test_broadcast_shapes_single_arg(self, validator):
        fn = validator._SHAPE_RULE_BUILTINS["broadcast_shapes"]
        assert fn((2, 3)) == (2, 3)

    def test_broadcast_shapes_incompatible_raises(self, validator):
        fn = validator._SHAPE_RULE_BUILTINS["broadcast_shapes"]
        with pytest.raises(ValueError, match="not broadcast-compatible"):
            fn((2, 3), (3, 3))

    def test_broadcast_shapes_accepts_lists(self, validator):
        """Tensor.shape may surface as list[int] depending on context."""
        fn = validator._SHAPE_RULE_BUILTINS["broadcast_shapes"]
        assert fn([1, 3], [2, 1]) == (2, 3)

    # -- is_broadcastable_to ---------------------------------------------

    def test_is_broadcastable_to_helper_registered(self, validator):
        assert "is_broadcastable_to" in validator._SHAPE_RULE_BUILTINS

    def test_is_broadcastable_to_equal_shapes(self, validator):
        fn = validator._SHAPE_RULE_BUILTINS["is_broadcastable_to"]
        assert fn((2, 3), (2, 3)) is True

    def test_is_broadcastable_to_size_one_expands(self, validator):
        fn = validator._SHAPE_RULE_BUILTINS["is_broadcastable_to"]
        assert fn((1, 3), (2, 3)) is True
        assert fn((3,), (2, 3)) is True
        assert fn((), (2, 3)) is True

    def test_is_broadcastable_to_unidirectional(self, validator):
        """``is_broadcastable_to(src, dst)`` is asymmetric.

        ``src`` may grow into ``dst`` but ``dst`` is fixed; expanding
        ``dst`` into a larger shape is not allowed.
        """
        fn = validator._SHAPE_RULE_BUILTINS["is_broadcastable_to"]
        # (2, 3) cannot broadcast *to* (3,) — dst is smaller.
        assert fn((2, 3), (3,)) is False
        # (2, 1) into (2, 3) — fine.
        assert fn((2, 1), (2, 3)) is True
        # (2, 3) into (2, 1) — would require shrinking dst dim 1.
        assert fn((2, 3), (2, 1)) is False

    def test_is_broadcastable_to_mismatched_dim_returns_false(self, validator):
        fn = validator._SHAPE_RULE_BUILTINS["is_broadcastable_to"]
        assert fn((2, 4), (2, 3)) is False

    def test_is_broadcastable_to_extra_leading_dim_returns_false(self, validator):
        """Source rank may not exceed destination rank."""
        fn = validator._SHAPE_RULE_BUILTINS["is_broadcastable_to"]
        assert fn((5, 2, 3), (2, 3)) is False

    # -- end-to-end via _eval_shape_rule ---------------------------------

    def test_broadcast_shapes_in_shape_rule(self, validator):
        ok, reason = validator._eval_shape_rule(
            "broadcast_shapes((1, 3), (2, 1)) == (2, 3)", {},
        )
        assert reason is None
        assert ok is True

    def test_is_broadcastable_to_in_shape_rule_true(self, validator):
        ok, reason = validator._eval_shape_rule(
            "is_broadcastable_to((1, 3), (2, 3))", {},
        )
        assert reason is None
        assert ok is True

    def test_is_broadcastable_to_in_shape_rule_false(self, validator):
        ok, reason = validator._eval_shape_rule(
            "is_broadcastable_to((2, 3), (2, 1))", {},
        )
        assert reason is None
        assert ok is False


# ---------------------------------------------------------------------------
# L3 extension: _validate_dtypes parity with dtype_combos / unions
# ---------------------------------------------------------------------------


def _make_op_cls_with_validate(validate_fn, *, name="FakeDtypeOp"):
    from tileops.ops.op_base import Op

    attrs = {
        "_validate_dtypes": validate_fn,
        "forward": lambda self, *a, **kw: None,
        "default_kernel_map": property(lambda self: {}),
    }
    return type(name, (Op,), attrs)


class TestValidateDtypesParity:
    """L3 extension: ``_validate_dtypes`` matches manifest dtype_combos/unions."""

    def test_no_override_skipped(self, validator):
        from tileops.ops.op_base import Op

        class BareOp(Op):
            def forward(self):
                return None

            @property
            def default_kernel_map(self):
                return {}

        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
            },
        }
        assert validator.check_l3_validate_dtypes_parity("Bare", entry, BareOp) == []

    def test_no_override_emits_missing_method_warning(self, validator):
        """F002 regression: missing override must not pass silently on L3."""
        from tileops.ops.op_base import Op

        class BareOp(Op):
            def forward(self):
                return None

            @property
            def default_kernel_map(self):
                return {}

        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
            },
        }
        warnings: list[str] = []
        errors = validator.check_l3_validate_dtypes_parity(
            "BareOp", entry, BareOp, warnings=warnings,
        )
        assert errors == []
        assert any(
            "does not override _validate_dtypes" in w for w in warnings
        ), warnings

    def test_no_override_opt_out_suppresses_warning(self, validator):
        """F002: parity_opt_out: [dtype_parity] suppresses the warning."""
        from tileops.ops.op_base import Op

        class BareOp(Op):
            def forward(self):
                return None

            @property
            def default_kernel_map(self):
                return {}

        entry = {
            "parity_opt_out": ["dtype_parity"],
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
            },
        }
        warnings: list[str] = []
        errors = validator.check_l3_validate_dtypes_parity(
            "BareOp", entry, BareOp, warnings=warnings,
        )
        assert errors == []
        assert not any(
            "does not override _validate_dtypes" in w for w in warnings
        ), warnings

    def test_union_accept_all_passes(self, validator):
        import torch

        def validate(self, x):
            if x.dtype not in (torch.float16, torch.bfloat16):
                raise ValueError(f"bad dtype {x.dtype}")

        cls = _make_op_cls_with_validate(validate)
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16 | bfloat16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
            },
        }
        assert validator.check_l3_validate_dtypes_parity("FakeDtypeOp", entry, cls) == []

    def test_union_reject_declared_fails(self, validator):
        """AC-3: rejects a dtype in the declared union -> parity error."""
        import torch

        def validate(self, x):
            if x.dtype != torch.float16:
                raise ValueError("only fp16")

        cls = _make_op_cls_with_validate(validate)
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16 | bfloat16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
            },
        }
        errors = validator.check_l3_validate_dtypes_parity("FakeDtypeOp", entry, cls)
        assert any("rejects valid combo" in e for e in errors), errors

    def test_dtype_combos_accept_listed_pass(self, validator):
        import torch

        def validate(self, x, w):
            allowed = {(torch.float16, torch.float16), (torch.bfloat16, torch.bfloat16)}
            if (x.dtype, w.dtype) not in allowed:
                raise ValueError("unlisted")

        cls = _make_op_cls_with_validate(validate)
        entry = {
            "signature": {
                "inputs": {
                    "x": {"dtype": "float16 | bfloat16"},
                    "w": {"dtype": "same_as(x)"},
                },
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "dtype_combos": [
                    {"x": "float16", "w": "float16"},
                    {"x": "bfloat16", "w": "bfloat16"},
                ],
            },
        }
        assert validator.check_l3_validate_dtypes_parity(
            "FakeDtypeOp", entry, cls,
        ) == []

    def test_dtype_combos_rejects_listed_fails(self, validator):
        import torch

        def validate(self, x, w):
            # Rejects the listed (bfloat16, bfloat16) combo.
            if x.dtype != torch.float16:
                raise ValueError("unlisted")

        cls = _make_op_cls_with_validate(validate)
        entry = {
            "signature": {
                "inputs": {
                    "x": {"dtype": "float16 | bfloat16"},
                    "w": {"dtype": "same_as(x)"},
                },
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "dtype_combos": [
                    {"x": "float16", "w": "float16"},
                    {"x": "bfloat16", "w": "bfloat16"},
                ],
            },
        }
        errors = validator.check_l3_validate_dtypes_parity(
            "FakeDtypeOp", entry, cls,
        )
        assert any("rejects dtype_combos" in e for e in errors), errors

    def test_dtype_combos_accepts_unlisted_fails(self, validator):
        """AC-3: accepts a non-listed combo -> parity error."""
        def validate(self, x, w):
            return None  # accepts everything

        cls = _make_op_cls_with_validate(validate)
        entry = {
            "signature": {
                "inputs": {
                    "x": {"dtype": "float16 | bfloat16"},
                    "w": {"dtype": "float16 | bfloat16"},
                },
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "dtype_combos": [
                    {"x": "float16", "w": "float16"},
                ],
            },
        }
        errors = validator.check_l3_validate_dtypes_parity(
            "FakeDtypeOp", entry, cls,
        )
        assert any("accepts non-listed combo" in e for e in errors), errors

    def test_dtype_combos_first_rejected_later_accepted_fails(self, validator):
        """Regression: non-listed combos must all be checked, not just first.

        Previously the loop broke on the first rejection, letting a later
        accepted non-listed combo escape detection. Enumerate every
        non-listed combination and report each acceptance.
        """
        import torch

        def validate(self, x, w):
            # Reject the first non-listed combo (fp16, bf16) but accept a
            # later non-listed combo (bf16, fp16). Listed (fp16, fp16) is
            # also accepted.
            if x.dtype == torch.float16 and w.dtype == torch.bfloat16:
                raise ValueError("rejected early non-listed combo")

        cls = _make_op_cls_with_validate(validate)
        entry = {
            "signature": {
                "inputs": {
                    "x": {"dtype": "float16 | bfloat16"},
                    "w": {"dtype": "float16 | bfloat16"},
                },
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "dtype_combos": [
                    {"x": "float16", "w": "float16"},
                ],
            },
        }
        errors = validator.check_l3_validate_dtypes_parity(
            "FakeDtypeOp", entry, cls,
        )
        # Must flag the later accepted non-listed combo (bf16, fp16) and
        # may also flag (bf16, bf16). At minimum one 'accepts non-listed'
        # error must be present.
        assert any("accepts non-listed combo" in e for e in errors), (
            f"Expected later non-listed acceptance to be flagged, got: {errors}"
        )
        # Stronger check: the (bfloat16, float16) combo specifically is
        # surfaced — proves the loop did not stop at the first rejection.
        assert any(
            "'x': 'bfloat16'" in e and "'w': 'float16'" in e
            for e in errors
        ), (
            f"Expected (bfloat16, float16) combo in errors, got: {errors}"
        )

    def test_signature_mismatch_union_fails(self, validator):
        """AC-3 regression: _validate_dtypes with a wrong kwarg name must fail.

        Previously TypeError from a signature mismatch was silently
        downgraded to a warning, letting an unusable _validate_dtypes
        satisfy parity. The validator must surface this as a hard error.
        """
        def validate(self, wrong_name):
            return None

        cls = _make_op_cls_with_validate(validate)
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16 | bfloat16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
            },
        }
        warnings: list[str] = []
        errors = validator.check_l3_validate_dtypes_parity(
            "FakeDtypeOp", entry, cls, warnings=warnings,
        )
        assert any(
            "signature does not match manifest inputs" in e for e in errors
        ), (
            "Signature mismatch in _validate_dtypes must surface as a "
            f"parity error, got errors={errors} warnings={warnings}"
        )

    def test_signature_mismatch_dtype_combos_fails(self, validator):
        """Same signature-mismatch hard-fail on the dtype_combos branch."""
        def validate(self, wrong_name, other_wrong):
            return None

        cls = _make_op_cls_with_validate(validate)
        entry = {
            "signature": {
                "inputs": {
                    "x": {"dtype": "float16 | bfloat16"},
                    "w": {"dtype": "same_as(x)"},
                },
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "dtype_combos": [
                    {"x": "float16", "w": "float16"},
                    {"x": "bfloat16", "w": "bfloat16"},
                ],
            },
        }
        warnings: list[str] = []
        errors = validator.check_l3_validate_dtypes_parity(
            "FakeDtypeOp", entry, cls, warnings=warnings,
        )
        assert any(
            "signature does not match manifest inputs" in e for e in errors
        ), (
            "Signature mismatch must surface on dtype_combos branch too, "
            f"got errors={errors} warnings={warnings}"
        )

    def test_cartesian_product_over_bound_skipped_with_warning(
        self, validator, monkeypatch,
    ):
        """Enumerating every combo must stay within a configurable bound.

        Guards against future ops that declare many inputs × wide dtype
        unions from exploding CI wall-time. When the product exceeds
        ``_MAX_DTYPE_COMBOS`` the op is skipped deterministically with
        a warning naming input count × option sizes.
        """
        monkeypatch.setattr(validator, "_MAX_DTYPE_COMBOS", 4)

        def _accept_all(self, **kwargs):
            return None

        cls = _make_op_cls_with_validate(_accept_all, name="WideDtypeOp")
        entry = {
            "signature": {
                "inputs": {
                    "a": {"dtype": "float16 | bfloat16 | float32"},
                    "b": {"dtype": "float16 | bfloat16 | float32"},
                },
                "outputs": {"y": {"dtype": "same_as(a)"}},
            },
        }
        warnings: list[str] = []
        errors = validator.check_l3_validate_dtypes_parity(
            "WideDtypeOp", entry, cls, warnings=warnings,
        )
        assert errors == [], (
            f"Over-bound enumeration should skip, not error: {errors}"
        )
        assert any(
            "exceeds _MAX_DTYPE_COMBOS" in w for w in warnings
        ), f"Expected over-bound skip warning, got: {warnings}"

    def test_body_typeerror_is_rejection_not_signature_mismatch(self, validator):
        """TypeError raised inside _validate_dtypes body is a legitimate
        rejection, not a signature mismatch.

        Regression: previously a bare ``except (ValueError, TypeError)``
        could not distinguish between a kwarg-name mismatch (signature
        error) and a TypeError raised inside the body (legitimate
        rejection). The validator must pre-bind the signature and only
        flag the former as a signature mismatch.
        """
        # Signature matches (``x`` kwarg), but the body raises TypeError
        # on every call. This should be treated as a rejection of every
        # combo drawn from the union, not a signature mismatch — which in
        # turn means each Cartesian combo is reported as rejected.
        def validate(self, x):
            raise TypeError("dtype comparison not supported")

        cls = _make_op_cls_with_validate(validate)
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16 | bfloat16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
            },
        }
        warnings: list[str] = []
        errors = validator.check_l3_validate_dtypes_parity(
            "FakeDtypeOp", entry, cls, warnings=warnings,
        )
        # No signature-mismatch error.
        assert not any(
            "signature does not match manifest inputs" in e for e in errors
        ), (
            "Body-level TypeError must not be misreported as signature "
            f"mismatch; errors={errors}"
        )
        # The body rejects every combo drawn from the union, so the
        # no-dtype_combos branch reports each as a parity violation.
        assert any(
            "rejects valid combo" in e for e in errors
        ), (
            "Body TypeError should surface as a rejection of declared "
            f"union combos; errors={errors}"
        )

    def test_dtype_combos_exhausts_union_emits_warning(self, validator):
        """When dtype_combos covers every Cartesian tuple, the validator
        emits the 'exhausts the union' warning even though no non-listed
        combo was checked.

        Regression: previously the warning fired only when
        ``checked_any and not rejected_at_least_one`` — which is
        impossible in the exhaustive case because every tuple hits the
        ``listed_combo_keys`` continue path and ``checked_any`` stays
        False.
        """
        import torch

        allowed = {torch.float16, torch.bfloat16}

        def validate(self, x, w):
            # Reject dtypes outside the declared union so the new
            # out-of-union probe does not produce parity errors.
            if x.dtype not in allowed or w.dtype not in allowed:
                raise ValueError("dtype out of union")
            return None

        cls = _make_op_cls_with_validate(validate)
        entry = {
            "signature": {
                "inputs": {
                    "x": {"dtype": "float16 | bfloat16"},
                    "w": {"dtype": "same_as(x)"},
                },
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "dtype_combos": [
                    {"x": "float16", "w": "float16"},
                    {"x": "float16", "w": "bfloat16"},
                    {"x": "bfloat16", "w": "float16"},
                    {"x": "bfloat16", "w": "bfloat16"},
                ],
            },
        }
        warnings: list[str] = []
        errors = validator.check_l3_validate_dtypes_parity(
            "FakeDtypeOp", entry, cls, warnings=warnings,
        )
        assert errors == [], f"Expected no errors, got: {errors}"
        assert any(
            "exhausts the union" in w for w in warnings
        ), (
            "Validator must warn when dtype_combos covers every "
            f"Cartesian combo; warnings={warnings}"
        )

    def test_no_combos_accepts_out_of_union_fails(self, validator):
        """AC-3 rejection side: when the op has no ``dtype_combos`` and
        ``_validate_dtypes`` accepts a dtype outside the declared union,
        the no-combos branch must emit a parity error.

        Regression for F010: previously the no-combos branch iterated
        only the union's Cartesian product and so could not detect an
        overly-permissive ``_validate_dtypes``.
        """
        # Overly-permissive implementation: accepts any dtype.
        def validate(self, x):
            return None

        cls = _make_op_cls_with_validate(validate)
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16 | bfloat16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
            },
        }
        warnings: list[str] = []
        errors = validator.check_l3_validate_dtypes_parity(
            "FakeDtypeOp", entry, cls, warnings=warnings,
        )
        assert any(
            "out-of-union" in e for e in errors
        ), (
            "Out-of-union dtype must surface as parity error when "
            f"_validate_dtypes accepts it; errors={errors}"
        )

    def test_no_combos_rejects_out_of_union_pass(self, validator):
        """Well-behaved op that rejects out-of-union dtypes produces no
        parity error.
        """
        import torch

        def validate(self, x):
            if x.dtype not in (torch.float16, torch.bfloat16):
                raise ValueError(f"unsupported dtype {x.dtype}")

        cls = _make_op_cls_with_validate(validate)
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16 | bfloat16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
            },
        }
        warnings: list[str] = []
        errors = validator.check_l3_validate_dtypes_parity(
            "FakeDtypeOp", entry, cls, warnings=warnings,
        )
        assert errors == [], (
            "Conforming op must not emit a parity error; "
            f"errors={errors}"
        )

    def test_no_combos_out_of_union_probe_respects_max(
        self, validator, monkeypatch,
    ):
        """Out-of-union probe must stay within ``_MAX_DTYPE_COMBOS``.

        With the cap tightened to 2, at most 2 out-of-union probes fire
        even though the sentinel pool contains 6 out-of-union dtypes
        for a {float16, bfloat16} union.
        """
        # Product size for a single-input {float16, bfloat16} union is 2,
        # so _MAX_DTYPE_COMBOS=2 keeps the Cartesian enumeration alive.
        monkeypatch.setattr(validator, "_MAX_DTYPE_COMBOS", 2)

        def validate(self, x):
            return None

        cls = _make_op_cls_with_validate(validate)
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16 | bfloat16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
            },
        }
        warnings: list[str] = []
        errors = validator.check_l3_validate_dtypes_parity(
            "FakeDtypeOp", entry, cls, warnings=warnings,
        )
        out_of_union_errs = [e for e in errors if "out-of-union" in e]
        # Sentinel pool has 6 out-of-union entries but probe budget is 2.
        assert len(out_of_union_errs) == 2, (
            "Out-of-union probe must be bounded by _MAX_DTYPE_COMBOS; "
            f"got {len(out_of_union_errs)} errors: {out_of_union_errs}"
        )

    def test_no_combos_accepts_same_as_violation_fails(self, validator):
        """AC-3 / R3: when ``_validate_dtypes`` accepts a same_as
        identity violation, the no-combos branch must surface it.

        Regression for F011: the union-iteration loop skips every
        same_as-violating candidate via ``_honours_same_as``, so a
        permissive op that fails to enforce same_as would go unflagged
        without a dedicated probe.
        """
        # Overly-permissive: does not check x.dtype == w.dtype.
        def validate(self, x, w):
            return None

        cls = _make_op_cls_with_validate(validate)
        entry = {
            "signature": {
                "inputs": {
                    "x": {"dtype": "float16 | bfloat16"},
                    "w": {"dtype": "same_as(x)"},
                },
                "outputs": {"y": {"dtype": "same_as(x)"}},
            },
        }
        warnings: list[str] = []
        errors = validator.check_l3_validate_dtypes_parity(
            "FakeDtypeOp", entry, cls, warnings=warnings,
        )
        assert any(
            "same_as violation" in e for e in errors
        ), (
            "same_as identity violation must surface as parity error "
            f"when _validate_dtypes accepts it; errors={errors}"
        )

    def test_no_combos_rejects_same_as_violation_pass(self, validator):
        """Well-behaved op that enforces same_as produces no parity
        error on the same_as probe.
        """
        import torch

        allowed = (torch.float16, torch.bfloat16)

        def validate(self, x, w):
            if x.dtype not in allowed or w.dtype not in allowed:
                raise ValueError(
                    f"unsupported dtype: x.dtype={x.dtype} "
                    f"w.dtype={w.dtype}"
                )
            if x.dtype != w.dtype:
                raise ValueError(
                    f"same_as violated: x.dtype={x.dtype} "
                    f"w.dtype={w.dtype}"
                )

        cls = _make_op_cls_with_validate(validate)
        entry = {
            "signature": {
                "inputs": {
                    "x": {"dtype": "float16 | bfloat16"},
                    "w": {"dtype": "same_as(x)"},
                },
                "outputs": {"y": {"dtype": "same_as(x)"}},
            },
        }
        warnings: list[str] = []
        errors = validator.check_l3_validate_dtypes_parity(
            "FakeDtypeOp", entry, cls, warnings=warnings,
        )
        assert errors == [], (
            "Conforming op must not emit a parity error; "
            f"errors={errors}"
        )

    def test_combos_branch_out_of_union_probe(self, validator):
        """Dtype_combos branch must fire the out-of-union probe.

        Regression: previously the out-of-union negative probe only ran
        in the no-dtype_combos branch, so a permissive ``_validate_dtypes``
        that accepts every dtype could pass parity as long as every
        listed combo was accepted. The probe now exercises the same
        rejection invariant in the dtype_combos branch.
        """
        # Overly-permissive implementation: accepts any dtype combo.
        def validate(self, x):
            return None

        cls = _make_op_cls_with_validate(validate)
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16 | bfloat16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "dtype_combos": [
                    {"x": "float16"},
                    {"x": "bfloat16"},
                ],
            },
        }
        warnings: list[str] = []
        errors = validator.check_l3_validate_dtypes_parity(
            "FakeDtypeOp", entry, cls, warnings=warnings,
        )
        assert any(
            "out-of-union" in e for e in errors
        ), (
            "Out-of-union probe must fire in the dtype_combos branch; "
            f"errors={errors}"
        )

    def test_invalid_dtype_combo_value_is_hard_error(self, validator):
        """A ``dtype_combos`` entry naming a non-existent dtype must be
        surfaced as a hard L3 error, not downgraded to a parity-skip
        warning.

        Regression: previously an invalid dtype name reached the
        ``cannot build mock tensor`` warning branch inside the parity
        loop, which silently disabled the check and hid a manifest data
        bug. The upfront validation pass now rejects entries that are
        neither in ``_TORCH_DTYPES`` nor a resolvable ``same_as`` ref.
        """
        def validate(self, x, w):
            return None

        cls = _make_op_cls_with_validate(validate)
        entry = {
            "signature": {
                "inputs": {
                    "x": {"dtype": "float16 | bfloat16"},
                    "w": {"dtype": "same_as(x)"},
                },
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "dtype_combos": [
                    {"x": "not_a_real_dtype", "w": "not_a_real_dtype"},
                ],
            },
        }
        warnings: list[str] = []
        errors = validator.check_l3_validate_dtypes_parity(
            "FakeDtypeOp", entry, cls, warnings=warnings,
        )
        assert any(
            "not a valid dtype" in e and "not_a_real_dtype" in e
            for e in errors
        ), (
            "Invalid dtype in dtype_combos must produce a hard error "
            f"mentioning the invalid dtype name; errors={errors}"
        )
        # Must not appear as a parity-skip warning either.
        assert not any(
            "cannot build mock tensor" in w for w in warnings
        ), (
            "Invalid dtype name must not be downgraded to "
            f"'cannot build mock tensor' warning; warnings={warnings}"
        )

    def test_unresolved_same_as_in_dtype_combo_is_hard_error(self, validator):
        """``dtype_combos`` entry with ``same_as(unknown_ref)`` must be
        a hard L3 error.

        Regression: previously unresolved ``same_as`` references inside
        combo values silently skipped parity. The upfront validation now
        routes them through ``_dtype_options_for_tensor`` — unresolved
        references return None and become an error.
        """
        def validate(self, x, w):
            return None

        cls = _make_op_cls_with_validate(validate)
        entry = {
            "signature": {
                "inputs": {
                    "x": {"dtype": "float16 | bfloat16"},
                    "w": {"dtype": "float16 | bfloat16"},
                },
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "dtype_combos": [
                    {"x": "float16", "w": "same_as(unknown_ref)"},
                ],
            },
        }
        errors = validator.check_l3_validate_dtypes_parity(
            "FakeDtypeOp", entry, cls,
        )
        assert any(
            "not a valid dtype" in e and "unknown_ref" in e
            for e in errors
        ), (
            "Unresolved same_as ref in dtype_combos must produce a hard "
            f"error mentioning the ref name; errors={errors}"
        )

    def test_valid_dtype_combo_reaches_build_mock_tensor(
        self, validator, monkeypatch,
    ):
        """Valid dtype names continue to reach the build-mock-tensor
        branch; the ``cannot build mock tensor`` warning is still
        reserved for valid names that the local torch build genuinely
        cannot materialize.

        Simulates a torch build lacking support for a declared dtype by
        monkeypatching ``_make_mock_tensor`` to return None for
        ``float8_e4m3fn`` while the combo itself is valid.
        """
        def validate(self, x):
            return None

        cls = _make_op_cls_with_validate(validate)
        entry = {
            "signature": {
                "inputs": {
                    "x": {"dtype": "float16 | float8_e4m3fn"},
                },
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "dtype_combos": [
                    {"x": "float8_e4m3fn"},
                ],
            },
        }
        original = validator._make_mock_tensor

        def fake(name):
            if name == "float8_e4m3fn":
                return None
            return original(name)

        monkeypatch.setattr(validator, "_make_mock_tensor", fake)
        warnings: list[str] = []
        errors = validator.check_l3_validate_dtypes_parity(
            "FakeDtypeOp", entry, cls, warnings=warnings,
        )
        # Valid dtype that the local build can't materialize: no hard
        # error, but the parity-skip warning path still fires.
        assert not any(
            "not a valid dtype" in e for e in errors
        ), (
            "Valid dtype name must not be flagged as invalid by the "
            f"upfront validation pass; errors={errors}"
        )
        assert any(
            "cannot build mock tensor" in w for w in warnings
        ), (
            "Valid-name-but-unmaterializable dtype must still reach the "
            f"'cannot build mock tensor' warning path; warnings={warnings}"
        )

    def test_combo_missing_input_is_manifest_error(self, validator):
        """A combo that omits an input entry remains a manifest error
        surfaced via the parity loop (not a rejection, not a skip).
        """
        def validate(self, x, w):
            return None

        cls = _make_op_cls_with_validate(validate)
        entry = {
            "signature": {
                "inputs": {
                    "x": {"dtype": "float16 | bfloat16"},
                    "w": {"dtype": "same_as(x)"},
                },
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "dtype_combos": [
                    {"x": "float16"},  # missing 'w'
                ],
            },
        }
        errors = validator.check_l3_validate_dtypes_parity(
            "FakeDtypeOp", entry, cls,
        )
        assert any(
            "is missing declared input" in e or "combo missing input" in e
            for e in errors
        ), (
            "Combo missing an input entry must be reported as a "
            f"manifest error; errors={errors}"
        )
        assert not any(
            "rejects dtype_combos[0]" in e for e in errors
        ), (
            "missing-input skip must not be reported as rejection; "
            f"errors={errors}"
        )

    def test_validate_dtypes_reads_self_dtype_attr(self, validator):
        """``_validate_dtypes`` that compares ``x.dtype != self.dtype``
        must accept every listed combo when mock_self.dtype is populated.

        Regression (review thread 1): ``_build_mock_self`` previously
        installed only ``signature.params`` defaults, so
        ``self.dtype`` fell through to the base-class ``Op.dtype = None``
        and the comparison always raised — causing the parity check to
        mark every listed combo as rejected. With the dtype axis now
        populated from the candidate combo, listed combos are accepted
        end-to-end.
        """
        def validate(self, x):
            # The generated pattern under test: compare the input dtype
            # against ``self.dtype`` (set in __init__ via a dtype param).
            if x.dtype != self.dtype:
                raise ValueError(
                    f"x.dtype {x.dtype} does not match self.dtype {self.dtype}"
                )

        cls = _make_op_cls_with_validate(validate)
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16 | bfloat16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
            },
        }
        warnings: list[str] = []
        errors = validator.check_l3_validate_dtypes_parity(
            "FakeDtypeOp", entry, cls, warnings=warnings,
        )
        assert errors == [], (
            "With self.dtype populated from the combo, listed combos "
            f"must be accepted; errors={errors} warnings={warnings}"
        )


class TestDtypeCombosDataHardening:
    """Hardening regressions for ``check_l3_dtype_combos_data``.

    Covers PR #1005 review findings #1 (combo-row completeness) and #5
    (reject union dtype expressions in combo values).
    """

    def test_combo_missing_input_is_hard_error(self, validator):
        """Finding #1: every combo row must cover every declared input.

        Omitting a declared input from a combo row used to silently pass
        when the op had no ``_validate_dtypes`` override — the parity
        loop never ran, so the omission was invisible. The data-level
        check must flag it independently.
        """
        sig = {
            "inputs": {
                "x": {"dtype": "float16 | bfloat16"},
                "w": {"dtype": "float16 | bfloat16"},
            },
            "outputs": {"y": {"dtype": "same_as(x)"}},
            "dtype_combos": [
                {"x": "float16", "w": "float16"},
                {"x": "bfloat16"},  # missing 'w'
            ],
        }
        errors = validator.check_l3_dtype_combos_data("FakeOp", sig)
        assert any(
            "dtype_combos[1]" in e and "missing declared input 'w'" in e
            for e in errors
        ), f"expected completeness error, got {errors}"

    def test_combo_value_union_is_hard_error(self, validator):
        """Finding #5: a union expression in a combo value is rejected.

        Per manifest.md R4, ``dtype_combos[i].<key>`` must be a single
        concrete dtype token (or a ``same_as(ref)`` resolving to one);
        ``"float16 | bfloat16"`` is a union and must fail the data
        check rather than silently expanding to multiple dtypes.
        """
        sig = {
            "inputs": {
                "x": {"dtype": "float16 | bfloat16"},
            },
            "outputs": {"y": {"dtype": "same_as(x)"}},
            "dtype_combos": [
                {"x": "float16 | bfloat16"},
            ],
        }
        errors = validator.check_l3_dtype_combos_data("FakeOp", sig)
        assert any(
            "combo values must be a single concrete dtype" in e for e in errors
        ), f"expected union rejection, got {errors}"


class TestStaticDimShapeParity:
    """Finding #3 regression: static_dims values must pin expected output sizes."""

    def test_static_dim_output_shape_catches_bad_infer(self, validator):
        """A generated _infer_output_shapes returning arbitrary integers
        for a static-dim-bound output position must fail parity.

        Previously the declared-output-shape comparison only checked
        input-bound symbols — ``static_dims`` keys were reclassified as
        output-only, so only rank/consistency was enforced. A bad impl
        returning e.g. ``(999, 999)`` for a declared ``[N, N]`` output
        with ``static_dims: {N: "x.shape[-1]"}`` would pass.
        """
        def bad_infer(self, x_shape):
            # Declared output shape is [N, N]; static_dims says N =
            # x.shape[-1] (=4 under mock). A correct impl would return
            # (4, 4); the bug returns (999, 999), which the old code
            # failed to catch because N was treated as output-only.
            return {"y": (999, 999)}

        cls = _make_op_cls_with_infer(bad_infer, name="StaticDimBadOp")
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16", "shape": "[M, N]"}},
                "outputs": {
                    "y": {"dtype": "same_as(x)", "shape": "[N, N]"},
                },
                "static_dims": {"N": "x.shape[-1]"},
            },
        }
        errors = validator.check_l2_infer_parity(
            "StaticDimBadOp", entry, cls,
        )
        assert any(
            "dim[0]=999" in e or "dim[1]=999" in e for e in errors
        ), f"expected static-dim parity error, got {errors}"


class TestUnexpectedValidateDtypesException:
    """Finding #4 regression: body-level unexpected exceptions become hard L3 errors."""

    def test_runtime_error_from_validate_body_is_hard_error(self, validator):
        """_validate_dtypes raising RuntimeError for every valid combo
        must produce a hard L3 parity error, not a warning.
        """
        def bad_validate(self, x):
            raise RuntimeError("simulated bug")

        cls = _make_op_cls_with_validate(bad_validate, name="BadValidateOp")
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16 | bfloat16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "dtype_combos": [
                    {"x": "float16"},
                    {"x": "bfloat16"},
                ],
            },
        }
        warnings: list[str] = []
        errors = validator.check_l3_validate_dtypes_parity(
            "BadValidateOp", entry, cls, warnings=warnings,
        )
        assert any(
            "raised unexpected exception" in e and "RuntimeError" in e
            for e in errors
        ), f"expected hard L3 error, got errors={errors} warnings={warnings}"

    def test_runtime_error_opt_out_downgrades_to_skip(self, validator):
        """``parity_opt_out: [dtype_parity]`` downgrades the body-raise
        to a silent skip for documented GPU-only cases.
        """
        def bad_validate(self, x):
            raise RuntimeError("needs GPU state")

        cls = _make_op_cls_with_validate(bad_validate, name="OptOutValidateOp")
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16 | bfloat16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "dtype_combos": [
                    {"x": "float16"},
                    {"x": "bfloat16"},
                ],
            },
            "parity_opt_out": ["dtype_parity"],
        }
        warnings: list[str] = []
        errors = validator.check_l3_validate_dtypes_parity(
            "OptOutValidateOp", entry, cls, warnings=warnings,
        )
        assert not any(
            "raised unexpected exception" in e for e in errors
        ), f"opt-out must suppress hard error; errors={errors}"

    def test_runtime_error_no_combos_is_hard_error(self, validator):
        """Same policy in the no-dtype_combos Cartesian branch."""
        def bad_validate(self, x):
            raise RuntimeError("simulated bug")

        cls = _make_op_cls_with_validate(
            bad_validate, name="BadValidateNoCombosOp",
        )
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16 | bfloat16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
            },
        }
        warnings: list[str] = []
        errors = validator.check_l3_validate_dtypes_parity(
            "BadValidateNoCombosOp", entry, cls, warnings=warnings,
        )
        assert any(
            "raised unexpected exception" in e and "RuntimeError" in e
            for e in errors
        ), f"expected hard L3 error, got errors={errors} warnings={warnings}"


class TestSameAsCycleHardError:
    """PR #1005 follow-up: pure ``same_as`` cycles must surface a hard L3 error.

    Previously, ``check_l3_dtype_combos_data`` returned silently when
    ``_resolve_tensor_dtype_options`` returned None, relying on
    ``check_l3`` to have flagged the culprit. But a pure cycle like
    ``x: same_as(y)`` / ``y: same_as(x)`` satisfies per-token validation
    and the R3 identity check, so combo validation would be silently
    skipped and invalid combo data passes.
    """

    def test_pure_same_as_cycle_emits_hard_error(self, validator):
        """A 2-cycle between two inputs must surface a diagnosed L3 error."""
        sig = {
            "inputs": {
                "x": {"dtype": "same_as(y)"},
                "y": {"dtype": "same_as(x)"},
            },
            "outputs": {"z": {"dtype": "same_as(x)"}},
            "dtype_combos": [
                {"x": "float16", "y": "float16"},
            ],
        }
        errors = validator.check_l3_dtype_combos_data("CycleOp", sig)
        assert any(
            "same_as cycle" in e and "'x'" in e and "'y'" in e
            for e in errors
        ), f"expected cycle diagnosis naming x and y, got {errors}"

    def test_dangling_same_as_emits_hard_error(self, validator):
        """A ``same_as(missing)`` reference is reported as dangling.

        The per-token ``_validate_dtype_token`` check already flags this
        at L3, but combo validation must surface its own hard error so
        callers never see a silent pass.
        """
        sig = {
            "inputs": {
                "x": {"dtype": "same_as(nope)"},
            },
            "outputs": {"z": {"dtype": "same_as(x)"}},
            "dtype_combos": [
                {"x": "float16"},
            ],
        }
        errors = validator.check_l3_dtype_combos_data("DanglingOp", sig)
        assert any(
            "dangling reference" in e and "same_as(nope)" in e
            for e in errors
        ), f"expected dangling diagnosis, got {errors}"


class TestParamDefaultOutputShapePin:
    """PR #1005 follow-up: param defaults must pin declared output-shape dims.

    A param with a concrete integer default (e.g. ``params.k.default = 4``)
    is a compile-time-known value just like ``static_dims``. Declared
    output ``shape: "[k]"`` must compare against the default, so a bad
    ``_infer_output_shapes`` returning ``(999,)`` is caught by exact-value
    comparison rather than only rank/consistency.
    """

    def test_param_default_pins_output_dim(self, validator):
        """Bad infer returning ``(999,)`` for declared ``[k]`` with
        ``params.k.default = 4`` must produce a hard L2 error."""
        def bad_infer(self, x_shape):
            return {"y": (999,)}

        cls = _make_op_cls_with_infer(bad_infer, name="ParamDefaultBadOp")
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16", "shape": "[M]"}},
                "outputs": {
                    "y": {"dtype": "same_as(x)", "shape": "[k]"},
                },
                "params": {"k": {"type": "int", "default": 4}},
            },
        }
        errors = validator.check_l2_infer_parity(
            "ParamDefaultBadOp", entry, cls,
        )
        assert any(
            "dim[0]=999" in e and "k=4" in e for e in errors
        ), f"expected param-default parity error, got {errors}"

    def test_param_default_pins_output_dim_pass(self, validator):
        """Correct infer returning ``(4,)`` for declared ``[k]`` with
        ``params.k.default = 4`` passes parity."""
        def good_infer(self, x_shape):
            return {"y": (4,)}

        cls = _make_op_cls_with_infer(good_infer, name="ParamDefaultGoodOp")
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16", "shape": "[M]"}},
                "outputs": {
                    "y": {"dtype": "same_as(x)", "shape": "[k]"},
                },
                "params": {"k": {"type": "int", "default": 4}},
            },
        }
        errors = validator.check_l2_infer_parity(
            "ParamDefaultGoodOp", entry, cls,
        )
        assert errors == [], (
            f"expected no parity errors on correct impl, got {errors}"
        )


class TestOutOfUnionProbeEngulfment:
    """PR #1005 follow-up: out-of-union probe must not be engulfed by wide unions.

    A prior implementation used a fixed 8-dtype ``_DTYPE_SENTINELS`` pool.
    An op declaring exactly those 8 dtypes for an input left the probe
    with no candidate, so an over-permissive ``_validate_dtypes``
    accepting e.g. ``uint8`` would go undetected. The probe now derives
    candidates from ``sorted(_TORCH_DTYPES - declared)``, guaranteeing a
    non-empty pool whenever declared does not cover the entire torch
    dtype universe.
    """

    _ALL_EIGHT = (
        "float16 | bfloat16 | float32 | float64 | "
        "int8 | int16 | int32 | int64"
    )

    def test_eight_sentinel_coverage_still_probes_out_of_union(self, validator):
        """Declared union covers all 8 legacy sentinels but not uint8.

        An over-permissive ``_validate_dtypes`` accepting ``uint8`` must
        surface a hard L3 error because ``uint8 ∈ _TORCH_DTYPES -
        declared``.
        """
        def accept_all(self, x):
            return True  # over-permissive: accepts any dtype

        cls = _make_op_cls_with_validate(accept_all, name="WideEightDtypeOp")
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": self._ALL_EIGHT}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
            },
        }
        warnings: list[str] = []
        errors = validator.check_l3_validate_dtypes_parity(
            "WideEightDtypeOp", entry, cls, warnings=warnings,
        )
        assert any(
            "accepts out-of-union dtype" in e for e in errors
        ), (
            f"expected out-of-union rejection error despite 8-dtype "
            f"union; errors={errors} warnings={warnings}"
        )

    def test_eight_sentinel_coverage_probes_in_combos_branch(self, validator):
        """Same gap in the dtype_combos branch: declared combos cover all
        8 legacy sentinels but permissive impl still accepts uint8."""
        def accept_all(self, x):
            return True

        cls = _make_op_cls_with_validate(
            accept_all, name="WideEightCombosOp",
        )
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": self._ALL_EIGHT}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "dtype_combos": [
                    {"x": "float16"}, {"x": "bfloat16"},
                    {"x": "float32"}, {"x": "float64"},
                    {"x": "int8"}, {"x": "int16"},
                    {"x": "int32"}, {"x": "int64"},
                ],
            },
        }
        warnings: list[str] = []
        errors = validator.check_l3_validate_dtypes_parity(
            "WideEightCombosOp", entry, cls, warnings=warnings,
        )
        assert any(
            "accepts out-of-union dtype" in e for e in errors
        ), (
            f"expected out-of-union rejection in dtype_combos branch; "
            f"errors={errors} warnings={warnings}"
        )

    def test_full_torch_coverage_emits_skip_warning(self, validator):
        """Declared union == full torch dtype set → warning, no vacuous pass.

        The probe cannot produce a candidate so it skips with a warning
        naming the op/input. No hard error is emitted because the
        ``_validate_dtypes`` impl is free to accept anything in this
        (wildly permissive) spec.
        """
        full_union = " | ".join(sorted(validator._TORCH_DTYPES))

        def accept_all(self, x):
            return True

        cls = _make_op_cls_with_validate(
            accept_all, name="FullCoverageOp",
        )
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": full_union}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
            },
        }
        warnings: list[str] = []
        errors = validator.check_l3_validate_dtypes_parity(
            "FullCoverageOp", entry, cls, warnings=warnings,
        )
        assert not any("accepts out-of-union dtype" in e for e in errors), (
            f"full-coverage spec must not produce a probe error; "
            f"errors={errors}"
        )
        assert any(
            "out-of-union probe skipped" in w and "'x'" in w
            for w in warnings
        ), (
            f"expected skip warning naming input 'x'; warnings={warnings}"
        )


# ---------------------------------------------------------------------------
# Bench checks
# ---------------------------------------------------------------------------
class TestBench:
    """bench checks that bench files use manifest workloads and op roofline."""

    def test_bench_with_load_workloads_passes(self, validator, tmp_path):
        bench_file = tmp_path / "bench_test.py"
        bench_file.write_text(
            "from tileops.manifest import load_workloads\n"
            "workloads = load_workloads('test_op')\n"
            "op.eval_roofline()\n"
        )
        errors = validator.check_l4_benchmark("test_op", str(bench_file), REPO_ROOT)
        assert errors == []

    def test_bench_with_load_workloads_only_fails(self, validator, tmp_path):
        """Bench using load_workloads but not op eval_roofline fails bench validation."""
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
            from tileops.manifest import load_workloads
            workloads = load_workloads('wrong_op')
            op.eval_roofline()
        """))
        errors = validator.check_l4_benchmark("test_op", str(bench_file), REPO_ROOT)
        assert any("load_workloads" in e for e in errors)

    def test_syntax_error_in_bench_file_fails_l4(self, validator, tmp_path):
        """A bench file with syntax errors produces an bench error."""
        bench_file = tmp_path / "bench_test.py"
        bench_file.write_text("def broken(\n")
        errors = validator.check_l4_benchmark("test_op", str(bench_file), REPO_ROOT)
        assert any("syntax error" in e for e in errors)

    def test_bench_indirect_helpers_pass(self, validator, tmp_path):
        """Importing workloads_to_params/ManifestBenchmark from benchmarks.benchmark_base passes."""
        bench_file = tmp_path / "bench_test.py"
        bench_file.write_text(textwrap.dedent("""\
            from benchmarks.benchmark_base import workloads_to_params, ManifestBenchmark
            params = workloads_to_params('test_op')
            ManifestBenchmark('test_op', op, params[0])
        """))
        errors = validator.check_l4_benchmark("test_op", str(bench_file), REPO_ROOT)
        assert errors == []

    def test_bench_indirect_wrong_op_fails(self, validator, tmp_path):
        """Indirect helpers called with wrong op name must still fail."""
        bench_file = tmp_path / "bench_test.py"
        bench_file.write_text(textwrap.dedent("""\
            from benchmarks.benchmark_base import workloads_to_params, ManifestBenchmark
            params = workloads_to_params('wrong_op')
            ManifestBenchmark('wrong_op', op, params[0])
        """))
        errors = validator.check_l4_benchmark("test_op", str(bench_file), REPO_ROOT)
        assert any("load_workloads" in e for e in errors)
        assert any("eval_roofline" in e for e in errors)


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
        manifest_file.write_text(yaml.safe_dump({"my_op": entry}))

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
        manifest_file.write_text(yaml.safe_dump({"my_op": entry}))

        errors, warnings = validator.validate_manifest(
            manifest_path=manifest_file,
            repo_root=tmp_path,
        )
        # No signature/shape/dtype/bench errors for spec-only
        non_schema = [e for e in errors if "[schema]" not in e]
        assert non_schema == [], (
            f"Spec-only op should only have schema errors (if any), got: {non_schema}"
        )

    def test_missing_status_rejected_at_schema_level(self, validator, tmp_path):
        """Entry without status field is rejected by schema validation."""
        entry = _make_entry(status=None)

        manifest_file = tmp_path / "ops_manifest.yaml"
        import yaml
        manifest_file.write_text(yaml.safe_dump({"my_op": entry}))

        errors, warnings = validator.validate_manifest(
            manifest_path=manifest_file,
            repo_root=tmp_path,
        )
        status_errors = [e for e in errors if "status" in e]
        assert len(status_errors) > 0, (
            f"Missing-status op should produce a schema error, got: {errors}"
        )

    def test_check_op_nonexistent_op_reports_error(self, validator, tmp_path):
        """--check-op with a name not in manifest reports an error."""
        entry = _make_entry()

        manifest_file = tmp_path / "ops_manifest.yaml"
        import yaml
        manifest_file.write_text(yaml.safe_dump({"my_op": entry}))

        errors, warnings = validator.validate_manifest(
            manifest_path=manifest_file,
            repo_root=tmp_path,
            check_op="nonexistent_op",
        )
        assert any("nonexistent_op" in e and "not found" in e for e in errors), (
            f"Expected error about 'nonexistent_op' not found in manifest, got: {errors}"
        )

    def test_manifest_path_non_mapping_root_reports_error(self, validator, tmp_path):
        """A manifest yaml whose root is not a mapping yields a schema error,
        not an AttributeError on .items()."""
        import yaml

        manifest_file = tmp_path / "ops_manifest.yaml"
        # Top-level sequence — common malformed shape (e.g. accidental list).
        manifest_file.write_text(yaml.safe_dump(["my_op", "other_op"]))

        errors, _ = validator.validate_manifest(
            manifest_path=manifest_file,
            repo_root=tmp_path,
        )
        assert any("top-level mapping" in e for e in errors), (
            f"Expected a top-level-mapping error, got: {errors}"
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
            {"target_op": target_entry, "other_op": other_entry},
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
            {"target_op": target_entry, "other_op": other_entry},
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

        Regression: an agent could modify a variant to break R16 variant
        rules, and --check-op <primary> would still pass because variants
        were excluded from the validation scope.
        """
        import yaml

        primary = _make_entry(source_kernel="shared_kernel.py")
        # Variant shares source with primary (valid)
        valid_variant = _make_entry(source_kernel="shared_kernel.py")
        valid_variant["variant_of"] = "primary_op"

        # Broken variant: different source.kernel violates R16
        broken_variant = _make_entry(source_kernel="different_kernel.py")
        broken_variant["variant_of"] = "primary_op"

        manifest_file = tmp_path / "ops_manifest.yaml"
        manifest_file.write_text(yaml.safe_dump({
            "primary_op": primary,
            "good_variant": valid_variant,
            "bad_variant": broken_variant,
        }))

        # check_op="primary_op" must catch the R16 violation on bad_variant
        errors, _ = validator.validate_manifest(
            manifest_path=manifest_file,
            repo_root=tmp_path,
            check_op="primary_op",
        )
        r16_errors = [e for e in errors if "bad_variant" in e and "R16" in e]
        assert len(r16_errors) > 0, (
            f"--check-op on primary must catch R16 violation in variant, "
            f"got errors: {errors}"
        )

        # good_variant should NOT have R16 errors
        good_r16 = [e for e in errors if "good_variant" in e and "R16" in e]
        assert good_r16 == [], (
            f"good_variant should pass R16, got: {good_r16}"
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
        manifest_file.write_text(yaml.safe_dump({
            "primary_op": primary,
            "broken_var": broken_variant,
        }))

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
        assert validator._parse_check_op(["--check-op", "SoftmaxFwdOp"]) == "SoftmaxFwdOp"
        assert validator._parse_check_op(["--check-op=SoftmaxFwdOp"]) == "SoftmaxFwdOp"
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
# _resolve_op_class: multi-class file resolution
# ---------------------------------------------------------------------------

class TestResolveOpClass:
    """_resolve_op_class correctly resolves op names to classes in multi-class files."""

    def test_single_class_file_exact_match(self, validator):
        """Single-class files resolve only when manifest key matches class name."""
        result = validator._resolve_op_class(
            "tileops/ops/reduction/softmax.py", "SoftmaxFwdOp",
        )
        assert result.cls is not None
        assert result.cls.__name__ == "SoftmaxFwdOp"

    def test_single_class_file_rejects_mismatched_name(self, validator):
        """Single-class files reject mismatched manifest keys — no bypass."""
        result = validator._resolve_op_class(
            "tileops/ops/reduction/softmax.py", "SoftmaxBwdOp",
        )
        assert result.cls is None
        assert result.warning is not None

    def test_exact_match_required_multi_class(self, validator):
        """Multi-class files require exact cls.__name__ == manifest key."""
        result = validator._resolve_op_class(
            "tileops/ops/reduction/reduce.py", "SumFwdOp",
        )
        assert result.cls is not None
        assert result.cls.__name__ == "SumFwdOp"

    @pytest.mark.parametrize(
        "op_name",
        [
            "SumFwdOp", "MeanFwdOp", "AmaxFwdOp", "AminFwdOp",
            "ProdFwdOp", "VarFwdOp", "StdFwdOp", "VarMeanFwdOp",
        ],
    )
    def test_reduce_ops_resolve_by_exact_name(self, validator, op_name):
        """Reduce Op classes resolve correctly by exact cls.__name__ match."""
        result = validator._resolve_op_class(
            "tileops/ops/reduction/reduce.py", op_name,
        )
        assert result.cls is not None
        assert result.cls.__name__ == op_name

    def test_nonexistent_module_returns_import_error(self, validator):
        """Module that cannot be imported returns import_error=True."""
        result = validator._resolve_op_class(
            "tileops/ops/nonexistent.py", "some_op",
        )
        assert result.import_error

    def test_module_with_no_op_classes_returns_none(self, validator):
        """Module with no forward()-bearing classes returns cls=None."""
        result = validator._resolve_op_class(
            "tileops/__init__.py", "some_op",
        )
        assert result.cls is None

    def test_ambiguous_fallback_returns_none_with_warning(self, validator):
        """When multiple candidates exist but no heuristic matches, return cls=None."""
        import importlib
        import types

        # Create a fake module with two candidate classes whose names don't
        # match any heuristic for the given op_name.
        fake_mod = types.ModuleType("tileops.ops.fake_ambiguous")
        fake_mod.__name__ = "tileops.ops.fake_ambiguous"

        class AlphaKernel:
            @staticmethod
            def forward():
                pass

        class BetaKernel:
            @staticmethod
            def forward():
                pass

        AlphaKernel.__module__ = fake_mod.__name__
        BetaKernel.__module__ = fake_mod.__name__
        fake_mod.AlphaKernel = AlphaKernel
        fake_mod.BetaKernel = BetaKernel

        # Patch importlib.import_module to return the fake module
        original_import = importlib.import_module

        def patched_import(name):
            if name == "tileops.ops.fake_ambiguous":
                return fake_mod
            return original_import(name)

        import unittest.mock as mock

        with (
            mock.patch.object(importlib, "import_module", side_effect=patched_import),
            pytest.warns(UserWarning, match="No class named"),
        ):
            result = validator._resolve_op_class(
                "tileops/ops/fake_ambiguous.py", "mystery_fwd",
            )
        # Should return empty result (no cls) since no exact match
        assert result.cls is None
        assert not result.import_error
        assert "No class named" in result.warning

    def test_ambiguous_warning_plumbed_through_check_l1(self, validator):
        """Ambiguity warning surfaces in check_l1's structured warnings list."""
        import importlib
        import types
        import unittest.mock as mock

        fake_mod = types.ModuleType("tileops.ops.fake_ambiguous")
        fake_mod.__name__ = "tileops.ops.fake_ambiguous"

        class AlphaKernel:
            @staticmethod
            def forward():
                pass

        class BetaKernel:
            @staticmethod
            def forward():
                pass

        AlphaKernel.__module__ = fake_mod.__name__
        BetaKernel.__module__ = fake_mod.__name__
        fake_mod.AlphaKernel = AlphaKernel
        fake_mod.BetaKernel = BetaKernel

        original_import = importlib.import_module

        def patched_import(name):
            if name == "tileops.ops.fake_ambiguous":
                return fake_mod
            return original_import(name)

        entry = {
            "source": {"op": "tileops/ops/fake_ambiguous.py"},
            "signature": {"inputs": {}, "params": {}},
        }
        warn_list: list[str] = []

        with (
            mock.patch.object(importlib, "import_module", side_effect=patched_import),
            pytest.warns(UserWarning, match="No class named"),
        ):
            errors = validator.check_l1("mystery_fwd", entry, warnings=warn_list)

        assert any("No class named" in w for w in warn_list)
        assert any("could not resolve" in e for e in errors)

    def test_suffix_match_ambiguity_emits_warning(self, validator):
        """When no class matches the manifest key exactly, emit warning."""
        import importlib
        import types
        import unittest.mock as mock

        fake_mod = types.ModuleType("tileops.ops.fake_no_match")
        fake_mod.__name__ = "tileops.ops.fake_no_match"

        class AlphaFwdOp:
            @staticmethod
            def forward():
                pass

        class BetaFwdOp:
            @staticmethod
            def forward():
                pass

        AlphaFwdOp.__module__ = fake_mod.__name__
        BetaFwdOp.__module__ = fake_mod.__name__
        fake_mod.AlphaFwdOp = AlphaFwdOp
        fake_mod.BetaFwdOp = BetaFwdOp

        original_import = importlib.import_module

        def patched_import(name):
            if name == "tileops.ops.fake_no_match":
                return fake_mod
            return original_import(name)

        with (
            mock.patch.object(importlib, "import_module", side_effect=patched_import),
            pytest.warns(UserWarning, match="No class named"),
        ):
            result = validator._resolve_op_class(
                "tileops/ops/fake_no_match.py", "GammaFwdOp",
            )
        # No exact match: cls should be None and warning emitted
        assert result.cls is None
        assert not result.import_error
        assert "No class named" in result.warning

    def test_direct_match_resolves_exact_class_name(self, validator):
        """Direct match resolves when cls.__name__ == manifest key.

        For op_name='SumFwdOp' with both _SumHelper and SumFwdOp in the module,
        the exact match finds SumFwdOp. No heuristic fallback.
        """
        import importlib
        import types
        import unittest.mock as mock

        fake_mod = types.ModuleType("tileops.ops.fake_priority")
        fake_mod.__name__ = "tileops.ops.fake_priority"

        class _SumHelper:
            @staticmethod
            def forward():
                pass

        class SumFwdOp:
            @staticmethod
            def forward():
                pass

        _SumHelper.__module__ = fake_mod.__name__
        SumFwdOp.__module__ = fake_mod.__name__
        fake_mod._SumHelper = _SumHelper
        fake_mod.SumFwdOp = SumFwdOp

        original_import = importlib.import_module

        def patched_import(name):
            if name == "tileops.ops.fake_priority":
                return fake_mod
            return original_import(name)

        with mock.patch.object(importlib, "import_module", side_effect=patched_import):
            result = validator._resolve_op_class(
                "tileops/ops/fake_priority.py", "SumFwdOp",
            )
        # Direct match finds SumFwdOp when the class name IS the manifest key
        assert result.cls is SumFwdOp, (
            f"Expected SumFwdOp (direct match) but got {result.cls.__name__}"
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

    def test_schema_validation_no_errors_on_real_manifest(self, validator):
        """Schema-level validation on the checked-in manifest produces no errors.

        Warnings (e.g. missing kernel_map for implemented ops) are acceptable
        since populating kernel_map for all ops is tracked separately.
        """
        errors, warnings = validator.validate_manifest(
            levels=frozenset({"schema"}),
        )
        assert errors == [], (
            f"Schema validation produced {len(errors)} error(s) on the "
            f"checked-in manifest:\n" + "\n".join(errors)
        )


# ---------------------------------------------------------------------------
# tileops.manifest.shape_rules helper module + validator integration
# ---------------------------------------------------------------------------


class TestShapeRuleHelpers:
    """Unit tests for :mod:`tileops.manifest.shape_rules` predicates."""

    def test_dim_range_validity_none_passes(self):
        from tileops.manifest.shape_rules import dim_range_validity

        x = type("X", (), {"ndim": 3})()
        assert dim_range_validity(x, None) is True

    def test_dim_range_validity_in_range_int_passes(self):
        from tileops.manifest.shape_rules import dim_range_validity

        x = type("X", (), {"ndim": 4})()
        for d in (-4, -1, 0, 3):
            assert dim_range_validity(x, d) is True, d

    def test_dim_range_validity_out_of_range_int_fails(self):
        from tileops.manifest.shape_rules import dim_range_validity

        x = type("X", (), {"ndim": 4})()
        for d in (-5, 4, 99):
            assert dim_range_validity(x, d) is False, d

    def test_dim_range_validity_sequence_all_in_range_passes(self):
        from tileops.manifest.shape_rules import dim_range_validity

        x = type("X", (), {"ndim": 4})()
        assert dim_range_validity(x, [0, -1, 2]) is True
        assert dim_range_validity(x, (0, 1)) is True

    def test_dim_range_validity_sequence_with_one_out_of_range_fails(self):
        from tileops.manifest.shape_rules import dim_range_validity

        x = type("X", (), {"ndim": 3})()
        assert dim_range_validity(x, [0, 5]) is False

    def test_dim_range_validity_malformed_dim_propagates_inline_error(self):
        """Malformed sequence dims propagate the same TypeError the inline form raises.

        The validator classifies eval errors as warnings (parity check
        skipped), so behaviour matches the pre-migration inline form
        end-to-end. Helper unit tests only need to confirm the exception
        propagates rather than being silenced.
        """
        import pytest

        from tileops.manifest.shape_rules import dim_range_validity

        x = type("X", (), {"ndim": 3})()
        # Iterating a string yields characters; ``-3 <= "0"`` raises
        # TypeError, matching the inline ``all(... for d in dim)`` form.
        with pytest.raises(TypeError):
            dim_range_validity(x, "0")
        # A list element that cannot be ordered against int raises the
        # same TypeError the inline expression would.
        with pytest.raises(TypeError):
            dim_range_validity(x, ["2"])

    def test_dim_range_validity_inline_quirks_preserved(self):
        """Exact parity with inline ``isinstance(dim, int)`` semantics.

        ``bool`` subclasses ``int``, so the inline expression treats
        ``True`` as a one-element axis list — and so must the helper.
        A non-list, non-tuple iterable (e.g. ``set``) was iterated
        directly by the inline form; the helper preserves that quirk.
        """
        from tileops.manifest.shape_rules import dim_range_validity

        x = type("X", (), {"ndim": 3})()
        # bool path: True -> [True] -> -3 <= 1 < 3 -> True.
        assert dim_range_validity(x, True) is True
        # set path: -3 <= 0 < 3 -> True.
        assert dim_range_validity(x, {0}) is True

    def test_dim_uniqueness_int_or_none_passes(self):
        from tileops.manifest.shape_rules import dim_uniqueness

        x = type("X", (), {"ndim": 4})()
        assert dim_uniqueness(x, None) is True
        assert dim_uniqueness(x, 0) is True
        assert dim_uniqueness(x, -2) is True

    def test_dim_uniqueness_unique_sequence_passes(self):
        from tileops.manifest.shape_rules import dim_uniqueness

        x = type("X", (), {"ndim": 4})()
        assert dim_uniqueness(x, [0, 1, 3]) is True
        assert dim_uniqueness(x, [0, -1]) is True  # -1 normalises to 3

    def test_dim_uniqueness_duplicate_after_normalization_fails(self):
        from tileops.manifest.shape_rules import dim_uniqueness

        x = type("X", (), {"ndim": 4})()
        # -4 and 0 collapse to the same axis.
        assert dim_uniqueness(x, [-4, 0]) is False
        assert dim_uniqueness(x, [1, 1]) is False

    def test_dim_uniqueness_empty_sequence_passes(self):
        from tileops.manifest.shape_rules import dim_uniqueness

        x = type("X", (), {"ndim": 3})()
        assert dim_uniqueness(x, []) is True

    def test_reduced_axes_none_returns_full_range(self):
        from tileops.manifest.shape_rules import reduced_axes

        x = type("X", (), {"ndim": 3})()
        assert reduced_axes(x, None) == frozenset({0, 1, 2})

    def test_reduced_axes_int_returns_single_normalized_axis(self):
        from tileops.manifest.shape_rules import reduced_axes

        x = type("X", (), {"ndim": 4})()
        assert reduced_axes(x, 0) == frozenset({0})
        assert reduced_axes(x, -1) == frozenset({3})
        assert reduced_axes(x, -4) == frozenset({0})

    def test_reduced_axes_non_empty_sequence_returns_normalized_set(self):
        from tileops.manifest.shape_rules import reduced_axes

        x = type("X", (), {"ndim": 4})()
        assert reduced_axes(x, [0, 2]) == frozenset({0, 2})
        assert reduced_axes(x, (-1, -2)) == frozenset({2, 3})

    def test_reduced_axes_empty_sequence_returns_full_range(self):
        from tileops.manifest.shape_rules import reduced_axes

        x = type("X", (), {"ndim": 3})()
        assert reduced_axes(x, []) == frozenset({0, 1, 2})
        assert reduced_axes(x, ()) == frozenset({0, 1, 2})

    def test_reduced_axes_matches_inline_expression_for_sum_cases(self):
        """Bit-identical parity with the expression migrated out of SumFwdOp.

        The inline form was
        ``{dim % x.ndim} if isinstance(dim, int) else
          {d % x.ndim for d in dim} if isinstance(dim, (list, tuple)) and len(dim) > 0
          else set(range(x.ndim))``.
        Drive both forms with the same inputs and assert equality so any
        future helper edit that drifts from the manifest's prior semantics
        fails loudly.
        """
        from tileops.manifest.shape_rules import reduced_axes

        def inline(x, dim):
            if isinstance(dim, int):
                return frozenset({dim % x.ndim})
            if isinstance(dim, (list, tuple)) and len(dim) > 0:
                return frozenset(d % x.ndim for d in dim)
            return frozenset(range(x.ndim))

        x = type("X", (), {"ndim": 4})()
        cases = [None, 0, -1, 2, [0, 2], (-1, -2), [], ()]
        for d in cases:
            assert reduced_axes(x, d) == inline(x, d), d

    def test_helpers_match_inline_expressions_over_full_case_matrix(self):
        """Helpers and inline expressions agree on results AND raised exceptions.

        Drives every helper against the literal inline expression that
        was migrated out of the manifest, over a case matrix that
        includes well-formed *and* malformed inputs. For each input, both
        forms must either return the same value or raise the same
        exception type — anything else is a behavioural drift.

        Covers the malformed cases the validator previously surfaced as
        eval-error warnings (``dim=["2"]``, ``dim=[1.5]``), as well as
        the contract-spec edge cases (``dim=None`` for "all axes" and an
        empty tuple).
        """
        from tileops.manifest.shape_rules import (
            dim_range_validity,
            dim_uniqueness,
            reduced_axes,
        )

        def inline_range(x, dim):
            return dim is None or all(
                -x.ndim <= d < x.ndim
                for d in ([dim] if isinstance(dim, int) else dim)
            )

        def inline_uniqueness(x, dim):
            return isinstance(dim, (int, type(None))) or (
                len({d % x.ndim for d in dim}) == len(dim)
            )

        def inline_axes(x, dim):
            if isinstance(dim, int):
                return frozenset({dim % x.ndim})
            if isinstance(dim, (list, tuple)) and len(dim) > 0:
                return frozenset(d % x.ndim for d in dim)
            return frozenset(range(x.ndim))

        x = type("X", (), {"ndim": 3})()
        cases = [
            None,
            0,
            -1,
            2,
            [0, 2],
            (-1, -2),
            [],
            (),
            # Reviewer reproducer: a string element triggers TypeError in
            # both ``-3 <= "2"`` and ``"2" % 3``; the helper must
            # propagate so the validator's eval-error path keeps treating
            # this as a warning, not a hard shape mismatch.
            ["2"],
            # A float element survives ordering and modulo; both forms
            # return the same set.
            [1.5],
            # ``dim=None`` is the contract-spec "all axes" sentinel.
            None,
            # Empty tuple: explicit "all axes" per the inline else clause.
            (),
        ]
        pairs = [
            (dim_range_validity, inline_range),
            (dim_uniqueness, inline_uniqueness),
            (reduced_axes, inline_axes),
        ]
        for helper, inline in pairs:
            for d in cases:
                helper_exc: type[BaseException] | None = None
                inline_exc: type[BaseException] | None = None
                helper_val = inline_val = None
                try:
                    helper_val = helper(x, d)
                except Exception as exc:  # noqa: BLE001
                    helper_exc = type(exc)
                try:
                    inline_val = inline(x, d)
                except Exception as exc:  # noqa: BLE001
                    inline_exc = type(exc)
                assert helper_exc is inline_exc, (
                    helper.__name__, d, helper_exc, inline_exc,
                )
                if helper_exc is None:
                    assert helper_val == inline_val, (
                        helper.__name__, d, helper_val, inline_val,
                    )

    def test_helper_rule_validator_warns_on_malformed_dim(self, validator):
        """Validator integration: helper rules surface malformed dims as warnings.

        The reviewer's reproducer (``dim=["2"]``) raised TypeError under
        the inline expression, which the validator classified as an
        eval-error warning ("could not be evaluated"). The helper-based
        rule must hit the same path: the parity check is skipped with a
        warning, not turned into a hard shape error.
        """
        def infer(self, x_shape, *, dim=None, keepdim=False):  # noqa: ARG001
            return {"y": x_shape}

        cls = _make_op_cls_with_infer(infer, name="HelperMalformedDimOp")
        sig_common = {
            "inputs": {"x": {"dtype": "float16"}},
            "outputs": {"y": {"dtype": "same_as(x)"}},
            "params": {
                "dim": {
                    "type": "int | list[int] | tuple[int, ...] | None",
                    "default": ["2"],
                },
                "keepdim": {"type": "bool", "default": False},
            },
        }
        entry_inline = {
            "signature": {
                **sig_common,
                "shape_rules": [
                    "dim is None or all(-x.ndim <= d < x.ndim for d in "
                    "([dim] if isinstance(dim, int) else dim))",
                    "isinstance(dim, (int, type(None))) or "
                    "len({d % x.ndim for d in dim}) == len(dim)",
                ],
            },
        }
        entry_helper = {
            "signature": {
                **sig_common,
                "shape_rules": [
                    "helper:dim_range_validity(x, dim)",
                    "helper:dim_uniqueness(x, dim)",
                ],
            },
        }
        warn_inline: list[str] = []
        warn_helper: list[str] = []
        errs_inline = validator.check_l2_infer_parity(
            "HelperMalformedDimOp", entry_inline, cls, warnings=warn_inline,
        )
        errs_helper = validator.check_l2_infer_parity(
            "HelperMalformedDimOp", entry_helper, cls, warnings=warn_helper,
        )
        # Both forms classify the malformed dim as an eval error and
        # emit a "could not be evaluated" warning; neither raises a hard
        # parity error.
        assert errs_inline == [] == errs_helper, (errs_inline, errs_helper)
        assert any("could not be evaluated" in w for w in warn_inline), (
            warn_inline
        )
        assert any("could not be evaluated" in w for w in warn_helper), (
            warn_helper
        )

    def test_helpers_registry_exposes_all_helpers(self):
        from tileops.manifest.shape_rules import (
            HELPERS,
            dim_range_validity,
            dim_uniqueness,
            reduced_axes,
        )

        assert HELPERS["dim_range_validity"] is dim_range_validity
        assert HELPERS["dim_uniqueness"] is dim_uniqueness
        assert HELPERS["reduced_axes"] is reduced_axes

    def test_helpers_have_docstrings(self):
        """Each helper documents its contract."""
        from tileops.manifest.shape_rules import (
            dim_range_validity,
            dim_uniqueness,
            reduced_axes,
        )

        for fn in (dim_range_validity, dim_uniqueness, reduced_axes):
            doc = (fn.__doc__ or "").strip()
            assert doc, f"{fn.__name__} missing docstring"
            # Non-trivial: at least a sentence beyond the summary line.
            assert len(doc.splitlines()) > 1, fn.__name__


class TestValidatorHelperResolution:
    """Validator integration of the ``helper:`` URI scheme."""

    def test_check_l2_accepts_helper_prefixed_rule(self, validator):
        """L2 syntax check parses ``helper:NAME(args)`` rules without error."""
        entry = {
            "signature": {
                "shape_rules": ["helper:dim_range_validity(x, dim)"],
            },
        }
        errors = validator.check_l2("op", entry)
        assert errors == [], errors

    def test_check_l0_rejects_unknown_helper_name(self, validator):
        """Unknown helper names must be flagged at L0 (typo defence)."""
        entry = _make_entry()
        entry["signature"]["shape_rules"] = ["helper:no_such_helper(x, dim)"]
        errors = validator.check_l0("op", entry)
        assert any(
            "shape_rules" in e and "no_such_helper" in e for e in errors
        ), errors

    def test_check_l0_rejects_malformed_helper_uri(self, validator):
        """Helper rules must parse as ``NAME(args)`` after the prefix."""
        entry = _make_entry()
        entry["signature"]["shape_rules"] = ["helper:not a call"]
        errors = validator.check_l0("op", entry)
        assert any("shape_rules" in e for e in errors), errors

    def test_check_l0_accepts_known_helper(self, validator):
        """A valid helper reference produces no L0 errors."""
        entry = _make_entry()
        entry["signature"]["shape_rules"] = [
            "helper:dim_range_validity(x, dim)",
            "helper:dim_uniqueness(x, dim)",
        ]
        errors = validator.check_l0("op", entry)
        assert errors == [], errors

    def test_l2_parity_helper_equivalent_to_inline(self, validator):
        """Helper rule produces identical parity outcome to the inline form."""
        # An op with a single-int dim that satisfies the range. Both the
        # inline-string rule and the helper-prefixed rule must accept the
        # same _infer_output_shapes result (i.e., parity behaviour is
        # bit-for-bit the same).
        def infer(self, x_shape):
            return {"y": x_shape}

        cls = _make_op_cls_with_infer(infer, name="HelperParityOp")
        sig_common = {
            "inputs": {"x": {"dtype": "float16"}},
            "outputs": {"y": {"dtype": "same_as(x)"}},
            "params": {"dim": {"type": "int | None", "default": -1}},
        }
        entry_inline = {
            "signature": {
                **sig_common,
                "shape_rules": [
                    "dim is None or -x.ndim <= dim < x.ndim",
                    "y.shape == x.shape",
                ],
            },
        }
        entry_helper = {
            "signature": {
                **sig_common,
                "shape_rules": [
                    "helper:dim_range_validity(x, dim)",
                    "y.shape == x.shape",
                ],
            },
        }
        errs_inline = validator.check_l2_infer_parity(
            "HelperParityOp", entry_inline, cls,
        )
        errs_helper = validator.check_l2_infer_parity(
            "HelperParityOp", entry_helper, cls,
        )
        assert errs_inline == errs_helper == []

    def test_l2_parity_helper_detects_out_of_range_default(self, validator):
        """Helper rule fires when the configured dim is invalid for the rank.

        Mirrors what the inline expression would do — a wrong default
        ``dim`` against a 2-D mock input must surface a parity failure
        via the helper resolution path.
        """
        def infer(self, x_shape):
            return {"y": x_shape}

        cls = _make_op_cls_with_infer(infer, name="HelperBadDimOp")
        # Mock inputs for a single-rank-2 tensor; default dim=9 is out of
        # range. The helper rule must fail under mock evaluation.
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "params": {"dim": {"type": "int", "default": 9}},
                "shape_rules": [
                    "x.shape == (B, S)",
                    "helper:dim_range_validity(x, dim)",
                    "y.shape == x.shape",
                ],
            },
        }
        warnings: list[str] = []
        errors = validator.check_l2_infer_parity(
            "HelperBadDimOp", entry, cls, warnings=warnings,
        )
        # Out-of-range dim is an input-only precondition that mock inputs
        # violate; the validator classifies that as a skip with a
        # warning (not a hard error). Concrete expected outcome:
        #   - no parity errors (the helper rule must not blame a correct
        #     ``_infer_output_shapes``),
        #   - no "could not be evaluated" warning (the helper resolved and
        #     ran — failure was a real predicate result, not an eval skip),
        #   - exactly one "input-only precondition" warning citing the
        #     helper rule itself, proving the helper-resolution path
        #     produced the same classification as the inline form would.
        assert errors == [], errors
        assert not any(
            "could not be evaluated" in w for w in warnings
        ), warnings
        precondition_hits = [
            w for w in warnings
            if "input-only precondition" in w
            and "helper:dim_range_validity(x, dim)" in w
        ]
        assert len(precondition_hits) == 1, warnings

    def test_unmigrated_inline_rules_unchanged(self, validator):
        """Constraint: unmigrated rules must keep behaving identically.

        An inline expression equivalent to the helper produces identical
        parity output regardless of whether the helper module is
        importable; this pins the opt-in property.
        """
        def infer(self, x_shape):
            return {"y": x_shape}

        cls = _make_op_cls_with_infer(infer, name="UnmigratedOp")
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "params": {"dim": {"type": "int | None", "default": -1}},
                "shape_rules": [
                    "dim is None or -x.ndim <= dim < x.ndim",
                    "y.shape == x.shape",
                ],
            },
        }
        assert validator.check_l2_infer_parity(
            "UnmigratedOp", entry, cls,
        ) == []

    def test_unprefixed_predicate_helper_raises_name_error(self, validator):
        """Opt-in property: predicate helpers are NOT in the unprefixed eval scope.

        An unprefixed shape_rule that names ``dim_range_validity(x, dim)``
        must surface an eval-error warning (NameError) rather than
        silently resolving the call. This pins the contract that
        callers must go through ``helper:`` to access predicate helpers,
        so the L0 helper-name validation in ``_check_helper_rule``
        cannot be bypassed.
        """
        import types
        ok, reason = validator._eval_shape_rule(
            "dim_range_validity(x, dim)",
            {"x": types.SimpleNamespace(ndim=4), "dim": 0},
        )
        assert ok is False
        assert reason is not None
        assert "NameError" in reason
        assert "dim_range_validity" in reason

    def test_unprefixed_dim_uniqueness_raises_name_error(self, validator):
        """Same opt-in contract for ``dim_uniqueness``."""
        import types
        ok, reason = validator._eval_shape_rule(
            "dim_uniqueness(x, dim)",
            {"x": types.SimpleNamespace(ndim=4), "dim": [0, 1]},
        )
        assert ok is False
        assert reason is not None
        assert "NameError" in reason
        assert "dim_uniqueness" in reason

    def test_unprefixed_reduced_axes_remains_callable(self, validator):
        """``reduced_axes`` is a value extractor, exposed as a regular builtin.

        Unlike the predicate helpers, ``reduced_axes`` is referenced
        unprefixed inside larger output-shape expressions
        (e.g. ``output.ndim == x.ndim - len(reduced_axes(x, dim))``),
        so it must remain callable in the unprefixed eval scope to
        preserve pre-migration parity for SumFwdOp.
        """
        import types
        ok, reason = validator._eval_shape_rule(
            "len(reduced_axes(x, dim)) == 1",
            {"x": types.SimpleNamespace(ndim=4), "dim": 0},
        )
        assert reason is None, reason
        assert ok is True

    def test_helper_prefixed_predicate_resolves(self, validator):
        """Sanity: ``helper:`` prefix exposes predicate helpers as expected."""
        import types
        ok, reason = validator._eval_shape_rule(
            "helper:dim_range_validity(x, dim)",
            {"x": types.SimpleNamespace(ndim=4), "dim": 0},
        )
        assert reason is None, reason
        assert ok is True

    def test_migrated_sum_rules_parity_with_pre_migration_inline(self, validator):
        """Regression: SumFwdOp's helper-based rules match the pre-migration inline form.

        Captures the proof-of-concept end-to-end migration. The pre-
        migration manifest entry encoded the same five reduction-dim
        rules as inline Python. The post-migration entry replaces the
        first two with ``helper:`` predicates and lifts the shared
        "set of normalized reduction axes" expression into
        :func:`reduced_axes`. Driving both forms through
        :func:`check_l2_infer_parity` against the same mock op class
        must produce equal error lists; any future drift in helper
        semantics will fail this test.
        """
        def infer(self, x_shape, *, dim=None, keepdim=False):  # noqa: ARG001
            # Identity output is enough to exercise the rule eval path
            # under mock inputs; the rules themselves don't reach this.
            return {"output": x_shape}

        cls = _make_op_cls_with_infer(infer, name="SumParityOp")
        sig_common = {
            "inputs": {"x": {"dtype": "float16"}},
            "outputs": {"output": {"dtype": "same_as(x)"}},
            "params": {
                "dim": {
                    "type": "int | list[int] | tuple[int, ...] | None",
                    "default": None,
                },
                "keepdim": {"type": "bool", "default": False},
            },
        }
        inline_axes = (
            "({dim % x.ndim} if isinstance(dim, int) else "
            "{d % x.ndim for d in dim} if isinstance(dim, (list, tuple)) "
            "and len(dim) > 0 else set(range(x.ndim)))"
        )
        entry_pre = {
            "signature": {
                **sig_common,
                "shape_rules": [
                    "dim is None or all(-x.ndim <= d < x.ndim for d in "
                    "([dim] if isinstance(dim, int) else dim))",
                    "isinstance(dim, (int, type(None))) or "
                    "len({d % x.ndim for d in dim}) == len(dim)",
                    f"output.ndim == (x.ndim if keepdim else x.ndim - "
                    f"len({inline_axes}))",
                ],
            },
        }
        entry_post = {
            "signature": {
                **sig_common,
                "shape_rules": [
                    "helper:dim_range_validity(x, dim)",
                    "helper:dim_uniqueness(x, dim)",
                    "output.ndim == (x.ndim if keepdim else x.ndim - "
                    "len(reduced_axes(x, dim)))",
                ],
            },
        }
        errs_pre = validator.check_l2_infer_parity(
            "SumParityOp", entry_pre, cls,
        )
        errs_post = validator.check_l2_infer_parity(
            "SumParityOp", entry_post, cls,
        )
        # Error strings quote the rule text verbatim, so post-migration
        # entries naturally differ from pre-migration ones in the rule
        # body. What must stay identical is the validator's classification
        # at each rule index: the same number of errors, the same severity
        # tags ("[shape]"), and the same indices flagged. That captures
        # the AC-3 contract — bit-identical validator behaviour pre/post
        # — without coupling the test to literal rule wording.
        def _classify(errs: list[str]) -> list[str]:
            tags = []
            for e in errs:
                if "shape_rules[0]" in e:
                    tags.append("[shape] rule[0]")
                elif "shape_rules[1]" in e:
                    tags.append("[shape] rule[1]")
                elif "shape_rules[2]" in e:
                    tags.append("[shape] rule[2]")
                else:
                    tags.append(e.split(" ", 1)[0])
            return tags

        assert _classify(errs_pre) == _classify(errs_post), (
            errs_pre, errs_post,
        )
        assert len(errs_pre) == len(errs_post), (errs_pre, errs_post)
