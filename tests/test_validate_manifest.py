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


# Import the validator module dynamically (it lives in scripts/, not a package)

@pytest.fixture(scope="module")
def validator():
    """Import validate_manifest as a module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("validate_manifest", VALIDATOR_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# schema: YAML structure validation

def _make_entry(*, inputs=None, outputs=None, params=None, dtype_combos=None,
                 source_kernel="k.py", status="spec-only", kernel_map=None,
                 **extra):
    """Build a minimal valid manifest entry for testing, with overrides.

    Use ``status=None`` to explicitly omit the status field (for testing
    that the validator rejects entries without status).
    ``kernel_map`` is placed under ``source`` per the manifest spec.
    """
    sig = {
        "inputs": inputs if inputs is not None else {"x": {"dtype": "float16"}},
        "outputs": outputs if outputs is not None else {"y": {"dtype": "same_as(x)"}},
        "shape_rules": ["y.shape == x.shape"],
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
        "workloads": [
            {"x_shape": [1, 4096], "dtypes": ["float16"]},
            {"x_shape": [8, 8192], "dtypes": ["float16"]},
        ],
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

    def test_missing_or_mistyped_fields_rejected(self, validator):
        """Case table: each row mutates one field and pins its schema branch."""
        def _set_input(entry, attrs):
            entry["signature"]["inputs"] = {"x": attrs}

        cases = [
            # (description, entry mutator, substrings expected in one error)
            ("missing ref_api",
             lambda e: e.pop("ref_api"), ["ref_api"]),
            ("ref_api non-string",
             lambda e: e.update(ref_api=123), ["ref_api", "string"]),
            ("missing family",
             lambda e: e.pop("family"), ["family"]),
            ("signature missing outputs",
             lambda e: e["signature"].pop("outputs"), ["outputs"]),
            ("roofline needs (flops + bytes) or func",
             lambda e: e.update(roofline={"flops": "2 * M"}), ["roofline"]),
            ("params as list",
             lambda e: e["signature"].update(params=["training", "epsilon"]),
             ["params", "schema"]),
            ("tensor missing dtype",
             lambda e: _set_input(e, {}), ["dtype"]),
            ("param missing type",
             lambda e: e["signature"].update(params={"eps": {"default": 1e-6}}),
             ["params.eps", "type"]),
            ("status non-string",
             lambda e: e.update(status=123), ["status", "string"]),
            ("dtype_combos references unknown tensor (R4)",
             lambda e: e["signature"].update(
                 dtype_combos=[{"x": "float16", "nonexistent": "bfloat16"}]),
             ["nonexistent", "dtype_combos"]),
            ("source.kernel non-string non-list",
             lambda e: e["source"].update(kernel=42), ["source.kernel"]),
            ("unknown signature key (init_dims)",
             lambda e: e["signature"].update(init_dims={"N": "x.shape[-1]"}),
             ["init_dims", "unknown signature keys"]),
            ("unknown top-level key (parity_opt_out)",
             lambda e: e.update(parity_opt_out=True),
             ["parity_opt_out", "unknown entry keys"]),
            ("unrecognized layout value (R19)",
             lambda e: _set_input(e, {"dtype": "float16", "layout": "nchw"}),
             ["layout", "nchw"]),
            ("both inputs and params empty",
             lambda e: e["signature"].update(inputs={}, params={}),
             ["input", "param"]),
            ("outputs empty",
             lambda e: e["signature"].update(
                 outputs={}, params={"dtype": {"type": "torch.dtype"}}),
             ["outputs must declare at least one tensor"]),
        ]
        for desc, mutate, substrings in cases:
            entry = _make_entry()
            mutate(entry)
            errors = validator.check_l0("test_op", entry)
            assert any(
                all(s in e for s in substrings) for e in errors
            ), f"{desc}: expected error with {substrings}, got: {errors}"

    def test_key_variant_word_after_direction_op_rejected(self, validator):
        """Key format: variant words must precede the direction suffix."""
        errors = validator.check_l0("GroupNormFwdOpNoAffine", _make_entry())
        assert any(
            "NoAffine" in e and "precede" in e for e in errors
        ), errors

    def test_key_missing_direction_suffix_with_sibling_rejected(self, validator):
        """Key format: direction suffix is required when a direction sibling exists."""
        errors = validator.check_l0(
            "SoftmaxOp", _make_entry(),
            all_op_names={"SoftmaxOp", "SoftmaxFwdOp"},
        )
        assert any(
            "direction suffix" in e and "SoftmaxFwdOp" in e for e in errors
        ), errors

    def test_valid_forms_pass(self, validator):
        """Case table: entry variations the schema explicitly permits."""
        # Tensor with valid layout field (R19).
        entry = _make_entry(
            inputs={"x": {"dtype": "float16", "shape": "[N, H, W, C]",
                          "layout": "channels_last"}},
        )
        assert validator.check_l0("test_op", entry) == []

        # source.kernel as a list of strings.
        entry = _make_entry(source_kernel=["k1.py", "k2.py"])
        assert validator.check_l0("test_op", entry) == []

        # signature.inputs == {} with non-empty params: generative ops
        # (ALiBi / sinusoidal positional encodings) synthesize the output
        # entirely from construction-time parameters. The schema gate is
        # ``len(outputs) >= 1 AND (len(inputs) >= 1 OR len(params) >= 1)``.
        entry = _make_entry(
            inputs={},
            outputs={"output": {"dtype": "float16 | bfloat16 | float32"}},
            params={
                "seq_len": {"type": "int"},
                "dtype": {"type": "torch.dtype"},
            },
        )
        entry["workloads"] = [
            {"seq_len": 4096, "dtype": "float16", "dtypes": ["float16"]},
            {"seq_len": 8192, "dtype": "float16", "dtypes": ["float16"]},
        ]
        assert validator.check_l0("inputs_empty_op", entry) == []

    def test_static_dims_valid_forms_pass(self, validator):
        """R20: single-axis references via int literal or declared param."""
        entry = _make_entry()
        entry["signature"]["static_dims"] = {"N": "x.shape[-1]"}
        assert validator.check_l0("test_op", entry) == []

        entry = _make_entry(params={"dim": {"type": "int", "default": -1}})
        entry["signature"]["static_dims"] = {"N": "x.shape[dim]"}
        assert validator.check_l0("test_op", entry) == []

    def test_static_dims_invalid_forms_rejected(self, validator):
        """R20 case table: malformed static_dims produce schema errors."""
        cases = [
            # (description, params, static_dims value, expected substrings)
            ("non-dict static_dims", None, ["N"],
             ["static_dims", "must be a mapping"]),
            ("non-string value", None, {"N": {"from": "x.shape[-1]"}},
             ["static_dims.N", "string expression"]),
            ("multi-axis product form",
             {"dim": {"type": "int | None", "default": -1}},
             {"N": "product(x.shape[i] for i in range(x.ndim))"},
             ["static_dims.N", "single-axis reference"]),
            ("unknown tensor reference", None, {"N": "weight.shape[0]"},
             ["static_dims.N", "'weight'", "inputs"]),
            ("unknown param axis", None, {"N": "x.shape[dim]"},
             ["static_dims.N", "'dim'", "param"]),
        ]
        for desc, params, sdims, substrings in cases:
            entry = _make_entry(params=params)
            entry["signature"]["static_dims"] = sdims
            errors = validator.check_l0("test_op", entry)
            assert any(
                all(s in e for s in substrings) for e in errors
            ), f"{desc}: expected error with {substrings}, got: {errors}"

        # Malformed signature (absent or non-mapping inputs) must not crash
        # the static_dims check; other schema layers own those diagnostics.
        for inputs_val in (None, "not a mapping"):
            entry = {
                "signature": {
                    "outputs": {"y": {"dtype": "float32"}},
                    "static_dims": {"N": "x.shape[0]"},
                },
            }
            if inputs_val is not None:
                entry["signature"]["inputs"] = inputs_val
            errors = validator.check_l0("BadOp", entry)
            assert isinstance(errors, list)

    def test_kernel_map_status_gating(self, validator):
        """kernel_map is advisory-missing on implemented, optional on
        spec-only, and an empty mapping is valid."""
        # status: implemented without kernel_map -> warning, not error.
        entry = _make_entry(status="implemented")
        entry["source"].pop("kernel_map", None)
        warnings = []
        errors = validator.check_l0("test_op", entry, warnings=warnings)
        assert not any("kernel_map" in e for e in errors), errors
        assert any("kernel_map" in w for w in warnings), warnings

        # status: spec-only without kernel_map -> no kernel_map diagnostics.
        entry = _make_entry(status="spec-only")
        errors = validator.check_l0("test_op", entry)
        assert [e for e in errors if "kernel_map" in e] == []

        # Empty dict is a valid mapping of str -> str.
        entry = _make_entry(status="implemented", kernel_map={})
        assert validator.check_l0("test_op", entry) == []

    def test_kernel_map_malformed_rejected(self, validator):
        """Non-mapping kernel_map and non-str entries produce schema errors."""
        entry = _make_entry(status="implemented", kernel_map="not_a_dict")
        errors = validator.check_l0("test_op", entry)
        assert any("kernel_map" in e and "mapping" in e for e in errors)

        entry = _make_entry(status="implemented", kernel_map={"fwd": 123})
        errors = validator.check_l0("test_op", entry)
        assert any("kernel_map" in e for e in errors)

    def test_shape_rule_expressions_pass_l0(self, validator):
        """Registered builtins and attribute calls pass the L0 callable gate."""
        entry = _make_entry(
            inputs={"x": {"dtype": "float16"}, "y": {"dtype": "float16"}},
        )
        entry["signature"]["shape_rules"] = [
            "len(x.shape) == 2",
            "broadcast_shapes(x.shape, y.shape) == x.shape",
            "all(d > 0 for d in x.shape)",
            # Method/attribute calls are out of scope for the gate.
            "x.shape.count(1) == 0",
        ]
        assert validator.check_l0("test_op", entry) == []

    def test_shape_rule_expressions_rejected_at_l0(self, validator):
        """Unknown callables and syntax errors fail with [schema] errors;
        repeated misspellings in one rule are reported once per name."""
        entry = _make_entry()
        entry["signature"]["shape_rules"] = ["totally_unknown_helper(x.shape) == 0"]
        errors = validator.check_l0("test_op", entry)
        assert any(
            "[schema]" in e and "shape_rules[0]" in e
            and "totally_unknown_helper" in e
            for e in errors
        ), errors

        entry = _make_entry()
        entry["signature"]["shape_rules"] = ["x.shape == ("]
        errors = validator.check_l0("test_op", entry)
        assert any(
            "[schema]" in e and "shape_rules[0]" in e and "syntax" in e.lower()
            for e in errors
        ), errors

        entry = _make_entry()
        entry["signature"]["shape_rules"] = [
            "totally_unknown_helper(x.shape) and totally_unknown_helper(x.shape)"
        ]
        errors = validator.check_l0("test_op", entry)
        matching = [
            e for e in errors
            if "[schema]" in e and "shape_rules[0]" in e
            and "totally_unknown_helper" in e
        ]
        assert len(matching) == 1, (
            f"Expected one schema error for the duplicated unknown callable, "
            f"got {len(matching)}: {matching}"
        )


class TestWorkloadPolicy:
    """Workload-count and required-param coverage rules."""

    def test_implemented_needs_two_workloads(self, validator):
        """Implemented ops need >= 2 workloads; spec-only ops are exempt."""
        entry = _make_entry(status="implemented", kernel_map={})
        entry["workloads"] = entry["workloads"][:1]
        errors = validator.check_l0("test_op", entry)
        assert any(
            "at least 2 workloads" in e for e in errors
        ), f"Expected workload-count error, got: {errors}"

        entry = _make_entry(status="spec-only")
        entry["workloads"] = entry["workloads"][:1]
        errors = validator.check_l0("test_op", entry)
        assert not any("at least 2 workloads" in e for e in errors), errors

    def test_workload_missing_required_param_fails(self, validator):
        """Params without a default must appear in every workload."""
        entry = _make_entry(params={"dim": {"type": "int"}})
        entry["workloads"] = [
            {"x_shape": [1, 4096], "dtypes": ["float16"], "dim": -1},
            {"x_shape": [8, 8192], "dtypes": ["float16"]},  # dim missing
        ]
        errors = validator.check_l0("test_op", entry)
        assert any(
            "workloads[1]" in e and "required param" in e and "dim" in e
            for e in errors
        ), f"Expected required-param error, got: {errors}"

    def test_workload_defaulted_param_may_be_omitted(self, validator):
        """Params with a default are not required in workloads."""
        entry = _make_entry(params={"dim": {"type": "int", "default": -1}})
        errors = validator.check_l0("test_op", entry)
        assert errors == [], f"Unexpected schema errors: {errors}"


class TestOutputShapeDeclaration:
    """Every output declares a shape, or the signature has shape_rules."""

    def test_output_without_shape_or_rules_fails(self, validator):
        entry = _make_entry()
        del entry["signature"]["shape_rules"]
        errors = validator.check_l0("test_op", entry)
        assert any(
            "output" in e and "'y'" in e and "shape" in e for e in errors
        ), f"Expected output-shape error, got: {errors}"

    def test_output_with_declared_shape_passes(self, validator):
        entry = _make_entry(
            outputs={"y": {"dtype": "same_as(x)", "shape": "[M, N]"}},
        )
        del entry["signature"]["shape_rules"]
        errors = validator.check_l0("test_op", entry)
        assert errors == [], f"Unexpected schema errors: {errors}"


class TestTensorConstraints:
    """Tensor ``constraints`` keys must name dims of the declared shape."""

    def test_constraint_key_outside_shape_dims_fails(self, validator):
        entry = _make_entry(
            inputs={"x": {"dtype": "float16", "shape": "[M, N]",
                          "constraints": {"K": "K % 2 == 0"}}},
        )
        errors = validator.check_l0("test_op", entry)
        assert any(
            "constraints" in e and "'K'" in e for e in errors
        ), f"Expected constraint-key error, got: {errors}"

    def test_constraints_without_shape_fails(self, validator):
        entry = _make_entry(
            inputs={"x": {"dtype": "float16",
                          "constraints": {"N": "N % 2 == 0"}}},
        )
        errors = validator.check_l0("test_op", entry)
        assert any(
            "constraints" in e and "shape" in e for e in errors
        ), f"Expected constraints-without-shape error, got: {errors}"

    def test_constraint_keys_matching_shape_dims_pass(self, validator):
        entry = _make_entry(
            inputs={"x": {"dtype": "float16", "shape": "[M, N]",
                          "constraints": {"N": "N % 2 == 0"}}},
        )
        errors = validator.check_l0("test_op", entry)
        assert errors == [], f"Unexpected schema errors: {errors}"


class TestSourcePathExistence:
    """source path values of non-spec-only ops must point at real files."""

    def test_missing_source_file_fails_for_implemented(self, validator, tmp_path):
        import yaml

        entry = _make_entry(status="implemented", kernel_map={})
        manifest_file = tmp_path / "ops_manifest.yaml"
        manifest_file.write_text(yaml.safe_dump({"my_op": entry}))

        errors, _ = validator.validate_manifest(
            manifest_path=manifest_file,
            repo_root=tmp_path,
            levels=frozenset({"schema"}),
        )
        assert any(
            "source." in e and "not a file" in e for e in errors
        ), f"Expected source-path error, got: {errors}"

    def test_spec_only_source_paths_are_placeholders(self, validator, tmp_path):
        import yaml

        entry = _make_entry(status="spec-only")
        manifest_file = tmp_path / "ops_manifest.yaml"
        manifest_file.write_text(yaml.safe_dump({"my_op": entry}))

        errors, _ = validator.validate_manifest(
            manifest_path=manifest_file,
            repo_root=tmp_path,
            levels=frozenset({"schema"}),
        )
        assert not any("not a file" in e for e in errors), errors

    def test_existing_source_files_pass(self, validator, tmp_path):
        import yaml

        for name in ("k.py", "o.py", "t.py", "b.py"):
            (tmp_path / name).write_text("# placeholder\n")
        entry = _make_entry(status="implemented", kernel_map={})
        manifest_file = tmp_path / "ops_manifest.yaml"
        manifest_file.write_text(yaml.safe_dump({"my_op": entry}))

        errors, _ = validator.validate_manifest(
            manifest_path=manifest_file,
            repo_root=tmp_path,
            levels=frozenset({"schema"}),
        )
        assert errors == [], f"Unexpected errors: {errors}"


class TestSingleInputWorkloadKeys:
    """R21: workload shape keys derive from signature.inputs."""

    @staticmethod
    def _sig(input_name="input", params=("dim",)):
        return {
            "inputs": {input_name: {"dtype": "float16"}},
            "outputs": {"output": {"dtype": f"same_as({input_name})"}},
            "params": {p: {"type": "int"} for p in params},
        }

    def test_violations_are_reported(self, validator):
        sig = self._sig()
        wrong_key = validator._check_single_input_workload_keys(
            "op", sig, [{"x_shape": [8], "dtypes": ["float16"]}])
        unknown_key = validator._check_single_input_workload_keys(
            "op", sig, [{"input_shape": [8], "dtypes": ["float16"], "dmi": 0}])
        collision = validator._check_single_input_workload_keys(
            "op", self._sig(params=("dtypes",)),
            [{"input_shape": [8], "dtypes": ["float16"]}])
        assert any("input_shape" in e for e in wrong_key), wrong_key
        assert any("dmi" in e for e in unknown_key), unknown_key
        assert any("collide" in e for e in collision), collision

    def test_out_of_scope_shapes_pass(self, validator):
        multi = {"inputs": {"q": {"dtype": "float16"},
                            "k": {"dtype": "float16"}}, "params": {}}
        assert validator._check_single_input_workload_keys(
            "op", multi,
            [{"q_shape": [8], "kv_shape": [8], "dtypes": ["float16"]}]) == []
        assert validator._check_single_input_workload_keys(
            "op", self._sig(params=("num_tokens",)),
            [{"num_tokens": 4096, "dtypes": ["float16"]}]) == []

    def test_non_string_workload_key_is_schema_error_not_crash(self, validator):
        entry = _make_entry()
        entry["workloads"] = [{"x_shape": [8], 1: "bad", "dtypes": ["float16"]}]
        errors = validator.check_l0("op", entry)
        assert any("non-string" in e for e in errors), errors

    def test_malformed_signature_fields_report_not_crash(self, validator):
        """check_l0 stays total on garbage YAML, reporting schema errors instead of crashing."""
        entry = _make_entry()
        entry["signature"]["params"] = 7
        assert validator.check_l0("op", entry)

        entry = _make_entry(inputs=7)
        assert validator.check_l0("op", entry)

        entry = _make_entry(inputs={1: {"dtype": "float16"}})
        assert any("non-string" in e for e in validator.check_l0("op", entry))

        entry = _make_entry(params={3: {"type": "int"}})
        assert any("non-string" in e for e in validator.check_l0("op", entry))

        entry = _make_entry()
        entry["signature"][1] = "junk"
        entry["signature"]["zzz"] = "junk"
        entry[2] = "junk"
        entry["yyy"] = "junk"
        errors = validator.check_l0("op", entry)
        assert any("unknown signature keys" in e for e in errors), errors
        assert any("unknown entry keys" in e for e in errors), errors


# torch_compile_fullgraph: capability flag schema

class TestRooflineStructuralRules:
    """L0 roofline reject branches, one guard each.

    Accept paths are owned by TestIntegration: the shipped manifest
    exercises inline and func modes through the same checks.
    """

    def _entry(self, validator, roofline):
        entry = _make_entry()
        entry["roofline"] = roofline
        return validator.check_l0("my_op", entry)

    def test_mixed_modes_fail(self, validator):
        errors = self._entry(
            validator, {"flops": "2*M*N", "bytes": "M*N", "func": "tileops.perf.formulas.gemm"})
        assert any("exclusive" in e for e in errors)

    def test_non_string_field_fails(self, validator):
        """Integer placeholders were the shipped violation shape."""
        errors = self._entry(validator, {"flops": 0, "bytes": "M*N"})
        assert any("non-empty string" in e for e in errors)

    def test_vars_non_mapping_fails(self, validator):
        errors = self._entry(
            validator, {"flops": "2*M*N", "bytes": "M*N", "vars": ["M"]})
        assert any("vars must be a mapping" in e for e in errors)

    def test_vars_non_string_value_fails(self, validator):
        errors = self._entry(
            validator, {"flops": "2*M*N", "bytes": "M*N", "vars": {"M": 4}})
        assert any("vars" in e and "non-empty string" in e for e in errors)

    def test_vars_non_string_key_fails(self, validator):
        errors = self._entry(
            validator, {"flops": "2*M*N", "bytes": "M*N", "vars": {4: "M"}})
        assert any("key" in e and "must be a string" in e for e in errors)

    def test_unresolvable_func_fails(self, validator):
        errors = self._entry(validator, {"func": "tileops.perf.formulas.no_such_formula"})
        assert any("does not resolve" in e for e in errors)

    def test_non_callable_func_fails(self, validator):
        """Distinguishes the callable() predicate from a hasattr regression."""
        errors = self._entry(validator, {"func": "tileops.perf.formulas.__doc__"})
        assert any("does not resolve" in e for e in errors)


class TestTorchCompileFullgraph:
    """torch_compile_fullgraph accepts only literal true on implemented ops."""

    def test_valid_spellings_pass(self, validator):
        """Absence (the only 'no promise' spelling) and literal true on an
        implemented op both pass the schema check."""
        for extra in ({}, {"torch_compile_fullgraph": True}):
            entry = _make_entry(status="implemented", kernel_map={}, **extra)
            errors = validator.check_l0("test_op", entry)
            assert not any("torch_compile_fullgraph" in e for e in errors), (
                f"Unexpected torch_compile_fullgraph errors for {extra}: {errors}"
            )

    def test_invalid_spellings_rejected(self, validator):
        """false, non-bool values, and spec-only placement are all rejected."""
        # Non-true values: absence is the only spelling of "no promise".
        for value in (False, "true", 1, None):
            entry = _make_entry(
                status="implemented", kernel_map={},
                torch_compile_fullgraph=value,
            )
            errors = validator.check_l0("test_op", entry)
            assert any(
                "torch_compile_fullgraph" in e and "true" in e for e in errors
            ), f"Expected literal-true error for {value!r}, got: {errors}"

        # The field is invalid on status: spec-only entries.
        entry = _make_entry(status="spec-only", torch_compile_fullgraph=True)
        errors = validator.check_l0("test_op", entry)
        assert any(
            "torch_compile_fullgraph" in e and "spec-only" in e for e in errors
        ), f"Expected spec-only rejection, got: {errors}"


# variant_of: cross-entry consistency (R16)

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


# signature: Op.forward() consistency


def _run_check_l1(validator, monkeypatch, cls, signature):
    """Drive the public ``check_l1`` entry point with a synthetic Op class.

    ``_resolve_op_class`` is monkeypatched to hand back *cls*, so the
    synthetic entry needs no importable module; ``__init__``/``forward``
    parameters are read off the class by ``check_l1`` itself.
    """
    monkeypatch.setattr(
        validator, "_resolve_op_class",
        lambda op_file, op_name: validator._ResolveResult(cls=cls),
    )
    entry = {
        "signature": signature,
        "source": {"op": "tileops/ops/synthetic.py"},
    }
    return validator.check_l1(cls.__name__, entry, warnings=[])


class TestSignature:
    """signature checks that Op.forward() params match manifest inputs."""

    def test_spec_only_null_source_op_skips_class_resolution(self, validator):
        """Spec-only entries with source.op: null skip L1 implementation checks."""
        entry = _make_entry(status="spec-only")
        entry["source"]["op"] = None
        warn_list = []

        errors = validator.check_l1("MissingSpecOnlyOp", entry, warnings=warn_list)

        assert errors == []
        assert len(warn_list) == 1
        assert "spec-only" in warn_list[0] and "null" in warn_list[0]

    def test_implemented_null_source_op_fails(self, validator):
        """Implemented entries still require a resolvable source.op."""
        entry = _make_entry(status="implemented")
        entry["source"]["op"] = None

        errors = validator.check_l1("MissingImplementedOp", entry)

        assert errors == ["[signature] MissingImplementedOp: missing source.op"]

    def test_signature_parity_accepted_forms(self, validator, monkeypatch):
        """Case table: inputs / params / static_dims placements the L1
        signature check accepts, driven through the public ``check_l1``
        entry point with synthetic Op classes."""
        class FwdMatchOp:
            def __init__(self): pass
            def forward(self, x, weight): return None

        class FwdParamOp:
            def __init__(self): pass
            def forward(self, x, weight, training=True): return None

        class InitParamOp:
            def __init__(self, M, N, dtype, eps=1e-6): pass
            def forward(self, x): return None

        class StaticDimInitOp:
            def __init__(self, N, dtype, dim=-1): pass
            def forward(self, x): return None

        cases = [
            # (description, Op class, manifest signature)
            ("forward params match manifest inputs", FwdMatchOp, {
                "inputs": {"x": {"dtype": "float16"},
                           "weight": {"dtype": "same_as(x)"}},
                "params": {},
            }),
            ("manifest param appears as forward() arg", FwdParamOp, {
                "inputs": {"x": {"dtype": "float16"},
                           "weight": {"dtype": "float32"}},
                "params": {"training": {"type": "bool", "default": True}},
            }),
            ("manifest param appears only in __init__()", InitParamOp, {
                "inputs": {"x": {"dtype": "float16"}},
                "params": {"eps": {"type": "float", "default": 1e-6}},
            }),
            ("static_dims key appears in __init__() (R20)", StaticDimInitOp, {
                "inputs": {"x": {"dtype": "float16"}},
                "params": {"dim": {"type": "int", "default": -1}},
                "static_dims": {"N": "x.shape[dim]"},
            }),
        ]
        for desc, cls, signature in cases:
            errors = _run_check_l1(validator, monkeypatch, cls, signature)
            assert errors == [], f"{desc}: unexpected errors: {errors}"

    def test_signature_parity_rejected_forms(self, validator, monkeypatch):
        """Case table: mismatches and malformed fields the L1 signature
        check reports (never crashes on), driven through the public
        ``check_l1`` entry point with synthetic Op classes."""
        class ForwardOnlyOp:
            def __init__(self): pass
            def forward(self, x): return None

        class ForwardWithParamOp:
            def __init__(self): pass
            def forward(self, x, training=True): return None

        class InitWithoutDimOp:
            def __init__(self, M, N, dtype, eps=1e-6): pass
            def forward(self, x): return None

        class InitWithoutStaticDimOp:
            def __init__(self, dtype, dim=-1): pass
            def forward(self, x): return None

        cases = [
            # (description, Op class, manifest signature,
            #  substrings expected in one error)
            ("forward() missing a manifest input", ForwardOnlyOp, {
                "inputs": {"x": {"dtype": "float16"},
                           "weight": {"dtype": "same_as(x)"}},
                "params": {},
            }, ["do not match"]),
            ("params as list reported, not crash", ForwardWithParamOp, {
                "inputs": {"x": {"dtype": "float16"}},
                "params": ["training"],
            }, ["signature", "params"]),
            ("param missing from both __init__ and forward", InitWithoutDimOp, {
                "inputs": {"x": {"dtype": "float16"}},
                "params": {"dim": {"type": "int", "default": -1}},
            }, ["dim"]),
            ("param-less __init__ leaves only forward()", ForwardOnlyOp, {
                "inputs": {"x": {"dtype": "float16"}},
                "params": {"eps": {"type": "float", "default": 1e-6}},
            }, ["eps"]),
            ("static_dims key missing from __init__ (R20)",
             InitWithoutStaticDimOp, {
                "inputs": {"x": {"dtype": "float16"}},
                "params": {"dim": {"type": "int", "default": -1}},
                "static_dims": {"N": "x.shape[dim]"},
            }, ["static_dims", "'N'"]),
            ("non-dict static_dims reported", InitWithoutStaticDimOp, {
                "inputs": {"x": {"dtype": "float16"}},
                "params": {},
                "static_dims": ["N"],
            }, ["static_dims"]),
        ]
        for desc, cls, signature, substrings in cases:
            errors = _run_check_l1(validator, monkeypatch, cls, signature)
            lowered = [e.lower() for e in errors]
            assert any(
                all(s.lower() in e for s in substrings) for e in lowered
            ), f"{desc}: expected error with {substrings}, got: {errors}"


# dtype: dtype string conformance


class TestDtype:
    """dtype checks that dtype strings are valid torch dtype names."""

    def test_invalid_dtype_tokens_are_hard_l3_errors(self, validator):
        """Unrecognized dtype names in workloads or dtype_combos fail hard.

        The dtype_combos case must not depend on a ``_validate_dtypes``
        override — the data-level check alone reports it.
        """
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
            },
            "workloads": [{"dtypes": ["not_a_dtype"]}],
        }
        errors = validator.check_l3("test_op", entry)
        assert any("not_a_dtype" in e and "dtype" in e for e in errors), errors

        entry = {
            "status": "implemented",
            "signature": {
                "inputs": {"x": {"dtype": "float16 | bfloat16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "dtype_combos": [{"x": "not_a_real_dtype"}],
            },
            "workloads": [{"dtypes": ["float16"]}],
        }
        errors = validator.check_l3("test_op", entry)
        assert any(
            "not_a_real_dtype" in e and "dtype_combos" in e for e in errors
        ), f"Expected hard L3 error for invalid combo value, got: {errors}"

    def test_dtype_combos_same_as_identity(self, validator):
        """R3 identity: same_as-bound tensors must match in every combo;
        a combo omitting the reference cannot be verified and fails."""
        def entry_with_combos(combos):
            return {
                "signature": {
                    "inputs": {
                        "x": {"dtype": "float16 | bfloat16"},
                        "w": {"dtype": "same_as(x)"},
                    },
                    "outputs": {"y": {"dtype": "same_as(x)"}},
                    "dtype_combos": combos,
                },
                "workloads": [{"dtypes": ["float16"]}],
            }

        # Matching dtypes for same_as-bound tensors pass.
        errors = validator.check_l3("test_op", entry_with_combos(
            [{"x": "float16", "w": "float16"},
             {"x": "bfloat16", "w": "bfloat16"}],
        ))
        assert errors == []

        # Mismatched dtypes violate the identity constraint.
        errors = validator.check_l3("test_op", entry_with_combos(
            [{"x": "float16", "w": "bfloat16"}],
        ))
        assert any("same_as" in e and "identity" in e for e in errors), errors

        # A combo naming the bound tensor without its reference fails.
        errors = validator.check_l3("test_op", entry_with_combos(
            [{"w": "float16"}],
        ))
        assert any("without its reference" in e for e in errors), errors

    def test_resolver_semantics(self, validator):
        """``_resolve_tensor_dtype_options`` resolves forward same_as refs
        (R3 is an identity constraint, not an ordering rule) and expands
        ``promote_int_to_float`` per R3a."""
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

        sig = {
            "inputs": {
                "input": {
                    "dtype": (
                        "float16 | bfloat16 | float32 | "
                        "int8 | int16 | int32 | int64 | uint8"
                    ),
                },
            },
            "outputs": {"output": {"dtype": "promote_int_to_float(input)"}},
        }
        resolved = validator._resolve_tensor_dtype_options(sig)
        assert resolved is not None
        # All integral options collapse to a single float32 entry; float
        # options stay as themselves (order-preserving de-dup).
        assert resolved["output"] == ["float16", "bfloat16", "float32"]

    def test_promote_int_to_float_rejected_outside_outputs(self, validator):
        """``promote_int_to_float`` is output-side only (R3a); unknown refs
        and malformed args are rejected wherever they appear."""
        cases = [
            # (description, sig, workloads, substrings expected in one error)
            ("unknown tensor reference", {
                "inputs": {"x": {"dtype": "float16 | int32"}},
                "outputs": {"y": {"dtype": "promote_int_to_float(z)"}},
            }, [{"dtypes": ["float16"]}],
             ["promote_int_to_float(z)",
              "must reference a signature input tensor"]),
            ("malformed empty arg", {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "promote_int_to_float()"}},
            }, [{"dtypes": ["float16"]}],
             ["unrecognized dtype", "promote_int_to_float()"]),
            ("input-side use", {
                "inputs": {
                    "x": {"dtype": "int8 | int32 | float32"},
                    "y": {"dtype": "promote_int_to_float(x)"},
                },
                "outputs": {"out": {"dtype": "float32"}},
            }, [{"dtypes": ["float32"]}],
             ["promote_int_to_float", " y ", "output-side only"]),
            ("workload-dtype use", {
                "inputs": {"x": {"dtype": "int8 | int32 | float32"}},
                "outputs": {"y": {"dtype": "promote_int_to_float(x)"}},
            }, [{"dtypes": ["promote_int_to_float(x)"]}],
             ["promote_int_to_float", "workloads[0].dtypes[0]",
              "output-side only"]),
        ]
        for desc, sig, workloads, substrings in cases:
            entry = {"signature": sig, "workloads": workloads}
            errors = validator.check_l3("test_op", entry)
            assert any(
                all(s in e for s in substrings) for e in errors
            ), f"{desc}: expected error with {substrings}, got: {errors}"

    def test_promote_int_to_float_signature_accepts(self, validator):
        """``promote_int_to_float(ref)`` is a recognized output dtype token."""
        entry = {
            "signature": {
                "inputs": {
                    "input": {
                        "dtype": (
                            "float16 | bfloat16 | float32 | "
                            "int8 | int16 | int32 | int64 | uint8"
                        ),
                    },
                },
                "outputs": {
                    "output": {"dtype": "promote_int_to_float(input)"},
                },
            },
            "workloads": [{"dtypes": ["float16"]}],
        }
        errors = validator.check_l3("PromoteOp", entry)
        assert errors == [], (
            f"promote_int_to_float on declared input must validate, got: {errors}"
        )


    def test_check_l3_with_non_dict_signature_does_not_crash(self, validator):
        """check_l3 must tolerate malformed signature.inputs/outputs.

        A list or string in place of the expected dict triggered an
        unguarded ``.update()`` / ``.keys()`` crash; treat as empty so
        the schema layer's own diagnostics surface unmasked.
        """
        for inputs_val in ([{"x": {}}], "not a mapping", None):
            for outputs_val in ([{"y": {}}], "nope", None):
                entry = {
                    "signature": {
                        "inputs": inputs_val,
                        "outputs": outputs_val,
                    },
                }
                errors = validator.check_l3("BadOp", entry)
                assert isinstance(errors, list)

        # Non-dict entry value inside an otherwise-well-formed inputs/outputs
        # mapping (e.g. ``inputs: {x: "float16"}``) must also be tolerated.
        entry = {
            "signature": {
                "inputs": {"x": "float16"},
                "outputs": {"y": ["float16"]},
            },
        }
        errors = validator.check_l3("BadOp", entry)
        assert isinstance(errors, list)


# L2 extension: _infer_output_shapes parity with shape_rules


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


    def test_no_override_emits_missing_method_warning(self, validator):
        """Missing override must not pass silently.

        Implemented ops whose class does not define
        ``_infer_output_shapes`` must surface a warning naming the gap;
        silently skipping such ops would leave the parity check with no
        visible coverage signal.
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


    def test_symbolic_dim_rule_detects_wrong_output(self, validator):
        """Rules like ``o.shape == (B, S, H, D)`` must evaluate, not warn-skip.

        Symbolic dimension names declared only via literal
        ``tensor.shape == (...)`` rules must be bound into the evaluation
        context; an unbound name raises NameError, downgrades the rule to
        a warning, and lets a wrong ``_infer_output_shapes`` pass parity.
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


    def test_incorrect_infer_fails(self, validator):
        """Parity error when _infer_output_shapes disagrees with shape_rules."""
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
        """R11 / R11a rules that call ``len`` or use comprehensions must be
        evaluable.

        Two eval-context invariants: the restricted builtins expose
        ``len`` / ``isinstance`` / ``set``, and ctx names are visible
        inside comprehension scopes (``eval`` resolves those against
        globals). Breaking either turns the rule into a NameError
        warning-skip and hides parity mismatches.
        """
        # Wrong: reduction op declares dim, keepdim=False so y should drop
        # rank(s), but _infer_output_shapes returns x_shape verbatim.
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

        # Comprehension scoping: generator / set comprehensions in the rule
        # body must not be skipped via a NameError warning.
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
                    # Actual parity rule expected to catch the mismatch.
                    "y.ndim == x.ndim - len(dim)",
                ],
            },
        }
        warnings: list[str] = []
        errors = validator.check_l2_infer_parity(
            "FakeOp", entry, cls, warnings=warnings,
        )
        assert not any(
            "could not be evaluated" in w for w in warnings
        ), f"Comprehension rule skipped via warning: {warnings}"
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

        If dims from different rules collide (e.g. ``A`` in
        ``x.shape == (A, B)`` and ``C`` in ``y.shape == (C, D)`` mapping
        to the same mock size), a downstream rule comparing
        ``x.shape[0]`` to ``y.shape[0]`` evaluates to a spurious
        True / False.
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

    def test_body_exception_is_hard_error_not_signature_mismatch(self, validator):
        """Exceptions raised inside the _infer_output_shapes body surface as
        hard L2 parity errors, never as signature mismatches or warnings.

        The signature is pre-bound via ``inspect.signature().bind`` so a
        TypeError from the body is distinguished from a signature
        mismatch; RuntimeError (e.g. ``'not ready'`` placeholders) follows
        the same hard-error policy so genuine bugs cannot silently pass.
        """
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "shape_rules": ["y.shape == x.shape"],
            },
        }
        for exc_cls, message in (
            (TypeError, "simulated implementation bug"),
            (RuntimeError, "not ready"),
        ):
            def infer(self, x_shape, _exc=exc_cls, _msg=message):
                # Signature matches; the body itself raises.
                raise _exc(_msg)

            cls = _make_op_cls_with_infer(infer)
            warnings: list[str] = []
            errors = validator.check_l2_infer_parity(
                "FakeOp", entry, cls, warnings=warnings,
            )
            assert not any(
                "signature does not match manifest inputs" in e for e in errors
            ), (
                f"Body {exc_cls.__name__} must not be misreported as "
                f"signature mismatch; errors={errors}"
            )
            assert any(
                f"raised {exc_cls.__name__}" in e for e in errors
            ), (
                f"Body {exc_cls.__name__} must surface as a hard L2 parity "
                f"error; errors={errors} warnings={warnings}"
            )


    def test_declared_output_shape_catches_wrong_infer(self, validator):
        """Declared output shapes alone (no shape_rules) must drive parity.

        Inferred outputs are compared against ``signature.outputs[*].shape``
        even when ``shape_rules`` is empty; a manifest that declares
        shapes only per-output (e.g. conv ops' ``"[N, C_out, L_out]"``)
        must still catch a broken ``_infer_output_shapes``.
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
        """Reading a non-manifest-param ``self`` attribute must not falsely skip parity.

        The mock ``self`` is built via ``cls.__new__(cls)`` so
        class-defined attributes stay reachable; a plain namespace mock
        raises AttributeError and silently skips the check.
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
        """Reading ``self.<static_dim>`` must exercise parity, not AttributeError-skip.

        ``_build_mock_self`` installs ``static_dims`` values resolved
        against the mock inputs in addition to ``signature.params``
        defaults, so a generated ``_infer_output_shapes`` that consults
        ``self.N`` runs end-to-end instead of AttributeError-skipping.
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
        """Correct infer with an output-only ``L_out`` symbol passes parity.

        Output-only symbols have no input-derived concrete size; they are
        checked only for rank and per-symbol consistency across outputs,
        never against an arbitrary ``dim_sizes`` entry.
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
        """Wrong output-only symbol value is flagged via the shape_rules defining it.

        A rule like ``L_out == L_in - kW + 1`` mentions no output tensor
        name, but ``L_out`` appears in a declared output shape, so it
        must classify as output-mentioning and be rebound from the
        inferred result before evaluation; treated as an input-only
        precondition it would fail in both contexts and be skipped.
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

    def test_conv_like_rank_and_consistency_still_caught(self, validator):
        """Loosening the output-only value check must not weaken the rank
        check, and an output-only symbol reused across multiple outputs
        must stay internally consistent."""
        # Rank disagreement against the declared output shape.
        def bad_rank_infer(self, x_shape, w_shape):
            # Wrong rank: drops the spatial dim entirely.
            return {"y": (x_shape[0], w_shape[0])}

        cls = _make_op_cls_with_infer(bad_rank_infer, name="ConvLikeBadOp")
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
        errors = validator.check_l2_infer_parity("ConvLikeBadOp", entry, cls)
        assert any(
            "rank" in e and "disagrees" in e for e in errors
        ), f"Expected rank error; got: {errors}"

        # Two outputs claiming ``L_out`` with different concrete sizes is
        # an internal inconsistency even though L_out is output-only.
        def inconsistent_infer(self, x_shape):
            return {
                "y1": (x_shape[0], x_shape[1] - 1),
                "y2": (x_shape[0], x_shape[1] - 2),
            }

        cls = _make_op_cls_with_infer(
            inconsistent_infer, name="InconsistentOutOnlyOp",
        )
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

        Returning ``[]`` instead would silently disable downstream dtype
        parity: ``_resolve_tensor_dtype_options`` bails only on None, and
        an empty list yields an empty Cartesian product.
        """
        out = validator._dtype_options_for_tensor(
            "y", "same_as(x)", resolved={},
        )
        assert out is None, (
            f"Pure same_as(unresolved) must return None, got {out!r}"
        )


# shape_rules broadcasting helpers (broadcast_shapes / is_broadcastable_to)


class TestShapeRuleBroadcastBuiltins:
    """Broadcasting helpers in shape_rules eval: torch.broadcast_shapes semantics, torch-free."""

    def test_broadcast_shapes_value_matrix(self, validator):
        """``broadcast_shapes`` produces the expected output across cases.

        Covers identical / scalar / size-1-expand / rank-promotion /
        variadic (0, 1, 3+ args) / list-input forms in one matrix. Hand-
        coded expectations (no torch dependency) so the test stays
        consistent with the validator's pure-Python implementation.
        """
        fn = validator._SHAPE_RULE_BUILTINS["broadcast_shapes"]
        cases: list[tuple[tuple, tuple]] = [
            (((2, 3), (2, 3)), (2, 3)),                  # identical
            (((), (4, 5)), (4, 5)),                      # scalar left
            (((4, 5), ()), (4, 5)),                      # scalar right
            (((1, 3), (2, 1)), (2, 3)),                  # size-1 expands
            (((3,), (2, 4, 3)), (2, 4, 3)),              # rank promotion
            (((1, 3), (2, 1), (1, 1)), (2, 3)),          # 3+ args
            ((), ()),                                    # no args
            (((2, 3),), (2, 3)),                         # single arg
            (([1, 3], [2, 1]), (2, 3)),                  # list inputs
        ]
        for args, expected in cases:
            assert fn(*args) == expected, (args, fn(*args), expected)

    def test_broadcast_shapes_incompatible_raises(self, validator):
        """Incompatible shapes raise ``ValueError`` (the only error path)."""
        fn = validator._SHAPE_RULE_BUILTINS["broadcast_shapes"]
        with pytest.raises(ValueError, match="not broadcast-compatible"):
            fn((2, 3), (3, 3))

    def test_is_broadcastable_to_value_matrix(self, validator):
        """``is_broadcastable_to(src, dst)`` returns True/False per cases.

        Pins the asymmetric semantics (src may grow into dst; dst is
        fixed) including the equal-shape, size-1-expand, dst-smaller,
        dst-shrink, dim-mismatch, and extra-leading-dim branches.
        """
        fn = validator._SHAPE_RULE_BUILTINS["is_broadcastable_to"]
        cases: list[tuple[tuple, tuple, bool]] = [
            ((2, 3), (2, 3), True),     # equal
            ((1, 3), (2, 3), True),     # size-1 expand
            ((3,), (2, 3), True),       # rank promotion
            ((), (2, 3), True),         # scalar source
            ((2, 3), (3,), False),      # dst smaller (asymmetry)
            ((2, 1), (2, 3), True),     # one-dim expand
            ((2, 3), (2, 1), False),    # would require shrinking dst
            ((2, 4), (2, 3), False),    # dim mismatch
            ((5, 2, 3), (2, 3), False), # extra leading dim
        ]
        for src, dst, expected in cases:
            assert fn(src, dst) is expected, (src, dst, fn(src, dst), expected)

    def test_broadcast_helpers_callable_from_shape_rule_eval(self, validator):
        """Both helpers resolve from inside ``_eval_shape_rule`` rule bodies.

        Pins the validator-integration contract: the helpers in
        ``_SHAPE_RULE_BUILTINS`` are reachable as bare names from rule
        text, returning the same value as direct calls. Drives one true
        and one false case for ``is_broadcastable_to`` (its bool return
        flows through the ok/reason pair) plus one ``broadcast_shapes``
        equality rule (its tuple return must support ``==`` comparison
        in the rule body).
        """
        cases: list[tuple[str, bool]] = [
            ("broadcast_shapes((1, 3), (2, 1)) == (2, 3)", True),
            ("is_broadcastable_to((1, 3), (2, 3))", True),
            ("is_broadcastable_to((2, 3), (2, 1))", False),
        ]
        for rule, expected_ok in cases:
            ok, reason = validator._eval_shape_rule(rule, {})
            assert reason is None, (rule, reason)
            assert ok is expected_ok, (rule, ok, expected_ok)


# L3 extension: _validate_dtypes parity with dtype_combos / unions


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


    def test_no_override_emits_missing_method_warning(self, validator):
        """Missing ``_validate_dtypes`` override must not pass silently on L3."""
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
        """Rejects a dtype in the declared union -> parity error."""
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
        """Accepts a non-listed combo -> parity error."""
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
        """Non-listed combos must all be checked, not just the first.

        Stopping at the first rejection would let a later accepted
        non-listed combo escape detection; every non-listed combination
        is probed and each acceptance reported.
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
        """_validate_dtypes with a wrong kwarg name must fail on both the
        union and the dtype_combos branch.

        A signature-mismatch TypeError is a hard parity error; downgraded
        to a warning it would let an uncallable ``_validate_dtypes``
        satisfy parity.
        """
        def validate(self, wrong_name, other_wrong=None):
            return None

        cls = _make_op_cls_with_validate(validate)
        union_entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16 | bfloat16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
            },
        }
        combos_entry = {
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
        for branch, entry in (("union", union_entry), ("combos", combos_entry)):
            warnings: list[str] = []
            errors = validator.check_l3_validate_dtypes_parity(
                "FakeDtypeOp", entry, cls, warnings=warnings,
            )
            assert any(
                "signature does not match manifest inputs" in e for e in errors
            ), (
                f"Signature mismatch in _validate_dtypes must surface as a "
                f"parity error on the {branch} branch, got errors={errors} "
                f"warnings={warnings}"
            )

    def test_body_unexpected_exception_is_hard_error(self, validator):
        """_validate_dtypes raising RuntimeError for every valid combo must
        produce a hard L3 parity error (not a warning) on both the
        dtype_combos and the no-combos Cartesian branch."""
        def bad_validate(self, x):
            raise RuntimeError("simulated bug")

        cls = _make_op_cls_with_validate(bad_validate, name="BadValidateOp")
        combos_entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16 | bfloat16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "dtype_combos": [
                    {"x": "float16"},
                    {"x": "bfloat16"},
                ],
            },
        }
        no_combos_entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16 | bfloat16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
            },
        }
        for branch, entry in (
            ("combos", combos_entry), ("no-combos", no_combos_entry),
        ):
            warnings: list[str] = []
            errors = validator.check_l3_validate_dtypes_parity(
                "BadValidateOp", entry, cls, warnings=warnings,
            )
            assert any(
                "raised unexpected exception" in e and "RuntimeError" in e
                for e in errors
            ), (
                f"expected hard L3 error on the {branch} branch, got "
                f"errors={errors} warnings={warnings}"
            )

    def test_wide_union_probe_pool_derived_from_torch_dtypes(self, validator):
        """The out-of-union probe pool is ``sorted(_TORCH_DTYPES - declared)``.

        The pool must stay non-empty for every union that does not cover
        the entire torch dtype universe: an op declaring a wide 8-dtype
        union still leaves ``uint8`` as a probe candidate, so an
        over-permissive ``_validate_dtypes`` that accepts it surfaces a
        hard L3 error.
        """
        def accept_all(self, x):
            return True  # over-permissive: accepts any dtype

        cls = _make_op_cls_with_validate(accept_all, name="WideEightDtypeOp")
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": (
                    "float16 | bfloat16 | float32 | float64 | "
                    "int8 | int16 | int32 | int64"
                )}},
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
        """Body TypeError is a legitimate rejection, not a signature mismatch.

        The signature is pre-bound before invocation; a bare
        ``except (ValueError, TypeError)`` around the call could not tell
        a kwarg-name mismatch from a TypeError raised inside the body.
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
        """Exhaustive dtype_combos still emit the 'exhausts the union' warning.

        When every Cartesian tuple is listed, no non-listed combo is ever
        checked, so the warning must not be conditioned on a non-listed
        probe having run.
        """
        import torch

        allowed = {torch.float16, torch.bfloat16}

        def validate(self, x, w):
            # Reject dtypes outside the declared union so the
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
        """Accepting an out-of-union dtype is a parity error on the no-combos branch.

        Iterating only the union's Cartesian product cannot detect an
        overly-permissive ``_validate_dtypes``; the branch needs its own
        out-of-union probe.
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
        """Op that rejects out-of-union dtypes produces no parity error."""
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
        """Accepting a same_as identity violation is a parity error.

        The union-iteration loop skips every same_as-violating candidate
        via ``_honours_same_as``, so a permissive op that fails to enforce
        same_as would go unflagged without a dedicated probe.
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
        """Op that enforces same_as passes the same_as probe."""
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

        Without it, a permissive ``_validate_dtypes`` that accepts every
        dtype passes parity as long as every listed combo is accepted.
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
        """A non-existent dtype in dtype_combos is a hard L3 error, not a skip warning.

        The upfront validation pass rejects entries that are neither in
        ``_TORCH_DTYPES`` nor a resolvable ``same_as`` ref; letting them
        reach the ``cannot build mock tensor`` branch would silently
        disable the check and hide a manifest data bug.
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
        assert not any(
            "cannot build mock tensor" in w for w in warnings
        ), (
            "Invalid dtype name must not be downgraded to "
            f"'cannot build mock tensor' warning; warnings={warnings}"
        )


    def test_valid_dtype_combo_reaches_build_mock_tensor(
        self, validator, monkeypatch,
    ):
        """The 'cannot build mock tensor' warning stays reserved for valid dtype names.

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
        """A combo omitting an input is a manifest error, never a rejection or silent skip."""
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
        """Comparing ``x.dtype != self.dtype`` must work with a populated mock self.

        ``_build_mock_self`` populates the dtype axis from the candidate
        combo; if ``self.dtype`` fell through to the base-class
        ``Op.dtype = None``, the comparison would raise and every listed
        combo would be marked rejected.
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


class TestDtypeCombosData:
    """Data-level hardening for ``check_l3_dtype_combos_data``.

    These checks run independently of any ``_validate_dtypes`` override, so
    manifest data bugs surface even when the parity loop never executes.
    """

    def test_malformed_combo_rows_are_hard_errors(self, validator):
        """Case table: incomplete rows and multi-dtype combo values fail.

        Per manifest.md R4, every combo row covers every declared input and
        each value is a single concrete dtype token (or a ``same_as(ref)``
        resolving to one); unions and ``promote_int_to_float(ref)`` expand
        to multiple dtypes and cannot pin a combo row.
        """
        cases = [
            # (description, sig, substrings expected in one error)
            ("combo row missing a declared input", {
                "inputs": {
                    "x": {"dtype": "float16 | bfloat16"},
                    "w": {"dtype": "float16 | bfloat16"},
                },
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "dtype_combos": [
                    {"x": "float16", "w": "float16"},
                    {"x": "bfloat16"},  # missing 'w'
                ],
            }, ["dtype_combos[1]", "missing declared input 'w'"]),
            ("union expression as combo value", {
                "inputs": {"x": {"dtype": "float16 | bfloat16"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
                "dtype_combos": [{"x": "float16 | bfloat16"}],
            }, ["combo values must be a single concrete dtype"]),
            ("promote_int_to_float as combo value", {
                "inputs": {"x": {"dtype": "float16 | int8"}},
                "outputs": {"y": {"dtype": "promote_int_to_float(x)"}},
                "dtype_combos": [
                    {"x": "float16", "y": "promote_int_to_float(x)"},
                ],
            }, ["promote_int_to_float(...) is allowed only on signature.outputs"]),
        ]
        for desc, sig, substrings in cases:
            errors = validator.check_l3_dtype_combos_data("FakeOp", sig)
            assert any(
                all(s in e for s in substrings) for e in errors
            ), f"{desc}: expected error with {substrings}, got: {errors}"

    def test_unresolvable_same_as_graph_is_hard_error(self, validator):
        """Pure ``same_as`` cycles and dangling refs must surface hard L3
        errors instead of silently skipping combo validation.

        A pure cycle like ``x: same_as(y)`` / ``y: same_as(x)`` satisfies
        per-token validation and the R3 identity check, so it needs a
        dedicated diagnosis to keep invalid combo data from passing.
        """
        cycle_sig = {
            "inputs": {
                "x": {"dtype": "same_as(y)"},
                "y": {"dtype": "same_as(x)"},
            },
            "outputs": {"z": {"dtype": "same_as(x)"}},
            "dtype_combos": [{"x": "float16", "y": "float16"}],
        }
        errors = validator.check_l3_dtype_combos_data("CycleOp", cycle_sig)
        assert any(
            "same_as cycle" in e and "'x'" in e and "'y'" in e
            for e in errors
        ), f"expected cycle diagnosis naming x and y, got {errors}"

        dangling_sig = {
            "inputs": {"x": {"dtype": "same_as(nope)"}},
            "outputs": {"z": {"dtype": "same_as(x)"}},
            "dtype_combos": [{"x": "float16"}],
        }
        errors = validator.check_l3_dtype_combos_data("DanglingOp", dangling_sig)
        assert any(
            "dangling reference" in e and "same_as(nope)" in e
            for e in errors
        ), f"expected dangling diagnosis, got {errors}"


class TestStaticDimShapeParity:
    """static_dims values must pin expected output sizes in L2 parity."""

    def test_static_dim_output_shape_catches_bad_infer(self, validator):
        """Arbitrary integers at a static-dim-bound output position must fail parity.

        ``static_dims`` keys in declared output shapes are checked by
        exact value (resolved against mock inputs), not only by
        rank/consistency; a bad impl returning ``(999, 999)`` for a
        declared ``[N, N]`` with ``static_dims: {N: "x.shape[-1]"}``
        must be caught.
        """
        def bad_infer(self, x_shape):
            # static_dims pins N = x.shape[-1] (=4 under mock); a correct
            # impl returns (4, 4).
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


class TestParamDefaultOutputShapePin:
    """Concrete param defaults pin declared output-shape dims in L2 parity."""

    def test_param_default_pins_output_dim(self, validator):
        """Bad infer returning ``(999,)`` for declared ``[k]`` with
        ``params.k.default = 4`` must produce a hard L2 error; a correct
        infer returning ``(4,)`` passes parity."""
        entry = {
            "signature": {
                "inputs": {"x": {"dtype": "float16", "shape": "[M]"}},
                "outputs": {
                    "y": {"dtype": "same_as(x)", "shape": "[k]"},
                },
                "params": {"k": {"type": "int", "default": 4}},
            },
        }

        def bad_infer(self, x_shape):
            return {"y": (999,)}

        cls = _make_op_cls_with_infer(bad_infer, name="ParamDefaultBadOp")
        errors = validator.check_l2_infer_parity(
            "ParamDefaultBadOp", entry, cls,
        )
        assert any(
            "dim[0]=999" in e and "k=4" in e for e in errors
        ), f"expected param-default parity error, got {errors}"

        def good_infer(self, x_shape):
            return {"y": (4,)}

        cls = _make_op_cls_with_infer(good_infer, name="ParamDefaultGoodOp")
        errors = validator.check_l2_infer_parity(
            "ParamDefaultGoodOp", entry, cls,
        )
        assert errors == [], (
            f"expected no parity errors on correct impl, got {errors}"
        )


# Bench checks
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
        """Manifest helpers called with a different op name must fail, on
        both the direct and the indirect (benchmarks.benchmark_base) path."""
        bench_file = tmp_path / "bench_test.py"
        bench_file.write_text(textwrap.dedent("""\
            from tileops.manifest import load_workloads
            workloads = load_workloads('wrong_op')
            op.eval_roofline()
        """))
        errors = validator.check_l4_benchmark("test_op", str(bench_file), REPO_ROOT)
        assert any("load_workloads" in e for e in errors)

        bench_file.write_text(textwrap.dedent("""\
            from benchmarks.benchmark_base import workloads_to_params, ManifestBenchmark
            params = workloads_to_params('wrong_op')
            ManifestBenchmark('wrong_op', op, params[0])
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


# --check-op: force all levels on a specific op, ignoring status


class TestCheckOp:
    """--check-op forces all validation levels on a named op, ignoring spec-only."""

    def test_spec_only_op_with_check_op_runs_all_levels(self, validator, tmp_path):
        """When check_op matches a spec-only op, L1-L4 checks run (not skipped)."""
        # Bench file guaranteed to fail L4 (no load_workloads).
        bench_file = tmp_path / "bench_test.py"
        bench_file.write_text("import pytest\n")

        entry = _make_entry(status="spec-only")
        entry["source"]["bench"] = str(bench_file)
        entry["source"]["bench_manifest_driven"] = True

        manifest_file = tmp_path / "ops_manifest.yaml"
        import yaml
        manifest_file.write_text(yaml.safe_dump({"my_op": entry}))

        # Without check_op: spec-only op skips L1-L4.
        errors_no_flag, warnings_no_flag = validator.validate_manifest(
            manifest_path=manifest_file,
            repo_root=tmp_path,
        )
        bench_errors_no_flag = [e for e in errors_no_flag if "[bench]" in e]
        assert bench_errors_no_flag == [], (
            f"Spec-only op should skip bench check without --check-op: {bench_errors_no_flag}"
        )

        # With check_op="my_op": all levels forced despite spec-only.
        errors_flag, warnings_flag = validator.validate_manifest(
            manifest_path=manifest_file,
            repo_root=tmp_path,
            check_op="my_op",
        )
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
        non_schema = [e for e in errors if "[schema]" not in e]
        assert non_schema == [], (
            f"Spec-only op should only have schema errors (if any), got: {non_schema}"
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
        """A non-mapping manifest root yields a schema error, not an AttributeError."""
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

        # other_op must be completely skipped — no import errors from its
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
        target_errors = [e for e in errors if "target_op" in e]
        assert len(target_errors) > 0, (
            "target_op should have validation errors from forced L4 check"
        )

    def test_check_op_ignores_unrelated_variant_of_errors(self, validator, tmp_path):
        """--check-op must not report variant_of errors from unrelated ops.

        check_variant_of_consistency must scope to the variant family;
        run across the full manifest it fails --check-op on unrelated ops
        with invalid variant_of references.
        """
        import yaml

        target_entry = _make_entry()
        other_entry = _make_entry()
        other_entry["variant_of"] = "nonexistent_primary"

        manifest_file = tmp_path / "ops_manifest.yaml"
        manifest_file.write_text(yaml.safe_dump(
            {"target_op": target_entry, "other_op": other_entry},
        ))

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

        Excluding variants from the scope would let a variant edit that
        breaks R16 pass --check-op <primary>.
        """
        import yaml

        primary = _make_entry(source_kernel="shared_kernel.py")
        # Valid variant: shares source with primary.
        valid_variant = _make_entry(source_kernel="shared_kernel.py")
        valid_variant["variant_of"] = "primary_op"

        # Broken variant: different source.kernel violates R16.
        broken_variant = _make_entry(source_kernel="different_kernel.py")
        broken_variant["variant_of"] = "primary_op"

        manifest_file = tmp_path / "ops_manifest.yaml"
        manifest_file.write_text(yaml.safe_dump({
            "primary_op": primary,
            "good_variant": valid_variant,
            "bad_variant": broken_variant,
        }))

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

        good_r16 = [e for e in errors if "good_variant" in e and "R16" in e]
        assert good_r16 == [], (
            f"good_variant should pass R16, got: {good_r16}"
        )


    def test_check_op_variant_family_runs_schema_on_variants(self, validator, tmp_path):
        """--check-op on primary runs per-op schema checks on variants too."""
        import yaml

        primary = _make_entry(source_kernel="shared.py")
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


# _resolve_op_class: multi-class file resolution

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
        """When multiple candidates exist but none matches the manifest key, return cls=None."""
        import importlib
        import types

        # Create a fake module with two candidate classes, neither named
        # after the given op_name.
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

        import unittest.mock as mock

        with (
            mock.patch.object(importlib, "import_module", side_effect=patched_import),
            pytest.warns(UserWarning, match="No class named"),
        ):
            result = validator._resolve_op_class(
                "tileops/ops/fake_ambiguous.py", "mystery_fwd",
            )
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
        assert result.cls is SumFwdOp, (
            f"Expected SumFwdOp (direct match) but got {result.cls.__name__}"
        )


# Integration: validate_manifest.py passes on the real codebase

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


# tileops.manifest.shape_rules helper module + validator integration


class TestShapeRuleHelpers:
    """Unit tests for :mod:`tileops.manifest.shape_rules` predicates."""


    def test_helper_rule_validator_warns_on_malformed_dim(self, validator):
        """Validator integration: helper rules surface malformed dims as warnings.

        A malformed dim default (``dim=["2"]``) raises TypeError from
        the helper, which the validator classifies as an eval-error
        warning ("could not be evaluated"). The contract: the parity
        check is skipped with a warning, not turned into a hard shape
        error — bit-identical to the equivalent inline form.
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
                    "dim_range_validity(x, dim)",
                    "dim_uniqueness(x, dim)",
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
        assert errs_inline == [] == errs_helper, (errs_inline, errs_helper)
        assert any("could not be evaluated" in w for w in warn_inline), (
            warn_inline
        )
        assert any("could not be evaluated" in w for w in warn_helper), (
            warn_helper
        )


class TestValidatorHelperResolution:
    """Validator integration of the shape_rules helper builtins."""

    def test_l2_parity_helper_detects_out_of_range_default(self, validator):
        """Out-of-range default ``dim`` surfaces as an input-only precondition.

        Mirrors what the inline expression would do: the predicate
        evaluates to False under mock inputs, but the validator
        classifies that as an *input* problem (the manifest's mock
        default ``dim`` is out of range for the mock ``x.ndim``), not
        as a parity failure of ``_infer_output_shapes``. The contract
        pinned here: ``errors == []`` (no parity blame on infer) plus
        exactly one ``"input-only precondition"`` warning citing the
        helper rule itself.
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
                    "dim_range_validity(x, dim)",
                    "y.shape == x.shape",
                ],
            },
        }
        warnings: list[str] = []
        errors = validator.check_l2_infer_parity(
            "HelperBadDimOp", entry, cls, warnings=warnings,
        )
        # No "could not be evaluated" warning: the helper resolved and ran;
        # the failure was a real predicate result, not an eval skip.
        assert errors == [], errors
        assert not any(
            "could not be evaluated" in w for w in warnings
        ), warnings
        precondition_hits = [
            w for w in warnings
            if "input-only precondition" in w
            and "dim_range_validity(x, dim)" in w
        ]
        assert len(precondition_hits) == 1, warnings

    def test_input_bound_symbols_tolerates_non_dict_inputs(self, validator):
        """``_input_bound_symbols`` must treat malformed inputs as empty.

        Schema-independent shape-rule extraction must not crash when
        ``signature.inputs`` is missing or non-mapping; the schema layer
        owns the structural error message.
        """
        result = validator._input_bound_symbols({
            "inputs": [{"x": {"shape": "[N]"}}],
            "shape_rules": ["x.shape == (N)"],
        })
        assert isinstance(result, set)
        result = validator._input_bound_symbols({
            "shape_rules": ["x.shape == (N)"],
        })
        assert isinstance(result, set)

    def test_shape_rules_helpers_callable_by_bare_name(self, validator):
        """Reduction-dim helpers from shape_rules.py are in the eval scope.

        Pin the public surface of ``tileops.manifest.shape_rules`` as
        seen by manifest YAML: each function listed in :data:`__all__`
        must be callable from a shape_rule body by bare name. Adding a
        new helper requires updating both ``__all__`` and the validator's
        ``_SHAPE_RULE_BUILTIN_PAIRS`` — this test fails loudly when one
        side moves without the other.
        """
        import types

        from tileops.manifest import shape_rules
        ctx = {"x": types.SimpleNamespace(ndim=4), "dim": 0}
        for name in shape_rules.__all__:
            ok, reason = validator._eval_shape_rule(f"{name}(x, dim)", ctx)
            assert reason is None, (name, reason)
            # Predicate helpers return bool; reduced_axes returns frozenset
            # — both are truthy on the canonical (ndim=4, dim=0) input.
            assert ok is True, name


# C1-C7 strict parity gates


class TestCtorSignatureParity:
    """C3: ctor signature parity (defaults + kw-only)."""

    def test_matching_defaults_pass(self, validator):
        from tileops.ops.op_base import Op

        class Op1(Op):
            def __init__(self, dim=-1, eps=1e-6, kernel_map=None): pass
            def forward(self, x): return None
            @property
            def default_kernel_map(self): return {}

        entry = {"signature": {
            "params": {
                "dim": {"type": "int", "default": -1},
                "eps": {"type": "float", "default": 1e-6},
            },
        }}
        assert validator.check_c3_ctor_signature_parity("Op1", entry, Op1) == []

        # compat_default: required manifest param with a ctor-only default.
        class OpCompat(Op):
            def __init__(self, num_experts=None, kernel_map=None): pass
            def forward(self, x): return None
            @property
            def default_kernel_map(self): return {}

        entry = {"signature": {
            "params": {
                "num_experts": {"type": "int", "compat_default": None},
            },
        }}
        assert validator.check_c3_ctor_signature_parity(
            "OpCompat", entry, OpCompat
        ) == []

    def test_ctor_mismatches_fail(self, validator):
        """Case table: missing default, compat_default mismatch, kw-only."""
        from tileops.ops.op_base import Op

        class OpNoDefault(Op):
            def __init__(self, dim, kernel_map=None): pass  # no default
            def forward(self, x): return None
            @property
            def default_kernel_map(self): return {}

        class OpCompatMismatch(Op):
            def __init__(self, num_experts=0, kernel_map=None): pass
            def forward(self, x): return None
            @property
            def default_kernel_map(self): return {}

        class OpKwOnly(Op):
            def __init__(self, *, dim=-1, kernel_map=None): pass
            def forward(self, x): return None
            @property
            def default_kernel_map(self): return {}

        cases = [
            ("param default missing on __init__", OpNoDefault,
             {"dim": {"type": "int", "default": -1}},
             "no default on __init__"),
            ("compat_default value mismatch", OpCompatMismatch,
             {"num_experts": {"type": "int", "compat_default": None}},
             "no manifest default"),
            ("kw_only mismatch", OpKwOnly,
             {"dim": {"type": "int", "default": -1, "kw_only": False}},
             "kw_only mismatch"),
        ]
        for desc, cls, params, substring in cases:
            entry = {"signature": {"params": params}}
            errs = validator.check_c3_ctor_signature_parity(
                cls.__name__, entry, cls,
            )
            assert any(substring in e for e in errs), (desc, errs)


class TestForwardSignatureParity:
    """C4: forward positional names match manifest inputs order."""

    def test_matching_passes(self, validator):
        from tileops.ops.op_base import Op

        class Op1(Op):
            def __init__(self): pass
            def forward(self, x, weight): return None
            @property
            def default_kernel_map(self): return {}

        entry = {"signature": {
            "inputs": {"x": {"dtype": "float16"}, "weight": {"dtype": "float16"}},
        }}
        assert validator.check_c4_forward_signature_parity(
            "Op1", entry, Op1,
        ) == []

    def test_wrong_order_fails(self, validator):
        from tileops.ops.op_base import Op

        class Op2(Op):
            def __init__(self): pass
            def forward(self, weight, x): return None  # swapped
            @property
            def default_kernel_map(self): return {}

        entry = {"signature": {
            "inputs": {"x": {"dtype": "float16"}, "weight": {"dtype": "float16"}},
        }}
        errs = validator.check_c4_forward_signature_parity("Op2", entry, Op2)
        assert any("do not start with" in e for e in errs), errs


class TestDispatchKernelInvariant:
    """C5: ``__init__`` complies with Slot S12 (kernel_map kwarg) + S13
    (body calls ``self.dispatch_kernel``). Pure static check on the
    Op subclass's source — no runtime construction."""

    def test_compliant_ctor_forms_pass(self, validator):
        """Case table: explicit kwarg, **kwargs absorption, and a
        dispatch_kernel call nested in a branch all satisfy S12+S13."""
        from tileops.ops.op_base import Op

        class GoodOp(Op):
            def __init__(self, kernel_map=None):
                self.dispatch_kernel(kernel_map)
            def forward(self, x): return None
            @property
            def default_kernel_map(self): return {}

        class VarKwOp(Op):
            # ``**kwargs`` absorbs ``kernel_map`` and satisfies S12.
            def __init__(self, **kwargs):
                self.dispatch_kernel(kwargs.get("kernel_map"))
            def forward(self, x): return None
            @property
            def default_kernel_map(self): return {}

        class BranchOp(Op):
            # The S13 walker is AST-recursive; a call inside a branch counts.
            def __init__(self, kernel_map=None, fast=False):
                if fast:
                    self.dispatch_kernel(kernel_map)
                else:
                    self.dispatch_kernel(kernel_map)
            def forward(self, x): return None
            @property
            def default_kernel_map(self): return {}

        for cls in (GoodOp, VarKwOp, BranchOp):
            assert validator.check_c5_dispatch_kernel_invariant(
                cls.__name__, {}, cls,
            ) == [], cls.__name__

    def test_non_compliant_ctor_forms_fail(self, validator):
        """Case table: dropped override, missing kwarg, and helper-only
        dispatch each violate the invariant."""
        from tileops.ops.op_base import Op

        class SilentDropOp(Op):
            # S13 violation: kwarg accepted but the body never calls
            # ``self.dispatch_kernel`` — the override is silently dropped.
            def __init__(self, kernel_map=None):
                pass
            def forward(self, x): return None
            @property
            def default_kernel_map(self): return {}

        class NoKwargOp(Op):
            # S12 violation: ``__init__`` does not accept ``kernel_map``.
            def __init__(self): pass
            def forward(self, x): return None
            @property
            def default_kernel_map(self): return {}

        class HelperOp(Op):
            # S13 requires dispatch_kernel in __init__ or super().__init__.
            def __init__(self, kernel_map=None):
                self._prepare(kernel_map)
            def _prepare(self, kernel_map=None):
                self.dispatch_kernel(kernel_map)
            def forward(self, x): return None
            @property
            def default_kernel_map(self): return {}

        cases = [
            (SilentDropOp, "Slot S13"),
            (NoKwargOp, "Slot S12"),
            (HelperOp, "does not call self.dispatch_kernel"),
        ]
        for cls, substring in cases:
            errs = validator.check_c5_dispatch_kernel_invariant(
                cls.__name__, {}, cls,
            )
            assert any(substring in e for e in errs), (cls.__name__, errs)


class TestStubOverrideGates:
    """C6 / C7: _validate_dtypes / eval_roofline must not be base stubs."""

    def test_base_stubs_detected(self, validator):
        from tileops.ops.op_base import Op

        class StubOp(Op):
            def __init__(self): pass
            def forward(self, x): return None
            @property
            def default_kernel_map(self): return {}

        errs = validator.check_c6_validate_dtypes_not_stub("StubOp", {}, StubOp)
        assert any("is the Op base stub" in e for e in errs), errs
        errs = validator.check_c7_eval_roofline_not_stub("StubOp", {}, StubOp)
        assert any("is the Op base stub" in e for e in errs), errs

    def test_overrides_pass(self, validator):
        from tileops.ops.op_base import Op

        class OverriddenOp(Op):
            def __init__(self): pass
            def forward(self, x): return None
            def _validate_dtypes(self, *args): pass
            def eval_roofline(self): return (0, 0)
            @property
            def default_kernel_map(self): return {}

        assert validator.check_c6_validate_dtypes_not_stub(
            "OverriddenOp", {}, OverriddenOp,
        ) == []
        assert validator.check_c7_eval_roofline_not_stub(
            "OverriddenOp", {}, OverriddenOp,
        ) == []


class TestStrictAdvisoryMode:
    """Advisory vs strict routing of C1-C7 failures through ``validate_manifest()``.

    Drives a synthetic single-file manifest pointing at a stub-only Op
    fixture: tests the routing itself, not the bare helpers, so the
    outcome is independent of the checked-in manifest's strict-parity
    backlog.
    """

    @pytest.fixture
    def stub_setup(self, tmp_path, monkeypatch, validator):
        """Synthetic single-file manifest wired to an in-process Op fixture failing C6/C7.

        Monkeypatches the validator's op-class resolver so the test does
        not depend on the synthetic op file being importable from
        sys.path.
        """
        from tileops.ops.op_base import Op

        class StubOp(Op):
            def __init__(self, N, dtype):
                self.N = N
                self.dtype = dtype
            def forward(self, x):  # noqa: ANN001
                return x
            @property
            def default_kernel_map(self):
                return {}

        # Schema-valid synthetic manifest entry. ``shape_rules`` is a
        # list of expression strings under ``signature``; all required
        # top-level fields (``ref_api``, ``roofline``, ``source.{op,
        # kernel, bench, test, kernel_map}``) are present so the test
        # would still parse if schema checks were enabled.
        manifest_yaml = tmp_path / "stub.yaml"
        manifest_yaml.write_text(
            "StubOp:\n"
            "  family: synth\n"
            "  status: implemented\n"
            "  ref_api: https://example.invalid/stub\n"
            "  signature:\n"
            "    params:\n"
            "      N: {type: int}\n"
            "      dtype: {type: torch.dtype}\n"
            "    inputs:\n"
            "      x: {dtype: float16}\n"
            "    outputs:\n"
            "      y: {dtype: 'same_as(x)'}\n"
            "    shape_rules:\n"
            "      - 'y.shape == x.shape'\n"
            "  workloads: []\n"
            "  roofline:\n"
            "    flops: 'N'\n"
            "    bytes: '2 * N'\n"
            "  source:\n"
            "    op: tests/__strict_parity_stub__.py\n"
            "    kernel: tileops/kernels/__strict_parity_stub__.py\n"
            "    bench: benchmarks/ops/__strict_parity_stub__.py\n"
            "    test: tests/__strict_parity_stub_test__.py\n"
            "    kernel_map:\n"
            "      stub: tileops.kernels.StubKernel\n"
        )

        def _fake_resolve(op_file, op_name):
            if op_name == "StubOp":
                return validator._ResolveResult(cls=StubOp)
            return validator._ResolveResult()

        monkeypatch.setattr(validator, "_resolve_op_class", _fake_resolve)
        return manifest_yaml

    def test_advisory_routes_strict_failures_to_warnings(
        self, validator, stub_setup,
    ):
        """Advisory mode routes strict-parity failures to warnings, not errors."""
        # Skip schema/L1 to keep the synthetic manifest minimal; the
        # checks we exercise here are the strict-parity ones (C5-C7),
        # gated by signature/dtype/bench.
        levels = frozenset({"signature", "shape", "dtype", "bench"})
        errors, warnings = validator.validate_manifest(
            manifest_path=stub_setup, strict_parity=False, levels=levels,
        )
        # No strict-parity-only tag should appear in errors. Use
        # STRICT_ONLY_TAGS, not STRICT_TAGS: ``[shape]`` / ``[dtype]``
        # are also emitted by non-strict L2 / L3 checks and may
        # legitimately reach errors regardless of advisory mode.
        leaked = [e for e in errors if any(t in e for t in validator.STRICT_ONLY_TAGS)]
        assert not leaked, (
            f"strict failures must not appear in errors in advisory mode; "
            f"leaked={leaked}"
        )
        # At least one strict-parity warning was raised (C6/C7 fixture
        # is guaranteed to fail both).
        strict_warnings = [
            w for w in warnings if "STRICT-PARITY (advisory)" in w
        ]
        assert strict_warnings, (
            f"advisory mode must surface strict-parity warnings; "
            f"warnings={warnings}"
        )

    def test_strict_routes_failures_to_errors(
        self, validator, stub_setup,
    ):
        """Strict mode routes the same failures to errors with no advisory prefix."""
        levels = frozenset({"signature", "shape", "dtype", "bench"})
        errors, warnings = validator.validate_manifest(
            manifest_path=stub_setup, strict_parity=True, levels=levels,
        )
        strict_errors = [e for e in errors if "[stub]" in e]
        assert strict_errors, (
            f"strict mode must surface strict-parity errors; "
            f"errors={errors}"
        )
        assert not any(
            "STRICT-PARITY (advisory)" in w for w in warnings
        ), f"strict mode must not demote to advisory; warnings={warnings}"


class TestCompileContractRegistry:
    """Enforcement point for the torch_compile_fullgraph contract.

    Must stay in this file: the always-on ``compile-contract-gate``
    preflight job runs pytest on this file on a CPU runner.
    """

    def test_declarations_match_registered_evidence(self):
        """Manifest declarations == registered compile-test evidence, exactly.

        A broken registration helper, a missing evidence module, or a
        typo'd op name all surface as a set difference here.
        """
        from tests.compile_contract import compile_contract_ops
        from tileops.manifest import load_manifest

        declared = {
            name for name, entry in load_manifest().items()
            if entry.get("torch_compile_fullgraph") is True
        }
        registered = compile_contract_ops()
        assert declared == registered, (
            f"evidence without declaration: {sorted(registered - declared)}; "
            f"declaration without evidence: {sorted(declared - registered)}"
        )
