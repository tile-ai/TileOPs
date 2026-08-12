"""Tests for scripts/validate_manifest.py.

Verifies that the manifest validator correctly implements schema/signature/shape/dtype/bench checks.
Uses synthetic manifest data to test individual check functions,
plus an integration test against the real ops manifest (src/tileops/manifest/).
"""

import contextlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

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


# Shared builders and drivers for the synthetic-op test scaffolding.

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


def _sig(inputs, outputs=None, **extra):
    """Build a signature dict from dtype-string shorthand.

    ``inputs`` / ``outputs`` map tensor name -> dtype string, or a full
    attr dict when the test needs ``shape`` / ``constraints`` fields.
    Extra keyword args (``params``, ``shape_rules``, ``dtype_combos``,
    ``static_dims``) are copied verbatim.
    """
    def _tensors(d):
        return {
            k: ({"dtype": v} if isinstance(v, str) else v)
            for k, v in d.items()
        }
    sig = {"inputs": _tensors(inputs)}
    if outputs is not None:
        sig["outputs"] = _tensors(outputs)
    sig.update(extra)
    return sig


def _conv_sig():
    """Conv-style signature with an output-only ``L_out`` symbol."""
    return _sig(
        {"x": {"dtype": "float16", "shape": "[N, C_in, L_in]"},
         "w": {"dtype": "float16", "shape": "[C_out, C_in, kW]"}},
        {"y": {"dtype": "float16", "shape": "[N, C_out, L_out]"}},
        shape_rules=["L_out == L_in - kW + 1"],
    )


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


def _make_op_cls_with_validate(validate_fn, *, name="FakeDtypeOp"):
    from tileops.ops.op_base import Op

    attrs = {
        "_validate_dtypes": validate_fn,
        "forward": lambda self, *a, **kw: None,
        "default_kernel_map": property(lambda self: {}),
    }
    return type(name, (Op,), attrs)


def _make_bare_op(name="BareOp"):
    """Op subclass overriding neither parity method."""
    from tileops.ops.op_base import Op

    return type(name, (Op,), {
        "forward": lambda self, *a, **kw: None,
        "default_kernel_map": property(lambda self: {}),
    })


def _infer_parity(validator, infer_fn, sig, *, name="FakeOp"):
    """Drive ``check_l2_infer_parity`` with a synthetic op class.

    Returns ``(errors, warnings)``.
    """
    cls = _make_op_cls_with_infer(infer_fn, name=name)
    warnings: list[str] = []
    errors = validator.check_l2_infer_parity(
        name, {"signature": sig}, cls, warnings=warnings,
    )
    return errors, warnings


def _dtype_parity(validator, validate_fn, sig, *, name="FakeDtypeOp"):
    """Drive ``check_l3_validate_dtypes_parity`` with a synthetic op class.

    Returns ``(errors, warnings)``.
    """
    cls = _make_op_cls_with_validate(validate_fn, name=name)
    warnings: list[str] = []
    errors = validator.check_l3_validate_dtypes_parity(
        name, {"signature": sig}, cls, warnings=warnings,
    )
    return errors, warnings


def _write_manifest(tmp_path, ops):
    """Serialize *ops* to a single-file synthetic manifest; return its path."""
    import yaml

    manifest_file = tmp_path / "ops_manifest.yaml"
    manifest_file.write_text(yaml.safe_dump(ops))
    return manifest_file


def _fake_op_module(mod_name, class_names):
    """Fake module holding forward()-bearing classes named *class_names*."""
    import types as _types

    mod = _types.ModuleType(mod_name)
    mod.__name__ = mod_name
    for cname in class_names:
        cls = type(cname, (), {"forward": staticmethod(lambda: None)})
        cls.__module__ = mod_name
        setattr(mod, cname, cls)
    return mod


@contextlib.contextmanager
def _patched_import(fake_mod):
    """Patch ``importlib.import_module`` to serve *fake_mod* by name."""
    import importlib
    import unittest.mock as mock

    original = importlib.import_module

    def patched(name):
        if name == fake_mod.__name__:
            return fake_mod
        return original(name)

    with mock.patch.object(importlib, "import_module", side_effect=patched):
        yield


# schema: YAML structure validation

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
        # synthesize the output entirely from construction-time params.
        # The schema gate is ``outputs >= 1 AND (inputs >= 1 OR params >= 1)``.
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
        """kernel_map is required on implemented, optional on spec-only,
        and an empty mapping is valid."""
        # status: implemented without kernel_map -> hard error.
        entry = _make_entry(status="implemented")
        entry["source"].pop("kernel_map", None)
        errors = validator.check_l0("test_op", entry)
        assert any("kernel_map is missing" in e for e in errors), errors

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
        cases = [
            # (rule, substring expected alongside shape_rules[0], count)
            ("totally_unknown_helper(x.shape) == 0",
             "totally_unknown_helper", None),
            ("x.shape == (", "syntax", None),
            ("totally_unknown_helper(x.shape) and "
             "totally_unknown_helper(x.shape)",
             "totally_unknown_helper", 1),
        ]
        for rule, substring, count in cases:
            entry = _make_entry()
            entry["signature"]["shape_rules"] = [rule]
            errors = validator.check_l0("test_op", entry)
            matching = [
                e for e in errors
                if "[schema]" in e and "shape_rules[0]" in e
                and substring in e.lower()
            ]
            if count is None:
                assert matching, (rule, errors)
            else:
                assert len(matching) == count, (rule, matching)


class TestWorkloadPolicy:
    """Workload-count and required-param coverage rules."""

    def test_implemented_needs_two_workloads(self, validator):
        """Implemented ops need >= 2 workloads; spec-only ops are exempt."""
        entry = _make_entry(status="implemented", kernel_map={})
        entry["workloads"] = entry["workloads"][:1]
        errors = validator.check_l0("test_op", entry)
        assert any("at least 2 workloads" in e for e in errors), errors

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
        ), errors

    def test_workload_defaulted_param_may_be_omitted(self, validator):
        """Params with a default are not required in workloads."""
        entry = _make_entry(params={"dim": {"type": "int", "default": -1}})
        assert validator.check_l0("test_op", entry) == []


class TestOutputShapeDeclaration:
    """Every output declares a shape, or the signature has shape_rules."""

    def test_output_shape_presence(self, validator):
        """Case table: no shape and no rules fails; declared shape passes."""
        entry = _make_entry()
        del entry["signature"]["shape_rules"]
        errors = validator.check_l0("test_op", entry)
        assert any(
            "output" in e and "'y'" in e and "shape" in e for e in errors
        ), errors

        entry = _make_entry(
            outputs={"y": {"dtype": "same_as(x)", "shape": "[M, N]"}},
        )
        del entry["signature"]["shape_rules"]
        assert validator.check_l0("test_op", entry) == []


class TestTensorConstraints:
    """Tensor ``constraints`` keys must name dims of the declared shape."""

    def test_constraint_key_shape_dim_matching(self, validator):
        """Case table: key outside dims / constraints without shape fail;
        keys matching the declared dims pass."""
        cases = [
            # (description, input tensor attrs, substrings or None=pass)
            ("constraint key outside shape dims",
             {"dtype": "float16", "shape": "[M, N]",
              "constraints": {"K": "K % 2 == 0"}}, ["constraints", "'K'"]),
            ("constraints without shape",
             {"dtype": "float16", "constraints": {"N": "N % 2 == 0"}},
             ["constraints", "shape"]),
            ("constraint keys matching shape dims",
             {"dtype": "float16", "shape": "[M, N]",
              "constraints": {"N": "N % 2 == 0"}}, None),
        ]
        for desc, attrs, substrings in cases:
            errors = validator.check_l0("test_op", _make_entry(inputs={"x": attrs}))
            if substrings is None:
                assert errors == [], (desc, errors)
            else:
                assert any(
                    all(s in e for s in substrings) for e in errors
                ), f"{desc}: expected error with {substrings}, got: {errors}"


class TestSourcePathExistence:
    """source path values of non-spec-only ops must point at real files."""

    def test_missing_source_file_fails_for_implemented(self, validator, tmp_path):
        manifest_file = _write_manifest(
            tmp_path, {"my_op": _make_entry(status="implemented", kernel_map={})},
        )
        errors, _ = validator.validate_manifest(
            manifest_path=manifest_file,
            repo_root=tmp_path,
            levels=frozenset({"schema"}),
        )
        assert any(
            "source." in e and "not a file" in e for e in errors
        ), errors

    def test_spec_only_source_paths_are_placeholders(self, validator, tmp_path):
        manifest_file = _write_manifest(
            tmp_path, {"my_op": _make_entry(status="spec-only")},
        )
        errors, _ = validator.validate_manifest(
            manifest_path=manifest_file,
            repo_root=tmp_path,
            levels=frozenset({"schema"}),
        )
        assert not any("not a file" in e for e in errors), errors

    def test_existing_source_files_pass(self, validator, tmp_path):
        # kernel/op resolve under src/; test/bench against the repo root.
        (tmp_path / "src").mkdir()
        for rel in ("src/k.py", "src/o.py", "t.py", "b.py"):
            (tmp_path / rel).write_text("# placeholder\n")
        manifest_file = _write_manifest(
            tmp_path, {"my_op": _make_entry(status="implemented", kernel_map={})},
        )
        errors, _ = validator.validate_manifest(
            manifest_path=manifest_file,
            repo_root=tmp_path,
            levels=frozenset({"schema"}),
        )
        assert errors == [], errors


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


class TestRooflineStructuralRules:
    """L0 roofline reject branches, one guard per row.

    Accept paths are owned by TestIntegration: the shipped manifest
    exercises inline and func modes through the same checks.
    """

    def test_reject_branches(self, validator):
        cases = [
            # (description, roofline value, substrings expected in one error)
            ("mixed inline+func modes",
             {"flops": "2*M*N", "bytes": "M*N",
              "func": "tileops.perf.formulas.gemm"}, ["exclusive"]),
            ("non-string field (shipped violation shape)",
             {"flops": 0, "bytes": "M*N"}, ["non-empty string"]),
            ("vars non-mapping",
             {"flops": "2*M*N", "bytes": "M*N", "vars": ["M"]},
             ["vars must be a mapping"]),
            ("vars non-string value",
             {"flops": "2*M*N", "bytes": "M*N", "vars": {"M": 4}},
             ["vars", "non-empty string"]),
            ("vars non-string key",
             {"flops": "2*M*N", "bytes": "M*N", "vars": {4: "M"}},
             ["key", "must be a string"]),
            ("unresolvable func",
             {"func": "tileops.perf.formulas.no_such_formula"},
             ["does not resolve"]),
            ("non-callable func (callable() predicate, not hasattr)",
             {"func": "tileops.perf.formulas.__doc__"},
             ["does not resolve"]),
        ]
        for desc, roofline, substrings in cases:
            entry = _make_entry()
            entry["roofline"] = roofline
            errors = validator.check_l0("my_op", entry)
            assert any(
                all(s in e for s in substrings) for e in errors
            ), f"{desc}: expected error with {substrings}, got: {errors}"


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
        for value in (False, "true", 1, None):
            entry = _make_entry(
                status="implemented", kernel_map={},
                torch_compile_fullgraph=value,
            )
            errors = validator.check_l0("test_op", entry)
            assert any(
                "torch_compile_fullgraph" in e and "true" in e for e in errors
            ), f"Expected literal-true error for {value!r}, got: {errors}"

        entry = _make_entry(status="spec-only", torch_compile_fullgraph=True)
        errors = validator.check_l0("test_op", entry)
        assert any(
            "torch_compile_fullgraph" in e and "spec-only" in e for e in errors
        ), errors


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
        assert validator.check_variant_of_consistency(ops) == []

    def test_malformed_entry_does_not_crash(self, validator):
        """Non-dict entry must not crash variant_of check."""
        ops = {"bad": 123, "ok": _make_entry()}
        assert validator.check_variant_of_consistency(ops) == []

    def test_violations_rejected(self, validator):
        """Case table: missing target, chaining, and shared-source
        mismatches all fail (R16)."""
        mismatched_op = _make_entry()
        mismatched_op["source"]["op"] = "different_op.py"
        mismatched_op["variant_of"] = "primary"
        cases = [
            ("variant target missing",
             {"v": {**_make_entry(), "variant_of": "nonexistent"}},
             ["nonexistent", "does not exist"]),
            ("variant chaining (single-level rule)",
             {"primary": _make_entry(),
              "variant_a": {**_make_entry(), "variant_of": "primary"},
              "variant_b": {**_make_entry(), "variant_of": "variant_a"}},
             ["chaining"]),
            ("mismatched source.kernel",
             {"primary": _make_entry(source_kernel="shared.py"),
              "variant": {**_make_entry(source_kernel="different.py"),
                          "variant_of": "primary"}},
             ["source.kernel", "R16"]),
            ("mismatched source.op",
             {"primary": _make_entry(), "variant": mismatched_op},
             ["source.op", "R16"]),
        ]
        for desc, ops, substrings in cases:
            errors = validator.check_variant_of_consistency(ops)
            assert any(
                all(s in e for s in substrings) for e in errors
            ), f"{desc}: expected error with {substrings}, got: {errors}"


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
        """Unrecognized dtype names in workloads or dtype_combos fail hard,
        independent of any ``_validate_dtypes`` override."""
        entry = {
            "signature": _sig({"x": "float16"}, {"y": "same_as(x)"}),
            "workloads": [{"dtypes": ["not_a_dtype"]}],
        }
        errors = validator.check_l3("test_op", entry)
        assert any("not_a_dtype" in e and "dtype" in e for e in errors), errors

        entry = {
            "status": "implemented",
            "signature": _sig(
                {"x": "float16 | bfloat16"}, {"y": "same_as(x)"},
                dtype_combos=[{"x": "not_a_real_dtype"}],
            ),
            "workloads": [{"dtypes": ["float16"]}],
        }
        errors = validator.check_l3("test_op", entry)
        assert any(
            "not_a_real_dtype" in e and "dtype_combos" in e for e in errors
        ), errors

    def test_dtype_combos_same_as_identity(self, validator):
        """R3 identity: same_as-bound tensors must match in every combo;
        a combo omitting the reference cannot be verified and fails."""
        def entry_with_combos(combos):
            return {
                "signature": _sig(
                    {"x": "float16 | bfloat16", "w": "same_as(x)"},
                    {"y": "same_as(x)"},
                    dtype_combos=combos,
                ),
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
        sig = _sig(
            {"x": "same_as(y)", "y": "float16 | bfloat16"},
            {"z": "same_as(y)"},
        )
        resolved = validator._resolve_tensor_dtype_options(sig)
        assert resolved is not None, "Forward same_as reference must resolve"
        assert resolved["x"] == ["float16", "bfloat16"]
        assert resolved["y"] == ["float16", "bfloat16"]
        assert resolved["z"] == ["float16", "bfloat16"]

        sig = _sig(
            {"input": ("float16 | bfloat16 | float32 | "
                       "int8 | int16 | int32 | int64 | uint8")},
            {"output": "promote_int_to_float(input)"},
        )
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
            ("unknown tensor reference",
             _sig({"x": "float16 | int32"}, {"y": "promote_int_to_float(z)"}),
             [{"dtypes": ["float16"]}],
             ["promote_int_to_float(z)",
              "must reference a signature input tensor"]),
            ("malformed empty arg",
             _sig({"x": "float16"}, {"y": "promote_int_to_float()"}),
             [{"dtypes": ["float16"]}],
             ["unrecognized dtype", "promote_int_to_float()"]),
            ("input-side use",
             _sig({"x": "int8 | int32 | float32",
                   "y": "promote_int_to_float(x)"}, {"out": "float32"}),
             [{"dtypes": ["float32"]}],
             ["promote_int_to_float", " y ", "output-side only"]),
            ("workload-dtype use",
             _sig({"x": "int8 | int32 | float32"},
                  {"y": "promote_int_to_float(x)"}),
             [{"dtypes": ["promote_int_to_float(x)"]}],
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
            "signature": _sig(
                {"input": ("float16 | bfloat16 | float32 | "
                           "int8 | int16 | int32 | int64 | uint8")},
                {"output": "promote_int_to_float(input)"},
            ),
            "workloads": [{"dtypes": ["float16"]}],
        }
        errors = validator.check_l3("PromoteOp", entry)
        assert errors == [], errors

    def test_check_l3_with_non_dict_signature_does_not_crash(self, validator):
        """check_l3 treats malformed signature.inputs/outputs as empty
        (schema layer owns the diagnostics) instead of crashing."""
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


class TestInferShapeParity:
    """L2 extension: ``_infer_output_shapes`` output must satisfy shape_rules."""

    def test_no_override_emits_missing_method_warning(self, validator):
        """Implemented op lacking the codegen-derived method warns; the gap
        must not pass silently."""
        entry = {"signature": _sig(
            {"x": "float16"}, {"y": "same_as(x)"},
            shape_rules=["y.shape == x.shape"],
        )}
        warnings: list[str] = []
        errors = validator.check_l2_infer_parity(
            "BareOp", entry, _make_bare_op(), warnings=warnings,
        )
        assert errors == []
        assert any(
            "does not override _infer_output_shapes" in w for w in warnings
        ), warnings

    def test_symbolic_dim_rule_detects_wrong_output(self, validator):
        """Rules like ``o.shape == (B, S, H, D)`` must evaluate, not
        warn-skip: an unbound symbolic dim would NameError into a warning
        and let a wrong impl pass parity."""
        def infer(self, q_shape, k_shape, v_shape):
            # Wrong: returns a 1-D shape instead of a 4-D shape.
            return {"o": (999,)}

        sig = _sig(
            {"q": "float16", "k": "float16", "v": "float16"},
            {"o": "float16"},
            shape_rules=[
                "q.shape == (B, S, H, D)",
                "k.shape == (B, S, H, D)",
                "v.shape == (B, S, H, D)",
                "o.shape == (B, S, H, D)",
            ],
        )
        errors, warnings = _infer_parity(validator, infer, sig, name="BadMHA")
        assert any(
            "_infer_output_shapes output violates" in e and "o.shape" in e
            for e in errors
        ), (errors, warnings)
        assert not any(
            "could not be evaluated" in w for w in warnings
        ), warnings

    def test_no_cls_skipped(self, validator):
        entry = {"signature": {"shape_rules": ["y.shape == x.shape"]}}
        assert validator.check_l2_infer_parity("Foo", entry, None) == []

    def test_incorrect_infer_fails(self, validator):
        """Parity error when _infer_output_shapes disagrees with shape_rules."""
        def infer(self, x_shape):
            # Wrong: drops a dim.
            return {"y": x_shape[:-1]}

        sig = _sig({"x": "float16"}, {"y": "same_as(x)"},
                   shape_rules=["y.shape == x.shape"])
        errors, _ = _infer_parity(validator, infer, sig)
        assert any("_infer_output_shapes output violates" in e for e in errors), errors

    def test_missing_output_fails(self, validator):
        def infer(self, x_shape):
            return {}  # missing y

        sig = _sig({"x": "float16"}, {"y": "same_as(x)"},
                   shape_rules=["y.shape == x.shape"])
        errors, _ = _infer_parity(validator, infer, sig)
        assert any("missing output" in e for e in errors), errors

    def test_signature_mismatch_reports(self, validator):
        def infer(self, a_shape):
            return {"y": a_shape}

        sig = _sig({"x": "float16"}, {"y": "same_as(x)"},
                   shape_rules=["y.shape == x.shape"])
        errors, _ = _infer_parity(validator, infer, sig)
        assert any("signature does not match" in e for e in errors), errors

    def test_tuple_literal_rule_rank(self, validator):
        """tensor.shape == (A, B) rules inform the mock input rank."""
        seen_rank: list[int] = []

        def infer(self, x_shape):
            seen_rank.append(len(x_shape))
            return {"y": x_shape}

        sig = _sig({"x": "float16"}, {"y": "same_as(x)"},
                   shape_rules=["x.shape == (B, S, H, D)",
                                "y.shape == x.shape"])
        errors, _ = _infer_parity(validator, infer, sig)
        assert errors == []
        assert seen_rank == [4], seen_rank

    def test_r11_style_rule_uses_len_helper(self, validator):
        """R11-style rules using ``len`` / comprehensions stay evaluable
        (restricted builtins + ctx visibility in comprehension scopes);
        breaking either hides mismatches behind NameError skips."""
        # Wrong: reduction op declares dim, keepdim=False so y should drop
        # rank(s), but _infer_output_shapes returns x_shape verbatim.
        def infer(self, x_shape):
            return {"y": x_shape}

        sig = _sig(
            {"x": "float16"}, {"y": "same_as(x)"},
            params={"dim": {"default": -1}, "keepdim": {"default": False}},
            shape_rules=["y.ndim == x.ndim - len({dim % x.ndim})"],
        )
        errors, _ = _infer_parity(validator, infer, sig)
        assert any("_infer_output_shapes output violates" in e for e in errors), errors

        # Comprehension scoping: generator / set comprehensions in the rule
        # body must not be skipped via a NameError warning.
        sig = _sig(
            {"x": "float16"}, {"y": "same_as(x)"},
            params={"dim": {"default": [-1]}, "keepdim": {"default": False}},
            shape_rules=[
                # Generator expression inside all(...): comprehension scope.
                "all(d % x.ndim in range(x.ndim) for d in dim)",
                # Set comprehension: also its own scope.
                "len({d % x.ndim for d in dim}) == len(dim)",
                # Actual parity rule expected to catch the mismatch.
                "y.ndim == x.ndim - len(dim)",
            ],
        )
        errors, warnings = _infer_parity(validator, infer, sig)
        assert not any(
            "could not be evaluated" in w for w in warnings
        ), warnings
        assert any(
            "_infer_output_shapes output violates" in e and "y.ndim" in e
            for e in errors
        ), errors

    def test_input_only_precondition_not_blamed_on_infer(self, validator):
        """Input-only preconditions violated by mock inputs are reported
        as skip warnings, never blamed on a correct impl."""
        def infer(self, x_shape, weight_shape):
            # Correct: y has the same shape as x.
            return {"y": x_shape}

        sig = _sig(
            {"x": "float16", "weight": "float16"}, {"y": "same_as(x)"},
            params={"dim": {"default": -1}},
            shape_rules=[
                # Form the mock synthesis will not satisfy, triggering the
                # precondition-violation path.
                "weight.shape == (x.shape[dim] + 1,)",
                "y.shape == x.shape",
            ],
        )
        errors, warnings = _infer_parity(validator, infer, sig)
        assert not any(
            "_infer_output_shapes output violates" in e for e in errors
        ), errors
        assert any("input-only precondition" in w for w in warnings), warnings

    def test_mock_input_shapes_cross_tensor_dims_distinct(self, validator):
        """Distinct symbolic dims across rules get distinct mock sizes;
        collisions would make cross-tensor rules spuriously True/False."""
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
        # Corollary: x and y disagree on the first dim.
        assert tuple(shapes["x"])[0] != tuple(shapes["y"])[0]

    def test_eval_shape_rule_rejects_dunder_attr(self, validator):
        """Evaluator rejects dunder attribute access (sandbox-escape
        defense against the restricted builtins)."""
        ok, reason = validator._eval_shape_rule(
            "().__class__ is None", {},
        )
        assert ok is False
        assert reason is not None
        assert "dunder attribute access not permitted" in reason

    def test_body_exception_is_hard_error_not_signature_mismatch(self, validator):
        """Body exceptions are hard L2 errors, never signature mismatches:
        the signature is pre-bound, so body TypeError / RuntimeError
        cannot masquerade as a binding failure."""
        sig = _sig({"x": "float16"}, {"y": "same_as(x)"},
                   shape_rules=["y.shape == x.shape"])
        for exc_cls, message in (
            (TypeError, "simulated implementation bug"),
            (RuntimeError, "not ready"),
        ):
            def infer(self, x_shape, _exc=exc_cls, _msg=message):
                # Signature matches; the body itself raises.
                raise _exc(_msg)

            errors, warnings = _infer_parity(validator, infer, sig)
            assert not any(
                "signature does not match manifest inputs" in e for e in errors
            ), (exc_cls, errors)
            assert any(
                f"raised {exc_cls.__name__}" in e for e in errors
            ), (exc_cls, errors, warnings)

    def test_declared_output_shape_catches_wrong_infer(self, validator):
        """Declared output shapes alone (no shape_rules) must drive
        parity against ``signature.outputs[*].shape``."""
        def infer(self, x_shape, w_shape):
            # Wrong: returns x_shape verbatim instead of [N, C_out, L_out].
            return {"y": tuple(x_shape)}

        sig = _sig(
            {"x": {"dtype": "float16", "shape": "[N, C_in, L_in]"},
             "w": {"dtype": "float16", "shape": "[C_out, C_in, kW]"}},
            {"y": {"dtype": "float16", "shape": "[N, C_out, L_out]"}},
        )
        errors, _ = _infer_parity(validator, infer, sig)
        assert any("disagrees with declared" in e for e in errors), errors

    def test_infer_reads_self_attr_uses_cls_new(self, validator):
        """Reading a class-defined ``self`` attribute must not skip parity:
        the mock self is built via ``cls.__new__(cls)``."""
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
                _ = self.some_attr
                return {"y": tuple(x_shape)}

        entry = {"signature": _sig(
            {"x": "float16"}, {"y": "same_as(x)"},
            shape_rules=["y.shape == x.shape"],
        )}
        warnings: list[str] = []
        errors = validator.check_l2_infer_parity(
            "SelfAttrOp", entry, SelfAttrOp, warnings=warnings,
        )
        assert errors == [], errors
        assert not any(
            "parity skipped" in w and "AttributeError" in w
            for w in warnings
        ), warnings

    def test_infer_reads_static_dim_attr_populated(self, validator):
        """Reading ``self.<static_dim>`` must exercise parity:
        ``_build_mock_self`` installs resolved static_dims values."""
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

        entry = {"signature": _sig(
            {"x": {"dtype": "float16", "shape": "[B, N]"}},
            {"y": {"dtype": "float16", "shape": "[N, N]"}},
            static_dims={"N": "x.shape[-1]"},
        )}
        warnings: list[str] = []
        errors = validator.check_l2_infer_parity(
            "StaticDimOp", entry, StaticDimOp, warnings=warnings,
        )
        assert errors == [], errors
        assert not any("AttributeError" in w for w in warnings), warnings

    def test_conv_like_output_only_symbol_not_blamed(self, validator):
        """Correct infer with an output-only ``L_out`` symbol passes:
        output-only symbols get rank/consistency checks only, never a
        comparison against arbitrary mock sizes."""
        def infer(self, x_shape, w_shape):
            # x: [N, C_in, L_in]; w: [C_out, C_in, kW]
            return {"y": (x_shape[0], w_shape[0], x_shape[2] - w_shape[2] + 1)}

        errors, _ = _infer_parity(
            validator, infer, _conv_sig(), name="ConvLikeOp",
        )
        assert errors == [], errors

    def test_conv_like_wrong_output_only_value_reported(self, validator):
        """Wrong output-only value is flagged via the defining rule:
        ``L_out`` classifies as output-mentioning (rebound from the
        inferred result), not as an input-only precondition."""
        def infer(self, x_shape, w_shape):
            # Deliberately wrong output-only L_out value (999).
            return {"y": (x_shape[0], w_shape[0], 999)}

        errors, _ = _infer_parity(
            validator, infer, _conv_sig(), name="ConvLikeWrongOutOnlyOp",
        )
        assert any(
            "L_out == L_in - kW + 1" in e and "violates shape_rules" in e
            for e in errors
        ), errors

    def test_conv_like_rank_and_consistency_still_caught(self, validator):
        """Loosening the output-only value check must not weaken the rank
        check, and an output-only symbol reused across multiple outputs
        must stay internally consistent."""
        # Rank disagreement against the declared output shape.
        def bad_rank_infer(self, x_shape, w_shape):
            # Wrong rank: drops the spatial dim entirely.
            return {"y": (x_shape[0], w_shape[0])}

        errors, _ = _infer_parity(
            validator, bad_rank_infer, _conv_sig(), name="ConvLikeBadOp",
        )
        assert any("rank" in e and "disagrees" in e for e in errors), errors

        # Two outputs claiming ``L_out`` with different concrete sizes is
        # an internal inconsistency even though L_out is output-only.
        def inconsistent_infer(self, x_shape):
            return {
                "y1": (x_shape[0], x_shape[1] - 1),
                "y2": (x_shape[0], x_shape[1] - 2),
            }

        sig = _sig(
            {"x": {"dtype": "float16", "shape": "[N, L_in]"}},
            {"y1": {"dtype": "float16", "shape": "[N, L_out]"},
             "y2": {"dtype": "float16", "shape": "[N, L_out]"}},
            shape_rules=["L_out == L_in - 1"],
        )
        errors, _ = _infer_parity(
            validator, inconsistent_infer, sig, name="InconsistentOutOnlyOp",
        )
        assert any(
            "output-only symbol" in e and "L_out" in e for e in errors
        ), errors


class TestDtypeOptionsHelper:
    """Unit tests for ``_dtype_options_for_tensor`` unresolved-ref contract."""

    def test_pure_same_as_unresolved_returns_none(self, validator):
        """same_as(unresolved ref) returns None — an empty list would
        silently disable dtype parity via an empty Cartesian product."""
        out = validator._dtype_options_for_tensor(
            "y", "same_as(x)", resolved={},
        )
        assert out is None, out


# shape_rules broadcasting helpers (broadcast_shapes / is_broadcastable_to)


class TestShapeRuleBroadcastBuiltins:
    """Broadcasting helpers in shape_rules eval: torch.broadcast_shapes semantics, torch-free."""

    def test_broadcast_shapes_value_matrix(self, validator):
        """``broadcast_shapes`` produces the expected output across cases.

        Covers identical / scalar / size-1-expand / rank-promotion /
        variadic (0, 1, 3+ args) / list-input forms. Hand-coded
        expectations (no torch dependency).
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
        """``is_broadcastable_to(src, dst)`` pins the asymmetric semantics
        (src may grow into dst; dst is fixed)."""
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
        """Both helpers resolve as bare names from inside ``_eval_shape_rule``
        rule bodies, returning the same value as direct calls."""
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


class TestValidateDtypesParity:
    """L3 extension: ``_validate_dtypes`` matches manifest dtype_combos/unions."""

    def test_no_override_emits_missing_method_warning(self, validator):
        """Missing ``_validate_dtypes`` override must not pass silently on L3."""
        entry = {"signature": _sig({"x": "float16"}, {"y": "same_as(x)"})}
        warnings: list[str] = []
        errors = validator.check_l3_validate_dtypes_parity(
            "BareOp", entry, _make_bare_op(), warnings=warnings,
        )
        assert errors == []
        assert any(
            "does not override _validate_dtypes" in w for w in warnings
        ), warnings

    def test_union_accept_reject_matrix(self, validator):
        """Accepting the whole union passes; rejecting a declared dtype
        is a parity error."""
        import torch

        def accept_union(self, x):
            if x.dtype not in (torch.float16, torch.bfloat16):
                raise ValueError(f"bad dtype {x.dtype}")

        def fp16_only(self, x):
            if x.dtype != torch.float16:
                raise ValueError("only fp16")

        sig = _sig({"x": "float16 | bfloat16"}, {"y": "same_as(x)"})
        errors, _ = _dtype_parity(validator, accept_union, sig)
        assert errors == []

        errors, _ = _dtype_parity(validator, fp16_only, sig)
        assert any("rejects valid combo" in e for e in errors), errors

    def test_dtype_combos_accept_listed_pass(self, validator):
        import torch

        def validate(self, x, w):
            allowed = {(torch.float16, torch.float16), (torch.bfloat16, torch.bfloat16)}
            if (x.dtype, w.dtype) not in allowed:
                raise ValueError("unlisted")

        errors, _ = _dtype_parity(
            validator, validate,
            _sig({"x": "float16 | bfloat16", "w": "same_as(x)"},
                 {"y": "same_as(x)"},
                 dtype_combos=[{"x": "float16", "w": "float16"},
                               {"x": "bfloat16", "w": "bfloat16"}]),
        )
        assert errors == []

    def test_dtype_combos_rejects_listed_fails(self, validator):
        import torch

        def validate(self, x, w):
            # Rejects the listed (bfloat16, bfloat16) combo.
            if x.dtype != torch.float16:
                raise ValueError("unlisted")

        errors, _ = _dtype_parity(
            validator, validate,
            _sig({"x": "float16 | bfloat16", "w": "same_as(x)"},
                 {"y": "same_as(x)"},
                 dtype_combos=[{"x": "float16", "w": "float16"},
                               {"x": "bfloat16", "w": "bfloat16"}]),
        )
        assert any("rejects dtype_combos" in e for e in errors), errors

    def test_dtype_combos_accepts_unlisted_fails(self, validator):
        """Accepts a non-listed combo -> parity error."""
        def validate(self, x, w):
            return None  # accepts everything

        errors, _ = _dtype_parity(
            validator, validate,
            _sig({"x": "float16 | bfloat16", "w": "float16 | bfloat16"},
                 {"y": "same_as(x)"},
                 dtype_combos=[{"x": "float16", "w": "float16"}]),
        )
        assert any("accepts non-listed combo" in e for e in errors), errors

    def test_dtype_combos_first_rejected_later_accepted_fails(self, validator):
        """Non-listed combos are all checked: stopping at the first
        rejection would let a later accepted combo escape."""
        import torch

        def validate(self, x, w):
            # Reject the first non-listed combo (fp16, bf16) but accept a
            # later non-listed combo (bf16, fp16).
            if x.dtype == torch.float16 and w.dtype == torch.bfloat16:
                raise ValueError("rejected early non-listed combo")

        errors, _ = _dtype_parity(
            validator, validate,
            _sig({"x": "float16 | bfloat16", "w": "float16 | bfloat16"},
                 {"y": "same_as(x)"},
                 dtype_combos=[{"x": "float16", "w": "float16"}]),
        )
        assert any("accepts non-listed combo" in e for e in errors), errors
        # The (bfloat16, float16) combo specifically is surfaced — proves
        # the loop did not stop at the first rejection.
        assert any(
            "'x': 'bfloat16'" in e and "'w': 'float16'" in e
            for e in errors
        ), errors

    def test_signature_mismatch_union_fails(self, validator):
        """_validate_dtypes with a wrong kwarg name must fail on both the
        union and the dtype_combos branch; downgraded to a warning it
        would let an uncallable ``_validate_dtypes`` satisfy parity."""
        def validate(self, wrong_name, other_wrong=None):
            return None

        union_sig = _sig({"x": "float16 | bfloat16"}, {"y": "same_as(x)"})
        combos_sig = _sig(
            {"x": "float16 | bfloat16", "w": "same_as(x)"},
            {"y": "same_as(x)"},
            dtype_combos=[{"x": "float16", "w": "float16"},
                          {"x": "bfloat16", "w": "bfloat16"}],
        )
        for branch, sig in (("union", union_sig), ("combos", combos_sig)):
            errors, warnings = _dtype_parity(validator, validate, sig)
            assert any(
                "signature does not match manifest inputs" in e for e in errors
            ), (branch, errors, warnings)

    def test_body_unexpected_exception_is_hard_error(self, validator):
        """_validate_dtypes raising RuntimeError for every valid combo must
        produce a hard L3 parity error (not a warning) on both the
        dtype_combos and the no-combos Cartesian branch."""
        def bad_validate(self, x):
            raise RuntimeError("simulated bug")

        combos_sig = _sig(
            {"x": "float16 | bfloat16"}, {"y": "same_as(x)"},
            dtype_combos=[{"x": "float16"}, {"x": "bfloat16"}],
        )
        no_combos_sig = _sig({"x": "float16 | bfloat16"}, {"y": "same_as(x)"})
        for branch, sig in (
            ("combos", combos_sig), ("no-combos", no_combos_sig),
        ):
            errors, warnings = _dtype_parity(
                validator, bad_validate, sig, name="BadValidateOp",
            )
            assert any(
                "raised unexpected exception" in e and "RuntimeError" in e
                for e in errors
            ), (branch, errors, warnings)

    def test_wide_union_probe_pool_derived_from_torch_dtypes(self, validator):
        """Probe pool is ``sorted(_TORCH_DTYPES - declared)``: a wide
        8-dtype union still leaves uint8 to catch permissive impls."""
        def accept_all(self, x):
            return True  # over-permissive: accepts any dtype

        errors, warnings = _dtype_parity(
            validator, accept_all,
            _sig({"x": ("float16 | bfloat16 | float32 | float64 | "
                        "int8 | int16 | int32 | int64")},
                 {"y": "same_as(x)"}),
            name="WideEightDtypeOp",
        )
        assert any(
            "accepts out-of-union dtype" in e for e in errors
        ), (errors, warnings)

    def test_full_torch_coverage_emits_skip_warning(self, validator):
        """Full-coverage union skips the probe with a warning naming the
        input — no vacuous pass, no hard error."""
        full_union = " | ".join(sorted(validator._TORCH_DTYPES))

        def accept_all(self, x):
            return True

        errors, warnings = _dtype_parity(
            validator, accept_all,
            _sig({"x": full_union}, {"y": "same_as(x)"}),
            name="FullCoverageOp",
        )
        assert not any("accepts out-of-union dtype" in e for e in errors), errors
        assert any(
            "out-of-union probe skipped" in w and "'x'" in w
            for w in warnings
        ), warnings

    def test_cartesian_product_over_bound_skipped_with_warning(
        self, validator, monkeypatch,
    ):
        """Enumeration must stay within ``_MAX_DTYPE_COMBOS``: over-bound
        ops skip deterministically with a warning naming input count ×
        option sizes (guards CI wall-time on many-input wide-union ops)."""
        monkeypatch.setattr(validator, "_MAX_DTYPE_COMBOS", 4)

        def _accept_all(self, **kwargs):
            return None

        errors, warnings = _dtype_parity(
            validator, _accept_all,
            _sig({"a": "float16 | bfloat16 | float32",
                  "b": "float16 | bfloat16 | float32"},
                 {"y": "same_as(a)"}),
            name="WideDtypeOp",
        )
        assert errors == [], errors
        assert any("exceeds _MAX_DTYPE_COMBOS" in w for w in warnings), warnings

    def test_body_typeerror_is_rejection_not_signature_mismatch(self, validator):
        """Body TypeError is a rejection, not a signature mismatch
        (the signature is pre-bound before invocation)."""
        def validate(self, x):
            raise TypeError("dtype comparison not supported")

        errors, _ = _dtype_parity(
            validator, validate,
            _sig({"x": "float16 | bfloat16"}, {"y": "same_as(x)"}),
        )
        assert not any(
            "signature does not match manifest inputs" in e for e in errors
        ), errors
        # The body rejects every combo drawn from the union, so the
        # no-dtype_combos branch reports each as a parity violation.
        assert any("rejects valid combo" in e for e in errors), errors

    def test_dtype_combos_exhausts_union_emits_warning(self, validator):
        """Exhaustive dtype_combos still emit the 'exhausts the union'
        warning even though no non-listed combo ever runs."""
        import torch

        allowed = {torch.float16, torch.bfloat16}

        def validate(self, x, w):
            # Reject dtypes outside the declared union so the
            # out-of-union probe does not produce parity errors.
            if x.dtype not in allowed or w.dtype not in allowed:
                raise ValueError("dtype out of union")
            return None

        errors, warnings = _dtype_parity(
            validator, validate,
            _sig({"x": "float16 | bfloat16", "w": "same_as(x)"},
                 {"y": "same_as(x)"},
                 dtype_combos=[
                     {"x": "float16", "w": "float16"},
                     {"x": "float16", "w": "bfloat16"},
                     {"x": "bfloat16", "w": "float16"},
                     {"x": "bfloat16", "w": "bfloat16"},
                 ]),
        )
        assert errors == [], errors
        assert any("exhausts the union" in w for w in warnings), warnings

    def test_no_combos_out_of_union_probe_matrix(self, validator):
        """No-combos branch fires its own out-of-union probe: a permissive
        op accepting any dtype fails (union iteration alone cannot catch
        it); an op rejecting out-of-union dtypes passes."""
        import torch

        def accept_all(self, x):
            return None

        def reject_out_of_union(self, x):
            if x.dtype not in (torch.float16, torch.bfloat16):
                raise ValueError(f"unsupported dtype {x.dtype}")

        sig = _sig({"x": "float16 | bfloat16"}, {"y": "same_as(x)"})
        errors, _ = _dtype_parity(validator, accept_all, sig)
        assert any("out-of-union" in e for e in errors), errors

        errors, _ = _dtype_parity(validator, reject_out_of_union, sig)
        assert errors == [], errors

    def test_no_combos_out_of_union_probe_respects_max(
        self, validator, monkeypatch,
    ):
        """Out-of-union probe is bounded by ``_MAX_DTYPE_COMBOS``: cap 2
        fires at most 2 probes from a 6-dtype pool (product size 2 keeps
        the Cartesian enumeration alive)."""
        monkeypatch.setattr(validator, "_MAX_DTYPE_COMBOS", 2)

        def validate(self, x):
            return None

        errors, _ = _dtype_parity(
            validator, validate,
            _sig({"x": "float16 | bfloat16"}, {"y": "same_as(x)"}),
        )
        out_of_union_errs = [e for e in errors if "out-of-union" in e]
        assert len(out_of_union_errs) == 2, out_of_union_errs

    def test_no_combos_same_as_probe_matrix(self, validator):
        """same_as identity has a dedicated negative probe: the union
        iteration skips same_as-violating candidates via
        ``_honours_same_as``, so a permissive op not enforcing same_as
        fails only through the probe; an enforcing op passes."""
        import torch

        allowed = (torch.float16, torch.bfloat16)

        def accept_all(self, x, w):
            return None  # does not check x.dtype == w.dtype

        def enforce_same_as(self, x, w):
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

        sig = _sig({"x": "float16 | bfloat16", "w": "same_as(x)"},
                   {"y": "same_as(x)"})
        errors, _ = _dtype_parity(validator, accept_all, sig)
        assert any("same_as violation" in e for e in errors), errors

        errors, _ = _dtype_parity(validator, enforce_same_as, sig)
        assert errors == [], errors

    def test_combos_branch_out_of_union_probe(self, validator):
        """Dtype_combos branch fires the out-of-union probe; otherwise a
        permissive impl passes once listed combos are accepted."""
        def validate(self, x):
            return None  # overly permissive

        errors, _ = _dtype_parity(
            validator, validate,
            _sig({"x": "float16 | bfloat16"}, {"y": "same_as(x)"},
                 dtype_combos=[{"x": "float16"}, {"x": "bfloat16"}]),
        )
        assert any("out-of-union" in e for e in errors), errors

    def test_invalid_dtype_combo_value_is_hard_error(self, validator):
        """A non-existent dtype in dtype_combos is a hard L3 error, not a
        'cannot build mock tensor' skip that would hide the data bug."""
        def validate(self, x, w):
            return None

        errors, warnings = _dtype_parity(
            validator, validate,
            _sig({"x": "float16 | bfloat16", "w": "same_as(x)"},
                 {"y": "same_as(x)"},
                 dtype_combos=[
                     {"x": "not_a_real_dtype", "w": "not_a_real_dtype"},
                 ]),
        )
        assert any(
            "not a valid dtype" in e and "not_a_real_dtype" in e
            for e in errors
        ), errors
        assert not any(
            "cannot build mock tensor" in w for w in warnings
        ), warnings

    def test_valid_dtype_combo_reaches_build_mock_tensor(
        self, validator, monkeypatch,
    ):
        """'cannot build mock tensor' stays reserved for valid dtype
        names on torch builds lacking them (simulated via monkeypatch)."""
        def validate(self, x):
            return None

        original = validator._make_mock_tensor

        def fake(name):
            if name == "float8_e4m3fn":
                return None
            return original(name)

        monkeypatch.setattr(validator, "_make_mock_tensor", fake)
        errors, warnings = _dtype_parity(
            validator, validate,
            _sig({"x": "float16 | float8_e4m3fn"}, {"y": "same_as(x)"},
                 dtype_combos=[{"x": "float8_e4m3fn"}]),
        )
        # Valid dtype that the local build can't materialize: no hard
        # error, but the parity-skip warning path still fires.
        assert not any("not a valid dtype" in e for e in errors), errors
        assert any(
            "cannot build mock tensor" in w for w in warnings
        ), warnings

    def test_combo_missing_input_is_manifest_error(self, validator):
        """A combo omitting an input is a manifest error, never a rejection or silent skip."""
        def validate(self, x, w):
            return None

        errors, _ = _dtype_parity(
            validator, validate,
            _sig({"x": "float16 | bfloat16", "w": "same_as(x)"},
                 {"y": "same_as(x)"},
                 dtype_combos=[{"x": "float16"}]),  # missing 'w'
        )
        assert any(
            "is missing declared input" in e or "combo missing input" in e
            for e in errors
        ), errors
        assert not any("rejects dtype_combos[0]" in e for e in errors), errors

    def test_validate_dtypes_reads_self_dtype_attr(self, validator):
        """``x.dtype != self.dtype`` works on the mock self: the dtype
        axis is populated from the candidate combo, not Op's None."""
        def validate(self, x):
            # The generated pattern under test: compare the input dtype
            # against ``self.dtype`` (set in __init__ via a dtype param).
            if x.dtype != self.dtype:
                raise ValueError(
                    f"x.dtype {x.dtype} does not match self.dtype {self.dtype}"
                )

        errors, warnings = _dtype_parity(
            validator, validate,
            _sig({"x": "float16 | bfloat16"}, {"y": "same_as(x)"}),
        )
        assert errors == [], (errors, warnings)


class TestDtypeCombosData:
    """Data-level hardening for ``check_l3_dtype_combos_data``.

    These checks run independently of any ``_validate_dtypes`` override, so
    manifest data bugs surface even when the parity loop never executes.
    """

    def test_malformed_combo_rows_are_hard_errors(self, validator):
        """Case table (manifest.md R4): rows must cover every declared
        input; unions and promote_int_to_float expand to multiple dtypes
        and cannot pin a combo row."""
        cases = [
            # (description, sig, substrings expected in one error)
            ("combo row missing a declared input",
             _sig({"x": "float16 | bfloat16", "w": "float16 | bfloat16"},
                  {"y": "same_as(x)"},
                  dtype_combos=[{"x": "float16", "w": "float16"},
                                {"x": "bfloat16"}]),  # missing 'w'
             ["dtype_combos[1]", "missing declared input 'w'"]),
            ("union expression as combo value",
             _sig({"x": "float16 | bfloat16"}, {"y": "same_as(x)"},
                  dtype_combos=[{"x": "float16 | bfloat16"}]),
             ["combo values must be a single concrete dtype"]),
            ("promote_int_to_float as combo value",
             _sig({"x": "float16 | int8"}, {"y": "promote_int_to_float(x)"},
                  dtype_combos=[
                      {"x": "float16", "y": "promote_int_to_float(x)"},
                  ]),
             ["promote_int_to_float(...) is allowed only on signature.outputs"]),
        ]
        for desc, sig, substrings in cases:
            errors = validator.check_l3_dtype_combos_data("FakeOp", sig)
            assert any(
                all(s in e for s in substrings) for e in errors
            ), f"{desc}: expected error with {substrings}, got: {errors}"

    def test_unresolvable_same_as_graph_is_hard_error(self, validator):
        """Pure same_as cycles and dangling refs surface hard L3 errors
        (they satisfy per-token + R3 checks, so need a dedicated diagnosis)."""
        cycle_sig = _sig(
            {"x": "same_as(y)", "y": "same_as(x)"}, {"z": "same_as(x)"},
            dtype_combos=[{"x": "float16", "y": "float16"}],
        )
        errors = validator.check_l3_dtype_combos_data("CycleOp", cycle_sig)
        assert any(
            "same_as cycle" in e and "'x'" in e and "'y'" in e
            for e in errors
        ), errors

        dangling_sig = _sig(
            {"x": "same_as(nope)"}, {"z": "same_as(x)"},
            dtype_combos=[{"x": "float16"}],
        )
        errors = validator.check_l3_dtype_combos_data("DanglingOp", dangling_sig)
        assert any(
            "dangling reference" in e and "same_as(nope)" in e
            for e in errors
        ), errors


class TestStaticDimShapeParity:
    """static_dims values must pin expected output sizes in L2 parity."""

    def test_static_dim_output_shape_catches_bad_infer(self, validator):
        """Static-dim-bound output positions are checked by exact value
        (resolved against mock inputs), not only rank/consistency."""
        def bad_infer(self, x_shape):
            # static_dims pins N = x.shape[-1] (=4 under mock); a correct
            # impl returns (4, 4).
            return {"y": (999, 999)}

        sig = _sig(
            {"x": {"dtype": "float16", "shape": "[M, N]"}},
            {"y": {"dtype": "same_as(x)", "shape": "[N, N]"}},
            static_dims={"N": "x.shape[-1]"},
        )
        errors, _ = _infer_parity(
            validator, bad_infer, sig, name="StaticDimBadOp",
        )
        assert any(
            "dim[0]=999" in e or "dim[1]=999" in e for e in errors
        ), errors


class TestParamDefaultOutputShapePin:
    """Concrete param defaults pin declared output-shape dims in L2 parity."""

    def test_param_default_pins_output_dim(self, validator):
        """Bad infer returning ``(999,)`` for declared ``[k]`` with
        ``params.k.default = 4`` must produce a hard L2 error; a correct
        infer returning ``(4,)`` passes parity."""
        sig = _sig(
            {"x": {"dtype": "float16", "shape": "[M]"}},
            {"y": {"dtype": "same_as(x)", "shape": "[k]"}},
            params={"k": {"type": "int", "default": 4}},
        )

        def bad_infer(self, x_shape):
            return {"y": (999,)}

        errors, _ = _infer_parity(
            validator, bad_infer, sig, name="ParamDefaultBadOp",
        )
        assert any(
            "dim[0]=999" in e and "k=4" in e for e in errors
        ), errors

        def good_infer(self, x_shape):
            return {"y": (4,)}

        errors, _ = _infer_parity(
            validator, good_infer, sig, name="ParamDefaultGoodOp",
        )
        assert errors == [], errors


# Bench checks
class TestBench:
    """bench checks that bench files use manifest workloads and op roofline.

    Case table: direct (``load_workloads`` + ``op.eval_roofline()``) and
    indirect (``benchmarks.benchmark_base`` helpers) usage passes; missing
    helpers, wrong op names, and syntax errors fail with the named
    diagnostics. Rows with ``None`` expect a clean pass.
    """

    def test_bench_file_usage_matrix(self, validator, tmp_path):
        cases = [
            # (description, bench file text, expected substrings or None)
            ("direct load_workloads + eval_roofline passes", """\
                from tileops.manifest import load_workloads
                workloads = load_workloads('test_op')
                op.eval_roofline()
            """, None),
            ("indirect base helpers pass", """\
                from benchmarks.benchmark_base import workloads_to_params, ManifestBenchmark
                params = workloads_to_params('test_op')
                ManifestBenchmark('test_op', op, params[0])
            """, None),
            ("load_workloads without eval_roofline fails", """\
                from tileops.manifest import load_workloads
                workloads = load_workloads('test_op')
            """, ["eval_roofline"]),
            ("no load_workloads fails", """\
                import pytest
                shapes = [(1024, 4096)]
            """, ["load_workloads"]),
            ("wrong op name fails (direct path)", """\
                from tileops.manifest import load_workloads
                workloads = load_workloads('wrong_op')
                op.eval_roofline()
            """, ["load_workloads"]),
            ("wrong op name fails (indirect path)", """\
                from benchmarks.benchmark_base import workloads_to_params, ManifestBenchmark
                params = workloads_to_params('wrong_op')
                ManifestBenchmark('wrong_op', op, params[0])
            """, ["load_workloads", "eval_roofline"]),
            ("syntax error fails", "def broken(\n", ["syntax error"]),
        ]
        for desc, text, expected in cases:
            bench_file = tmp_path / "bench_test.py"
            bench_file.write_text(textwrap.dedent(text))
            errors = validator.check_l4_benchmark(
                "test_op", str(bench_file), REPO_ROOT,
            )
            if expected is None:
                assert errors == [], (desc, errors)
            else:
                for substring in expected:
                    assert any(substring in e for e in errors), (
                        f"{desc}: expected {substring!r} in errors, got: {errors}"
                    )


# --check-op: force all levels on a specific op, ignoring status


    def test_bench_declaration_required_for_implemented_ops(self, validator):
        """Implemented ops may not opt out of the manifest-driven bench contract."""
        entry = {
            "status": "implemented",
            "source": {"bench": "benchmarks/ops/bench_x.py"},
        }
        errs = validator.check_bench_declaration("XFwdOp", entry)
        assert any("bench_manifest_driven must be declared" in e for e in errs), errs

        entry["source"]["bench_manifest_driven"] = True
        assert validator.check_bench_declaration("XFwdOp", entry) == []

        # spec-only ops and ops without a bench pointer are exempt.
        assert validator.check_bench_declaration(
            "XFwdOp", {"status": "spec-only", "source": {"bench": "b.py"}}) == []
        assert validator.check_bench_declaration(
            "XFwdOp", {"status": "implemented", "source": {}}) == []


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
        manifest_file = _write_manifest(tmp_path, {"my_op": entry})

        # Without check_op: spec-only op skips L1-L4.
        errors_no_flag, _ = validator.validate_manifest(
            manifest_path=manifest_file,
            repo_root=tmp_path,
        )
        bench_errors_no_flag = [e for e in errors_no_flag if "[bench]" in e]
        assert bench_errors_no_flag == [], bench_errors_no_flag

        # With check_op="my_op": all levels forced despite spec-only.
        errors_flag, _ = validator.validate_manifest(
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
        manifest_file = _write_manifest(
            tmp_path, {"my_op": _make_entry(status="spec-only")},
        )
        errors, _ = validator.validate_manifest(
            manifest_path=manifest_file,
            repo_root=tmp_path,
        )
        non_schema = [e for e in errors if "[schema]" not in e]
        assert non_schema == [], non_schema

    def test_check_op_nonexistent_op_reports_error(self, validator, tmp_path):
        """--check-op with a name not in manifest reports an error."""
        manifest_file = _write_manifest(tmp_path, {"my_op": _make_entry()})
        errors, _ = validator.validate_manifest(
            manifest_path=manifest_file,
            repo_root=tmp_path,
            check_op="nonexistent_op",
        )
        assert any("nonexistent_op" in e and "not found" in e for e in errors), errors

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
        assert any("top-level mapping" in e for e in errors), errors

    def test_check_op_scopes_to_single_op(self, validator, tmp_path):
        """--check-op validates only the named op; unrelated ops are not processed."""
        # target_op: spec-only with a real bench file -> L4 runs and fails.
        bench_file = tmp_path / "bench_target.py"
        bench_file.write_text("import pytest\n")
        target_entry = _make_entry(status="spec-only")
        target_entry["source"]["bench"] = str(bench_file)
        target_entry["source"]["bench_manifest_driven"] = True

        # other_op: implemented, points at a nonexistent kernel — if
        # validated, L1 would fail importing the missing module.
        other_entry = _make_entry(source_kernel="nonexistent_impl.py")
        manifest_file = _write_manifest(
            tmp_path, {"target_op": target_entry, "other_op": other_entry},
        )

        errors, _ = validator.validate_manifest(
            manifest_path=manifest_file,
            repo_root=tmp_path,
            check_op="target_op",
        )
        other_errors = [e for e in errors if "other_op" in e]
        assert other_errors == [], other_errors
        target_errors = [e for e in errors if "target_op" in e]
        assert len(target_errors) > 0, (
            "target_op should have validation errors from forced L4 check"
        )

    def test_check_op_ignores_unrelated_variant_of_errors(self, validator, tmp_path):
        """--check-op scopes variant_of checks to the variant family;
        unrelated ops with bad references must not fail the run."""
        other_entry = _make_entry()
        other_entry["variant_of"] = "nonexistent_primary"
        manifest_file = _write_manifest(
            tmp_path, {"target_op": _make_entry(), "other_op": other_entry},
        )

        errors, _ = validator.validate_manifest(
            manifest_path=manifest_file,
            repo_root=tmp_path,
            check_op="target_op",
        )
        variant_errors = [e for e in errors if "variant_of" in e]
        assert variant_errors == [], variant_errors

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
        """--check-op on a primary also validates its immediate variants,
        so a variant edit breaking R16 cannot slip through."""
        primary = _make_entry(source_kernel="shared_kernel.py")
        valid_variant = _make_entry(source_kernel="shared_kernel.py")
        valid_variant["variant_of"] = "primary_op"
        # Broken variant: different source.kernel violates R16.
        broken_variant = _make_entry(source_kernel="different_kernel.py")
        broken_variant["variant_of"] = "primary_op"
        manifest_file = _write_manifest(tmp_path, {
            "primary_op": primary,
            "good_variant": valid_variant,
            "bad_variant": broken_variant,
        })

        errors, _ = validator.validate_manifest(
            manifest_path=manifest_file,
            repo_root=tmp_path,
            check_op="primary_op",
        )
        r16_errors = [e for e in errors if "bad_variant" in e and "R16" in e]
        assert len(r16_errors) > 0, errors

        good_r16 = [e for e in errors if "good_variant" in e and "R16" in e]
        assert good_r16 == [], good_r16

    def test_check_op_variant_family_runs_schema_on_variants(self, validator, tmp_path):
        """--check-op on primary runs per-op schema checks on variants too."""
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
        manifest_file = _write_manifest(tmp_path, {
            "primary_op": primary,
            "broken_var": broken_variant,
        })

        errors, _ = validator.validate_manifest(
            manifest_path=manifest_file,
            repo_root=tmp_path,
            check_op="primary_op",
        )
        schema_errors = [e for e in errors if "broken_var" in e and "source" in e]
        assert len(schema_errors) > 0, errors

    def test_check_op_cli_parsing(self, validator):
        """_parse_check_op extracts the op name from argv; a missing
        value exits with status 2."""
        assert validator._parse_check_op(["--check-op", "SoftmaxFwdOp"]) == "SoftmaxFwdOp"
        assert validator._parse_check_op(["--check-op=SoftmaxFwdOp"]) == "SoftmaxFwdOp"
        assert validator._parse_check_op(["--verbose"]) is None
        assert validator._parse_check_op([]) is None
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
        fake_mod = _fake_op_module(
            "tileops.ops.fake_ambiguous", ["AlphaKernel", "BetaKernel"],
        )
        with (
            _patched_import(fake_mod),
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
        fake_mod = _fake_op_module(
            "tileops.ops.fake_ambiguous", ["AlphaKernel", "BetaKernel"],
        )
        entry = {
            "source": {"op": "tileops/ops/fake_ambiguous.py"},
            "signature": {"inputs": {}, "params": {}},
        }
        warn_list: list[str] = []
        with (
            _patched_import(fake_mod),
            pytest.warns(UserWarning, match="No class named"),
        ):
            errors = validator.check_l1("mystery_fwd", entry, warnings=warn_list)

        assert any("No class named" in w for w in warn_list)
        assert any("could not resolve" in e for e in errors)

    def test_direct_match_resolves_exact_class_name(self, validator):
        """Direct match resolves cls.__name__ == manifest key even next
        to sibling candidates. No heuristic fallback."""
        fake_mod = _fake_op_module(
            "tileops.ops.fake_priority", ["_SumHelper", "SumFwdOp"],
        )
        with _patched_import(fake_mod):
            result = validator._resolve_op_class(
                "tileops/ops/fake_priority.py", "SumFwdOp",
            )
        assert result.cls is fake_mod.SumFwdOp, result.cls


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
        """A malformed dim default (``dim=["2"]``) raises TypeError from
        the helper; the validator classifies it as an eval-error warning
        (parity skip), bit-identical to the equivalent inline form."""
        def infer(self, x_shape, *, dim=None, keepdim=False):
            return {"y": x_shape}

        params = {
            "dim": {
                "type": "int | list[int] | tuple[int, ...] | None",
                "default": ["2"],
            },
            "keepdim": {"type": "bool", "default": False},
        }
        sig_inline = _sig(
            {"x": "float16"}, {"y": "same_as(x)"}, params=params,
            shape_rules=[
                "dim is None or all(-x.ndim <= d < x.ndim for d in "
                "([dim] if isinstance(dim, int) else dim))",
                "isinstance(dim, (int, type(None))) or "
                "len({d % x.ndim for d in dim}) == len(dim)",
            ],
        )
        sig_helper = _sig(
            {"x": "float16"}, {"y": "same_as(x)"}, params=params,
            shape_rules=[
                "dim_range_validity(x, dim)",
                "dim_uniqueness(x, dim)",
            ],
        )
        errs_inline, warn_inline = _infer_parity(
            validator, infer, sig_inline, name="HelperMalformedDimOp",
        )
        errs_helper, warn_helper = _infer_parity(
            validator, infer, sig_helper, name="HelperMalformedDimOp",
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
        """Out-of-range default ``dim`` classifies as an input problem:
        ``errors == []`` plus exactly one input-only-precondition warning
        citing the helper rule — never parity blame on the impl."""
        def infer(self, x_shape):
            return {"y": x_shape}

        # Rank-2 mock input; default dim=9 is out of range, so the helper
        # rule must fail under mock evaluation.
        sig = _sig(
            {"x": "float16"}, {"y": "same_as(x)"},
            params={"dim": {"type": "int", "default": 9}},
            shape_rules=[
                "x.shape == (B, S)",
                "dim_range_validity(x, dim)",
                "y.shape == x.shape",
            ],
        )
        errors, warnings = _infer_parity(
            validator, infer, sig, name="HelperBadDimOp",
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
        """``_input_bound_symbols`` treats malformed inputs as empty
        instead of crashing (schema layer owns the diagnostics)."""
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
        """Every ``tileops.manifest.shape_rules.__all__`` helper is
        callable by bare name from rule bodies; fails loudly when
        ``__all__`` and ``_SHAPE_RULE_BUILTIN_PAIRS`` drift apart."""
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


def _strict_op(name, init=None, forward=None, **methods):
    """Op subclass with lambda-supplied ``__init__`` / ``forward``.

    C3 / C4 / C6 / C7 checks read signatures and method identity only,
    so lambdas suffice. (C5 parses ``__init__`` source and needs real
    ``def`` statements — see TestDispatchKernelInvariant.)
    """
    from tileops.ops.op_base import Op

    ns = {
        "forward": forward if forward is not None else (lambda self, x: None),
        "default_kernel_map": property(lambda self: {}),
    }
    if init is not None:
        ns["__init__"] = init
    ns.update(methods)
    return type(name, (Op,), ns)


class TestCtorSignatureParity:
    """C3: ctor signature parity (defaults + kw-only)."""

    def test_matching_defaults_pass(self, validator):
        cls = _strict_op(
            "Op1", init=lambda self, dim=-1, eps=1e-6, kernel_map=None: None,
        )
        entry = {"signature": {
            "params": {
                "dim": {"type": "int", "default": -1},
                "eps": {"type": "float", "default": 1e-6},
            },
        }}
        assert validator.check_c3_ctor_signature_parity("Op1", entry, cls) == []

    def test_dtype_default_compared_as_dtype(self, validator):
        """A ``torch.dtype`` default is spelled by name in YAML.

        ``torch.float16 != "float16"``, so comparing the spellings would make
        every dtype default a mismatch and leave the manifest unable to declare
        one at all.
        """
        cls = _strict_op(
            "OpDtypeDefault",
            init=lambda self, dtype=torch.float16, kernel_map=None: None,
        )
        entry = {"signature": {
            "params": {"dtype": {"type": "torch.dtype", "default": "float16"}},
        }}
        assert validator.check_c3_ctor_signature_parity(
            "OpDtypeDefault", entry, cls,
        ) == []

        # A genuinely different dtype still fails.
        entry_bad = {"signature": {
            "params": {"dtype": {"type": "torch.dtype", "default": "bfloat16"}},
        }}
        errs = validator.check_c3_ctor_signature_parity(
            "OpDtypeDefault", entry_bad, cls,
        )
        assert any("default mismatch" in e for e in errs), errs

    def test_ctor_mismatches_fail(self, validator):
        """Case table: missing default, undeclared ctor default, kw-only."""
        cases = [
            ("param default missing on __init__",
             _strict_op("OpNoDefault",
                        init=lambda self, dim, kernel_map=None: None),
             {"dim": {"type": "int", "default": -1}},
             "no default on __init__"),
            ("ctor default the manifest does not declare",
             _strict_op("OpUndeclaredDefault",
                        init=lambda self, num_experts=0, kernel_map=None: None),
             {"num_experts": {"type": "int"}},
             "no manifest default"),
            ("kw_only mismatch",
             _strict_op("OpKwOnly",
                        init=lambda self, *, dim=-1, kernel_map=None: None),
             {"dim": {"type": "int", "default": -1, "kw_only": False}},
             "kw_only mismatch"),
        ]
        for desc, cls, params, substring in cases:
            entry = {"signature": {"params": params}}
            errs = validator.check_c3_ctor_signature_parity(
                cls.__name__, entry, cls,
            )
            assert any(substring in e for e in errs), (desc, errs)

    def test_forward_order_matrix(self, validator):
        """Matching order passes; swapped positional names fail."""
        entry = {"signature": {
            "inputs": {"x": {"dtype": "float16"}, "weight": {"dtype": "float16"}},
        }}
        cls = _strict_op("Op1", forward=lambda self, x, weight: None)
        assert validator.check_c4_forward_signature_parity(
            "Op1", entry, cls,
        ) == []

        cls = _strict_op("Op2", forward=lambda self, weight, x: None)  # swapped
        errs = validator.check_c4_forward_signature_parity("Op2", entry, cls)
        assert any("do not start with" in e for e in errs), errs


class TestDispatchKernelInvariant:
    """C5: ``__init__`` complies with Slot S12 (kernel_map kwarg) + S13
    (body calls ``self.dispatch_kernel``). Pure static check on the
    Op subclass's source — no runtime construction."""

    def test_compliant_ctor_forms_pass(self, validator):
        """Case table: explicit kwarg, **kwargs absorption, and a
        branch-nested dispatch_kernel call all satisfy S12+S13. Pure
        static inspection of ``__init__`` — plain classes suffice."""
        class GoodOp:
            def __init__(self, kernel_map=None):
                self.dispatch_kernel(kernel_map)

        class VarKwOp:
            # ``**kwargs`` absorbs ``kernel_map`` and satisfies S12.
            def __init__(self, **kwargs):
                self.dispatch_kernel(kwargs.get("kernel_map"))

        class BranchOp:
            # The S13 walker is AST-recursive; a call inside a branch counts.
            def __init__(self, kernel_map=None, fast=False):
                if fast:
                    self.dispatch_kernel(kernel_map)
                else:
                    self.dispatch_kernel(kernel_map)

        for cls in (GoodOp, VarKwOp, BranchOp):
            assert validator.check_c5_dispatch_kernel_invariant(
                cls.__name__, {}, cls,
            ) == [], cls.__name__

    def test_non_compliant_ctor_forms_fail(self, validator):
        """Case table: dropped override, missing kwarg, and helper-only
        dispatch each violate the invariant."""
        class SilentDropOp:
            # S13 violation: kwarg accepted but the body never calls
            # ``self.dispatch_kernel`` — the override is silently dropped.
            def __init__(self, kernel_map=None):
                pass

        class NoKwargOp:
            # S12 violation: ``__init__`` does not accept ``kernel_map``.
            def __init__(self): pass

        class HelperOp:
            # S13 requires dispatch_kernel in __init__ or super().__init__.
            def __init__(self, kernel_map=None):
                self._prepare(kernel_map)
            def _prepare(self, kernel_map=None):
                self.dispatch_kernel(kernel_map)

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
        cls = _strict_op("StubOp", init=lambda self: None)
        errs = validator.check_c6_validate_dtypes_not_stub("StubOp", {}, cls)
        assert any("is the Op base stub" in e for e in errs), errs
        errs = validator.check_c7_eval_roofline_not_stub("StubOp", {}, cls)
        assert any("is the Op base stub" in e for e in errs), errs

    def test_overrides_pass(self, validator):
        cls = _strict_op(
            "OverriddenOp", init=lambda self: None,
            _validate_dtypes=lambda self, *args: None,
            eval_roofline=lambda self: (0, 0),
        )
        assert validator.check_c6_validate_dtypes_not_stub(
            "OverriddenOp", {}, cls,
        ) == []
        assert validator.check_c7_eval_roofline_not_stub(
            "OverriddenOp", {}, cls,
        ) == []


class TestStrictAdvisoryMode:
    """Advisory vs strict routing of C1-C7 failures through
    ``validate_manifest()``, driven by a stub-only synthetic manifest so
    the outcome is independent of the checked-in strict-parity backlog.
    """

    @pytest.fixture
    def stub_setup(self, tmp_path, monkeypatch, validator):
        """Synthetic single-file manifest wired to an in-process Op
        fixture failing C6/C7; the op-class resolver is monkeypatched so
        no importable module is needed."""
        from tileops.ops.op_base import Op

        class StubOp(Op):
            def __init__(self, N, dtype):
                self.N = N
                self.dtype = dtype
            def forward(self, x):
                return x
            @property
            def default_kernel_map(self):
                return {}

        # Schema-valid synthetic manifest entry: all required top-level
        # fields present so the test would still parse if schema checks
        # were enabled.
        entry = {
            "family": "synth",
            "status": "implemented",
            "ref_api": "https://example.invalid/stub",
            "signature": _sig(
                {"x": "float16"}, {"y": "same_as(x)"},
                params={"N": {"type": "int"},
                        "dtype": {"type": "torch.dtype"}},
                shape_rules=["y.shape == x.shape"],
            ),
            "workloads": [],
            "roofline": {"flops": "N", "bytes": "2 * N"},
            "source": {
                "op": "tests/__strict_parity_stub__.py",
                "kernel": "tileops/kernels/__strict_parity_stub__.py",
                "bench": "benchmarks/ops/__strict_parity_stub__.py",
                "test": "tests/__strict_parity_stub_test__.py",
                "kernel_map": {"stub": "tileops.kernels.StubKernel"},
            },
        }
        manifest_yaml = _write_manifest(tmp_path, {"StubOp": entry})

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
        # checks exercised here are the strict-parity ones (C5-C7).
        levels = frozenset({"signature", "shape", "dtype", "bench"})
        errors, warnings = validator.validate_manifest(
            manifest_path=stub_setup, strict_parity=False, levels=levels,
        )
        # No strict-parity-only tag may appear in errors. Use
        # STRICT_ONLY_TAGS, not STRICT_TAGS: ``[shape]`` / ``[dtype]``
        # are also emitted by non-strict L2 / L3 checks and may
        # legitimately reach errors regardless of advisory mode.
        leaked = [e for e in errors if any(t in e for t in validator.STRICT_ONLY_TAGS)]
        assert not leaked, leaked
        # At least one strict-parity warning was raised (C6/C7 fixture
        # is guaranteed to fail both).
        strict_warnings = [
            w for w in warnings if "STRICT-PARITY (advisory)" in w
        ]
        assert strict_warnings, warnings

    def test_strict_routes_failures_to_errors(
        self, validator, stub_setup,
    ):
        """Strict mode routes the same failures to errors with no advisory prefix."""
        levels = frozenset({"signature", "shape", "dtype", "bench"})
        errors, warnings = validator.validate_manifest(
            manifest_path=stub_setup, strict_parity=True, levels=levels,
        )
        strict_errors = [e for e in errors if "[stub]" in e]
        assert strict_errors, errors
        assert not any(
            "STRICT-PARITY (advisory)" in w for w in warnings
        ), warnings


class TestCompileContractRegistry:
    """Enforcement point for the torch_compile_fullgraph contract.

    Must stay in this file: the always-on ``compile-contract-gate``
    preflight job runs pytest on this file on a CPU runner.
    """

    def test_declarations_match_registered_evidence(self):
        """Manifest declarations == registered compile-test evidence;
        broken registration or typo'd op names surface as a set diff."""
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
