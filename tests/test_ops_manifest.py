"""Schema validation for ops_manifest.yaml.

Validates structural invariants across all ops in the manifest.
Not op-specific — tests apply to every entry.
"""

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "ops_manifest.yaml"


@pytest.fixture(scope="module")
def manifest():
    """Load and parse the ops manifest."""
    assert MANIFEST_PATH.exists(), f"ops_manifest.yaml not found at {MANIFEST_PATH}"
    with open(MANIFEST_PATH) as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), "Manifest root must be a YAML mapping"
    return data


@pytest.fixture(scope="module")
def all_ops(manifest):
    """Return the ops dict from the manifest."""
    assert "ops" in manifest, "Manifest must have top-level 'ops' key"
    assert isinstance(manifest["ops"], dict)
    return manifest["ops"]


class TestManifestStructure:
    """Manifest file exists, parses, and has the expected top-level structure."""

    def test_manifest_exists(self):
        assert MANIFEST_PATH.exists()

    def test_manifest_is_valid_yaml(self, manifest):
        assert manifest is not None

    def test_has_ops_key(self, manifest):
        assert "ops" in manifest
        assert isinstance(manifest["ops"], dict)


class TestOpSchema:
    """Every op entry has the required fields and valid sub-structure."""

    REQUIRED_TOP_FIELDS = {"family", "signature", "workloads", "roofline", "source"}

    def test_every_op_has_required_fields(self, all_ops):
        for op_name, entry in all_ops.items():
            missing = self.REQUIRED_TOP_FIELDS - set(entry.keys())
            assert not missing, f"{op_name} missing fields: {missing}"

    def test_every_signature_has_inputs_and_outputs(self, all_ops):
        for op_name, entry in all_ops.items():
            sig = entry["signature"]
            assert "inputs" in sig, f"{op_name}: signature missing 'inputs'"
            assert isinstance(sig["inputs"], dict), f"{op_name}: inputs must be a dict"
            assert len(sig["inputs"]) >= 1, f"{op_name}: must have at least 1 input"
            assert "outputs" in sig, f"{op_name}: signature missing 'outputs'"
            assert isinstance(sig["outputs"], dict), f"{op_name}: outputs must be a dict"

    def test_every_roofline_has_valid_mode(self, all_ops):
        for op_name, entry in all_ops.items():
            roofline = entry["roofline"]
            has_inline = "flops" in roofline and "bytes" in roofline
            has_func = "func" in roofline
            assert has_inline or has_func, (
                f"{op_name}: roofline must have (flops + bytes) or func"
            )

    def test_every_tensor_has_name_and_dtype(self, all_ops):
        for op_name, entry in all_ops.items():
            sig = entry["signature"]
            for direction in ("inputs", "outputs"):
                for i, tensor in enumerate(sig[direction]):
                    assert "name" in tensor, (
                        f"{op_name}: {direction}[{i}] missing 'name'"
                    )
                    assert "dtype" in tensor, (
                        f"{op_name}: {direction}[{i}] missing 'dtype'"
                    )

    def test_every_output_has_explicit_shape(self, all_ops):
        for op_name, entry in all_ops.items():
            sig = entry["signature"]
            has_shape_rules = "shape_rules" in sig and len(sig["shape_rules"]) > 0
            for tensor in sig["outputs"]:
                name = tensor.get("name", "?")
                has_shape = "shape" in tensor
                assert has_shape or has_shape_rules, (
                    f"{op_name}: output '{name}' must have 'shape' or "
                    f"signature must have 'shape_rules'"
                )

    def test_constraints_keys_match_shape_dims(self, all_ops):
        for op_name, entry in all_ops.items():
            sig = entry["signature"]
            for direction in ("inputs", "outputs"):
                for tensor in sig[direction]:
                    if "constraints" not in tensor:
                        continue
                    name = tensor.get("name", "?")
                    assert "shape" in tensor, (
                        f"{op_name}: tensor '{name}' has constraints "
                        f"but no shape"
                    )
                    shape_str = tensor["shape"]
                    # Extract dimension names from "[D1, D2, ...]"
                    dims = {
                        d.strip()
                        for d in shape_str.strip("[]").split(",")
                        if d.strip()
                    }
                    for ckey in tensor["constraints"]:
                        assert ckey in dims, (
                            f"{op_name}: tensor '{name}' constraint key "
                            f"'{ckey}' not in shape dims {dims}"
                        )

    def test_dtype_combos_reference_declared_tensors(self, all_ops):
        for op_name, entry in all_ops.items():
            sig = entry["signature"]
            if "dtype_combos" not in sig:
                continue
            declared = {
                t["name"]
                for t in sig["inputs"] + sig["outputs"]
                if "name" in t
            }
            for i, combo in enumerate(sig["dtype_combos"]):
                assert isinstance(combo, dict), (
                    f"{op_name}: dtype_combos[{i}] must be a mapping"
                )
                for tname in combo:
                    assert tname in declared, (
                        f"{op_name}: dtype_combos[{i}] references "
                        f"unknown tensor '{tname}'"
                    )

    def test_shape_rules_are_list_of_strings(self, all_ops):
        for op_name, entry in all_ops.items():
            sig = entry["signature"]
            if "shape_rules" not in sig:
                continue
            rules = sig["shape_rules"]
            assert isinstance(rules, list), (
                f"{op_name}: shape_rules must be a list"
            )
            for i, rule in enumerate(rules):
                assert isinstance(rule, str), (
                    f"{op_name}: shape_rules[{i}] must be a string"
                )

    def test_shape_rules_are_valid_expressions(self, all_ops):
        for op_name, entry in all_ops.items():
            sig = entry["signature"]
            if "shape_rules" not in sig:
                continue
            for rule in sig["shape_rules"]:
                try:
                    compile(rule, "<shape_rule>", "eval")
                except SyntaxError as exc:
                    pytest.fail(
                        f"{op_name}: invalid shape_rule: {rule!r} ({exc})"
                    )


class TestSourcePaths:
    """All source paths point to existing files."""

    def test_all_source_paths_exist(self, all_ops):
        for op_name, entry in all_ops.items():
            source = entry["source"]
            for key, rel_path in source.items():
                full_path = REPO_ROOT / rel_path
                assert full_path.is_file(), (
                    f"{op_name}: source.{key} is not a file: {rel_path}"
                )
