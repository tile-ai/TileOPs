"""Validation tests for attention op entries in ops_manifest.yaml.

Tests AC-1 through AC-4 for issue #718: attention operator family manifest entries.
"""

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "tileops" / "ops_manifest.yaml"

ATTENTION_OPS = [
    "mha_fwd",
    "mha_bwd",
    "gqa_fwd",
    "gqa_bwd",
    "mha_decode",
    "mha_decode_paged",
    "gqa_decode",
    "gqa_decode_paged",
    "gqa_sliding_window_fwd",
    "gqa_sliding_window_varlen_fwd",
    "deepseek_mla_decode",
    "deepseek_dsa_decode",
]


@pytest.fixture(scope="module")
def manifest():
    with open(MANIFEST_PATH) as f:
        data = yaml.safe_load(f)
    return data


@pytest.fixture(scope="module")
def all_ops(manifest):
    return manifest["ops"]


class TestAttentionOpsExist:
    """AC-1: All 12 attention ops present with status: implemented."""

    def test_all_12_attention_ops_present(self, all_ops):
        missing = [op for op in ATTENTION_OPS if op not in all_ops]
        assert not missing, f"Missing attention ops: {missing}"

    @pytest.mark.parametrize("op_name", ATTENTION_OPS)
    def test_attention_op_family_is_attention(self, all_ops, op_name):
        assert all_ops[op_name]["family"] == "attention"

    @pytest.mark.parametrize("op_name", ATTENTION_OPS)
    def test_attention_op_has_status_implemented(self, all_ops, op_name):
        assert all_ops[op_name].get("status") == "implemented", (
            f"{op_name} must have status: implemented"
        )


class TestStatusField:
    """AC-2: status field documented and present on all ops."""

    def test_schema_header_documents_status(self):
        """The manifest schema header comment must mention the status field."""
        text = MANIFEST_PATH.read_text()
        # Extract all comment lines at the top of the file (schema header)
        header_lines = []
        for line in text.splitlines():
            if line.startswith("#") or line.strip() == "":
                header_lines.append(line)
            else:
                break
        header = "\n".join(header_lines)
        assert "status" in header.lower(), (
            "Schema header must document the 'status' field"
        )

    def test_all_existing_norm_ops_have_status(self, all_ops):
        """All existing norm ops must have status: implemented."""
        norm_ops = {k: v for k, v in all_ops.items() if v["family"] == "norm"}
        assert len(norm_ops) > 0, "Expected at least one norm op"
        for op_name, entry in norm_ops.items():
            assert entry.get("status") == "implemented", (
                f"Norm op {op_name} must have status: implemented"
            )


class TestFormulasStubs:
    """AC-3: formulas.py exists with stub functions for attention roofline."""

    def test_formulas_module_importable(self):
        from tileops.perf import formulas
        assert formulas is not None

    @pytest.mark.parametrize("op_name", ATTENTION_OPS)
    def test_each_attention_op_has_roofline_func(self, all_ops, op_name):
        """Each attention op must reference a func in roofline section."""
        roofline = all_ops[op_name]["roofline"]
        assert "func" in roofline, (
            f"{op_name}: attention ops must use func-mode roofline"
        )
        func_path = roofline["func"]
        assert func_path.startswith("tileops.perf.formulas."), (
            f"{op_name}: func must reference tileops.perf.formulas module"
        )

    @pytest.mark.parametrize("op_name", ATTENTION_OPS)
    def test_roofline_func_is_callable(self, all_ops, op_name):
        """Referenced roofline function must exist and be callable."""
        import importlib

        roofline = all_ops[op_name]["roofline"]
        func_path = roofline["func"]
        module_path, func_name = func_path.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        func = getattr(mod, func_name, None)
        assert callable(func), (
            f"{op_name}: {func_path} must be a callable function"
        )

    @pytest.mark.parametrize("op_name", ATTENTION_OPS)
    def test_roofline_stub_returns_dict(self, all_ops, op_name):
        """Stub function must return dict with 'flops' and 'bytes' keys."""
        import importlib

        roofline = all_ops[op_name]["roofline"]
        func_path = roofline["func"]
        module_path, func_name = func_path.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        func = getattr(mod, func_name)
        result = func()
        assert isinstance(result, dict), f"{func_path} must return a dict"
        assert "flops" in result, f"{func_path} result must have 'flops'"
        assert "bytes" in result, f"{func_path} result must have 'bytes'"


class TestYamlValidity:
    """AC-4: Manifest is valid YAML."""

    def test_yaml_loads_cleanly(self):
        with open(MANIFEST_PATH) as f:
            data = yaml.safe_load(f)
        assert data is not None
        assert "ops" in data


class TestWorkloadCoverage:
    """Workloads include 8B + 70B configs, short + long sequences."""

    @pytest.mark.parametrize("op_name", ATTENTION_OPS)
    def test_at_least_two_workloads(self, all_ops, op_name):
        workloads = all_ops[op_name]["workloads"]
        assert len(workloads) >= 2, f"{op_name} must have >= 2 workloads"

    @pytest.mark.parametrize("op_name", ATTENTION_OPS)
    def test_workloads_have_dtypes(self, all_ops, op_name):
        for wl in all_ops[op_name]["workloads"]:
            assert "dtypes" in wl, f"{op_name}: every workload must have dtypes"

    @pytest.mark.parametrize("op_name", ATTENTION_OPS)
    def test_workloads_have_labels(self, all_ops, op_name):
        for wl in all_ops[op_name]["workloads"]:
            assert "label" in wl, f"{op_name}: every workload should have a label"
