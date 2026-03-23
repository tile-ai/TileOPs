"""Validation tests for ops_manifest.yaml."""

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


class TestManifestStructure:
    """AC-1: ops_manifest.yaml exists at repo root and is valid YAML."""

    def test_manifest_exists(self):
        assert MANIFEST_PATH.exists()

    def test_manifest_is_valid_yaml(self, manifest):
        assert manifest is not None

    def test_manifest_has_ops_key(self, manifest):
        assert "ops" in manifest, "Manifest must have top-level 'ops' key"
        assert isinstance(manifest["ops"], dict)


class TestRmsNormFwdEntry:
    """AC-2: rmsnorm_fwd entry contains all four fields."""

    REQUIRED_FIELDS = {"signature", "workloads", "roofline", "source"}

    def test_rmsnorm_fwd_exists(self, manifest):
        assert "rmsnorm_fwd" in manifest["ops"], (
            "rmsnorm_fwd entry missing from manifest"
        )

    def test_rmsnorm_fwd_has_all_fields(self, manifest):
        entry = manifest["ops"]["rmsnorm_fwd"]
        missing = self.REQUIRED_FIELDS - set(entry.keys())
        assert not missing, f"rmsnorm_fwd missing fields: {missing}"

    def test_signature_has_inputs(self, manifest):
        sig = manifest["ops"]["rmsnorm_fwd"]["signature"]
        assert "inputs" in sig
        assert isinstance(sig["inputs"], list)
        assert len(sig["inputs"]) >= 2, "rmsnorm_fwd needs at least x and weight inputs"

    def test_signature_has_outputs(self, manifest):
        sig = manifest["ops"]["rmsnorm_fwd"]["signature"]
        assert "outputs" in sig
        assert isinstance(sig["outputs"], list)
        assert len(sig["outputs"]) >= 1

    def test_signature_has_params(self, manifest):
        sig = manifest["ops"]["rmsnorm_fwd"]["signature"]
        assert "params" in sig
        assert isinstance(sig["params"], list)

    def test_shape_rules_are_valid_expressions(self, manifest):
        """Constraint: shape_rules must use Python expression syntax (compilable with mode='eval')."""
        sig = manifest["ops"]["rmsnorm_fwd"]["signature"]
        if "shape_rules" not in sig:
            pytest.skip("No shape_rules defined")
        for rule in sig["shape_rules"]:
            try:
                compile(rule, "<shape_rule>", "eval")
            except SyntaxError as exc:
                pytest.fail(
                    f"shape_rule is not a valid Python expression: {rule!r} ({exc})"
                )

    def test_roofline_field(self, manifest):
        roofline = manifest["ops"]["rmsnorm_fwd"]["roofline"]
        assert isinstance(roofline, dict)
        # Must have either both flops and memory, or func
        has_inline = "flops" in roofline and "memory" in roofline
        has_func = "func" in roofline
        assert has_inline or has_func, (
            "roofline must have both inline expressions (flops and memory) or func"
        )

    def test_roofline_flops_matches_benchmark(self, manifest):
        """Ensure manifest FLOPs formula is consistent with the benchmark cost model."""
        roofline = manifest["ops"]["rmsnorm_fwd"]["roofline"]
        flops_expr = roofline["flops"]
        M, N = 2048, 4096  # noqa: N806
        manifest_flops = eval(flops_expr)  # noqa: S307
        # Benchmark: benchmarks/ops/bench_rms_norm.py calculate_flops() = 4 * M * N
        benchmark_flops = 4 * M * N
        assert manifest_flops == benchmark_flops, (
            f"Manifest flops ({flops_expr}={manifest_flops}) != "
            f"benchmark flops (4*M*N={benchmark_flops})"
        )

    def test_source_field(self, manifest):
        source = manifest["ops"]["rmsnorm_fwd"]["source"]
        assert isinstance(source, dict)
        assert "kernel" in source
        assert "op" in source


class TestSourcePaths:
    """AC-3: All file paths in source field point to existing files."""

    def test_all_source_paths_exist(self, manifest):
        source = manifest["ops"]["rmsnorm_fwd"]["source"]
        for key, rel_path in source.items():
            full_path = REPO_ROOT / rel_path
            assert full_path.exists(), (
                f"source.{key} path does not exist: {rel_path}"
            )


class TestWorkloads:
    """AC-4: Workloads cover Llama-3.1-8B/70B/405B with prefill and decode."""

    REQUIRED_HIDDEN_DIMS = {4096, 8192, 16384}

    def test_workloads_is_list(self, manifest):
        wl = manifest["ops"]["rmsnorm_fwd"]["workloads"]
        assert isinstance(wl, list)
        assert len(wl) >= 6, (
            "Need at least 6 workloads (3 models x 2 phases)"
        )

    def test_covers_all_hidden_dims(self, manifest):
        wl = manifest["ops"]["rmsnorm_fwd"]["workloads"]
        hidden_dims = set()
        for entry in wl:
            assert "hidden" in entry, f"Workload entry missing 'hidden': {entry}"
            hidden_dims.add(entry["hidden"])
        missing = self.REQUIRED_HIDDEN_DIMS - hidden_dims
        assert not missing, f"Missing hidden dims: {missing}"

    @staticmethod
    def _get_phase(entry):
        """Return the phase of a workload entry, inferring from seq_len if needed."""
        if "phase" in entry:
            return entry["phase"]
        if "seq_len" in entry:
            return "prefill" if entry["seq_len"] >= 512 else "decode"
        return None

    def test_has_prefill_and_decode(self, manifest):
        wl = manifest["ops"]["rmsnorm_fwd"]["workloads"]
        phases = {self._get_phase(e) for e in wl} - {None}
        assert "prefill" in phases, "No prefill workload found"
        assert "decode" in phases, "No decode workload found"

    def test_each_hidden_dim_has_both_phases(self, manifest):
        wl = manifest["ops"]["rmsnorm_fwd"]["workloads"]
        # Group by hidden dim
        from collections import defaultdict
        dim_phases = defaultdict(set)
        for entry in wl:
            h = entry["hidden"]
            phase = self._get_phase(entry)
            if phase is not None:
                dim_phases[h].add(phase)
        for h in self.REQUIRED_HIDDEN_DIMS:
            assert "prefill" in dim_phases[h], (
                f"hidden={h} missing prefill workload"
            )
            assert "decode" in dim_phases[h], (
                f"hidden={h} missing decode workload"
            )

    def test_no_reference_field(self, manifest):
        """Constraint: No reference field in manifest."""
        entry = manifest["ops"]["rmsnorm_fwd"]
        assert "reference" not in entry, (
            "Manifest must not contain a reference field"
        )
