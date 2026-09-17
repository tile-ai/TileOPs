"""Tests for the programmatic manifest API (tileops.manifest).

Schema policies for manifest entries are owned by scripts/validate_manifest.py
(see tests/test_validate_manifest.py); this file covers only the package's
load/merge surface.
"""

from pathlib import Path

import pytest

from tileops.manifest import load_manifest, load_workloads, manifest_files

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = REPO_ROOT / "src" / "tileops" / "manifest"


class TestManifestStructure:
    """Manifest package exists and contributes at least one family file."""

    def test_manifest_dir_exists(self):
        assert MANIFEST_DIR.is_dir()

    def test_manifest_has_family_files(self):
        files = manifest_files()
        assert len(files) >= 1
        assert all(p.name.endswith(".yaml") for p in files)

    def test_manifest_loads(self):
        ops = load_manifest()
        assert isinstance(ops, dict)
        assert ops


class TestManifestAPI:
    """Load helpers accept only canonical PascalCase op keys."""

    def test_load_workloads_returns_list(self):
        workloads = load_workloads("RMSNormFwdOp")
        assert isinstance(workloads, list)
        assert len(workloads) >= 1
        assert "x_shape" in workloads[0]

    def test_load_workloads_unknown_op_raises(self):
        with pytest.raises(KeyError, match="NonexistentOp"):
            load_workloads("NonexistentOp")

    def test_load_workloads_snake_case_raises(self):
        """Legacy snake_case names are not resolved."""
        with pytest.raises(KeyError, match="rmsnorm_fwd"):
            load_workloads("rmsnorm_fwd")

    def test_manifest_does_not_expose_roofline_evaluator(self):
        import tileops.manifest as manifest

        for name in (
            "_safe_eval",
            "eval_roofline",
            "has_roofline_vars",
            "resolve_roofline_vars",
        ):
            assert not hasattr(manifest, name)


class TestForwardTensorArgumentCount:
    """A workspace is a forward argument wherever the manifest is read positionally.

    ``Op._refuse_empty_input`` zips the manifest's tensor names against the call's
    tensors and skips when the two lengths disagree, so an entry that declares a
    workspace outside ``signature.inputs`` silently loses the empty-input guard
    unless the workspace names are counted too.
    """

    @staticmethod
    def _manifest_tensor_names(entry: dict) -> list[str]:
        workspaces = (entry.get("resources") or {}).get("workspaces") or []
        return list(entry["signature"]["inputs"]) + [w["name"] for w in workspaces]

    @staticmethod
    def _op_class(op_name: str, entry: dict):
        import importlib

        module = entry["source"]["op"].removesuffix(".py").replace("/", ".")
        return getattr(importlib.import_module(module), op_name)

    def test_every_entry_with_workspaces_matches_its_forward_arity(self):
        import inspect

        checked = 0
        for op_name, entry in load_manifest().items():
            if entry.get("status") != "implemented" or not entry.get("resources"):
                continue
            cls = self._op_class(op_name, entry)
            params = [
                p.name
                for p in inspect.signature(cls.forward).parameters.values()
                if p.name != "self" and p.kind is not inspect.Parameter.VAR_KEYWORD
            ]
            expected = self._manifest_tensor_names(entry)
            assert params[: len(expected)] == expected, op_name
            checked += 1
        assert checked, "no implemented entry declares resources.workspaces"


class TestEmptyInputGuardCountsWorkspaces:
    """The guard must still fire for an op whose workspaces live under ``resources``.

    It compares the manifest's tensor names against the call's tensors by length and
    returns early when they disagree, so counting only ``signature.inputs`` would
    disable the refusal for every op that declares a workspace.
    """

    def test_guard_fires_with_workspaces_declared_outside_inputs(self):
        import torch

        from tileops.ops.moe.routed_expert.fused_routed_expert import FusedMoEExpertsFwdOp

        entry = load_manifest()["FusedMoEExpertsFwdOp"]
        assert entry["resources"]["workspaces"], "entry no longer exercises this path"

        op = FusedMoEExpertsFwdOp.__new__(FusedMoEExpertsFwdOp)
        empty = torch.zeros(0, 8)
        rest = torch.zeros(1, 8)
        # output, hidden_states, w_gate_up, w_down, topk_weights, topk_ids,
        # workspace1, workspace2 — eight tensors, six of them manifest inputs.
        call = (empty, empty, rest, rest, rest, rest, rest, rest)

        with pytest.raises(ValueError, match="does not support an empty tensor"):
            op._refuse_empty_input(call)
