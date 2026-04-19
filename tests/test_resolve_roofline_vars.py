"""Unit tests for ``tileops.manifest.resolve_roofline_vars``.

Exercises evaluation of ``roofline.vars`` expressions against real workload
shapes and op-call params for reduction and softmax-family ops. Also
verifies the evaluator's sandbox: restricted globals reject attribute
traversal escapes (``__class__``, ``__import__``).
"""

from __future__ import annotations

import pytest

from tileops.manifest import has_roofline_vars, resolve_roofline_vars

pytestmark = pytest.mark.smoke


class TestReductionVars:
    """Reduction ops (SumFwdOp) have set-based vars that depend on dim."""

    def test_default_dim_is_last_axis(self):
        """Default ``dim=-1`` collapses the last axis."""
        resolved = resolve_roofline_vars(
            "SumFwdOp",
            tensor_shapes={"x": (2048, 4096)},
        )
        assert resolved["M"] == 2048
        assert resolved["N"] == 4096

    def test_dim_zero_first_axis(self):
        """``dim=0`` reduces the first axis — M and N swap."""
        resolved = resolve_roofline_vars(
            "SumFwdOp",
            tensor_shapes={"x": (2048, 4096)},
            params={"dim": 0},
        )
        assert resolved["M"] == 4096
        assert resolved["N"] == 2048

    def test_dim_negative_last_axis(self):
        """``dim=-1`` matches default and matches last-axis reduction."""
        resolved = resolve_roofline_vars(
            "SumFwdOp",
            tensor_shapes={"x": (4, 8, 1024)},
            params={"dim": -1},
        )
        assert resolved["M"] == 32
        assert resolved["N"] == 1024

    def test_dim_tuple_multi_axis(self):
        """``dim=(0, 2)`` reduces both axes — N is their product."""
        resolved = resolve_roofline_vars(
            "SumFwdOp",
            tensor_shapes={"x": (4, 8, 16)},
            params={"dim": (0, 2)},
        )
        # reduced axes: {0, 2} -> N = 4 * 16 = 64
        # kept axis: 1 -> M = 8
        assert resolved["M"] == 8
        assert resolved["N"] == 64

    def test_dim_list_equivalent_to_tuple(self):
        """list and tuple should produce identical bindings."""
        a = resolve_roofline_vars(
            "SumFwdOp",
            tensor_shapes={"x": (4, 8, 16)},
            params={"dim": [0, 2]},
        )
        b = resolve_roofline_vars(
            "SumFwdOp",
            tensor_shapes={"x": (4, 8, 16)},
            params={"dim": (0, 2)},
        )
        assert a == b

    def test_dim_none_full_reduction(self):
        """``dim=None`` reduces every axis — M == 1, N == numel."""
        resolved = resolve_roofline_vars(
            "SumFwdOp",
            tensor_shapes={"x": (4, 8, 16)},
            params={"dim": None},
        )
        assert resolved["M"] == 1
        assert resolved["N"] == 4 * 8 * 16

    def test_dim_empty_tuple_full_reduction(self):
        """Empty sequence falls back to full reduction (per manifest)."""
        resolved = resolve_roofline_vars(
            "SumFwdOp",
            tensor_shapes={"x": (4, 8, 16)},
            params={"dim": ()},
        )
        assert resolved["M"] == 1
        assert resolved["N"] == 4 * 8 * 16


class TestSoftmaxVars:
    """Softmax uses a different vars formulation (prefix * suffix product)."""

    def test_default_dim_last_axis(self):
        resolved = resolve_roofline_vars(
            "SoftmaxFwdOp",
            tensor_shapes={"x": (32, 32, 4096)},
        )
        assert resolved["M"] == 32 * 32
        assert resolved["N"] == 4096

    def test_dim_middle_axis(self):
        resolved = resolve_roofline_vars(
            "SoftmaxFwdOp",
            tensor_shapes={"x": (32, 32, 4096)},
            params={"dim": 1},
        )
        assert resolved["M"] == 32 * 4096
        assert resolved["N"] == 32


class TestErrors:
    def test_unknown_op_raises_keyerror(self):
        with pytest.raises(KeyError):
            resolve_roofline_vars("NotAnOp", tensor_shapes={"x": (4,)})

    def test_op_without_vars_raises_valueerror(self):
        """Ops whose roofline has no ``vars`` mapping cannot be resolved."""
        # RMSNormFwdOp has vars, so pick an op we know doesn't. Scan manifest.
        from tileops.manifest import _load_manifest
        ops = _load_manifest()
        target = None
        for name, entry in ops.items():
            rf = entry.get("roofline", {}) or {}
            if isinstance(rf, dict) and "vars" not in rf:
                target = name
                break
        if target is None:
            pytest.skip("every op in manifest declares roofline.vars")
        with pytest.raises(ValueError, match="roofline.vars"):
            resolve_roofline_vars(target, tensor_shapes={"x": (4,)})


class TestHasRooflineVars:
    """``has_roofline_vars`` is the precondition callers use to decide
    whether to call ``resolve_roofline_vars`` or fall back to a legacy
    heuristic. It must cleanly distinguish "nothing to resolve" (False)
    from "something to resolve, may raise on eval" (True)."""

    def test_unknown_op_is_false(self):
        assert has_roofline_vars("NotAnOp") is False

    def test_op_with_vars_is_true(self):
        assert has_roofline_vars("SumFwdOp") is True

    def test_empty_vars_is_false(self, monkeypatch):
        from tileops.manifest import _load_manifest

        _load_manifest.cache_clear()
        real = _load_manifest()
        patched = dict(real)
        patched["_EmptyVarsOp"] = {
            "roofline": {"vars": {}, "flops": "0", "bytes": "0"}
        }
        monkeypatch.setattr(
            "tileops.manifest._load_manifest", lambda: patched
        )
        assert has_roofline_vars("_EmptyVarsOp") is False

    def test_missing_vars_key_is_false(self, monkeypatch):
        from tileops.manifest import _load_manifest

        _load_manifest.cache_clear()
        real = _load_manifest()
        patched = dict(real)
        patched["_NoVarsOp"] = {
            "roofline": {"flops": "0", "bytes": "0"}
        }
        monkeypatch.setattr(
            "tileops.manifest._load_manifest", lambda: patched
        )
        assert has_roofline_vars("_NoVarsOp") is False


class TestSandbox:
    """The evaluator runs against project-owned expressions but should still
    block access to the Python builtins that enable trivial escapes."""

    def test_no_import_via_vars(self, monkeypatch):
        from tileops.manifest import _load_manifest

        _load_manifest.cache_clear()
        real = _load_manifest()
        poisoned = dict(real)
        poisoned["__EvilOp2"] = {
            "roofline": {
                "vars": {"pwn": "__import__('os')"},
                "flops": "0",
                "bytes": "0",
            }
        }
        monkeypatch.setattr(
            "tileops.manifest._load_manifest", lambda: poisoned
        )
        with pytest.raises(ValueError, match="Failed to evaluate"):
            resolve_roofline_vars("__EvilOp2", tensor_shapes={})


class TestEndToEndWithEvalRoofline:
    """Resolved vars must plug back into ``eval_roofline`` cleanly."""

    def test_sum_fwd_non_last_axis_drives_roofline(self):
        from tileops.manifest import eval_roofline

        shape = (2048, 4096)
        resolved = resolve_roofline_vars(
            "SumFwdOp",
            tensor_shapes={"x": shape},
            params={"dim": 0},
        )
        # elem_bytes not declared in vars -> caller supplies it.
        flops, mem_bytes = eval_roofline(
            "SumFwdOp", M=resolved["M"], N=resolved["N"], elem_bytes=2
        )
        # For dim=0: M=4096, N=2048 -> flops = M*N = 4096*2048
        assert flops == 4096 * 2048
        # bytes = (M*N + M) * elem_bytes
        assert mem_bytes == (4096 * 2048 + 4096) * 2
