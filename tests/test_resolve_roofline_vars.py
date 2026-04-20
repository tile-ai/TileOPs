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
    """The evaluator runs against project-owned expressions but the
    sandbox must still reject every known class of Python-level escape so
    that a mis-authored or tampered manifest cannot execute arbitrary
    code."""

    def _poison(self, monkeypatch, expr: str) -> None:
        from tileops.manifest import _load_manifest

        _load_manifest.cache_clear()
        real = _load_manifest()
        poisoned = dict(real)
        poisoned["__EvilOp"] = {
            "roofline": {
                "vars": {"pwn": expr},
                "flops": "0",
                "bytes": "0",
            }
        }
        monkeypatch.setattr(
            "tileops.manifest._load_manifest", lambda: poisoned
        )

    @pytest.mark.parametrize(
        "expr",
        [
            # __import__ direct call (historic escape vector)
            "__import__('os')",
            # Dunder attribute traversal used by classic eval escapes
            "().__class__",
            "().__class__.__base__.__subclasses__()",
            "(1).__class__",
            # Builtins probe
            "__builtins__",
            "__builtins__['__import__']",
            # getattr / open / import keyword
            "getattr(x, 'shape')",
            "open('/etc/passwd')",
            # Immediately-invoked lambda (rejected because Lambda is not
            # on the whitelist)
            "(lambda: 0)()",
            # Attribute on a bare int — not a shape proxy, should still
            # reject before __class__ etc. is evaluated.
            "(0).__class__",
            # Subclass-traversal payload that exfiltrated os via the
            # previous eval sandbox (copilot PoC).
            "[c for c in ().__class__.__base__.__subclasses__() "
            "if c.__name__=='catch_warnings'][0]()._module."
            "__builtins__['__import__']('os').popen('echo X').read()",
        ],
    )
    def test_sandbox_rejects_escapes(self, monkeypatch, expr):
        self._poison(monkeypatch, expr)
        with pytest.raises(ValueError, match="Failed to evaluate"):
            resolve_roofline_vars(
                "__EvilOp", tensor_shapes={"x": (2, 3)}
            )

    def test_sandbox_rejects_import_statement_syntax(self, monkeypatch):
        """An ``import`` statement is not even a valid expression, but we
        assert that the failure path still routes through the standard
        ValueError instead of leaking a raw SyntaxError to callers."""
        self._poison(monkeypatch, "import os")
        with pytest.raises(ValueError, match="Failed to evaluate"):
            resolve_roofline_vars(
                "__EvilOp", tensor_shapes={"x": (2, 3)}
            )

    def test_sandbox_rejects_attribute_chain_on_shape_proxy(
        self, monkeypatch
    ):
        """Even the legit ``x`` binding must not leak object-graph access
        beyond the narrow ``.shape`` / ``.ndim`` whitelist."""
        self._poison(monkeypatch, "x.__class__")
        with pytest.raises(ValueError, match="Failed to evaluate"):
            resolve_roofline_vars(
                "__EvilOp", tensor_shapes={"x": (2, 3)}
            )


class TestNonMappingRoofline:
    """``has_roofline_vars`` / ``resolve_roofline_vars`` must handle
    manifest entries whose ``roofline`` value is not a mapping (e.g. an
    accidental list or string) without leaking AttributeError."""

    def _poison(self, monkeypatch, roofline_value):
        from tileops.manifest import _load_manifest

        _load_manifest.cache_clear()
        real = _load_manifest()
        poisoned = dict(real)
        poisoned["_OddRoofline"] = {"roofline": roofline_value}
        monkeypatch.setattr(
            "tileops.manifest._load_manifest", lambda: poisoned
        )

    def test_has_roofline_vars_false_for_string(self, monkeypatch):
        self._poison(monkeypatch, "not-a-dict")
        assert has_roofline_vars("_OddRoofline") is False

    def test_has_roofline_vars_false_for_list(self, monkeypatch):
        self._poison(monkeypatch, [])
        assert has_roofline_vars("_OddRoofline") is False

    def test_resolve_raises_valueerror_for_string(self, monkeypatch):
        self._poison(monkeypatch, "not-a-dict")
        with pytest.raises(ValueError, match="roofline.vars"):
            resolve_roofline_vars(
                "_OddRoofline", tensor_shapes={"x": (2, 3)}
            )

    def test_resolve_raises_valueerror_for_list(self, monkeypatch):
        self._poison(monkeypatch, [])
        with pytest.raises(ValueError, match="roofline.vars"):
            resolve_roofline_vars(
                "_OddRoofline", tensor_shapes={"x": (2, 3)}
            )


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
