"""Tests for manifest-driven ``_validate_dtypes`` codegen.

Covers ``tileops.ops._dtype_codegen.synthesize_validate_dtypes`` and the
``Op.__init_subclass__`` auto-installation hook in ``tileops.ops.op_base``.
"""

import pytest
import torch

from tileops.ops._dtype_codegen import synthesize_validate_dtypes
from tileops.ops.op_base import Op

pytestmark = pytest.mark.smoke


def _make_op(name, sig, *, status="implemented", extra_attrs=None):
    """Build a concrete Op subclass with a manifest signature attached."""
    attrs = {
        "default_kernel_map": property(lambda self: {}),
        "forward": lambda self, *a, **kw: None,
        "__manifest_signature__": sig,
        "__manifest_status__": status,
    }
    if extra_attrs:
        attrs.update(extra_attrs)
    return type(name, (Op,), attrs)


class TestSynthesizeSimpleUnion:
    def test_accepts_listed_dtypes(self):
        sig = {"inputs": {"x": {"dtype": "float16 | bfloat16"}}}
        fn = synthesize_validate_dtypes("FooOp", sig)

        class Mock:
            pass

        m = Mock()
        m.dtype = torch.float16
        fn(m, x=torch.empty(0, dtype=torch.float16))
        fn(m, x=torch.empty(0, dtype=torch.bfloat16))

    def test_rejects_out_of_union(self):
        sig = {"inputs": {"x": {"dtype": "float16 | bfloat16"}}}
        fn = synthesize_validate_dtypes("FooOp", sig)

        class Mock:
            pass

        m = Mock()
        m.dtype = torch.float16
        with pytest.raises(ValueError):
            fn(m, x=torch.empty(0, dtype=torch.float32))


class TestSynthesizeSameAs:
    def test_same_as_matching_pair_accepted(self):
        sig = {
            "inputs": {
                "x": {"dtype": "float16 | bfloat16"},
                "y": {"dtype": "same_as(x)"},
            }
        }
        fn = synthesize_validate_dtypes("FooOp", sig)

        class Mock:
            pass

        m = Mock()
        m.dtype = torch.float16
        fn(m, x=torch.empty(0, dtype=torch.float16),
              y=torch.empty(0, dtype=torch.float16))

    def test_same_as_mismatching_pair_rejected(self):
        sig = {
            "inputs": {
                "x": {"dtype": "float16 | bfloat16"},
                "y": {"dtype": "same_as(x)"},
            }
        }
        fn = synthesize_validate_dtypes("FooOp", sig)

        class Mock:
            pass

        m = Mock()
        m.dtype = torch.float16
        with pytest.raises(ValueError):
            fn(m, x=torch.empty(0, dtype=torch.float16),
                  y=torch.empty(0, dtype=torch.bfloat16))


class TestSynthesizeBoolDtype:
    def test_bool_input_only_accepts_bool(self):
        sig = {
            "inputs": {
                "cond": {"dtype": "bool"},
                "x": {"dtype": "float16 | bfloat16 | float32"},
                "y": {"dtype": "same_as(x)"},
            }
        }
        fn = synthesize_validate_dtypes("WhereLikeOp", sig)

        class Mock:
            pass

        m = Mock()
        m.dtype = torch.float16
        fn(m,
           cond=torch.empty(0, dtype=torch.bool),
           x=torch.empty(0, dtype=torch.float16),
           y=torch.empty(0, dtype=torch.float16))
        with pytest.raises(ValueError):
            fn(m,
               cond=torch.empty(0, dtype=torch.float16),
               x=torch.empty(0, dtype=torch.float16),
               y=torch.empty(0, dtype=torch.float16))


class TestSynthesizeSignature:
    def test_keyword_arg_names_match_manifest(self):
        """The synthesized function must accept inputs as keyword arguments
        (validator probes via kwargs)."""
        import inspect
        sig = {
            "inputs": {
                "q": {"dtype": "float16 | bfloat16"},
                "k": {"dtype": "same_as(q)"},
            }
        }
        fn = synthesize_validate_dtypes("MhaOp", sig)
        params = list(inspect.signature(fn).parameters.keys())
        # Self plus q, k
        assert params == ["self", "q", "k"]


class TestSynthesizeDtypeCombos:
    """When ``dtype_combos`` is present, it is the exhaustive list of
    accepted cross-tensor dtype rows (manifest.md R6)."""

    def _mock(self):
        class Mock:
            pass

        return Mock()

    def test_listed_combo_accepted(self):
        sig = {
            "inputs": {
                "x": {"dtype": "float16 | bfloat16"},
                "weight": {"dtype": "float16 | float8_e4m3fn | bfloat16"},
            },
            "dtype_combos": [
                {"x": "float16", "weight": "float16"},
                {"x": "float16", "weight": "float8_e4m3fn"},
                {"x": "bfloat16", "weight": "bfloat16"},
            ],
        }
        fn = synthesize_validate_dtypes("MixedGemmOp", sig)
        m = self._mock()
        fn(m,
           x=torch.empty(0, dtype=torch.float16),
           weight=torch.empty(0, dtype=torch.float16))
        fn(m,
           x=torch.empty(0, dtype=torch.float16),
           weight=torch.empty(0, dtype=torch.float8_e4m3fn))
        fn(m,
           x=torch.empty(0, dtype=torch.bfloat16),
           weight=torch.empty(0, dtype=torch.bfloat16))

    def test_unlisted_cross_combo_rejected(self):
        """A row that passes per-input unions but is not in
        ``dtype_combos`` must still be rejected."""
        sig = {
            "inputs": {
                "x": {"dtype": "float16 | bfloat16"},
                "weight": {"dtype": "float16 | float8_e4m3fn | bfloat16"},
            },
            "dtype_combos": [
                {"x": "float16", "weight": "float16"},
                {"x": "bfloat16", "weight": "bfloat16"},
            ],
        }
        fn = synthesize_validate_dtypes("MixedGemmOp", sig)
        m = self._mock()
        # Each input dtype is in its declared union, but the (x, weight)
        # pair is not in the combo list.
        with pytest.raises(ValueError, match="dtype_combos"):
            fn(m,
               x=torch.empty(0, dtype=torch.bfloat16),
               weight=torch.empty(0, dtype=torch.float16))

    def test_out_of_union_still_rejected_before_combo_check(self):
        sig = {
            "inputs": {
                "x": {"dtype": "float16 | bfloat16"},
                "weight": {"dtype": "float16 | bfloat16"},
            },
            "dtype_combos": [
                {"x": "float16", "weight": "float16"},
            ],
        }
        fn = synthesize_validate_dtypes("MixedGemmOp", sig)
        m = self._mock()
        with pytest.raises(ValueError):
            fn(m,
               x=torch.empty(0, dtype=torch.float32),
               weight=torch.empty(0, dtype=torch.float16))

    def test_combo_ignores_same_as_axis(self):
        """``same_as`` inputs do not contribute a combo axis; they are
        still validated by the per-input pass."""
        sig = {
            "inputs": {
                "x": {"dtype": "float16 | bfloat16"},
                "weight": {"dtype": "float16 | float8_e4m3fn"},
                "bias": {"dtype": "same_as(x)"},
            },
            "dtype_combos": [
                {"x": "float16", "weight": "float16"},
                {"x": "float16", "weight": "float8_e4m3fn"},
            ],
        }
        fn = synthesize_validate_dtypes("MixedGemmBiasOp", sig)
        m = self._mock()
        fn(m,
           x=torch.empty(0, dtype=torch.float16),
           weight=torch.empty(0, dtype=torch.float8_e4m3fn),
           bias=torch.empty(0, dtype=torch.float16))
        # same_as(x) violated
        with pytest.raises(ValueError):
            fn(m,
               x=torch.empty(0, dtype=torch.float16),
               weight=torch.empty(0, dtype=torch.float16),
               bias=torch.empty(0, dtype=torch.bfloat16))

    def test_same_as_token_inside_combo_row_resolved(self):
        """``same_as(ref)`` inside a combo row resolves to the sibling's
        concrete dtype before tuple comparison (validator R4/R6)."""
        sig = {
            "inputs": {
                "x": {"dtype": "float16 | bfloat16"},
                "weight": {"dtype": "float16 | bfloat16"},
            },
            "dtype_combos": [
                {"x": "float16", "weight": "same_as(x)"},
                {"x": "bfloat16", "weight": "same_as(x)"},
            ],
        }
        fn = synthesize_validate_dtypes("SameAsComboOp", sig)
        m = self._mock()
        fn(m,
           x=torch.empty(0, dtype=torch.float16),
           weight=torch.empty(0, dtype=torch.float16))
        fn(m,
           x=torch.empty(0, dtype=torch.bfloat16),
           weight=torch.empty(0, dtype=torch.bfloat16))
        # Mismatched pair is rejected because the resolved combo is
        # (float16, float16) or (bfloat16, bfloat16), not the mix.
        with pytest.raises(ValueError, match="dtype_combos"):
            fn(m,
               x=torch.empty(0, dtype=torch.float16),
               weight=torch.empty(0, dtype=torch.bfloat16))

    def test_rejects_same_as_cycle_in_combo_row(self):
        sig = {
            "inputs": {
                "x": {"dtype": "float16 | bfloat16"},
                "y": {"dtype": "float16 | bfloat16"},
            },
            "dtype_combos": [
                {"x": "same_as(y)", "y": "same_as(x)"},
            ],
        }
        with pytest.raises(ValueError, match="cycle"):
            synthesize_validate_dtypes("CycleOp", sig)

    def test_rejects_same_as_dangling_sibling_in_combo_row(self):
        sig = {
            "inputs": {
                "x": {"dtype": "float16 | bfloat16"},
                "weight": {"dtype": "float16 | bfloat16"},
            },
            "dtype_combos": [
                {"x": "float16", "weight": "same_as(missing)"},
            ],
        }
        with pytest.raises(ValueError, match="not present in the same"):
            synthesize_validate_dtypes("DanglingOp", sig)

    def test_rejects_unknown_input_in_combo_row(self):
        sig = {
            "inputs": {
                "x": {"dtype": "float16"},
            },
            "dtype_combos": [
                {"x": "float16", "missing": "float16"},
            ],
        }
        with pytest.raises(ValueError, match="unknown input"):
            synthesize_validate_dtypes("BadOp", sig)


class TestAutoInstallHook:
    def test_subclass_with_manifest_status_implemented_gets_validator(self):
        sig = {"inputs": {"x": {"dtype": "float16"}}}
        Cls = _make_op("AutoInstalledOp", sig, status="implemented")
        # The synthesized method must live in cls.__dict__, not be the base stub
        assert "_validate_dtypes" in Cls.__dict__
        assert Cls._validate_dtypes is not Op._validate_dtypes

    def test_subclass_spec_only_does_not_install(self):
        sig = {"inputs": {"x": {"dtype": "float16"}}}
        Cls = _make_op("SpecOnlyOp", sig, status="spec-only")
        assert "_validate_dtypes" not in Cls.__dict__
        assert Cls._validate_dtypes is Op._validate_dtypes

    def test_subclass_explicit_override_not_clobbered(self):
        sig = {"inputs": {"x": {"dtype": "float16"}}}

        def manual_validator(self, x):
            raise RuntimeError("manual")

        Cls = _make_op(
            "ManualOp", sig, status="implemented",
            extra_attrs={"_validate_dtypes": manual_validator},
        )
        assert Cls._validate_dtypes is manual_validator

    def test_subclass_without_manifest_metadata_leaves_stub(self):
        # No __manifest_signature__: hook can't synthesize; stub remains.
        class NoMetaOp(Op):
            @property
            def default_kernel_map(self):
                return {}

            def forward(self, *a, **kw):
                return None

        assert NoMetaOp._validate_dtypes is Op._validate_dtypes
