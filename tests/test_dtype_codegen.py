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
