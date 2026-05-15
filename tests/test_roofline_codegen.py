"""Tests for manifest-driven ``eval_roofline`` codegen.

Covers ``tileops.ops._roofline_codegen.synthesize_eval_roofline`` and the
``Op.__init_subclass__`` auto-installation hook in
``tileops.ops.op_base``.
"""


import pytest

from tileops.ops._roofline_codegen import (
    _resolve_tensor_binding,
    _ShapeProxy,
    synthesize_eval_roofline,
)
from tileops.ops.op_base import Op

pytestmark = pytest.mark.smoke


def _make_op(name, *, roofline, signature=None, status="implemented",
             extra_attrs=None):
    """Build a concrete Op subclass with a manifest roofline attached."""
    attrs = {
        "default_kernel_map": property(lambda self: {}),
        "forward": lambda self, *a, **kw: None,
        "__manifest_signature__": signature or {"inputs": {}},
        "__manifest_roofline__": roofline,
        "__manifest_status__": status,
    }
    if extra_attrs:
        attrs.update(extra_attrs)
    return type(name, (Op,), attrs)


class TestSynthesizeFuncMode:
    """Func mode: ``return <func>(self)`` — codegen does not introspect."""

    def test_returns_func_result(self):
        # A toy func module attribute we point the manifest at.
        import types
        mod = types.ModuleType("test_synth_funcmod")
        def my_roofline(op):
            return (op.flops_hint, op.bytes_hint)
        mod.my_roofline = my_roofline
        import sys
        sys.modules["test_synth_funcmod"] = mod
        try:
            fn = synthesize_eval_roofline(
                "FooOp",
                roofline={"func": "test_synth_funcmod.my_roofline"},
                signature={"inputs": {"x": {"dtype": "float16"}}},
            )

            class Mock:
                flops_hint = 42
                bytes_hint = 99
            assert fn(Mock()) == (42, 99)
        finally:
            del sys.modules["test_synth_funcmod"]

    def test_func_unresolvable_at_synthesis_raises(self):
        with pytest.raises(ValueError, match="cannot resolve"):
            synthesize_eval_roofline(
                "FooOp",
                roofline={"func": "no.such.module.fn"},
                signature={"inputs": {"x": {"dtype": "float16"}}},
            )


class TestSynthesizeInlineMode:
    """Inline mode: emits a vars-resolution block + return block."""

    def test_inline_no_vars(self):
        # M, N, K all come from op attributes (params).
        fn = synthesize_eval_roofline(
            "GemmOp",
            roofline={
                "flops": "2 * M * N * K",
                "bytes": "(M * K + K * N + M * N) * elem_bytes",
            },
            signature={
                "inputs": {"a": {"dtype": "float16"}},
                "params": {"M": {"type": "int"},
                           "N": {"type": "int"},
                           "K": {"type": "int"}},
            },
        )

        class Mock:
            # Input binding per the 2-tier contract; never read by the
            # arithmetic expression in this test, but the resolver must
            # find a valid shape.
            a_shape = (4, 3)
            M = 4
            N = 5
            K = 3
        import torch
        m = Mock()
        m.dtype = torch.float16
        flops, nbytes = fn(m)
        assert flops == 2 * 4 * 5 * 3
        assert nbytes == (4 * 3 + 3 * 5 + 4 * 5) * 2

    def test_inline_with_vars(self):
        # PReLU-style: vars derive shape-dependent values from inputs.
        fn = synthesize_eval_roofline(
            "PreluFwdOp",
            roofline={
                "vars": {
                    "N": "product(input.shape)",
                    "W": "1 if weight.ndim == 0 or weight.shape[0] == 1 "
                         "else weight.shape[0]",
                },
                "flops": "2 * N",
                "bytes": "(2 * N + W) * elem_bytes",
            },
            signature={
                "inputs": {
                    "input": {"dtype": "float16"},
                    "weight": {"dtype": "same_as(input)"},
                },
            },
        )

        import torch

        class FakeTensor:
            def __init__(self, shape, dtype=torch.float16):
                self.shape = shape
                self.dtype = dtype
                self.ndim = len(shape)

        class Mock:
            pass

        m = Mock()
        m.input = FakeTensor((8, 16, 4))  # N = 512
        m.weight = FakeTensor((16,))      # W = 16
        m.dtype = torch.float16
        flops, nbytes = fn(m)
        assert flops == 2 * 512
        assert nbytes == (2 * 512 + 16) * 2

    def test_inline_missing_flops_or_bytes_rejected(self):
        with pytest.raises(ValueError, match="must declare both"):
            synthesize_eval_roofline(
                "BadOp",
                roofline={"flops": "1"},
                signature={"inputs": {"x": {"dtype": "float16"}}},
            )


class TestAutoInstallHook:
    """``__init_subclass__`` wires up the validator for implemented ops."""

    def test_implemented_op_with_func_roofline_installs(self):
        import types
        mod = types.ModuleType("test_install_funcmod")
        def my_roofline(op):
            return (1, 2)
        mod.my_roofline = my_roofline
        import sys
        sys.modules["test_install_funcmod"] = mod
        try:
            Cls = _make_op(
                "AutoFuncOp",
                roofline={"func": "test_install_funcmod.my_roofline"},
            )
            assert "eval_roofline" in Cls.__dict__
            assert Cls.eval_roofline is not Op.eval_roofline
        finally:
            del sys.modules["test_install_funcmod"]

    def test_spec_only_does_not_install(self):
        Cls = _make_op(
            "SpecOnlyRoofOp",
            roofline={"flops": "1", "bytes": "1"},
            status="spec-only",
        )
        assert "eval_roofline" not in Cls.__dict__
        assert Cls.eval_roofline is Op.eval_roofline

    def test_explicit_override_not_clobbered(self):
        def manual(self):
            return (7, 11)
        Cls = _make_op(
            "ManualRoofOp",
            roofline={"flops": "1", "bytes": "1"},
            extra_attrs={"eval_roofline": manual},
        )
        assert Cls.eval_roofline is manual

    def test_inherited_override_not_clobbered(self):
        """An intermediate base class (e.g. UnaryOp) supplying its own
        ``eval_roofline`` must be honored — codegen must not synthesize
        a body on the leaf subclass and shadow the inherited working
        implementation."""
        def base_impl(self):
            return (123, 456)

        class IntermediateBase(Op):
            default_kernel_map = property(lambda self: {})

            def forward(self, *a, **kw):
                return None

            eval_roofline = base_impl

        Leaf = type(
            "LeafInheritsRoofline",
            (IntermediateBase,),
            {
                "__manifest_signature__": {"inputs": {}},
                "__manifest_roofline__": {"flops": "1", "bytes": "1"},
                "__manifest_status__": "implemented",
            },
        )
        # Codegen must not overwrite the inherited implementation.
        assert Leaf.eval_roofline is base_impl
        # And it must still be callable end-to-end.
        assert Leaf().eval_roofline() == (123, 456)

    def test_no_manifest_metadata_leaves_stub(self):
        class NoMetaOp(Op):
            @property
            def default_kernel_map(self):
                return {}

            def forward(self, *a, **kw):
                return None

        assert NoMetaOp.eval_roofline is Op.eval_roofline


class TestRealOpSmoke:
    """End-to-end checks on concrete ops with both inline-mode tiers:
    ``PreluFwdOp`` uses ``self.<input>_shape`` for both inputs;
    ``NanToNumFwdOp`` uses ``self.<input>_shape`` for its single input.
    """

    def test_prelu_fwd_op_eval_roofline_uses_shape_attrs(self):
        import torch

        from tileops.ops.elementwise.prelu import PreluFwdOp

        # PreluFwdOp.__init__ builds a kernel; bypass via __new__ so the
        # test stays smoke-light and CUDA-free.
        op = PreluFwdOp.__new__(PreluFwdOp)
        op.input_shape = (16, 256, 56, 56)
        op.weight_shape = (256,)
        op.dtype = torch.float16

        from math import prod as _prod
        N = _prod(op.input_shape)
        W = op.weight_shape[0]
        elem = op.dtype.itemsize
        flops, total_bytes = op.eval_roofline()
        assert flops == 2 * N
        assert total_bytes == (2 * N + W) * elem

    def test_nan_to_num_fwd_op_eval_roofline_uses_input_shape(self):
        import torch

        from tileops.ops.elementwise.nan_to_num import NanToNumFwdOp

        op = NanToNumFwdOp.__new__(NanToNumFwdOp)
        op.input_shape = (4096 * 4096,)
        op.dtype = torch.float16

        N = op.input_shape[0]
        elem = op.dtype.itemsize
        flops, total_bytes = op.eval_roofline()
        assert flops == 6 * N
        assert total_bytes == 2 * N * elem


class TestResolveTensorBinding:
    """``_resolve_tensor_binding`` honors the 2-tier contract.

    Tier 1: ``self.<name>`` exposing ``.shape``/``.ndim``.
    Tier 2: ``self.<name>_shape`` as a shape tuple/list.
    Anything else → ``ValueError`` naming op + input + attempted
    conventions.
    """

    def test_real_tensor_attr_wins(self):
        class Op:
            input = _ShapeProxy((4, 5))
        out = _resolve_tensor_binding(Op(), "input", "ToyOp")
        assert out.shape == (4, 5)

    def test_input_shape_attr_used(self):
        class Op:
            input_shape = (2, 3, 4)
        out = _resolve_tensor_binding(Op(), "input", "ToyOp")
        assert out.shape == (2, 3, 4)
        assert out.ndim == 3

    def test_input_shape_attr_accepts_list(self):
        # Shape tuples written as YAML lists survive the manifest load
        # path as Python lists; the resolver must accept either form.
        class Op:
            input_shape = [2, 3, 4]
        out = _resolve_tensor_binding(Op(), "input", "ToyOp")
        assert out.shape == (2, 3, 4)

    def test_no_binding_raises_with_op_and_conventions(self):
        class Op:
            pass
        with pytest.raises(ValueError) as exc:
            _resolve_tensor_binding(Op(), "input", "ToyOp")
        msg = str(exc.value)
        assert "ToyOp" in msg
        assert "'input'" in msg
        assert "self.input" in msg
        assert "self.input_shape" in msg

    def test_tier1_requires_both_shape_and_ndim(self):
        # A partially-conformant attribute (``.shape`` but no ``.ndim``)
        # must not pass tier 1 and bypass the ``<name>_shape`` fallback.
        class ShapeOnly:
            shape = (2, 3)

        class Op:
            input = ShapeOnly()
            input_shape = (7, 11)

        out = _resolve_tensor_binding(Op(), "input", "ToyOp")
        # Tier 1 rejected ShapeOnly, tier 2 used input_shape → proxy.
        assert out.shape == (7, 11)
        assert out.ndim == 2


class TestVarsLayerComprehensions:
    """Vars-layer comprehensions: target names bind to a child scope."""

    def test_generator_target_resolves(self):
        # ``sum(d for d in x.shape)`` was previously rejected because the
        # bare ``Name`` walk surfaced ``d`` as unknown; the scoped
        # validator now binds ``d`` for the comprehension subtree only.
        fn = synthesize_eval_roofline(
            "ProdOp",
            roofline={
                "vars": {"N": "sum(d for d in x.shape)"},
                "flops": "N",
                "bytes": "N * elem_bytes",
            },
            signature={"inputs": {"x": {"dtype": "float16"}}},
        )

        import torch

        class Mock:
            x = _ShapeProxy((2, 3, 5))
            dtype = torch.float16

        flops, nbytes = fn(Mock())
        assert flops == 2 + 3 + 5
        assert nbytes == (2 + 3 + 5) * 2

    def test_listcomp_target_resolves(self):
        fn = synthesize_eval_roofline(
            "Op",
            roofline={
                "vars": {"N": "sum([d * d for d in x.shape])"},
                "flops": "N",
                "bytes": "N * elem_bytes",
            },
            signature={"inputs": {"x": {"dtype": "float16"}}},
        )

        import torch

        class Mock:
            x = _ShapeProxy((2, 3))
            dtype = torch.float16

        flops, _ = fn(Mock())
        assert flops == 2 * 2 + 3 * 3

    def test_target_does_not_leak_to_outer_scope(self):
        # ``d`` is bound inside the comprehension. Referencing it outside
        # must still raise — the scope must pop.
        with pytest.raises(ValueError, match="unknown name 'd'"):
            synthesize_eval_roofline(
                "BadOp",
                roofline={
                    "vars": {
                        "N": "sum(d for d in x.shape)",
                        "M": "d",  # 'd' is not visible here
                    },
                    "flops": "N + M",
                    "bytes": "N * elem_bytes",
                },
                signature={"inputs": {"x": {"dtype": "float16"}}},
            )

    def test_target_not_bound_in_its_own_iterable(self):
        # Python evaluates each generator's iterable in the surrounding
        # scope, *before* binding its target. ``sum(d for d in d)``
        # raises NameError at runtime; the synthesis-time gate must
        # mirror that and reject the iterable's reference to ``d``.
        with pytest.raises(ValueError, match="unknown name 'd'"):
            synthesize_eval_roofline(
                "BadOp",
                roofline={
                    "vars": {"N": "sum(d for d in d)"},
                    "flops": "N",
                    "bytes": "N * elem_bytes",
                },
                signature={"inputs": {"x": {"dtype": "float16"}}},
            )


class TestVarsLayerAttributeWhitelist:
    """Vars-layer ``Attribute`` access is restricted to ``.shape`` / ``.ndim``,
    and only when taken directly from a declared tensor-input Name."""

    def test_arbitrary_attribute_rejected(self):
        with pytest.raises(ValueError, match="non-whitelisted attribute 'dtype'"):
            synthesize_eval_roofline(
                "BadOp",
                roofline={
                    "vars": {"N": "x.dtype"},
                    "flops": "N",
                    "bytes": "N * elem_bytes",
                },
                signature={"inputs": {"x": {"dtype": "float16"}}},
            )

    def test_chained_shape_attr_rejected(self):
        # ``x.shape.ndim`` would crash at runtime: ``x.shape`` is a
        # tuple, not a shape-bearing proxy. Reject at synthesis.
        with pytest.raises(ValueError, match="non-tensor-input operand"):
            synthesize_eval_roofline(
                "BadOp",
                roofline={
                    "vars": {"N": "x.shape.ndim"},
                    "flops": "N",
                    "bytes": "N * elem_bytes",
                },
                signature={"inputs": {"x": {"dtype": "float16"}}},
            )

    def test_subscript_chained_attr_rejected(self):
        # ``x.shape[0].shape`` — subscripting shape yields an int.
        with pytest.raises(ValueError, match="non-tensor-input operand"):
            synthesize_eval_roofline(
                "BadOp",
                roofline={
                    "vars": {"N": "x.shape[0].shape"},
                    "flops": "N",
                    "bytes": "N * elem_bytes",
                },
                signature={"inputs": {"x": {"dtype": "float16"}}},
            )

    def test_attr_on_local_rejected(self):
        # ``N.shape`` — ``N`` is a local vars-layer name, not a tensor.
        with pytest.raises(ValueError, match="non-tensor-input operand"):
            synthesize_eval_roofline(
                "BadOp",
                roofline={
                    "vars": {
                        "N": "x.shape[0]",
                        "M": "N.shape",  # ← invalid
                    },
                    "flops": "M",
                    "bytes": "M * elem_bytes",
                },
                signature={"inputs": {"x": {"dtype": "float16"}}},
            )


class TestVarsLayerTensorInputUsage:
    """Vars-layer rejects tensor inputs used as bare values."""

    def test_bare_tensor_in_helper_call_rejected(self):
        # ``sum(x)`` would iterate tensor data at runtime; vars layer is
        # shape-derived, so reject the bare reference at synthesis.
        with pytest.raises(ValueError, match="tensor input 'x' as a bare value"):
            synthesize_eval_roofline(
                "BadOp",
                roofline={
                    "vars": {"N": "sum(x)"},
                    "flops": "N",
                    "bytes": "N * elem_bytes",
                },
                signature={"inputs": {"x": {"dtype": "float16"}}},
            )

    def test_bare_tensor_subscript_rejected(self):
        # ``x[0]`` reads tensor data at runtime.
        with pytest.raises(ValueError, match="tensor input 'x' as a bare value"):
            synthesize_eval_roofline(
                "BadOp",
                roofline={
                    "vars": {"N": "x[0]"},
                    "flops": "N",
                    "bytes": "N * elem_bytes",
                },
                signature={"inputs": {"x": {"dtype": "float16"}}},
            )

    def test_shape_subscript_accepted(self):
        # ``x.shape[0]`` is shape access, not data access. Must pass.
        fn = synthesize_eval_roofline(
            "Op",
            roofline={
                "vars": {"N": "x.shape[0]"},
                "flops": "N",
                "bytes": "N * elem_bytes",
            },
            signature={"inputs": {"x": {"dtype": "float16"}}},
        )

        import torch

        class Mock:
            x = _ShapeProxy((4, 5))
            dtype = torch.float16

        flops, _ = fn(Mock())
        assert flops == 4


class TestVarsKeyCollision:
    """Vars keys must not shadow inputs / params / helpers / earlier vars."""

    def test_vars_key_collides_with_helper(self):
        with pytest.raises(ValueError, match="collides with an existing name"):
            synthesize_eval_roofline(
                "BadOp",
                roofline={
                    "vars": {"sum": "1", "N": "sum(x.shape)"},
                    "flops": "N",
                    "bytes": "N * elem_bytes",
                },
                signature={"inputs": {"x": {"dtype": "float16"}}},
            )

    def test_vars_key_collides_with_input(self):
        with pytest.raises(ValueError, match="collides with an existing name"):
            synthesize_eval_roofline(
                "BadOp",
                roofline={
                    "vars": {"x": "1"},
                    "flops": "x",
                    "bytes": "x * elem_bytes",
                },
                signature={"inputs": {"x": {"dtype": "float16"}}},
            )

    def test_vars_key_collides_with_param(self):
        with pytest.raises(ValueError, match="collides with an existing name"):
            synthesize_eval_roofline(
                "BadOp",
                roofline={
                    "vars": {"M": "1"},
                    "flops": "M",
                    "bytes": "M * elem_bytes",
                },
                signature={
                    "inputs": {"x": {"dtype": "float16"}},
                    "params": {"M": {"type": "int"}},
                },
            )


class TestArithmeticLayerForm:
    """Arithmetic layer rejects non-numeric AST forms."""

    def test_tuple_literal_rejected(self):
        with pytest.raises(ValueError, match="forbidden construct Tuple"):
            synthesize_eval_roofline(
                "BadOp",
                roofline={"flops": "(1, 2)", "bytes": "1 * elem_bytes"},
                signature={"inputs": {"x": {"dtype": "float16"}}},
            )

    def test_list_literal_rejected(self):
        with pytest.raises(ValueError, match="forbidden construct List"):
            synthesize_eval_roofline(
                "BadOp",
                roofline={"flops": "[1]", "bytes": "1 * elem_bytes"},
                signature={"inputs": {"x": {"dtype": "float16"}}},
            )


class TestParamContract:
    """``signature.params`` must be exposed as ``self.<param>``."""

    def test_missing_param_raises_attribute_error(self):
        fn = synthesize_eval_roofline(
            "ToyOp",
            roofline={
                "flops": "M * N",
                "bytes": "(M + N) * elem_bytes",
            },
            signature={
                "inputs": {"x": {"dtype": "float16"}},
                "params": {"M": {"type": "int"}, "N": {"type": "int"}},
            },
        )

        import torch

        class Mock:
            x_shape = (1,)
            M = 4
            # N intentionally omitted.
            dtype = torch.float16

        with pytest.raises(AttributeError, match="'N'"):
            fn(Mock())


class TestMissingRoofline:
    """Manifest entry without a roofline block should leave the stub."""

    def test_op_without_roofline_block_leaves_stub(self):
        Cls = _make_op(
            "NoRoofOp",
            roofline=None,
            extra_attrs={"__manifest_roofline__": None},
        )
        assert Cls.eval_roofline is Op.eval_roofline


class TestRealManifestParity:
    """Smoke-check: every status:implemented op in the live manifest is
    no longer the base stub after class load. Covers AC-1 in spirit."""

    def test_no_c7_stubs_for_implemented_ops(self):
        from tileops.manifest import load_manifest
        ops = load_manifest()
        # Trigger op-class imports.
        import tileops.ops  # noqa: F401
        stubs = []
        for op_name, entry in ops.items():
            if entry.get("status") != "implemented":
                continue
            # Resolve the class via the registered Op subclasses.
            cls = _find_op_class(op_name)
            if cls is None:
                continue
            if cls.eval_roofline is Op.eval_roofline:
                stubs.append(op_name)
        assert stubs == [], (
            f"{len(stubs)} implemented ops still have the base "
            f"eval_roofline stub: {stubs[:5]}..."
        )


def _find_op_class(op_name):
    """Walk Op.__subclasses__ recursively, return the class named ``op_name``."""
    seen = set()
    stack = list(Op.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        if cls.__name__ == op_name:
            return cls
        stack.extend(cls.__subclasses__())
    return None
