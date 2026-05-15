"""Tests for manifest-driven ``eval_roofline`` codegen.

Covers ``tileops.ops._roofline_codegen.synthesize_eval_roofline`` and the
``Op.__init_subclass__`` auto-installation hook in
``tileops.ops.op_base``.
"""


import pytest

from tileops.ops._roofline_codegen import (
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
        # M, N, K all come from op attributes (shape dims).
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

    def test_relu_fwd_op_uses_unary_base_roofline(self):
        """Regression: ``ReluFwdOp`` inherits ``UnaryOp.eval_roofline``;
        codegen must not replace it with a synthesized body that depends
        on attributes the leaf class never sets (e.g. an input tensor
        ``shape``)."""
        import torch

        from tileops.ops.elementwise import ReluFwdOp
        from tileops.ops.elementwise._base import UnaryOp

        # The class-level method must still be the UnaryOp implementation.
        assert ReluFwdOp.eval_roofline is UnaryOp.eval_roofline

        # And a constructed instance must produce sensible (flops, bytes).
        op = ReluFwdOp.__new__(ReluFwdOp)
        op.dtype = torch.float16
        op.output_dtype = torch.float16
        op.N_total = 32
        flops, total_bytes = op.eval_roofline()
        assert flops > 0
        assert total_bytes > 0

    def test_no_manifest_metadata_leaves_stub(self):
        class NoMetaOp(Op):
            @property
            def default_kernel_map(self):
                return {}

            def forward(self, *a, **kw):
                return None

        assert NoMetaOp.eval_roofline is Op.eval_roofline


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
