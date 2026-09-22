"""Real-op smoke tests for the generated ``eval_roofline``."""

import pytest

pytestmark = pytest.mark.smoke


class TestRealOpSmoke:
    def test_prelu_fwd_op_eval_roofline_uses_shape_attrs(self):
        import torch

        from tileops.ops.elementwise.prelu import PreluFwdOp

        # __new__ bypasses kernel construction so the smoke stays CUDA-free.
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

    def test_optional_input_presence_switches_the_formula(self):
        """R18.1: an optional input may appear as a bare presence test."""
        import torch

        from tileops.ops._roofline_codegen import synthesize_eval_roofline

        fn = synthesize_eval_roofline(
            "FakeOp",
            roofline={
                "vars": {"N": "x.shape[0]", "has_bias": "bias is not None"},
                "flops": "(2 if has_bias else 1) * N",
                "bytes": "(N + (8 if has_bias else 0)) * elem_bytes",
            },
            signature={
                "inputs": {
                    "x": {"dtype": "float16"},
                    "bias": {"dtype": "float16", "optional": True},
                },
            },
        )

        class FakeOp:
            dtype = torch.float16

            def __init__(self, bias):
                self.x_shape = (64,)
                self.bias = bias

        assert fn(FakeOp(torch.empty(8, dtype=torch.float16))) == (128, 144)
        assert fn(FakeOp(None)) == (64, 128)

    def test_vars_may_not_read_an_optional_input(self):
        """Only a presence test — its shape is unavailable on an absent call."""
        from tileops.ops._roofline_codegen import synthesize_eval_roofline

        for expr in ("bias.shape[0]", "bias is None or bias.ndim", "bias[0]"):
            with pytest.raises(ValueError, match="optional input"):
                synthesize_eval_roofline(
                    "FakeOp",
                    roofline={
                        "vars": {"n": expr},
                        "flops": "n",
                        "bytes": "n",
                    },
                    signature={
                        "inputs": {
                            "x": {"dtype": "float16"},
                            "bias": {"dtype": "float16", "optional": True},
                        },
                    },
                )

    def test_optional_input_the_op_never_exposes_still_raises(self):
        """An unexposed binding must not read as "the call omitted it"."""
        import torch

        from tileops.ops._roofline_codegen import synthesize_eval_roofline

        fn = synthesize_eval_roofline(
            "FakeOp",
            roofline={
                "vars": {"N": "x.shape[0]", "has_bias": "bias is not None"},
                "flops": "N",
                "bytes": "(N + (8 if has_bias else 0)) * elem_bytes",
            },
            signature={
                "inputs": {
                    "x": {"dtype": "float16"},
                    "bias": {"dtype": "float16", "optional": True},
                },
            },
        )

        class FakeOp:
            dtype = torch.float16
            x_shape = (64,)

        with pytest.raises(ValueError, match="cannot resolve roofline input"):
            fn(FakeOp())

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


class TestTotalContract:
    """Codegen answers a data defect with a verdict, never with a crash."""

    def test_a_non_mapping_signature_is_a_verdict(self):
        from tileops.ops._roofline_codegen import synthesize_eval_roofline

        with pytest.raises(ValueError, match="signature must be a mapping"):
            synthesize_eval_roofline(
                "FakeOp",
                roofline={"flops": "1", "bytes": "1"},
                signature="not a mapping",
            )

    def test_a_raising_func_attribute_is_a_verdict(self):
        import sys
        import types

        from tileops.ops._roofline_codegen import synthesize_eval_roofline

        class Boom(types.ModuleType):
            def __getattr__(self, name):
                raise RuntimeError("module said no")

        sys.modules["_codegen_boom"] = Boom("_codegen_boom")
        try:
            with pytest.raises(ValueError, match="RuntimeError: module said no"):
                synthesize_eval_roofline(
                    "FakeOp", roofline={"func": "_codegen_boom.fn"}, signature=None
                )
        finally:
            del sys.modules["_codegen_boom"]

    def test_checking_a_formula_needs_no_torch(self):
        """An `out_elem_bytes` formula binds the resolver, it does not import it."""
        import subprocess
        import sys
        import textwrap

        probe = textwrap.dedent(
            """
            import sys
            sys.modules["torch"] = None  # any import of it raises
            from tileops.ops._roofline_codegen import synthesize_eval_roofline
            synthesize_eval_roofline(
                "FakeOp",
                roofline={"vars": {"N": "product(x.shape)"},
                          "flops": "N", "bytes": "N * out_elem_bytes"},
                signature={"inputs": {"x": {"dtype": "float16"}},
                           "outputs": {"y": {"dtype": "bool"}}},
            )
            print("ok")
            """
        )
        out = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=False
        )
        assert out.returncode == 0, out.stderr
        assert "ok" in out.stdout


class TestEvaluatorOwnership:
    """Enforce evaluator ownership for implemented manifest entries."""

    @staticmethod
    def _owner(cls):
        for base in cls.__mro__:
            if "eval_roofline" in base.__dict__:
                return base
        return None

    def _implemented(self):
        import importlib

        from tileops.manifest import load_manifest

        for name, entry in load_manifest().items():
            if entry.get("status") != "implemented":
                continue
            module = entry["source"]["op"].removesuffix(".py").replace("/", ".")
            cls = getattr(importlib.import_module(module), name, None)
            if cls is not None:
                yield name, cls

    def test_each_op_owns_a_generated_evaluator(self):
        from tileops.ops._roofline_codegen import SYNTHESIZED_ATTR

        invalid = []
        for name, cls in self._implemented():
            owner = self._owner(cls)
            generated = getattr(owner.__dict__["eval_roofline"], SYNTHESIZED_ATTR, False)
            if owner is not cls or not generated:
                invalid.append(f"{name} (owned by {owner.__name__})")
        assert not invalid, f"ops without their own generated evaluator: {invalid}"


class TestCallPayload:
    """Call-bound formula inputs override construction-bound state."""

    def test_requires_prior_forward(self):
        from tileops.ops.attention.gqa import GroupedQueryAttentionDenseFwdOp

        op = GroupedQueryAttentionDenseFwdOp.__new__(GroupedQueryAttentionDenseFwdOp)
        op._roofline_kwargs = None
        with pytest.raises(RuntimeError, match="requires a prior forward"):
            op.eval_roofline()

    def test_rejects_non_mapping_payload(self):
        from tileops.perf.formulas import _shape_or_attrs

        class _Miswired:
            def __init__(self):
                self._roofline_kwargs = (1, 2)

        with pytest.raises(ValueError, match="mapping"):
            _shape_or_attrs(_Miswired(), {})

    def test_payload_overrides_instance_state(self):
        import torch

        from tileops.perf.formulas import _shape_or_attrs

        class _Op:
            def __init__(self):
                self.out_dtype = torch.float32
                self._roofline_kwargs = {"out_dtype": torch.float16, "q_shape": (1, 2, 3, 4)}

        data = _shape_or_attrs(_Op(), {})
        assert data["out_dtype"] is torch.float16
        assert data["q_shape"] == (1, 2, 3, 4)

    def test_payload_preserves_other_instance_state(self):
        from tileops.perf.formulas import _shape_or_attrs

        class _Op:
            def __init__(self):
                self.is_causal = True
                self._roofline_kwargs = {"q_shape": (1, 2, 3, 4)}

        assert _shape_or_attrs(_Op(), {})["is_causal"] is True


class TestInheritedEvaluator:
    """A subclass with a manifest entry owns its generated evaluator."""

    @staticmethod
    def _op(name, bytes_expr, base=None):
        from tileops.ops.op_base import Op

        return type(
            name,
            (base or Op,),
            {
                "__manifest_status__": "implemented",
                "__manifest_signature__": {
                    "inputs": {"x": {"dtype": "float16", "shape": "[N]"}},
                    "outputs": {"y": {"dtype": "same_as(x)"}},
                },
                "__manifest_roofline__": {
                    "vars": {"N": "x.shape[0]"},
                    "flops": "N",
                    "bytes": bytes_expr,
                },
                "forward": lambda self, *a, **kw: None,
                "_infer_output_shapes": lambda self, x_shape: {"y": x_shape},
                "_validate_dtypes": lambda self, *a: None,
                "default_kernel_map": property(lambda self: {}),
            },
        )

    def test_a_subclass_runs_its_own_entry(self):
        import torch

        from tileops.ops._roofline_codegen import SYNTHESIZED_ATTR

        parent = self._op("_ParentOp", "2 * N * elem_bytes")
        child = self._op("_ChildOp", "4 * N * elem_bytes", base=parent)

        assert getattr(child.__dict__.get("eval_roofline"), SYNTHESIZED_ATTR, False), (
            "the child inherited the parent's generated evaluator"
        )
        instance = child.__new__(child)
        instance.x_shape, instance.dtype = (128,), torch.float16
        assert instance.eval_roofline()[1] == 4 * 128 * 2

    def test_an_explicit_parent_does_not_replace_the_child_entry(self):
        import torch

        from tileops.ops.op_base import Op

        class _Base(Op):
            def eval_roofline(self):
                return (1, 2)

        child = self._op("_HandChildOp", "9 * N * elem_bytes", base=_Base)
        instance = child.__new__(child)
        instance.x_shape, instance.dtype = (8,), torch.float16
        assert instance.eval_roofline() == (8, 9 * 8 * 2)
