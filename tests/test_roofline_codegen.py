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


# Ops whose `eval_roofline` the installer stands aside for. The value names the
# class that owns the method and the codegen capability its absence rests on:
# when that capability lands, the entry goes and the method with it.
HAND_EVALUATED = {
    "MoePrePermuteFwdOp": (
        "MoePrePermuteFwdOp",
        "its output extents follow the layout spec the call passes, and the "
        "vars layer binds inputs and params only",
    ),
}


class TestEvaluatorOwnership:
    """Who owns `eval_roofline`, op by op.

    The installer stands aside without a word for a class that defines the
    method itself, so an op can leave the manifest behind by adding one method.
    That is how 98 of 175 entries came to state a formula nothing ran.
    """

    @staticmethod
    def _owner(cls):
        """The class whose `eval_roofline` an instance of *cls* would call."""
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

    def test_the_manifest_entry_is_what_runs_unless_an_op_is_registered(self):
        from tileops.ops._roofline_codegen import SYNTHESIZED

        unregistered = []
        for name, cls in self._implemented():
            owner = self._owner(cls)
            if getattr(owner.__dict__["eval_roofline"], SYNTHESIZED, False):
                # Generated for this class, not inherited from another op's entry.
                assert owner is cls, f"{name} runs {owner.__name__}'s entry, not its own"
                continue
            if name not in HAND_EVALUATED:
                unregistered.append(f"{name} (owned by {owner.__name__})")
        assert not unregistered, (
            f"these ops evaluate their own roofline and say nowhere why: {unregistered}; "
            "their manifest entry states a formula nothing runs"
        )

    def test_a_registered_op_owns_the_method_where_it_says(self):
        from tileops.ops._roofline_codegen import SYNTHESIZED

        implemented = dict(self._implemented())
        stale = sorted(set(HAND_EVALUATED) - set(implemented))
        assert not stale, f"registered but not implemented: {stale}"
        for name, (expected_owner, reason) in HAND_EVALUATED.items():
            owner = self._owner(implemented[name])
            assert not getattr(owner.__dict__["eval_roofline"], SYNTHESIZED, False), (
                f"{name} is registered as hand-evaluated and codegen now serves it; drop the entry"
            )
            assert owner.__name__ == expected_owner, f"{name}: {owner.__name__}"
            assert reason and not reason.endswith("."), name


class TestCallPayload:
    """A formula reads the call an op ran, not what its constructor defaulted to."""

    def test_an_op_that_has_not_run_says_so(self):
        """Distinct from the ValueError an unwired op raises: this one is the
        caller's sequencing, and the audit reports it as such."""
        from tileops.ops.attention.gqa import GroupedQueryAttentionDenseFwdOp

        op = GroupedQueryAttentionDenseFwdOp.__new__(GroupedQueryAttentionDenseFwdOp)
        op._roofline_kwargs = None
        with pytest.raises(RuntimeError, match="requires a prior forward"):
            op.eval_roofline()

    def test_a_payload_that_is_not_a_mapping_is_the_author_s_wiring(self):
        from tileops.perf.formulas import _shape_or_attrs

        class _Miswired:
            def __init__(self):
                self._roofline_kwargs = (1, 2)

        with pytest.raises(ValueError, match="mapping"):
            _shape_or_attrs(_Miswired(), {})

    def test_the_payload_wins_over_a_construction_default(self):
        """`out_dtype` is settled at construction and restated by the call; the
        call is what moved the bytes."""
        import torch

        from tileops.perf.formulas import _shape_or_attrs

        class _Op:
            def __init__(self):
                self.out_dtype = torch.float32
                self._roofline_kwargs = {"out_dtype": torch.float16, "q_shape": (1, 2, 3, 4)}

        data = _shape_or_attrs(_Op(), {})
        assert data["out_dtype"] is torch.float16
        assert data["q_shape"] == (1, 2, 3, 4)

    def test_an_attribute_the_payload_omits_survives(self):
        from tileops.perf.formulas import _shape_or_attrs

        class _Op:
            def __init__(self):
                self.is_causal = True
                self._roofline_kwargs = {"q_shape": (1, 2, 3, 4)}

        assert _shape_or_attrs(_Op(), {})["is_causal"] is True


class TestInheritedEvaluator:
    """An op that subclasses another op answers its own entry.

    The installer stands aside for a hand-written method, and a generated one on
    a parent looks the same from the child's side. Standing aside there would
    run the parent's formula for the child's entry, which is the bypass this
    module exists to prevent.
    """

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

        from tileops.ops._roofline_codegen import SYNTHESIZED

        parent = self._op("_ParentOp", "2 * N * elem_bytes")
        child = self._op("_ChildOp", "4 * N * elem_bytes", base=parent)

        assert getattr(child.__dict__.get("eval_roofline"), SYNTHESIZED, False), (
            "the child inherited the parent's generated evaluator"
        )
        instance = child.__new__(child)
        instance.x_shape, instance.dtype = (128,), torch.float16
        assert instance.eval_roofline()[1] == 4 * 128 * 2

    def test_a_hand_written_parent_still_serves_its_subclass(self):
        from tileops.ops.op_base import Op

        class _Base(Op):
            def eval_roofline(self):
                return (1, 2)

        child = self._op("_HandChildOp", "9 * N * elem_bytes", base=_Base)
        assert "eval_roofline" not in child.__dict__
        assert child.__new__(child).eval_roofline() == (1, 2)
