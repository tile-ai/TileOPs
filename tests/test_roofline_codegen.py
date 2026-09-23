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
        """Reported whether or not the formula reads anything from it."""
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"flops": "1", "bytes": "1"},
            signature="not a mapping",
        )
        assert [d.code for d in result.diagnostics] == ["signature.not-a-mapping"]

    def test_a_non_mapping_signature_stops_a_formula_that_reads_it(self):
        """The whole block settles nothing, so it is needed exactly when a part
        of it would have been."""
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"vars": {"N": "product(x.shape)"}, "flops": "N", "bytes": "N"},
            signature="not a mapping",
        )
        assert result.plan is None
        assert ("signature.not-a-mapping", True) in {
            (d.code, d.blocking) for d in result.diagnostics
        }

    def test_a_non_mapping_signature_does_not_stop_a_formula_that_does_not(self):
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"flops": "1", "bytes": "1"},
            signature="not a mapping",
        )
        assert result.plan is not None
        assert not result.blocking

    @pytest.mark.parametrize("block", ["inputs", "outputs", "params"])
    def test_a_non_mapping_signature_block_is_a_verdict(self, block):
        """A non-mapping reads as empty, which would make a name look declared.

        The verdict stands whether or not the formula reaches for that block;
        whether it also stops emission is
        `TestPartialSignature`'s subject.
        """
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"flops": "1", "bytes": "1"},
            signature={"inputs": {}, block: 5},
        )
        assert [d.code for d in result.diagnostics] == [f"signature.{block}.not-a-mapping"]
        assert f"signature.{block} must be a mapping" in result.diagnostics[0].message

    @pytest.mark.parametrize("bad", [[], "", False, 0, ["a"]], ids=repr)
    def test_a_non_mapping_vars_is_a_verdict(self, bad):
        """A falsey one must not read as an absent one."""
        from tileops.ops._roofline_codegen import synthesize_eval_roofline

        with pytest.raises(ValueError, match="roofline.vars must be a mapping"):
            synthesize_eval_roofline(
                "FakeOp",
                roofline={"vars": bad, "flops": "1", "bytes": "1"},
                signature={"inputs": {"x": {"dtype": "float16"}}},
            )

    def test_mixing_both_modes_is_a_verdict(self):
        from tileops.ops._roofline_codegen import synthesize_eval_roofline

        with pytest.raises(ValueError, match="cannot mix func and inline"):
            synthesize_eval_roofline(
                "FakeOp",
                roofline={"func": "tileops.perf.formulas.fused_moe_fwd_bytes", "flops": "1"},
                signature=None,
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


class TestPartialSignature:
    """A block the formula never reads may be unreadable without stopping it."""

    def test_unread_outputs_does_not_stop_emission(self):
        """`out_elem_bytes` is the only thing `outputs` settles for a formula."""
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"flops": "1", "bytes": "1"},
            signature={"inputs": {}, "outputs": 9},
        )
        assert [d.code for d in result.diagnostics] == ["signature.outputs.not-a-mapping"]
        assert result.plan is not None
        assert not result.blocking

    def test_read_outputs_does_stop_emission(self):
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"flops": "1", "bytes": "out_elem_bytes"},
            signature={"inputs": {}, "outputs": 9},
        )
        assert result.plan is None
        assert any(
            d.code == "signature.outputs.not-a-mapping" and d.blocking for d in result.diagnostics
        )

    def test_a_malformed_signature_does_not_hide_the_formula_defect(self):
        """The defect this restructure exists to stop hiding."""
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"flops": "NOPE", "bytes": "1"},
            signature={"inputs": 5, "outputs": {"y": {}}},
        )
        codes = {d.code for d in result.diagnostics}
        assert "arith.unknown-name" in codes
        assert "signature.inputs.not-a-mapping" in codes

    def test_two_defects_in_one_expression_are_two_diagnostics(self):
        """`subject` is what keeps the second from collapsing into the first."""
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"flops": "AAA + BBB", "bytes": "1"},
            signature={"inputs": {}, "outputs": {"y": {}}},
        )
        unknown = [d for d in result.diagnostics if d.code == "arith.unknown-name"]
        assert sorted(d.subject for d in unknown) == ["AAA", "BBB"]
        assert len({d.identity for d in unknown}) == 2

    @pytest.mark.parametrize("name", [7, "class"], ids=["non-string", "keyword"])
    def test_a_name_that_cannot_bind_a_local_is_a_verdict(self, name):
        """Reached `sorted()` as a TypeError before, or emitted a SyntaxError.

        No expression can name either -- one is not a string and the other does
        not parse -- so the formula cannot be reaching for it, and the evaluator
        still emits while the entry still draws the line.
        """
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"flops": "1", "bytes": "1"},
            signature={"inputs": {}, "outputs": {"y": {}}, "params": {name: {"type": "int"}}},
        )
        assert [d.code for d in result.diagnostics] == [
            "signature.non-string-name" if name == 7 else "signature.unusable-name"
        ]
        assert not result.blocking
        assert result.plan is not None


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
        from tileops.ops._roofline_emit import SYNTHESIZED_ATTR

        invalid = []
        for name, cls in self._implemented():
            owner = self._owner(cls)
            generated = getattr(owner.__dict__["eval_roofline"], SYNTHESIZED_ATTR, False)
            if owner is not cls or not generated:
                invalid.append(f"{name} (owned by {owner.__name__})")
        assert not invalid, f"ops without their own generated evaluator: {invalid}"


class TestInstallOutcomes:
    """What ``maybe_install_eval_roofline`` does with each kind of entry."""

    def test_a_subclass_without_an_entry_installs_nothing(self):
        from tileops.ops._roofline_codegen import maybe_install_eval_roofline
        from tileops.ops.op_base import Op

        class _NotAManifestOp(Op):
            pass

        maybe_install_eval_roofline(_NotAManifestOp)
        assert "eval_roofline" not in _NotAManifestOp.__dict__

    def test_a_spec_only_entry_installs_nothing(self):
        from tileops.ops._roofline_codegen import maybe_install_eval_roofline

        class _SpecOnly:
            __manifest_status__ = "spec-only"
            __manifest_roofline__ = {"flops": "N", "bytes": "N * elem_bytes"}
            __manifest_signature__ = {"inputs": {}, "outputs": {}}

        maybe_install_eval_roofline(_SpecOnly)
        assert "eval_roofline" not in _SpecOnly.__dict__

    def test_an_implemented_entry_with_an_empty_roofline_raises(self):
        """An empty block is as absent as no key: ``roofline`` is required of
        every entry, so it is not a configuration codegen may pass over."""
        from tileops.ops._roofline_codegen import maybe_install_eval_roofline

        class _EmptyRoofline:
            __manifest_status__ = "implemented"
            __manifest_roofline__ = {}
            __manifest_signature__ = {"inputs": {}, "outputs": {}}

        with pytest.raises(ValueError, match="declares no roofline block"):
            maybe_install_eval_roofline(_EmptyRoofline)

    @pytest.mark.parametrize("block", [None, "flops: N", []])
    def test_a_loaded_entry_without_a_usable_roofline_raises(self, block, monkeypatch):
        """The production path: the entry comes from the manifest loader. An
        absent key reaches here as ``None``, which the class-attached path
        cannot express — it falls back to the loader instead."""
        import tileops.ops._roofline_codegen as codegen

        entry = {"status": "implemented", "signature": {"inputs": {}, "outputs": {}}}
        if block is not None:
            entry["roofline"] = block
        monkeypatch.setattr(codegen, "try_load_entry", lambda name: entry)

        class _Loaded:
            pass

        with pytest.raises(ValueError, match="declares no roofline block"):
            codegen.maybe_install_eval_roofline(_Loaded)

    def test_an_unsynthesizable_formula_installs_the_base_stub(self):
        from tileops.ops._roofline_codegen import maybe_install_eval_roofline
        from tileops.ops.op_base import Op

        class _BadFormula:
            __manifest_status__ = "implemented"
            __manifest_roofline__ = {"flops": "NOPE * 2", "bytes": "N * elem_bytes"}
            __manifest_signature__ = {"inputs": {}, "outputs": {}}

        maybe_install_eval_roofline(_BadFormula)
        assert _BadFormula.__dict__["eval_roofline"] is Op.eval_roofline

    def test_a_refused_formula_does_not_fall_through_to_the_parent(self):
        """Binding the stub is what keeps a child off its parent's formula.
        Leaving the attribute alone hands the answer to MRO lookup, and the
        child would be priced by an entry that is not its own."""
        import torch

        from tileops.ops.op_base import Op

        common = {
            "__manifest_status__": "implemented",
            "__manifest_signature__": {
                "inputs": {"x": {"dtype": "float16", "shape": "[N]"}},
                "outputs": {"y": {"dtype": "same_as(x)"}},
            },
            "forward": lambda self, *a, **kw: None,
            "_infer_output_shapes": lambda self, x_shape: {"y": x_shape},
            "_validate_dtypes": lambda self, *a: None,
            "default_kernel_map": property(lambda self: {}),
        }
        parent = type(
            "_PricedParent",
            (Op,),
            {**common, "__manifest_roofline__": {
                "vars": {"N": "x.shape[0]"}, "flops": "N", "bytes": "7 * N * elem_bytes"}},
        )  # fmt: skip
        child = type(
            "_RefusedChild",
            (parent,),
            {**common, "__manifest_roofline__": {
                "vars": {"N": "x.shape[0]"}, "flops": "NOPE * 2", "bytes": "N * elem_bytes"}},
        )  # fmt: skip

        assert child.eval_roofline is Op.eval_roofline
        op = parent.__new__(parent)
        op.x_shape, op.dtype = (10,), torch.float16
        assert op.eval_roofline() == (10, 140)
        with pytest.raises(TypeError, match="abstract"):
            child()


class TestThroughClassCreation:
    """What a real ``class X(Op)`` ends up with, not what the installer returns.

    ``Op.__init_subclass__`` runs three other codegen passes around this one;
    calling the installer alone would not exercise the path an op arrives by.
    """

    BASE = {
        "__manifest_status__": "implemented",
        "__manifest_signature__": {
            "inputs": {"x": {"dtype": "float16", "shape": "[N]"}},
            "outputs": {"y": {"dtype": "same_as(x)"}},
        },
        "forward": lambda self, *a, **kw: None,
        "_infer_output_shapes": lambda self, x_shape: {"y": x_shape},
        "_validate_dtypes": lambda self, *a: None,
        "default_kernel_map": property(lambda self: {}),
    }
    GOOD = {"vars": {"N": "product(x.shape)"}, "flops": "N", "bytes": "N * elem_bytes"}
    BAD = {"vars": {}, "flops": "NOPE", "bytes": "1"}

    def _build(self, name, roofline, base=None, signature=None):
        from tileops.ops.op_base import Op

        body = {**self.BASE, "__manifest_roofline__": roofline}
        if signature is not None:
            body["__manifest_signature__"] = signature
        return type(name, (base or Op,), body)

    def test_a_sound_entry_arrives_with_a_generated_evaluator(self):
        import torch

        from tileops.ops._roofline_emit import SYNTHESIZED_ATTR

        cls = self._build("_ClassGood", self.GOOD)
        assert getattr(cls.__dict__["eval_roofline"], SYNTHESIZED_ATTR, False)
        op = cls.__new__(cls)
        op.x_shape, op.dtype = (10,), torch.float16
        assert op.eval_roofline() == (10, 20)

    def test_a_refused_entry_arrives_abstract(self):
        from tileops.ops.op_base import Op

        cls = self._build("_ClassBad", self.BAD)
        assert cls.__dict__["eval_roofline"] is Op.eval_roofline
        with pytest.raises(TypeError, match="abstract"):
            cls()

    def test_a_refused_child_does_not_inherit_its_parent_formula(self):
        from tileops.ops.op_base import Op

        parent = self._build("_ClassParent", self.GOOD)
        child = self._build("_ClassChild", self.BAD, base=parent)
        assert child.eval_roofline is Op.eval_roofline

    def test_an_unread_malformed_block_still_arrives_generated(self):
        """`outputs` settles only `out_elem_bytes`, which this formula never says."""
        from tileops.ops._roofline_emit import SYNTHESIZED_ATTR

        cls = self._build(
            "_ClassPartial",
            self.GOOD,
            signature={"inputs": {"x": {"dtype": "float16", "shape": "[N]"}}, "outputs": 9},
        )
        assert getattr(cls.__dict__["eval_roofline"], SYNTHESIZED_ATTR, False)


class TestTotality:
    """The analysis answers whatever YAML produced, without raising."""

    @pytest.mark.parametrize(
        "roofline",
        [None, 5, "flops: N", [], {}, {"flops": {"nested": 1}, "bytes": 2}],
        ids=["none", "scalar", "string", "list", "empty", "nested"],
    )
    @pytest.mark.parametrize(
        "signature",
        [None, 5, {}, {"inputs": 5}, {"inputs": {7: {}}}, {"outputs": []}],
        ids=["none", "scalar", "empty", "bad-inputs", "bad-key", "bad-outputs"],
    )
    def test_no_shape_of_entry_raises(self, roofline, signature):
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline("FakeOp", roofline=roofline, signature=signature)
        assert isinstance(result.diagnostics, tuple)
        # Nothing emits from an entry this broken.
        assert result.plan is None or not result.blocking

    @pytest.mark.parametrize(
        "expr",
        ["1" + "+1" * 500, "-" * 10000 + "1"],
        ids=["deep-binop", "deep-unary"],
    )
    def test_an_expression_too_deep_to_parse_is_a_verdict(self, expr):
        """Exhausting the parser raised RecursionError or MemoryError, which the
        installer would have let take down the importing module."""
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"vars": {"N": expr}, "flops": "N", "bytes": "1"},
            signature={"inputs": {}, "outputs": {"y": {}}},
        )
        assert result.plan is None
        assert result.diagnostics

    def test_a_self_referential_entry_does_not_hang(self):
        from tileops.manifest.roofline_analysis import analyze_roofline

        loop: dict = {"flops": "1", "bytes": "1"}
        loop["vars"] = loop
        analyze_roofline("FakeOp", roofline=loop, signature={"inputs": {}})

    def test_every_message_names_the_op(self):
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"vars": {"N": "nope.shape[0]"}, "flops": "ALSO_NOPE", "bytes": "1"},
            signature={"inputs": {}, "outputs": {"y": {}}},
        )
        assert result.diagnostics
        for d in result.diagnostics:
            assert d.message.startswith("FakeOp: "), d.message


class TestNameSafety:
    """A declared name must mean in the emitted body what it meant in the entry."""

    SIG = {"inputs": {}, "outputs": {"y": {}}}

    def test_two_names_python_reads_as_one_collide(self):
        """The parser normalizes identifiers, so `K` and `KELVIN SIGN` are one name.

        Comparing the raw strings let the second assignment overwrite the first
        with no collision found, and the evaluator returned the later value.
        """
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"vars": {"K": "1", "\u212a": "2"}, "flops": "K", "bytes": "1"},
            signature=self.SIG,
        )
        assert [d.code for d in result.diagnostics] == ["vars.collision"]
        assert result.plan is None

    @pytest.mark.parametrize("name", sorted({"self", "elem_bytes", "_flops"}))
    def test_a_name_the_body_binds_for_itself_is_refused(self, name):
        """A param called `self` emits `self = self.self`, and every later line
        then reads the param where the op was meant."""
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"vars": {}, "flops": name, "bytes": "1"},
            signature={**self.SIG, "params": {name: {"type": "int"}}},
        )
        assert result.plan is None
        assert any(d.code == "signature.reserved-name" for d in result.diagnostics)

    def test_a_key_that_normalizes_to_a_keyword_is_refused(self):
        """`\uff49\uff46` normalizes to `if`; the check ran on the original
        spelling, so the plan carried a name the body could not assign to."""
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"vars": {"\uff49\uff46": "1"}, "flops": "1", "bytes": "1"},
            signature={"inputs": {}, "outputs": {"y": {}}},
        )
        assert result.plan is None
        assert [d.code for d in result.diagnostics] == ["vars.key-keyword"]

    def test_a_vars_key_the_body_binds_for_itself_is_refused(self):
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"vars": {"_flops": "1"}, "flops": "1", "bytes": "1"},
            signature=self.SIG,
        )
        assert [d.code for d in result.diagnostics] == ["vars.key-reserved"]
        assert result.plan is None


class TestOneNameOneSource:
    """A name the formula reads must come from one place.

    The body binds inputs then params, so a name in both binds twice and the
    second wins; a name shared with a helper binds over it. Neither shows in the
    entry and both fail only when the evaluator runs.
    """

    OUT = {"y": {}}

    def test_a_name_in_both_blocks_is_refused_when_read(self):
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"vars": {"N": "product(x.shape)"}, "flops": "N", "bytes": "N"},
            signature={"inputs": {"x": {}}, "params": {"x": {}}, "outputs": self.OUT},
        )
        assert result.plan is None
        assert [d.code for d in result.diagnostics] == ["signature.name-in-two-blocks"]

    def test_a_declared_name_shadowing_a_helper_is_refused_when_read(self):
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"vars": {"N": "sum(range(3))"}, "flops": "N", "bytes": "1"},
            signature={"inputs": {}, "params": {"sum": {}}, "outputs": self.OUT},
        )
        assert result.plan is None
        assert [d.code for d in result.diagnostics] == ["signature.name-shadows-helper"]

    def test_declaring_a_helper_name_without_reading_it_is_allowed(self):
        """Real entries declare `min` and `max` without saying them in a formula."""
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"vars": {}, "flops": "1", "bytes": "1"},
            signature={"inputs": {}, "params": {"min": {}}, "outputs": self.OUT},
        )
        assert result.plan is not None
        assert not result.diagnostics


class TestNormalizedNamesResolve:
    """A name Python normalizes must resolve, not just collide."""

    @pytest.mark.parametrize("where", ["vars", "params"])
    def test_a_name_needing_normalization_is_found(self, where):
        """`\u212a` is the Kelvin sign; Python reads it as `K`, and so must the
        allowed-name set, or a valid formula is refused for an unknown name."""
        from tileops.manifest.roofline_analysis import analyze_roofline

        kelvin = "\u212a"
        roofline = {"vars": {}, "flops": "K", "bytes": "1"}
        signature = {"inputs": {}, "outputs": {"y": {}}}
        if where == "vars":
            roofline["vars"] = {kelvin: "1"}
        else:
            signature["params"] = {kelvin: {"type": "int"}}
        result = analyze_roofline("FakeOp", roofline=roofline, signature=signature)
        assert not result.diagnostics
        assert result.plan is not None


class TestComprehensionLocals:
    """A comprehension's target is its own, not a declared input of that name."""

    def test_a_comprehension_local_is_not_an_input_read(self):
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"vars": {"N": "len([1 for x in range(3)])"}, "flops": "N", "bytes": "1"},
            signature={"inputs": {"x": 5}, "outputs": {"y": {}}},
        )
        # The unreadable declaration is still reported; it just does not refuse a
        # formula that never reads that input.
        assert result.plan is not None

    def test_a_comprehension_local_is_not_bound(self):
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"vars": {"N": "len([1 for x in range(3)])"}, "flops": "N", "bytes": "1"},
            signature={"inputs": {"x": {"dtype": "float16"}}, "outputs": {"y": {}}},
        )
        assert result.plan is not None
        assert [b.name for b in result.plan.bindings] == []


class TestComprehensionShadowing:
    """A comprehension target is a local, whatever an outer name of that
    spelling happens to be.

    Tracking only that a name is bound reads `[x.shape for x in range(1)]` as a
    tensor read when an input is called `x`, and lets `[sum(...) for sum in ...]`
    call an integer. Tracking the kind settles both, and stops a plain
    `[x for x in range(3)]` being refused because an unrelated input shares the
    name.
    """

    IN = {"x": {"dtype": "float16"}}
    OUT = {"y": {}}

    def _run(self, expr, inputs):
        from tileops.manifest.roofline_analysis import analyze_roofline

        return analyze_roofline(
            "FakeOp",
            roofline={"vars": {"N": expr}, "flops": "N", "bytes": "1"},
            signature={"inputs": inputs, "outputs": self.OUT},
        )

    def test_a_target_shadowing_an_input_is_not_a_tensor(self):
        result = self._run("len([x.shape for x in range(1)])", self.IN)
        assert result.plan is None
        assert [d.code for d in result.diagnostics] == ["vars.attribute-operand"]

    def test_a_target_shadowing_a_helper_is_not_the_helper(self):
        result = self._run("len([sum(range(2)) for sum in range(1)])", {})
        assert result.plan is None
        assert [d.code for d in result.diagnostics] == ["vars.unknown-helper"]

    def test_a_target_sharing_an_input_name_is_still_legal(self):
        from tileops.ops._roofline_emit import emit_eval_roofline

        result = self._run("len([x for x in range(3)])", self.IN)
        assert not result.diagnostics
        assert result.plan is not None
        emit_eval_roofline(result.plan)


class TestUnreadableInputsAreNotGuessed:
    """While the declared inputs cannot be read, which names are tensors is not
    known, so the judgments that turn on it are left to the unjudged line."""

    def test_no_verdict_is_claimed_on_a_name_that_might_be_an_input(self):
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"vars": {"N": "product(x.shape)"}, "flops": "N", "bytes": "N"},
            signature={"inputs": 5, "outputs": {"y": {}}},
        )
        assert result.plan is None
        assert [d.code for d in result.diagnostics] == ["signature.inputs.not-a-mapping"]


class TestUnreadableInputAttributes:
    """An input whose attributes will not read states no optionality."""

    def test_an_unread_optionality_is_not_invented(self):
        """Binding it as not-optional would decide the fact rather than read it."""
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"vars": {"N": "product(x.shape)"}, "flops": "N", "bytes": "N"},
            signature={"inputs": {"x": 5}, "outputs": {"y": {}}},
        )
        assert [d.code for d in result.diagnostics] == ["signature.input-attributes"]
        assert result.plan is None
        assert [u.missing for u in result.unjudged] == ["signature.inputs"]

    def test_an_unread_optionality_on_an_input_nothing_binds_still_emits(self):
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"vars": {}, "flops": "1", "bytes": "1"},
            signature={"inputs": {"x": 5}, "outputs": {"y": {}}},
        )
        assert [d.code for d in result.diagnostics] == ["signature.input-attributes"]
        assert result.plan is not None


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

        from tileops.ops._roofline_emit import SYNTHESIZED_ATTR

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
