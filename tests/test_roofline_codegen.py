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
        """The whole block settles nothing, so it is needed when a part is."""
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

    def test_mixing_both_modes_does_not_hide_either_half(self):
        """Both halves are present, so both are judged: stopping at the mode
        verdict leaves whichever half is also wrong unreported."""
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"func": "no.such.callable", "flops": "NOPE", "bytes": "1"},
            signature={"inputs": {}, "outputs": {"y": {}}},
        )
        assert result.plan is None
        assert {d.code for d in result.diagnostics} == {
            "roofline.mixed-modes",
            "func.import",
            "arith.unknown-name",
        }

    def test_mixing_both_modes_emits_nothing_even_when_both_halves_are_sound(self):
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={
                "func": "tileops.perf.formulas._binary_broadcast_roofline",
                "flops": "1",
                "bytes": "1",
            },
            signature={"inputs": {}, "outputs": {"y": {}}},
        )
        assert result.plan is None
        assert [d.code for d in result.diagnostics] == ["roofline.mixed-modes"]

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

    @pytest.mark.parametrize(
        "inputs",
        [{"x": 5}, {7: {}}],
        ids=["unreadable-attributes", "unreadable-key"],
    )
    def test_a_block_read_in_part_stops_a_formula_that_reads_it(self, inputs):
        """A block read in part states an incomplete set of names, which is what
        a formula reading one of them needs."""
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"vars": {"N": "product(x.shape)"}, "flops": "N", "bytes": "N"},
            signature={"inputs": inputs, "outputs": {"y": {}}},
        )
        assert result.plan is None
        assert {u.missing for u in result.unjudged} == {"signature.inputs"}

    @pytest.mark.parametrize(
        "inputs",
        [{"x": 5}, {7: {}}],
        ids=["unreadable-attributes", "unreadable-key"],
    )
    def test_a_block_read_in_part_does_not_stop_a_formula_that_does_not(self, inputs):
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"vars": {}, "flops": "1", "bytes": "1"},
            signature={"inputs": inputs, "outputs": {"y": {}}},
        )
        assert result.plan is not None
        assert not result.unjudged

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
        """No expression can name either: one is not a string, the other does not
        parse. The formula is not reaching for it, so the entry draws the line
        and the evaluator still emits.
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

    def test_no_verdict_is_claimed_on_a_name_that_might_be_an_input(self):
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"vars": {"N": "product(x.shape)"}, "flops": "N", "bytes": "N"},
            signature={"inputs": 5, "outputs": {"y": {}}},
        )
        assert result.plan is None
        assert [d.code for d in result.diagnostics] == ["signature.inputs.not-a-mapping"]


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
        """The loader path. An absent key reaches here as ``None``, which the
        class-attached path cannot express."""
        import tileops.ops._roofline_codegen as codegen

        entry = {"status": "implemented", "signature": {"inputs": {}, "outputs": {}}}
        if block is not None:
            entry["roofline"] = block
        monkeypatch.setattr(codegen, "try_load_entry", lambda name: entry)

        class _Loaded:
            pass

        with pytest.raises(ValueError, match="declares no roofline block"):
            codegen.maybe_install_eval_roofline(_Loaded)


class TestThroughClassCreation:
    """What a real ``class X(Op)`` ends up with.

    ``Op.__init_subclass__`` runs three other codegen passes around this one, so
    this is the path an op actually arrives by.
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


class TestNoDefectHidesAnother:
    """Every defect that appears alone appears beside any other.

    Suppression has arrived by three different routes -- a malformed signature,
    a mixed mode, a judgment raised at the point of refusal -- so the property
    is held by a cross product rather than by a case per route. Adding an
    injector below extends the matrix against every existing one.
    """

    SIG = {
        "inputs": {"x": {"dtype": "float16", "shape": "[N]"}},
        "outputs": {"y": {"dtype": "same_as(x)"}},
        "params": {"alpha": {"type": "float"}},
    }
    ROOFLINE = {"vars": {"N": "product(x.shape)"}, "flops": "N", "bytes": "N * elem_bytes"}

    # name -> (slot, inject). Two injectors sharing a slot overwrite each other,
    # which is a conflict in the case rather than a defect in the analysis.
    # name -> (slot, inject, signal). Two injectors sharing a slot overwrite
    # each other, which is a conflict in the case rather than a defect in the
    # analysis. The signal is what this defect must draw on its own: without it
    # the matrix would compare the analysis against itself and miss a judgment
    # that stopped being made at all.
    INJECT = {
        "func-unresolvable": ("mode", lambda r, g: r.update(func="no.such.mod.fn"), "func.import"),
        "flops-unknown-name": ("flops", lambda r, g: r.update(flops="NOPE"), "arith.unknown-name"),
        "flops-syntax": ("flops", lambda r, g: r.update(flops="1 +"), "flops.syntax"),
        "bytes-forbidden": (
            "bytes",
            lambda r, g: r.update(bytes="N[0]"),
            "arith.forbidden-construct",
        ),
        "bytes-syntax": ("bytes", lambda r, g: r.update(bytes="1 +"), "bytes.syntax"),
        "vars-bad-attribute": (
            "vars-m",
            lambda r, g: r["vars"].update(M="N.real"),
            "vars.attribute",
        ),
        "vars-unknown-helper": (
            "vars-p",
            lambda r, g: r["vars"].update(P="open(1)"),
            "vars.unknown-helper",
        ),
        "vars-key-keyword": (
            "vars-k",
            lambda r, g: r["vars"].update(**{"if": "1"}),
            "vars.key-keyword",
        ),
        "vars-key-reserved": (
            "vars-r",
            lambda r, g: r["vars"].update(_flops="1"),
            "vars.key-reserved",
        ),
        "inputs-not-mapping": (
            "inputs",
            lambda r, g: g.update(inputs=5),
            "signature.inputs.not-a-mapping",
        ),
        "input-attributes-unreadable": (
            "inputs",
            lambda r, g: g.update(inputs={"x": 5}),
            "unjudged:signature.inputs",
        ),
        "outputs-not-mapping": (
            "outputs",
            lambda r, g: g.update(outputs=5),
            "signature.outputs.not-a-mapping",
        ),
        "params-not-mapping": (
            "params",
            lambda r, g: g.update(params=5),
            "signature.params.not-a-mapping",
        ),
        "param-non-string": (
            "params",
            lambda r, g: g.update(params={7: {"type": "int"}}),
            "signature.non-string-name",
        ),
        "param-reserved": (
            "params",
            lambda r, g: g.update(params={"self": {"type": "int"}}),
            "signature.reserved-name",
        ),
    }

    def _seen(self, names):
        """What the entry drew: diagnostic codes, plus the facts left unjudged.

        A judgment the analysis declines for want of a fact is not a lost
        verdict, so the fact it names counts as covering it.
        """
        import copy

        from tileops.manifest.roofline_analysis import analyze_roofline

        roofline = copy.deepcopy(self.ROOFLINE)
        signature = copy.deepcopy(self.SIG)
        for name in names:
            self.INJECT[name][1](roofline, signature)
        result = analyze_roofline("FakeOp", roofline=roofline, signature=signature)
        return {d.code for d in result.diagnostics} | {
            f"unjudged:{u.missing}" for u in result.unjudged
        }

    def test_the_sound_entry_draws_nothing(self):
        assert self._seen([]) == set()

    @pytest.mark.parametrize("name", sorted(INJECT))
    def test_each_defect_draws_its_signal_alone(self, name):
        """Pinned rather than merely non-empty: the matrix below compares the
        analysis against itself, so a judgment that stopped being made would
        look consistent to it."""
        assert self.INJECT[name][2] in self._seen([name])

    def test_no_pair_loses_what_either_draws_alone(self):
        """A verdict either still stands beside the other defect, or the
        analysis says the other defect left it unjudged. Silently dropping it is
        what this forbids."""
        import itertools

        alone = {name: self._seen([name]) for name in self.INJECT}
        lost = []
        for a, b in itertools.combinations(sorted(self.INJECT), 2):
            if self.INJECT[a][0] == self.INJECT[b][0]:
                continue
            both = self._seen([a, b])
            # A fact the pair leaves unreadable that neither left unreadable
            # alone: the judgment it carried moved rather than vanished.
            excused = {c for c in both if c.startswith("unjudged:")} - (alone[a] | alone[b])
            for name in (a, b):
                if excused:
                    continue
                for code in sorted(alone[name] - both):
                    lost.append(f"{a} + {b} lost {name}'s {code}")
        assert not lost, lost


class TestEveryPlanRuns:
    """A plan the analysis builds emits, compiles and returns two ints.

    The other half of the boundary's contract, and the one that failed by
    example three times -- an async comprehension, a param named `self`, a
    fullwidth keyword. A cross product holds it instead: the entry is named
    for a real single-output op, because `out_elem_bytes` resolves the
    declared output's dtype through the manifest by the class's name.
    """

    OP = "SiluAndMulFwdOp"
    SIG = {
        "inputs": {"x": {"dtype": "float16", "shape": "[N]"}},
        "outputs": {"output": {"dtype": "same_as(x)"}},
        "params": {"alpha": {"type": "float"}},
    }
    ROOFLINE = {"vars": {"N": "product(x.shape)"}, "flops": "N", "bytes": "N * elem_bytes"}

    # Defects and rarely-written-but-legal forms alike: the ones that yield a
    # plan are the ones this holds.
    INJECT = {
        "async-comprehension": (
            "vars-a",
            lambda r, g: r["vars"].update(A="len([q async for q in range(1)])"),
        ),
        "soft-keyword": ("vars-t", lambda r, g: r["vars"].update(type="1")),
        "nfkc-name": ("vars-k", lambda r, g: r["vars"].update(**{"\u212a": "1"})),
        "keyword-fullwidth": ("vars-w", lambda r, g: r["vars"].update(**{"\uff49\uff46": "1"})),
        "param-self": (
            "params",
            lambda r, g: (
                g.update(params={"self": {"type": "int"}}),
                r.update(flops="N * self"),
            ),
        ),
        "param-shadows-helper": (
            "params",
            lambda r, g: (
                g.update(params={"sum": {"type": "int"}}),
                r["vars"].update(S="sum(d for d in x.shape)"),
            ),
        ),
        "name-in-two-blocks": ("params2", lambda r, g: g.update(params={"x": {"type": "int"}})),
        "out-elem-bytes": ("bytes", lambda r, g: r.update(bytes="N * out_elem_bytes")),
        "optional-input": (
            "inputs",
            lambda r, g: g["inputs"].update(b={"dtype": "float16", "optional": True}),
        ),
        "reads-optional": ("vars-o", lambda r, g: r["vars"].update(H="1 if b is None else 2")),
        "comprehension": ("vars-c", lambda r, g: r["vars"].update(C="sum(d for d in x.shape)")),
        "helper-chain": (
            "vars-h",
            lambda r, g: r["vars"].update(Hh="max(1, min(2, len(x.shape)))"),
        ),
        "param-read": ("flops", lambda r, g: r.update(flops="N * alpha")),
        "deep-expression": ("vars-d", lambda r, g: r["vars"].update(D="1" + "+1" * 400)),
        "unused-param": (
            "params3",
            lambda r, g: g.setdefault("params", {}).update(u={"type": "int"}),
        ),
        "vars-key-reserved": (
            "vars-r",
            lambda r, g: (r["vars"].update(_flops="1"), r.update(flops="_flops")),
        ),
    }

    def _probe_op(self):
        import torch

        op = type(self.OP, (), {})()
        op.x_shape, op.x = (8,), torch.empty((8,), dtype=torch.float16, device="meta")
        op.b_shape, op.b = None, None
        op.dtype, op.alpha, op.u = torch.float16, 2.0, 1
        # Names a declared param may legally carry, which the body must not read
        # in place of its own bindings.
        op.self = 1
        op.sum = 1
        return op

    def _plan(self, names):
        """The plan for these injectors, or None if the entry was refused."""
        import copy

        from tileops.manifest.roofline_analysis import analyze_roofline

        roofline, signature = copy.deepcopy(self.ROOFLINE), copy.deepcopy(self.SIG)
        for name in names:
            self.INJECT[name][1](roofline, signature)
        return analyze_roofline(self.OP, roofline=roofline, signature=signature).plan

    @staticmethod
    def _plan_defects(plan):
        """Ways a plan cannot be emitted soundly, whatever a probe object holds.

        One attribute cannot be both a tensor and a param's value, so a name
        bound twice is not observable by running the evaluator against any one
        object. It is observable here.
        """
        from tileops.manifest.roofline_analysis import EMITTER_NAMES, VARS_HELPERS

        names = [b.name for b in plan.bindings]
        defects = []
        if len(names) != len(set(names)):
            defects.append(f"binds a name twice: {names}")
        for name in names:
            if name in EMITTER_NAMES:
                defects.append(f"binds {name!r}, which the body binds for itself")
            if name in VARS_HELPERS:
                defects.append(f"binds {name!r}, which shadows the helper of that name")
        for name, _ in plan.vars_program:
            if name in EMITTER_NAMES or name in VARS_HELPERS:
                defects.append(f"assigns {name!r}, which is not the author's to bind")
        return defects

    def _run(self, names):
        """The value the plan for these injectors computes, or None if refused."""
        import copy

        from tileops.manifest.roofline_analysis import analyze_roofline
        from tileops.ops._roofline_emit import emit_eval_roofline

        roofline, signature = copy.deepcopy(self.ROOFLINE), copy.deepcopy(self.SIG)
        for name in names:
            self.INJECT[name][1](roofline, signature)
        result = analyze_roofline(self.OP, roofline=roofline, signature=signature)
        if result.plan is None:
            return None
        return emit_eval_roofline(result.plan)(self._probe_op())

    def test_the_sound_entry_runs(self):
        assert self._run([]) == (8, 16)

    @pytest.mark.parametrize("name", sorted(INJECT))
    def test_each_injector_alone_either_refuses_or_runs(self, name):
        value = self._run([name])
        assert value is None or (
            isinstance(value, tuple) and [type(v) for v in value] == [int, int]
        )

    def test_no_pair_yields_a_plan_that_cannot_run(self):
        import itertools

        broken = []
        for a, b in itertools.combinations(sorted(self.INJECT), 2):
            if self.INJECT[a][0] == self.INJECT[b][0]:
                continue
            try:
                value = self._run([a, b])
            except Exception as exc:  # noqa: BLE001 - that it raises is the failure
                broken.append(f"{a} + {b}: {type(exc).__name__}: {exc}")
                continue
            if value is not None and [type(v) for v in value] != [int, int]:
                broken.append(f"{a} + {b}: returned {value!r}")
            plan = self._plan([a, b])
            if plan is not None:
                broken += [f"{a} + {b}: {d}" for d in self._plan_defects(plan)]
        assert not broken, broken

    @pytest.mark.parametrize("name", sorted(INJECT))
    def test_no_plan_binds_a_name_it_does_not_own(self, name):
        plan = self._plan([name])
        assert plan is None or not self._plan_defects(plan)


class TestNothingLegalIsRefused:
    """A formula the spec allows is accepted, and draws nothing.

    The other two matrices ask what a wrong entry draws. This one asks what a
    right one does not: a gate tightened against a defect can refuse a form the
    spec permits, and no defect corpus would notice.
    """

    OP = "SiluAndMulFwdOp"
    SIG = {
        "inputs": {"x": {"dtype": "float16", "shape": "[N]"}},
        "outputs": {"output": {"dtype": "same_as(x)"}},
        "params": {"alpha": {"type": "float"}},
    }
    ROOFLINE = {"vars": {"N": "product(x.shape)"}, "flops": "N", "bytes": "N * elem_bytes"}

    # Every one of these is legal: it appears in the manifest, or §4.4.3 permits it.
    LEGAL = {
        "optional-input": (
            "in-b",
            lambda r, g: g["inputs"].update(b={"dtype": "float16", "optional": True}),
        ),
        "presence-test": (
            "v-h",
            lambda r, g: (
                g["inputs"].update(b={"dtype": "float16", "optional": True}),
                r["vars"].update(H="1 if b is None else 2"),
            ),
        ),
        "comprehension": ("v-c", lambda r, g: r["vars"].update(C="sum(d for d in x.shape)")),
        "nested-helpers": ("v-n", lambda r, g: r["vars"].update(Nn="max(1, min(2, len(x.shape)))")),
        "isinstance-branch": (
            "v-i",
            lambda r, g: r["vars"].update(
                I="alpha[0] if isinstance(alpha, (tuple, list)) else alpha"
            ),
        ),
        "earlier-var": ("v-e", lambda r, g: r["vars"].update(E="N * 2")),
        "shape-index": ("v-s", lambda r, g: r["vars"].update(S="x.shape[0]")),
        "ndim": ("v-d", lambda r, g: r["vars"].update(D="x.ndim")),
        "soft-keyword-name": ("v-t", lambda r, g: r["vars"].update(type="1")),
        "nfkc-name": ("v-k", lambda r, g: r["vars"].update(**{"\u212a": "1"})),
        "out-elem-bytes": ("bytes", lambda r, g: r.update(bytes="N * out_elem_bytes")),
        "param-in-arithmetic": ("flops", lambda r, g: r.update(flops="N * alpha")),
        "numeric-helpers": (
            "flops2",
            lambda r, g: r.update(flops="ceil(N / 2) + floor(N / 3) + log2(N + 1)"),
        ),
        "conditional-arithmetic": (
            "bytes2",
            lambda r, g: r.update(bytes="(N if N > 0 else 1) * elem_bytes"),
        ),
        "unread-param": ("p-u", lambda r, g: g["params"].update(unused={"type": "int"})),
        "unread-input": ("in-u", lambda r, g: g["inputs"].update(w={"dtype": "float16"})),
    }

    def _refusal(self, names):
        """Why these legal forms were refused, or None if they were accepted."""
        import copy

        from tileops.manifest.roofline_analysis import analyze_roofline

        roofline, signature = copy.deepcopy(self.ROOFLINE), copy.deepcopy(self.SIG)
        for name in names:
            self.LEGAL[name][1](roofline, signature)
        result = analyze_roofline(self.OP, roofline=roofline, signature=signature)
        if result.plan is None:
            return [d.code for d in result.diagnostics] or ["refused without saying why"]
        if result.diagnostics:
            return [f"accepted but still drew {[d.code for d in result.diagnostics]}"]
        return None

    @pytest.mark.parametrize("name", sorted(LEGAL))
    def test_each_legal_form_is_accepted_alone(self, name):
        assert self._refusal([name]) is None

    def test_no_pair_of_legal_forms_is_refused(self):
        import itertools

        refused = []
        for a, b in itertools.combinations(sorted(self.LEGAL), 2):
            if self.LEGAL[a][0] == self.LEGAL[b][0]:
                continue
            why = self._refusal([a, b])
            if why is not None:
                refused.append(f"{a} + {b}: {why}")
        assert not refused, refused


class TestTotality:
    """The analysis answers whatever YAML produced, without raising."""

    ROOFLINES = [None, 5, "flops: N", [], {}, {"flops": {"nested": 1}, "bytes": 2}]
    SIGNATURES = [None, 5, {}, {"inputs": 5}, {"inputs": {7: {}}}, {"outputs": []}]

    def test_no_shape_of_entry_raises(self):
        from tileops.manifest.roofline_analysis import analyze_roofline

        raised = []
        for roofline in self.ROOFLINES:
            for signature in self.SIGNATURES:
                try:
                    result = analyze_roofline("FakeOp", roofline=roofline, signature=signature)
                except Exception as exc:  # noqa: BLE001 - that it raises is the failure
                    raised.append((roofline, signature, exc))
                    continue
                # Nothing emits from an entry this broken.
                assert result.plan is None or not result.blocking
        assert not raised, raised

    @pytest.mark.parametrize(
        "expr",
        ["1" + "+1" * 500, "-" * 10000 + "1"],
        ids=["deep-binop", "deep-unary"],
    )
    def test_an_expression_too_deep_to_parse_is_a_verdict(self, expr):
        """Exhausting the parser is a verdict, not an exception out of the
        analysis."""
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


class TestNameSafety:
    """A declared name must mean in the emitted body what it meant in the entry."""

    SIG = {"inputs": {}, "outputs": {"y": {}}}

    def test_two_names_python_reads_as_one_collide(self):
        """The parser normalizes identifiers, so `K` and `KELVIN SIGN` are one
        name, and the second assignment would shadow the first."""
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
        """A param called `self` emits `self = self.self`, after which every line
        reads the param rather than the op."""
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"vars": {}, "flops": name, "bytes": "1"},
            signature={**self.SIG, "params": {name: {"type": "int"}}},
        )
        assert result.plan is None
        assert any(d.code == "signature.reserved-name" for d in result.diagnostics)

    def test_a_key_that_normalizes_to_a_keyword_is_refused(self):
        """`\uff49\uff46` normalizes to `if`, which the body cannot assign to."""
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
        """Declaring a helper name is allowed; reading it is not."""
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"vars": {}, "flops": "1", "bytes": "1"},
            signature={"inputs": {}, "params": {"min": {}}, "outputs": self.OUT},
        )
        assert result.plan is not None
        assert not result.diagnostics

    @pytest.mark.parametrize("where", ["vars", "params"])
    def test_a_name_needing_normalization_is_found(self, where):
        """`\u212a` is the Kelvin sign; Python reads it as `K`, and so must the
        allowed-name set."""
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


class TestEmittableByConstruction:
    """A plan the analysis builds must be one emission can compile and run."""

    OUT = {"y": {}}

    def test_an_async_comprehension_is_refused(self):
        """The generated body is a plain function, so it cannot hold one."""
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={
                "vars": {"N": "len([x async for x in range(1)])"},
                "flops": "N",
                "bytes": "1",
            },
            signature={"inputs": {}, "outputs": self.OUT},
        )
        assert result.plan is None
        assert [d.code for d in result.diagnostics] == ["vars.async-comprehension"]

    @pytest.mark.parametrize("name", ["type", "match", "case"])
    def test_a_soft_keyword_is_an_ordinary_name(self, name):
        """`type` is a plausible param name, and `type = 1` is valid Python."""
        from tileops.manifest.roofline_analysis import analyze_roofline
        from tileops.ops._roofline_emit import emit_eval_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"vars": {name: "1"}, "flops": name, "bytes": "1"},
            signature={"inputs": {}, "outputs": self.OUT},
        )
        assert not result.diagnostics
        assert result.plan is not None
        emit_eval_roofline(result.plan)

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
