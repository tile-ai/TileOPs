"""The ``eval_roofline`` generated from a legacy entry's roofline block."""

import pytest

pytestmark = pytest.mark.smoke


class TestLegacyEvaluator:
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

    def test_a_non_mapping_signature_block_is_a_verdict(self):
        """A non-mapping reads as empty, which would make a name look declared.

        The verdict stands whether or not the formula reaches for that block;
        whether it also stops emission is `TestPartialSignature`'s subject.
        """
        from tileops.manifest.roofline_analysis import analyze_roofline

        for block in ("inputs", "outputs", "params"):
            result = analyze_roofline(
                "FakeOp",
                roofline={"flops": "1", "bytes": "1"},
                signature={"inputs": {}, block: 5},
            )
            assert [d.code for d in result.diagnostics] == [f"signature.{block}.not-a-mapping"]
            assert f"signature.{block} must be a mapping" in result.diagnostics[0].message

    def test_a_non_mapping_vars_is_a_verdict(self):
        """A falsey one must not read as an absent one."""
        from tileops.ops._roofline_codegen import synthesize_eval_roofline

        for bad in ([], "", False, 0, ["a"]):
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
        # Sound on both sides is still two modes, and still emits nothing.
        sound = analyze_roofline(
            "FakeOp",
            roofline={
                "func": "tileops.perf.formulas.gqa_dense_fwd_roofline",
                "flops": "1",
                "bytes": "1",
            },
            signature={"inputs": {}, "outputs": {"y": {}}},
        )
        assert sound.plan is None
        assert [d.code for d in sound.diagnostics] == ["roofline.mixed-modes"]

    def test_mixing_both_modes_is_a_verdict(self):
        from tileops.ops._roofline_codegen import synthesize_eval_roofline

        with pytest.raises(ValueError, match="cannot mix func and inline"):
            synthesize_eval_roofline(
                "FakeOp",
                roofline={"func": "tileops.perf.formulas.fused_moe_fwd_roofline", "flops": "1"},
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

    def test_a_block_read_in_part_does_not_stop_a_formula_that_does_not_read_it(self):
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(
            "FakeOp",
            roofline={"flops": "1", "bytes": "1"},
            signature={"inputs": {"x": 5}, "outputs": {"y": {}}},
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
        from tileops.manifest import load_manifest
        from tileops.manifest.registry import op_class
        from tileops.manifest.signature import is_legacy

        for name, entry in load_manifest().items():
            if entry.get("status") == "implemented" and is_legacy(entry):
                yield name, op_class(name, entry)

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

    def test_a_loaded_entry_without_a_usable_roofline_raises(self, monkeypatch):
        """The loader path. An absent key reaches here as ``None``, which the
        class-attached path cannot express."""
        import tileops.ops._roofline_codegen as codegen

        for block in (None, "flops: N", []):
            entry = {"status": "implemented", "signature": {"inputs": {}, "outputs": {}}}
            if block is not None:
                entry["roofline"] = block
            monkeypatch.setattr(codegen, "try_load_entry", lambda name, e=entry: e)

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


def _slots(label):
    """The manifest keys an injector writes. Two injectors sharing one overwrite
    each other, which is a conflict in the case rather than a defect found."""
    return frozenset(label.split(","))


# Verdicts that rest on which names a block declares, so an unreadable block
# leaves them unreachable: a name is looked for in inputs and in params, and
# either block being unreadable is enough to stop the lookup. Every other
# verdict is settled by the expression alone, and no unjudged block excuses
# losing one.
_NAME_RESOLUTION_CODES = frozenset(
    {
        "arith.unknown-name",
        "vars.unknown-name",
        "signature.name-in-two-blocks",
    }
)


class TestNoDefectHidesAnother:
    """Every defect in the entry text that appears alone appears beside any other.

    Suppression has arrived by three different routes -- a malformed signature,
    a mixed mode, a judgment raised at the point of refusal -- so the property
    is held by a cross product rather than by a case per route. Adding an
    injector below extends the matrix against every existing one.

    Scoped to what the entry text settles. Whether the assembled formula runs is
    settled only once there is a plan to run, so no entry that is refused
    carries that judgment, and `TestEveryPlanRuns` holds it instead.
    """

    SIG = {
        "inputs": {"x": {"dtype": "float16", "shape": "[N]"}},
        "outputs": {"y": {"dtype": "same_as(x)"}},
        "params": {"alpha": {"type": "float"}},
    }
    ROOFLINE = {"vars": {"N": "product(x.shape)"}, "flops": "N", "bytes": "N * elem_bytes"}

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
        "vars-call-form": (
            "vars-cf",
            lambda r, g: r["vars"].update(W="range(stop=3)"),
            "vars.call-form",
        ),
        "vars-call-form-nested": (
            "vars-cn",
            lambda r, g: r["vars"].update(Wn="len(min(1, 2, default=0))"),
            "vars.call-form",
        ),
        "vars-call-form-in-comprehension": (
            "vars-cc",
            lambda r, g: r["vars"].update(Wc="sum(len(d, d) for d in x.shape)"),
            "vars.call-form",
        ),
        "vars-call-form-starred": (
            "vars-cs",
            lambda r, g: r["vars"].update(Ws="len(*[x.shape, x.shape])"),
            "vars.call-form",
        ),
        "vars-call-form-kwargs": (
            "vars-ck",
            lambda r, g: r["vars"].update(Wk="len(**{})"),
            "vars.call-form",
        ),
        "arith-call-form": (
            "flops",
            lambda r, g: r.update(flops="ceil(N, 2)"),
            "arith.call-form",
        ),
        "vars-async-comprehension": (
            "vars-a",
            lambda r, g: r["vars"].update(A="len([q async for q in range(1)])"),
            "vars.async-comprehension",
        ),
        "name-in-two-blocks": (
            "params,flops",
            lambda r, g: (
                g.update(params={"x": {"type": "int"}}),
                r.update(flops="N * x"),
            ),
            "signature.name-in-two-blocks",
        ),
        "param-shadows-helper": (
            "params,vars-s,flops",
            lambda r, g: (
                g.update(params={"sum": {"type": "int"}}),
                r["vars"].update(S="sum(d for d in x.shape)"),
                r.update(flops="N * sum"),
            ),
            "signature.name-shadows-helper",
        ),
        "param-emitter-local": (
            "params",
            lambda r, g: g.update(params={"elem_bytes": {"type": "int"}}),
            "signature.reserved-name",
        ),
        "param-keyword": (
            "params",
            lambda r, g: g.update(params={"class": {"type": "int"}}),
            "signature.unusable-name",
        ),
        # Python reads an identifier NFKC-normalized, so each of these is the
        # name beside it once the generated body is parsed.
        "param-keyword-nfkc": (
            "params",
            lambda r, g: g.update(params={"\uff43lass": {"type": "int"}}),
            "signature.unusable-name",
        ),
        "param-emitter-nfkc": (
            "params",
            lambda r, g: g.update(params={"\uff53elf": {"type": "int"}}),
            "signature.reserved-name",
        ),
        "param-duplicate-nfkc": (
            "params",
            lambda r, g: g.update(params={"K": {"type": "int"}, "\u212a": {"type": "int"}}),
            "signature.duplicate-name",
        ),
        "vars-key-keyword-nfkc": (
            "vars-kw",
            lambda r, g: r["vars"].update(**{"\uff49\uff46": "1"}),
            "vars.key-keyword",
        ),
        "vars-key-reserved-nfkc": (
            "vars-rn",
            lambda r, g: r["vars"].update(**{"_\uff46lops": "1"}),
            "vars.key-reserved",
        ),
        "input-key-unreadable": (
            "inputs",
            lambda r, g: g.update(inputs={7: {}}),
            "unjudged:signature.inputs",
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
        # Full identity. Production calls two diagnostics one defect only when
        # code, path and subject agree, so comparing codes alone would not see
        # one of two same-code verdicts go missing.
        return {d.identity for d in result.diagnostics} | {
            ("unjudged", u.missing, u.judgment) for u in result.unjudged
        }

    def test_the_sound_entry_draws_nothing(self):
        assert self._seen([]) == set()

    def test_no_defect_is_lost_alone_or_beside_another(self):
        """Each defect draws its signal alone, and still draws it beside any
        other -- or the analysis says the other left it unjudged. The signal is
        pinned because the pairs below compare the analysis against itself, so a
        judgment that stopped being made would look consistent to them."""
        import itertools

        alone = {name: self._seen([name]) for name in self.INJECT}
        lost = [
            f"{name} alone no longer draws {signal}"
            for name, (_, _, signal) in sorted(self.INJECT.items())
            if signal not in {i[0] for i in alone[name]}
            and signal not in {f"unjudged:{i[1]}" for i in alone[name]}
        ]
        for a, b in itertools.combinations(sorted(self.INJECT), 2):
            if _slots(self.INJECT[a][0]) & _slots(self.INJECT[b][0]):
                continue
            both = self._seen([a, b])
            # A verdict may move to an unjudged line, but only a verdict that
            # reads the block left unreadable. Which names a block declares is
            # what the name-resolution verdicts rest on, and nothing else here
            # does: a form verdict is settled by the expression alone, so no
            # unjudged block excuses losing one.
            unreadable = {i[1] for i in both if i[0] == "unjudged"}
            resolves_names = bool(unreadable & {"signature.inputs", "signature.params"})
            for name in (a, b):
                for ident in sorted(alone[name] - both):
                    if resolves_names and ident[0] in _NAME_RESOLUTION_CODES:
                        continue
                    lost.append(f"{a} + {b} lost {name}'s {ident}")
        assert not lost, lost


class TestEveryPlanRuns:
    """A plan the analysis builds emits, compiles and returns two ints.

    The other half of the boundary's contract, and the one that failed by
    example three times -- an async comprehension, a param named `self`, a
    fullwidth keyword. A cross product holds it instead. `out_elem_bytes`
    resolves the declared output's dtype through the manifest by the class's
    name, so the synthetic entry is served under that name.
    """

    OP = "SyntheticLegacyFwdOp"
    SIG = {
        "inputs": {"x": {"dtype": "float16", "shape": "[N]"}},
        "outputs": {"output": {"dtype": "same_as(x)"}},
        "params": {"alpha": {"type": "float"}},
    }
    ROOFLINE = {"vars": {"N": "product(x.shape)"}, "flops": "N", "bytes": "N * elem_bytes"}

    @pytest.fixture(autouse=True)
    def _serve_the_entry(self, monkeypatch):
        from tileops.ops import _output_dtype

        entries = {self.OP: {"signature": self.SIG}}
        monkeypatch.setattr(_output_dtype, "load_manifest", lambda: entries)

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
            "params,flops",
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
        "name-in-two-blocks": ("params", lambda r, g: g.update(params={"x": {"type": "int"}})),
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
            "params",
            lambda r, g: g.setdefault("params", {}).update(u={"type": "int"}),
        ),
        "helper-called-wrong": (
            "vars-cf",
            lambda r, g: r["vars"].update(W="range(stop=3)"),
        ),
        "helper-wrong-arity": (
            "vars-ar",
            lambda r, g: r["vars"].update(Wa="len(x.shape, x.shape)"),
        ),
        "vars-key-reserved": (
            "vars-r,flops",
            lambda r, g: (r["vars"].update(_flops="1"), r.update(flops="_flops")),
        ),
        "helper-no-args": ("vars-na", lambda r, g: r["vars"].update(Na="min()")),
        "helper-starred": ("vars-st", lambda r, g: r["vars"].update(St="len(*[x.shape])")),
        "arith-call-form": ("flops", lambda r, g: r.update(flops="ceil(N, 2)")),
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

    def test_a_plan_whose_source_does_not_compile_is_a_verdict(self):
        """The plan's source is compiled before the plan is accepted, so a text
        Python will not take is one verdict however it fails to be Python: a
        leading newline, a comment line, a keyword given twice."""
        from tileops.manifest.roofline_analysis import analyze_roofline

        for block in (
            {"vars": {"N": "\n1"}, "flops": "N", "bytes": "N"},
            {"vars": {"N": "\n# c\n1"}, "flops": "N", "bytes": "N"},
            {"vars": {"N": "min([1], key=abs, key=len)"}, "flops": "N", "bytes": "N"},
            {"vars": {"N": "1"}, "flops": "\nN", "bytes": "N"},
        ):
            result = analyze_roofline(self.OP, roofline=block, signature=self.SIG)
            assert result.plan is None, block
            assert [d.code for d in result.diagnostics] == ["roofline.does-not-compile"], block

    def test_a_func_taking_more_than_the_op_says_so(self):
        """The emitted method calls it with the op and nothing else, so a
        callable that needs a second argument is refused rather than resolved."""
        from tileops.manifest.roofline_analysis import analyze_roofline

        result = analyze_roofline(self.OP, roofline={"func": "builtins.pow"}, signature=self.SIG)
        assert result.plan is None
        assert [d.code for d in result.diagnostics] == ["func.arity"]

    def test_a_func_returning_the_wrong_shape_says_so(self):
        """What a func-mode formula returns is settled only by calling it, so
        the wrapper keeps that contract: the reading never travels as a number
        that is not a (flops, bytes) pair."""
        from tileops.manifest.roofline_analysis import analyze_roofline
        from tileops.ops._roofline_emit import emit_eval_roofline

        result = analyze_roofline(self.OP, roofline={"func": "builtins.id"}, signature=self.SIG)
        assert result.plan is not None
        with pytest.raises(TypeError, match=self.OP):
            emit_eval_roofline(result.plan)(self._probe_op())

    def test_no_plan_is_one_that_cannot_run(self):
        """Alone and in every non-overwriting pair."""
        import itertools

        broken = []
        cases = [(n,) for n in sorted(self.INJECT)] + [
            (a, b)
            for a, b in itertools.combinations(sorted(self.INJECT), 2)
            if not _slots(self.INJECT[a][0]) & _slots(self.INJECT[b][0])
        ]
        for names in cases:
            a = " + ".join(names)
            try:
                value = self._run(list(names))
            except Exception as exc:  # noqa: BLE001 - that it raises is the failure
                broken.append(f"{a}: {type(exc).__name__}: {exc}")
                continue
            if value is not None and [type(v) for v in value] != [int, int]:
                broken.append(f"{a}: returned {value!r}")
            plan = self._plan(list(names))
            if plan is not None:
                broken += [f"{a}: {d}" for d in self._plan_defects(plan)]
        assert not broken, broken


class TestCallFormTableMatchesPython:
    """Every row of CALL_FORMS says what the helper itself accepts.

    The table states forms Python exposes no signature for, so it is checked
    against the callable rather than against itself: each generated call is put
    to both, and a row too narrow or too wide shows up as a disagreement.
    """

    # A value legal at that position for that helper, so a TypeError from the
    # call is about the form and not about the value.
    POSITIONAL = {
        "product": [(1, 2)] * 5,
        "isinstance": [tuple] * 5,
        "len": [()] * 5,
        "set": [()] * 5,
        "tuple": [()] * 5,
        "list": [()] * 5,
        "range": [1, 2, 3, 4, 5],
        "int": ["10", 2, 3, 4, 5],
        "float": ["1", 2, 3, 4, 5],
        "bool": [1, 2, 3, 4, 5],
        "min": [((1,), (2, 3))] * 5,
        "max": [((1,), (2, 3))] * 5,
        "sum": [(1, 2), 0, 1, 2, 3],
        "abs": [1, 2, 3, 4, 5],
        "log2": [1, 2, 3, 4, 5],
        "ceil": [1, 2, 3, 4, 5],
        "floor": [1, 2, 3, 4, 5],
    }
    # Every keyword a row names, every parameter name these callables document --
    # so a row that forgot a keyword its helper does take is a disagreement
    # rather than a cell nobody generates -- and one name none of them takes.
    KEYWORD = {
        "default": 0, "key": abs, "start": 0, "base": 2, "x": 1, "obj": 1,
        "iterable": (1, 2), "a": 1, "b": 1, "stop": 1, "step": 1, "object": 1,
        "class_or_tuple": tuple, "nope": 1,
    }  # fmt: skip
    # A key must suit the values above, or its TypeError would be about them.
    KEYWORD_BY_HELPER = {"min": {"key": len}, "max": {"key": len}}
    KEYWORD_SETS = [(), ("default", "key")] + [(name,) for name in KEYWORD]

    def test_no_row_is_narrower_or_wider_than_the_helper(self):
        import ast

        from tileops.manifest.roofline_analysis import (
            CALL_FORMS,
            VARS_HELPERS,
            _call_form_defect,
        )

        disagreed = []
        for name, helper in VARS_HELPERS.items():
            for count in range(5):
                for keywords in self.KEYWORD_SETS:
                    call = ast.parse(
                        f"{name}({', '.join(['1'] * count + [f'{k}=1' for k in keywords])})",
                        mode="eval",
                    ).body
                    table_accepts = _call_form_defect(call, name) is None
                    try:
                        helper(
                            *self.POSITIONAL[name][:count],
                            **{
                                k: self.KEYWORD_BY_HELPER.get(name, {}).get(k, self.KEYWORD[k])
                                for k in keywords
                            },
                        )
                        python_accepts = True
                    except TypeError:
                        python_accepts = False
                    except Exception:  # noqa: BLE001 - the form was taken; the value was not
                        python_accepts = True
                    if table_accepts != python_accepts:
                        disagreed.append(
                            f"{name}({count} positional, {keywords}): "
                            f"table {table_accepts}, python {python_accepts}"
                        )
        assert not disagreed, disagreed
        assert set(CALL_FORMS) == set(VARS_HELPERS)


class TestNothingLegalIsRefused:
    """A formula the spec allows is accepted, and draws nothing.

    The other two matrices ask what a wrong entry draws. This one asks what a
    right one does not: a gate tightened against a defect can refuse a form the
    spec permits, and no defect corpus would notice.
    """

    OP = "SyntheticLegacyFwdOp"
    SIG = {
        "inputs": {"x": {"dtype": "float16", "shape": "[N]"}},
        "outputs": {"output": {"dtype": "same_as(x)"}},
        "params": {"alpha": {"type": "float"}},
    }
    ROOFLINE = {"vars": {"N": "product(x.shape)"}, "flops": "N", "bytes": "N * elem_bytes"}

    @pytest.fixture(autouse=True)
    def _serve_the_entry(self, monkeypatch):
        from tileops.ops import _output_dtype

        entries = {self.OP: {"signature": self.SIG}}
        monkeypatch.setattr(_output_dtype, "load_manifest", lambda: entries)

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
        "soft-keyword-name": (
            "v-t",
            lambda r, g: r["vars"].update(**{"type": "1", "match": "1", "case": "1"}),
        ),
        "nfkc-name": ("v-k", lambda r, g: r["vars"].update(**{"\u212a": "1"})),
        "out-elem-bytes": ("bytes", lambda r, g: r.update(bytes="N * out_elem_bytes")),
        "param-in-arithmetic": ("flops", lambda r, g: r.update(flops="N * alpha")),
        "numeric-helpers": (
            "flops",
            lambda r, g: r.update(flops="ceil(N / 2) + floor(N / 3) + log2(N + 1)"),
        ),
        "conditional-arithmetic": (
            "bytes",
            lambda r, g: r.update(bytes="(N if N > 0 else 1) * elem_bytes"),
        ),
        "unread-param": ("p-u", lambda r, g: g["params"].update(unused={"type": "int"})),
        "helper-keyword-argument": (
            "v-kw",
            lambda r, g: r["vars"].update(Kw="min([1], default=0) + product(x.shape, start=2)"),
        ),
        "helper-optional-argument": (
            "v-oa",
            lambda r, g: r["vars"].update(Oa='int("10", 2) + sum([1], 0) + len(range(1, 9, 2))'),
        ),
        "helper-other-overload": (
            "v-ov",
            lambda r, g: r["vars"].update(
                Ov='int("10", base=2) + sum([1], start=0) + min(-1, 2, key=abs)'
            ),
        ),
        "helper-called-empty": (
            "v-ce",
            lambda r, g: r["vars"].update(Ce="len(set()) + len(tuple()) + int() + len(list())"),
        ),
        "value-dependent-arithmetic": (
            "flops",
            lambda r, g: r.update(flops="N // (alpha + 1)"),
        ),
        "helper-named-unread-param": (
            "p-h",
            lambda r, g: g["params"].update(bool={"type": "int"}),
        ),
        "nfkc-param-name": (
            "p-k",
            lambda r, g: (
                g["params"].update(**{"\uff2a": {"type": "int"}}),
                r["vars"].update(Jj="J"),
            ),
        ),
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

    def test_no_legal_form_is_refused(self):
        """Alone and in every non-overwriting pair."""
        import itertools

        cases = [(n,) for n in sorted(self.LEGAL)] + [
            (a, b)
            for a, b in itertools.combinations(sorted(self.LEGAL), 2)
            if not _slots(self.LEGAL[a][0]) & _slots(self.LEGAL[b][0])
        ]
        refused = [
            f"{' + '.join(names)}: {why}"
            for names in cases
            if (why := self._refusal(list(names))) is not None
        ]
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

    def test_an_expression_too_deep_to_parse_is_a_verdict(self):
        """Exhausting the parser is a verdict, not an exception out of the
        analysis."""
        import sys

        from tileops.manifest.roofline_analysis import analyze_roofline

        for expr in ("1" + "+1" * 500, "-" * 10000 + "1"):
            # How deep is too deep depends on the recursion limit, which another
            # test in the same process may have raised. Pinned, so this
            # expression is over it whatever ran first.
            limit = sys.getrecursionlimit()
            sys.setrecursionlimit(min(limit, 1000))
            try:
                result = analyze_roofline(
                    "FakeOp",
                    roofline={"vars": {"N": expr}, "flops": "N", "bytes": "1"},
                    signature={"inputs": {}, "outputs": {"y": {}}},
                )
            finally:
                sys.setrecursionlimit(limit)
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

    OUT = {"y": {}}


class TestEmittableByConstruction:
    """A plan the analysis builds must be one emission can compile and run."""

    OUT = {"y": {}}

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
        from tileops.perf.formulas import _shape_or_attrs

        class _Unrun:
            def __init__(self):
                self._roofline_kwargs = None

        with pytest.raises(RuntimeError, match="requires a prior forward"):
            _shape_or_attrs(_Unrun(), {})

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
