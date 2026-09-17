"""Tests for the validator's derived-fact layer.

The facts are what every check reads instead of re-deriving. A fact that is
wrong here is wrong everywhere at once, so these run without YAML and without
the validator around them.
"""

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _manifest_facts as F  # noqa: E402


def _entry(**over):
    entry = {
        "status": "implemented",
        "signature": {
            "inputs": {"x": {"dtype": "float16 | bfloat16"}},
            "outputs": {"y": {"dtype": "same_as(x)"}},
        },
    }
    entry.update(over)
    return entry


class TestCallVersusValue:
    """The two lists this layer exists to keep apart."""

    def test_workspace_is_in_the_call_but_not_the_value_contract(self):
        f = F.build(
            "Op",
            _entry(resources={"workspaces": [{"name": "ws", "dtype": "float16"}]}),
        )
        assert f.call_names == ("x", "ws")
        assert f.combo_columns == ("x",)

    def test_a_premerged_signature_gives_the_same_answer(self):
        """A caller may hand over a signature the workspaces are already in.

        The marker is what tells them apart; without honouring it the workspace
        would be read back as a caller-visible input.
        """
        entry = _entry(resources={"workspaces": [{"name": "ws", "dtype": "float16"}]})
        direct = F.build("Op", entry)
        premerged = F.build(
            "Op",
            {
                "status": "implemented",
                "signature": {
                    "inputs": {
                        "x": {"dtype": "float16 | bfloat16"},
                        "ws": {"dtype": "float16", F.WORKSPACE_ATTR: True},
                    },
                    "outputs": {"y": {"dtype": "same_as(x)"}},
                },
            },
        )
        assert premerged.call_names == direct.call_names
        assert premerged.combo_columns == direct.combo_columns

    def test_optional_input_is_not_a_required_combo_column(self):
        f = F.build(
            "Op",
            _entry(
                signature={
                    "inputs": {
                        "x": {"dtype": "float16"},
                        "bias": {"dtype": "float16", "optional": True},
                    },
                    "outputs": {"y": {"dtype": "same_as(x)"}},
                }
            ),
        )
        assert f.combo_columns == ("x", "bias")
        assert f.required_combo_columns == ("x",)
        assert f.optional_names == frozenset({"bias"})


class TestDeclaredShapes:
    """A shape is a fact only where it can be bound as mock dimension names."""

    @staticmethod
    def _with_output_shape(shape):
        return F.build(
            "Op",
            {
                "status": "implemented",
                "signature": {
                    "inputs": {"x": {"dtype": "float16"}},
                    "outputs": {"y": {"dtype": "same_as(x)", "shape": shape}},
                },
            },
        )

    def test_identifier_list_is_a_declaration(self):
        assert self._with_output_shape("[N, C, L]").declared_output_shapes == {"y": ("N", "C", "L")}

    @pytest.mark.parametrize(
        "shape",
        ["[4, d]", "[2 * N]", "[]", "", None],
        ids=["literal", "arithmetic", "empty", "blank", "absent"],
    )
    def test_anything_else_declares_nothing(self, shape):
        """Not 'an empty declaration' — no declaration at all.

        A consumer skips its check when an output declares no bindable shape
        and runs it when the shape is stated, so collapsing the two would turn
        a skipped check into a reported violation.
        """
        assert self._with_output_shape(shape).declared_output_shapes == {}


class TestDtypeFacts:
    def test_same_as_spans_inputs_and_outputs(self):
        f = F.build("Op", _entry())
        assert f.same_as_map == {"y": "x"}

    def test_a_plain_union_has_no_reference(self):
        assert F.build("Op", _entry()).arg("x").same_as is None

    @pytest.mark.parametrize(
        "dtype",
        ["float16 | same_as(x)", "same_as(x) | float16", "notsame_as(x)"],
        ids=["union_trailing", "union_leading", "other_call"],
    )
    def test_only_a_bare_same_as_is_a_reference(self, dtype):
        """A dtype that merely mentions same_as does not follow another tensor.

        The negative probes substitute a dtype on one tensor and expect the ops
        that follow it to reject; treating a union as a reference would make
        them expect a rejection the op never makes.
        """
        f = F.build(
            "Op",
            _entry(
                signature={
                    "inputs": {"x": {"dtype": "float16"}, "w": {"dtype": dtype}},
                    "outputs": {"y": {"dtype": "same_as(x)"}},
                }
            ),
        )
        assert "w" not in f.same_as_map

    def test_call_scope_excludes_outputs(self):
        f = F.build("Op", _entry(resources={"workspaces": [{"name": "ws", "dtype": "same_as(x)"}]}))
        assert f.same_as_map == {"y": "x", "ws": "x"}
        assert f.call_same_as_map == {"ws": "x"}


class TestStatusAndSeverity:
    """Status decides whether a check may probe code; severity decides how it reports."""

    @pytest.mark.parametrize(
        "status, spec_only, implemented",
        [
            ("implemented", False, True),
            ("spec-only", True, False),
            (None, True, False),
            (3, True, False),
            ("garbage", False, False),
        ],
        ids=["implemented", "spec_only", "absent", "not_a_string", "unknown_string"],
    )
    def test_status_branches(self, status, spec_only, implemented):
        """An unknown string is neither: schema reports it, facts do not guess.

        Treating it as spec-only would silently stand down every check that
        needs an implementation, on an entry whose status is simply a typo.
        """
        entry = {"signature": {"inputs": {}, "outputs": {}}}
        if status is not None:
            entry["status"] = status
        f = F.build("Op", entry)
        assert f.spec_only is spec_only
        assert f.implemented is implemented

    def test_bench_severity_defaults_to_the_softer_one(self):
        assert not F.build("Op", _entry()).bench_manifest_driven
        assert F.build("Op", _entry(source={"bench_manifest_driven": True})).bench_manifest_driven

    @pytest.mark.parametrize(
        "roofline, mode",
        [
            ({"func": "pkg.f"}, "func"),
            ({"flops": "1", "bytes": "2"}, "inline"),
            ({"flops": "1"}, "none"),
            ({}, "none"),
        ],
        ids=["func", "inline", "half_inline", "empty"],
    )
    def test_roofline_mode(self, roofline, mode):
        assert F.build("Op", _entry(roofline=roofline)).roofline_mode == mode


class TestMutationContract:
    """Which inputs the op writes in place — the workspaces are never part of it."""

    def test_only_a_marked_caller_input_counts(self):
        f = F.build(
            "Op",
            {
                "status": "implemented",
                "signature": {
                    "inputs": {
                        "x": {"dtype": "float16"},
                        "out": {"dtype": "float16", "mutated": True},
                    },
                    "outputs": {"y": {"dtype": "same_as(x)"}},
                },
            },
        )
        assert f.mutated_input_names == frozenset({"out"})

    def test_a_workspace_is_never_a_mutated_input(self):
        """Every call writes one, so marking it would state nothing."""
        f = F.build(
            "Op",
            {
                "status": "implemented",
                "signature": {
                    "inputs": {"x": {"dtype": "float16"}},
                    "outputs": {"y": {"dtype": "same_as(x)"}},
                },
                "resources": {"workspaces": [{"name": "ws", "dtype": "float16", "mutated": True}]},
            },
        )
        assert f.mutated_input_names == frozenset()


class TestKeyTaker:
    """The parser is the schema: whatever it does not read is an unknown key."""

    def test_a_key_nothing_reads_is_unknown(self):
        f = F.build("Op", _entry(made_up_field=1))
        assert f.unknown_keys[F.Section.ENTRY] == ("made_up_field",)

    def test_every_declared_key_is_accepted(self):
        entry = _entry(
            ref_api="none",
            workloads=[],
            roofline={"flops": "1", "bytes": "1"},
            source={"kernel": "k.py", "op": "o.py", "test": "t.py", "bench": "b.py"},
            composition={"kind": "composite", "stages": []},
            resources={"workspaces": []},
            torch_compile_fullgraph=True,
            family="test",
        )
        assert F.build("Op", entry).unknown_keys[F.Section.ENTRY] == ()

    def test_the_message_and_the_parser_read_one_declaration(self):
        """The accepted set a diagnostic prints is the set the parser consumed.

        Two lists would drift: a field added to one and not the other either
        reads as unknown while the message calls it valid, or the reverse.
        """
        f = F.build("Op", _entry())
        assert set(f.accepted_keys[F.Section.ENTRY]) == set(F.SECTION_KEYS[F.Section.ENTRY])

    @pytest.mark.parametrize("section", list(F.Section))
    def test_every_section_declares_its_keys(self, section):
        assert F.SECTION_KEYS[section], section

    def test_unknown_keys_of_ignores_a_non_mapping(self):
        assert F.unknown_keys_of(F.Section.STAGE, "not a mapping") == ()
