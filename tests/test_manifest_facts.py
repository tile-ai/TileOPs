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


class TestParsingAccumulates:
    """A field that cannot be read must not stop the rest from being read."""

    def test_a_broken_field_leaves_the_others_intact(self):
        f = F.build("Op", _entry(composition={"kind": "composite", "stages": "not a list"}))
        assert "composition.stages" in f.invalid
        assert f.call_names == ("x",)
        assert f.outputs[0].name == "y"

    def test_invalid_records_what_reports_it(self):
        f = F.build(
            "Op",
            _entry(
                signature={"inputs": {"x": {"dtype": "f16"}}, "outputs": {}, "dtype_combos": "nope"}
            ),
        )
        invalid = f.invalid["signature.dtype_combos"]
        assert F.DiagnosticKind.DTYPE_COMBO_DATA in invalid.covers


class TestCanSilence:
    """A consumer falls silent only for a problem already reported."""

    KIND = F.DiagnosticKind.DTYPE_COMBO_DATA

    def test_silent_when_the_covering_diagnostic_was_emitted(self):
        invalid = F.Invalid("bad", (self.KIND,))
        assert F.can_silence(invalid, [self.KIND], [self.KIND])

    def test_reports_when_the_levels_in_force_produce_nothing(self):
        invalid = F.Invalid("bad", (self.KIND,))
        assert not F.can_silence(invalid, [self.KIND], [])

    def test_reports_when_the_consumer_is_not_blocked_by_it(self):
        invalid = F.Invalid("bad", (self.KIND,))
        assert not F.can_silence(invalid, [F.DiagnosticKind.SHAPE_RULE_SYNTAX], [self.KIND])

    def test_a_usable_fact_never_silences(self):
        assert not F.can_silence(None, [self.KIND], [self.KIND])
