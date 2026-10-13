"""Unit tests for ``scripts/lint/manifest_label_lint.py``."""

import manifest_label_lint as lint
import pytest

pytestmark = pytest.mark.smoke

# One entry with a dtype parameter, opening every probe file.
_HEADER = (
    "FooFwdOp:\n"
    "  family: foo\n"
    "  signature:\n"
    "    params:\n"
    '      out_dtype: {type: "float16 | bfloat16"}\n'
    "  workloads:\n"
)


def _file(*rows: str, op_header: str = _HEADER) -> str:
    return op_header + "".join(f"  - {row}\n" for row in rows)


def _findings(text: str, base: str | None = None) -> list[str]:
    return lint.findings("foo.yaml", text, base, adts={})


def test_rows_sharing_a_label_and_differing_in_shape_are_reported() -> None:
    (found,) = _findings(_file("{M: 1, label: a}", "{M: 2, label: a}"))

    assert "foo.yaml:8: FooFwdOp: rows labelled 'a' differ in M" in found


DTYPE_ONLY = [
    pytest.param(
        ("{M: 1, out_dtype: float16, label: a}", "{M: 1, out_dtype: bfloat16, label: a}"),
        id="dtype-param",
    ),
    pytest.param(
        (
            "{M: 1, dtype_cases: [{T: float16}], label: a}",
            "{M: 1, dtype_cases: [{T: bfloat16}], label: a}",
        ),
        id="dtype-cases",
    ),
]


@pytest.mark.parametrize("rows", DTYPE_ONLY)
def test_rows_sharing_a_label_and_differing_in_dtype_are_left_alone(rows) -> None:
    assert _findings(_file(*rows)) == []


def test_one_call_under_two_labels_is_reported() -> None:
    (found,) = _findings(_file("{M: 1, label: a}", "{M: 1, label: b}"))

    assert "row 'b' is the same call as row 'a'" in found


def test_a_label_shared_by_two_entries_is_left_alone() -> None:
    bar = _HEADER.replace("FooFwdOp", "BarFwdOp")

    assert _findings(_file("{M: 1, label: a}") + _file("{M: 2, label: a}", op_header=bar)) == []


def test_a_row_whose_parameters_change_under_its_label_is_reported() -> None:
    (found,) = _findings(_file("{M: 2, label: a}"), base=_file("{M: 1, label: a}"))

    assert "FooFwdOp: row 'a' changed M since the merge base" in found


def test_a_changed_row_under_a_new_label_is_left_alone() -> None:
    assert _findings(_file("{M: 2, label: b}"), base=_file("{M: 1, label: a}")) == []


def test_a_label_reused_for_another_row_is_reported() -> None:
    base = _file("{M: 1, label: a}", "{M: 2, label: b}")

    (found,) = _findings(_file("{M: 2, label: a}"), base=base)

    assert "row 'a' changed M" in found


def test_a_dtype_added_under_a_label_is_left_alone() -> None:
    base = _file("{M: 1, out_dtype: float16, label: a}")
    head = _file("{M: 1, out_dtype: float16, label: a}", "{M: 1, out_dtype: bfloat16, label: a}")

    assert _findings(head, base=base) == []
