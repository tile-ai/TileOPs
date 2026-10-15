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


def _findings(text: str) -> list[str]:
    return lint.findings("foo.yaml", text, adts={})


def test_rows_sharing_a_label_and_differing_in_shape_are_reported() -> None:
    (found,) = _findings(_file("{M: 1, label: a}", "{M: 2, label: a}"))

    assert found.startswith("foo.yaml:8: FooFwdOp: rows labelled 'a' differ in M")
    assert found.endswith('or add "# lint: manifest-label allow: <reason>" on the line')


SHARED_LABEL_OTHER_DIFFERENCE = [
    pytest.param(("{M: 1, N: 4, label: a}", "{M: 1, N: 8, label: a}"), id="second-shape-key"),
    pytest.param(("{M: 1, label: a}", "{M: 1, causal: true, label: a}"), id="key-only-one-row-has"),
    pytest.param(
        ("{M: 1, label: a}", "{M: 1, label: b}", "{M: 2, label: a}"),
        id="rows-not-adjacent",
    ),
]


@pytest.mark.parametrize("rows", SHARED_LABEL_OTHER_DIFFERENCE)
def test_rows_sharing_a_label_and_differing_beyond_dtype_are_reported(rows) -> None:
    assert len(_findings(_file(*rows))) == 1, rows


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
    pytest.param(("{M: 1, N: 4, label: a}", "{N: 4, M: 1, label: a}"), id="keys-reordered"),
]


@pytest.mark.parametrize("rows", DTYPE_ONLY)
def test_rows_whose_shared_label_spans_only_dtypes_are_left_alone(rows) -> None:
    assert _findings(_file(*rows)) == []


def test_one_call_under_two_labels_is_left_alone() -> None:
    assert _findings(_file("{M: 1, label: a}", "{M: 1, label: b}")) == []


def test_a_label_shared_by_two_entries_is_left_alone() -> None:
    bar = _HEADER.replace("FooFwdOp", "BarFwdOp")

    assert _findings(_file("{M: 1, label: a}") + _file("{M: 2, label: a}", op_header=bar)) == []


def test_a_row_allowance_with_a_reason_suppresses_the_finding() -> None:
    rows = ("{M: 1, label: a}", "{M: 2, label: a}  # lint: manifest-label allow: legacy history")

    assert _findings(_file(*rows)) == []


def test_a_bare_row_allowance_does_not_suppress_the_finding() -> None:
    assert _findings(_file("{M: 1, label: a}", "{M: 2, label: a}  # lint: manifest-label allow:"))
