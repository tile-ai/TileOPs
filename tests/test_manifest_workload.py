"""Workload rows (docs/design/manifest.md § Workloads)."""

import copy
from pathlib import Path

import pytest
import torch
import yaml

from tileops.manifest.plan import entry_plan
from tileops.manifest.primitives import GENERATORS, PREDICATES
from tileops.manifest.workload import RowError, check_workloads, instantiate

pytestmark = pytest.mark.smoke

_CASES = yaml.safe_load((Path(__file__).parent / "manifest_cases.yaml").read_text())
_ADTS = _CASES["adts"]
_ENTRIES = _CASES["entries"]


_MARKED = [
    (name, row)
    for name in sorted(_ENTRIES)
    for row in _ENTRIES[name].get("workloads", [])
    if row.get("expect_fail")
]


@pytest.mark.parametrize(("name", "row"), _MARKED, ids=[row["label"] for _, row in _MARKED])
def test_a_marked_row_is_rejected(name, row):
    plan = entry_plan(name, _ENTRIES[name], _ADTS)
    row = {k: v for k, v in row.items() if k != "expect_fail"}
    for case in row.get("dtype_cases") or [{}]:
        with pytest.raises(RowError):
            instantiate(plan, row, case)


@pytest.mark.parametrize("name", sorted(_ENTRIES))
def test_fixture_rows_pass_the_validator(name):
    entry = copy.deepcopy(_ENTRIES[name])
    entry["workloads"] = [r for r in entry.get("workloads", []) if not r.get("expect_fail")]
    if entry["workloads"]:
        assert check_workloads(name, entry, _ADTS) == []


def test_case_id_is_label_then_dtype_cases_then_dtype_parameters():
    entry = _two_dtypes([{"U": "float16", "T": "bfloat16"}])
    sig = entry_plan("Pair", entry, _ADTS)
    call = instantiate(sig, entry["workloads"][0], entry["workloads"][0]["dtype_cases"][0])
    assert call.case_id == "r-bfloat16-float16"
    plan = entry_plan("GemmFp8FwdOp", _ENTRIES["GemmFp8FwdOp"], _ADTS)
    row = _ENTRIES["GemmFp8FwdOp"]["workloads"][1]
    assert instantiate(plan, row, {}).case_id == f"{row['label']}-float16"


def test_an_unimportable_constructor_class_is_a_row_error():
    adts = copy.deepcopy(_ADTS)
    for ctor in adts["MGroupedLayout"]["sum"].values():
        ctor["python"] = "nonexistent_module.Layout"
    errors = check_workloads("MoePrePermuteFwdOp", _entry("MoePrePermuteFwdOp"), adts)
    assert any("a constructor class" in e for e in errors), errors


def _first_call(name):
    sig = entry_plan(name, _ENTRIES[name], _ADTS)
    row = next(r for r in _ENTRIES[name]["workloads"] if not r.get("expect_fail"))
    return instantiate(sig, row, (row.get("dtype_cases") or [{}])[0])


def test_arguments_are_constructor_values():
    from tileops.ops.moe.contracts import ContiguousLayoutSpec, ContiguousPacking

    pooling = _first_call("MeanPoolingFwdOp")
    assert pooling.arguments(pooling.materialize("cpu"))["accum_dtype"] is torch.float32
    moe = _first_call("MoePrePermuteFwdOp")
    layout = moe.arguments(moe.materialize("cpu"))["layout"]
    assert isinstance(layout, ContiguousLayoutSpec) and layout.packing is ContiguousPacking.TIGHT
    entry = _with_param(
        _varlen(L=[1], q={"masked": {"max_m": 4}}),
        q={"type": "MGroupedLayout | None"},
        w={"dtype": "float16", "shape": "[2]"},
    )
    sig = entry_plan("Varlen", entry, _ADTS)
    call = instantiate(sig, entry["workloads"][0], entry["workloads"][0]["dtype_cases"][0])
    tensors = call.materialize("cpu")
    arguments = call.arguments(tensors)
    assert arguments["q"].max_m == 4 and arguments["w"] is tensors["w"]


def test_materialize_draws_every_dtype_category():
    from tileops.manifest.workload import Call, TensorSpec

    specs = {d: TensorSpec((4,), d) for d in ("bool", "int8", "float16", "complex64")}
    tensors = Call("c", {}, specs).materialize(device="cpu")
    assert {n: str(t.dtype) for n, t in tensors.items()} == {d: f"torch.{d}" for d in specs}
    assert set(tensors["bool"].tolist()) <= {False, True}


def test_paged_fits_is_false_on_mismatched_lengths():
    assert PREDICATES["attn.paged_fits"]([1, 2], [0, 1], 8) is False


def test_packed_positions_restart_at_each_sequence():
    assert GENERATORS["packed_positions"]([2, 3]) == [0, 1, 0, 1, 2]
    for lengths in ([], [2, 0]):
        with pytest.raises(ValueError, match="non-empty positive list"):
            GENERATORS["packed_positions"](lengths)


def test_generated_metadata_is_deterministic_and_materializes():
    sig = entry_plan("MoePrePermuteFwdOp", _ENTRIES["MoePrePermuteFwdOp"], _ADTS)
    row = _ENTRIES["MoePrePermuteFwdOp"]["workloads"][0]
    first, second = (instantiate(sig, row, {"T": "float16"}) for _ in range(2))
    assert first.specs["local_expert_ids"].values == second.specs["local_expert_ids"].values
    tensors = first.materialize(device="cpu")
    assert tensors["hidden_states"].dtype == torch.float16
    assert tensors["local_expert_ids"].tolist() == first.specs["local_expert_ids"].values


def _entry(name, **changes):
    entry = copy.deepcopy(_ENTRIES[name])
    entry["workloads"] = [
        {k: v for k, v in r.items() if k != "expect_fail"}
        for r in entry["workloads"]
        if not r.get("expect_fail")
    ]
    for i, row in enumerate(entry["workloads"]):
        row.setdefault("label", f"r{i}")
    entry.update(changes)
    return entry


def _requires_without_values():
    entry = _entry("GQAPrefillVarlenFwdOp")
    del entry["signature"]["inputs"]["cu_seqlens_q"]["values"]
    return entry


def _varlen(**row):
    """Offsets and chunk indices generated from one list of sequence lengths."""
    signature = {
        "forall": {"B": "Dim", "N": "Dim", "L": "Seq[Int]", "T": "DType[float16]"},
        "inputs": {
            "offsets": {"dtype": "int32", "shape": "[B + 1]", "values": "prefix_sum(L)"},
            "chunks": {"dtype": "int32", "shape": "[N, 2]", "values": "chunk_indices(L, 4)"},
            "x": {"dtype": "T", "shape": "[B]"},
        },
        "outputs": {"y": {"dtype": "T", "shape": "[B]"}},
    }
    row = {"dtype_cases": [{"T": "float16"}], "label": "r", **row}
    return {"signature": signature, "workloads": [row]}


def _two_dtypes(cases):
    """Two dtype indices, restricted by `dtype_combos`."""
    signature = {
        "forall": {"M": "Dim", "T": "DType[float16 | bfloat16]", "U": "DType[float16 | bfloat16]"},
        "inputs": {"a": {"dtype": "T", "shape": "[M]"}, "b": {"dtype": "U", "shape": "[M]"}},
        "outputs": {"y": {"dtype": "T", "shape": "[M]"}},
        "dtype_combos": [{"T": "float16", "U": "float16"}, {"T": "bfloat16", "U": "float16"}],
    }
    return {"signature": signature, "workloads": [{"M": 4, "label": "r", "dtype_cases": cases}]}


def _with_param(entry, **params):
    entry["signature"]["params"] = params
    return entry


def _varlen_cross_requires():
    """Two optional metadata tensors, one constrained by the other's values."""
    entry = _varlen(L=[2, 3], some=["a", "b"])
    inputs = entry["signature"]["inputs"]
    for name in ("a", "b"):
        inputs[name] = {**inputs["offsets"], "optional": True}
    inputs["a"]["requires"] = ["attn.paged_fits(b, 8)"]
    return entry


def _varlen_divided(**row):
    """An output axis divided by an index a refinement keeps positive."""
    entry = _varlen(L=[2, 3], M=4, **row)
    entry["signature"]["forall"] |= {"M": "Dim", "K": "Dim"}
    entry["signature"]["inputs"]["w"] = {"dtype": "T", "shape": "[M, K]"}
    entry["signature"]["outputs"]["z"] = {"dtype": "T", "shape": "[M // K]"}
    entry["signature"]["shape_rules"] = ["K > 0"]
    return entry


def _varlen_empty_chunks():
    """An empty list of lengths yields a [0, 2] chunk table: B and C are both solved."""
    entry = _varlen(L=[])
    entry["signature"]["forall"]["C"] = "Dim"
    entry["signature"]["inputs"]["offsets"].update(shape="[B, C]", values="chunk_indices(L, 4)")
    return entry


def _varlen_generated(dtype="int32", requires=(), some=()):
    """An optional generated tensor, passed by the row when *some* names it."""
    entry = _varlen(L=[2, 3], some=list(some))
    entry["signature"]["inputs"]["m"] = {
        "dtype": dtype,
        "shape": "[N, 2]",
        "values": "chunk_indices(L, 4)",
        "optional": True,
        **({"requires": list(requires)} if requires else {}),
    }
    return entry


def _varlen_starred(p="tuple[int, int, int]", value=None):
    """A generated tensor whose starred axes a fixed-length tuple type counts."""
    entry = _with_param(_varlen_generated(some=["m"]), p={"type": p})
    if value is not None:
        entry["workloads"][0]["p"] = value
    entry["signature"]["inputs"]["m"]["shape"] = "[*p]"
    return entry


def _varlen_chained():
    """Counts generated from an index another generator solves."""
    entry = _varlen(L=[2, 3])
    entry["signature"]["forall"]["N"] = "Dim"
    entry["signature"]["inputs"]["chunks"] = {
        "dtype": "int32",
        "shape": "[N]",
        "values": "as_tensor(balanced_sizes(B, 1))",
    }
    return entry


def _varlen_with(shape="[B + 1]", values="prefix_sum(L)", requires=(), B=None, L=(2, 3), **forall):
    """`_varlen` with its offsets declared by *shape*, *values* and *requires*; *B* in the row."""
    entry = _varlen(L=list(L), **({} if B is None else {"B": B}))
    entry["signature"]["forall"].update(forall)
    entry["signature"]["inputs"]["offsets"].update(shape=shape, values=values)
    if requires:
        entry["signature"]["inputs"]["offsets"]["requires"] = list(requires)
    if "C" in forall:
        entry["signature"]["inputs"]["x"]["shape"] = "[B, C]"
        entry["workloads"][0].update(B=2, C=3, L=[1, 2, 3, 4, 5])
    return entry


def _varlen_spliced_requires():
    """A `requires` on a generated tensor whose rank a splice of unknown length leaves open."""
    entry = _varlen_starred(p="list[int]", value=[2, 2])
    entry["signature"]["inputs"]["m"]["requires"] = ["max_segment(9)"]
    return entry


_ENTRY_ERRORS = [
    (
        "GemmFwdOp",
        _entry(
            "GemmFwdOp",
            workloads=[
                {
                    "M": 1,
                    "N": 1,
                    "K": 1,
                    "trans_a": False,
                    "trans_b": False,
                    "dtype_cases": [{"T": "float16"}],
                    "label": "a b",
                }
            ],
        ),
        "`label`",
    ),
    (
        "GemmFwdOp",
        _entry(
            "GemmFwdOp",
            workloads=[
                {
                    "M": 1,
                    "N": 1,
                    "K": 1,
                    "trans_a": False,
                    "trans_b": False,
                    "dtype_cases": [{"T": "float16"}],
                    "label": "x",
                }
            ]
            * 2,
        ),
        "repeats",
    ),
    (
        "ClampFwdOp",
        _entry(
            "ClampFwdOp",
            status="implemented",
            workloads=[
                {
                    "A": [4],
                    "L": [4],
                    "U": [4],
                    "some": ["min", "max"],
                    "dtype_cases": [{"T": "float16"}],
                    "label": "x",
                }
            ],
        ),
        "no row omits optional tensor 'max'",
    ),
    (
        "ClampFwdOp",
        _entry(
            "ClampFwdOp",
            status="implemented",
            workloads=[{"A": [4], "dtype_cases": [{"T": "float16"}], "label": "x"}],
        ),
        "no row passes optional tensor 'max'",
    ),
    (
        "GemmFwdOp",
        _entry(
            "GemmFwdOp",
            workloads=[
                {
                    "N": 8,
                    "K": 8,
                    "trans_a": False,
                    "trans_b": True,
                    "label": "x",
                    "dtype_cases": [{"T": "float16"}],
                }
            ],
        ),
        "row misses ['M']",
    ),
    ("Pair", _two_dtypes([{"U": "bfloat16", "T": "float16"}]), "is not a row of dtype_combos"),
    ("GQAPrefillVarlenFwdOp", _requires_without_values(), "has `requires` but no `values`"),
    ("Varlen", _varlen(L=[2**31 - 1, 1]), "outside int32"),
    ("Varlen", _varlen(L=[1], some=["x"]), "`some` names ['x']"),
    ("Varlen", _varlen(L=[1], dtype_cases=[]), "`dtype_cases` must be"),
    ("Varlen", _varlen_with(values="prefix_sum()"), "misses argument 1"),
    (
        "Varlen",
        _varlen_with(values="prefix_sum(L)", requires=["attn.paged_fits(x, 8)"]),
        "is not declared",
    ),
    ("Varlen", _varlen_chained(), None),
    ("Varlen", _varlen_empty_chunks(), None),
    ("Varlen", _varlen_with(values="bogus(L)"), "is not a call of a built-in"),
    ("Varlen", _varlen_divided(K=0), "refinement fails: K > 0"),
    ("Varlen", _varlen_generated(dtype="float32", some=["m"]), "need an int32 or int64 dtype"),
    ("Varlen", _varlen_generated(dtype="int64", some=["m"]), None),
    ("Varlen", _varlen_starred(value=[1, 2, 3]), "yields rank 2"),
    (
        "Varlen",
        _with_param(
            _varlen(
                L=[1],
                q={"contiguous": {"packing": "tight", "metadata_kind": "per_row", "alignment": 2}},
            ),
            q={"type": "MGroupedLayout | None"},
        ),
        "is not a MGroupedLayout | None",
    ),
    ("Varlen", _varlen_generated(requires=["max_segment(9)"]), "reads rank 1"),
    ("Varlen", _varlen_cross_requires(), "reads 'b' where it is absent"),
    ("Varlen", _varlen_spliced_requires(), "needs a fixed-rank tensor"),
    (
        "Varlen",
        _with_param(_varlen(L=[1], p=[0]), p={"type": "int | tuple[int, int]"}),
        "is not a int | tuple",
    ),
    ("Varlen", _varlen_with(shape="[B + 10]", values="as_tensor(L)"), "is not B + 10 for a Dim"),
    ("Varlen", _varlen_with(values="as_tensor(balanced_sizes(L, 1))"), "expected Int"),
    (
        "Varlen",
        _varlen_with(values="prefix_sum(L)", requires=["max_segment(1)"]),
        "requires max_segment(1) fails",
    ),
]


@pytest.mark.parametrize(
    ("name", "entry", "message"), _ENTRY_ERRORS, ids=[m or "clean" for *_, m in _ENTRY_ERRORS]
)
def test_check_workloads(name, entry, message):
    errors = check_workloads(name, entry, _ADTS)
    if message is None:
        assert errors == []
    else:
        assert any(message in e for e in errors), errors
