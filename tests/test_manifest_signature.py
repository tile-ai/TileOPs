"""Static signature checks (docs/design/manifest.md § Validation)."""

import copy
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tileops.manifest._plan import check_adts, check_entry, signature_schema_errors
from tileops.manifest.primitives import PRIMITIVES

pytestmark = pytest.mark.smoke

_CASES = yaml.safe_load((Path(__file__).parent / "manifest_cases.yaml").read_text())
_ADTS = _CASES["adts"]
_ENTRIES = _CASES["entries"]


@pytest.mark.parametrize("name", sorted(_ENTRIES))
def test_accepts_case(name):
    assert check_entry(name, _ENTRIES[name], _ADTS) == ([], [])


def _edit(name, *changes):
    entry = copy.deepcopy(_ENTRIES[name])
    for change in changes:
        change(entry["signature"])
    return name, entry


def _set(path, value):
    def change(sig):
        node = sig
        for key in path[:-1]:
            node = node[key] if isinstance(node, list) else node.setdefault(key, {})
        node[path[-1]] = value

    return change


def _without_masked(sig):
    for family in sig["types"].values():
        family["cases"] = [c for c in family["cases"] if c["when"] != {"masked": "_"}]


# A family no shape applies, covering both values of its discriminant.
_UNUSED = {
    "params": {"p": "Bool", "M": "Dim"},
    "match": "p",
    "cases": [{"when": True, "is": "[M]"}, {"when": False, "is": "[M]"}],
}

# ClampOut's cases with a tensor bound; a call with none matches no case.
_CLAMP_CASES = [
    {"when": [True, True], "is": "[*broadcast(A, L, U)]"},
    {"when": [True, False], "is": "[*broadcast(A, L)]"},
    {"when": [False, True], "is": "[*broadcast(A, U)]"},
]

_EDITS = [
    (
        _edit("ClampFwdOp", _set(("shape_rules",), ["min is not None or max is not None"])),
        "outside the expression language",
    ),
    (_edit("GEMMFwdOp", _set(("shape_rules",), ["a.shape == (M, K)"])), "read as a value"),
    (
        _edit("SumFwdOp", _set(("shape_rules",), ["isinstance(dim, int)"])),
        "not a built-in primitive",
    ),
    (_edit("GEMMFwdOp", _set(("outputs", "d", "shape"), "[*broadcast(M)]")), "expected Shape"),
    (_edit("GEMMFwdOp", _set(("shape_rules",), ["M + 1"])), "expected Bool"),
    (_edit("GEMMFwdOp", _set(("forall", "T"), "DType[banana]")), "unknown kind"),
    (_edit("GEMMFwdOp", _set(("let", "L"), "(")), "cannot parse"),
    (_edit("MaxPool2dFwdOp", _set(("let", "kH"), "sH + 1")), "let cycle"),
    (
        _edit(
            "ClampFwdOp",
            _set(("shape_rules",), []),
            _set(("types", "ClampOut", "cases"), _CLAMP_CASES[:3]),
        ),
        "0 cases match",
    ),
    (
        _edit(
            "MoEPrePermuteFwdOp",
            _without_masked,
            _set(("shape_rules",), ["K > 0 and layout.kind == 'contiguous'"]),
        ),
        "0 cases match",
    ),
    (
        _edit(
            "MoEPrePermuteFwdOp",
            _without_masked,
            _set(("shape_rules",), ["layout.kind == 'contiguous'"]),
        ),
        None,
    ),
    (
        _edit("SiLUAndMulFwdOp", _set(("inputs", "x", "shape"), "[M, N * N]")),
        "'N' cannot be solved",
    ),
    # A known coefficient: a construction parameter whose sign the call checks.
    (
        _edit(
            "SiLUAndMulFwdOp",
            _set(("params", "k"), {"type": "int"}),
            _set(("inputs", "x", "shape"), "[M, k * N]"),
        ),
        None,
    ),
    (
        _edit("GEMMFwdOp", _set(("params", "w"), {"type": "Tensor | None"})),
        "a tensor is a call-time input",
    ),
    (
        _edit(
            "ClampFwdOp",
            _set(("inputs", "min", "type"), "float | Tensor"),
            _set(("inputs", "min", "default"), True),
        ),
        "default True is not a value of 'float | Tensor'",
    ),
    (
        _edit(
            "ClampFwdOp",
            _set(("inputs", "min", "type"), "Number | Tensor | None"),
            _set(("params", "min__number"), {"type": "int"}),
        ),
        "operator slot 'min__number' takes the name of another argument",
    ),
    (
        _edit(
            "ClampFwdOp",
            _set(("inputs", "min", "type"), "int | Tensor | None"),
            _set(("let", "Z"), "min + 1"),
            _set(("outputs", "output", "shape"), "[Z]"),
        ),
        "let 'Z' reads a call-time scalar",
    ),
    (
        _edit(
            "ClampFwdOp",
            _set(("inputs", "min", "type"), "int | Tensor | None"),
            _set(("outputs", "output", "shape"), "[min]"),
        ),
        "tensor 'min' is read as a value",
    ),
    # A combination outside promote's domain is left out of the output's dtypes.
    (
        _edit(
            "ConcatRowsFwdOp",
            _set(("forall", "T"), "Seq[DType[float8_e4m3fn | float16]]"),
            _set(("inputs", "head", "dtype"), "float16"),
            _set(("outputs", "output", "dtype"), "promote(T)"),
        ),
        None,
    ),
    # Each member guard narrows what the earlier ones left.
    (
        _edit(
            "ClampFwdOp",
            _set(("inputs", "min", "type"), "int | Tensor | None"),
            _set(("inputs", "min", "shape"), "[*A]"),
            _set(("inputs", "max", "shape"), "[*A]"),
            _set(("outputs", "output", "shape"), "[*A]"),
            _set(("types",), {}),
            _set(
                ("shape_rules",),
                ["member(min) == 'Tensor' or member(min) == 'None' or min >= 0"],
            ),
        ),
        None,
    ),
    (
        _edit(
            "GEMMFwdOp",
            _set(("forall", "U"), "DType[float16]"),
            _set(("outputs", "d", "dtype"), "U"),
        ),
        "'U' cannot be solved",
    ),
    (
        _edit(
            "FusedMoESharedExpertFwdOp",
            _set(("outputs", "routed_output", "shape"), "[shared_ffn_size, hidden_size]"),
        ),
        "shared_ffn_size has kind NoneType, expected Int",
    ),
    (
        _edit("GQAPrefillVarlenFwdOp", _set(("shape_rules",), ["sum(q_lens) == total_q"])),
        "value list",
    ),
    (
        _edit("RMSNormFwdOp", _set(("params", "normalized_shape", "type"), "float")),
        "expected Seq[Int]",
    ),
    (_edit("GEMMFwdOp", _set(("inputs", "a", "shape"), "Mat[trans_a, M]")), "takes 3 arguments"),
    (_edit("GEMMFwdOp", _set(("forall", "d"), "Dim")), "declared twice"),
    (_edit("GEMMFwdOp", _set(("shape_rules",), ["present(trans_a)"])), "names no tensor"),
    (_edit("GEMMFwdOp", _set(("outputs", "d", "shape"), "[-trans_a, N]")), "expected an integer"),
    (_edit("GEMMFwdOp", _set(("let", "L"), "max(default=M)")), "misses argument 1"),
    (_edit("GEMMFwdOp", _set(("types", "Mat", "cases", 0, "is"), "[M, C]")), "'M' is not declared"),
    (
        _edit("GEMMFwdOp", _set(("inputs", "a", "mutated"), "M > 0")),
        "reads more than discriminants",
    ),
    (_edit("GEMMFwdOp", _set(("shape_rules",), ["M == None"])), "compares with None"),
    (_edit("GEMMFwdOp", _set(("types", "Mat", "match"), "len(())")), "not Bool, an enum or an ADT"),
    (
        _edit("GEMMFwdOp", _set(("types", "Mat", "params", "t"), "Maybe[Bool]")),
        "has kind Maybe",
    ),
    (_edit("GEMMFwdOp", _set(("outputs", "d", "optional"), True)), "unknown key 'optional'"),
    # Input order is the generated signature's parameter order, so a required input after an
    # optional one emits a parameter without a default after one with a default.
    (
        _edit(
            "GEMMFwdOp",
            _set(("inputs", "a", "default"), None),
            _set(("inputs", "a", "type"), "Tensor | None"),
        ),
        "an input without a default may not follow 'a'",
    ),
    (
        _edit(
            "MoEPrePermuteFwdOp",
            _set(("types", "ExpertRows", "cases", 0, "when"), {"masked": {"max_m": 1}}),
        ),
        "not a finite field",
    ),
    (
        _edit(
            "MoEPrePermuteFwdOp",
            _set(("types", "ExpertRows", "cases", 1, "when"), {"contiguous": {"bogus": 1}}),
        ),
        "has no field 'bogus'",
    ),
    (_edit("GEMMFwdOp", _set(("outputs", "d", "shape"), "[M[0], N]")), "not a sequence"),
    (_edit("GEMMFwdOp", _set(("dtype_combos",), [{"T": "float32"}])), "not in the set of 'T'"),
    (
        _edit(
            "MoEPrePermuteFwdOp",
            _set(("types", "ExpertRows", "cases", 1, "is"), "[P, layout.max_m]"),
        ),
        "constructor 'contiguous' lacks",
    ),
    (
        _edit(
            "FusedMoESharedExpertFwdOp",
            _set(
                ("outputs", "routed_output", "shape"),
                "[shared_ffn_size if present(shared_ffn_size) else num_tokens, hidden_size]",
            ),
        ),
        None,
    ),
    (
        _edit(
            "GEMMFwdOp",
            _set(("let", "L"), "trans_a or M > 0"),
            _set(("shape_rules",), ["not L"]),
            _set(("types", "Mat", "cases"), [{"when": False, "is": "[R, C]"}]),
        ),
        "0 cases match",
    ),
    (
        _edit(
            "GEMMFwdOp",
            _set(("params", "p"), {"type": "int | 'x'"}),
            _set(("shape_rules",), ["p < 2"]),
        ),
        "compares",
    ),
    (_edit("GEMMFwdOp", _set(("let", "z"), "1 if trans_a else 'x'")), "has arms of kinds"),
    (
        _edit(
            "MoEPrePermuteFwdOp",
            _set(("params", "q"), {"type": "MGroupedLayout | None"}),
            _set(("let", "z"), "q.max_m if present(q) else 0"),
        ),
        "reads q.max_m, which constructor 'contiguous' lacks",
    ),
    (
        _edit(
            "GEMMFwdOp",
            _set(("params", "eps"), {"type": "float"}),
            _set(("outputs", "d", "shape"), "[M if eps > 0.5 else N, N]"),
        ),
        "takes no part in types",
    ),
    (_edit("GEMMFwdOp", _set(("types", "Unused"), _UNUSED)), "Unused is applied by no shape"),
    (_edit("GEMMFwdOp", _set(("forall", "out"), "Dim")), "'out' is reserved"),
    (
        _edit(
            "GEMMFwdOp",
            _set(("params", "p"), {"type": "int | 'auto' | None"}),
            _set(
                ("outputs", "d", "shape"),
                "[M, N if not present(p) else (1 if p == 'auto' else p)]",
            ),
        ),
        None,
    ),
    (_edit("GEMMFwdOp", _set(("shape_rules",), ["T == 'bfloat16' or M > 0"])), None),
    (_edit("GEMMFwdOp", _set(("shape_rules",), ["T == 'bflaot16' or M > 0"])), "compares"),
    (_edit("GEMMFwdOp", _set(("params", "p"), {"type": "list[int | None]"})), None),
    (_edit("GEMMFwdOp", _set(("params", "p"), {"type": "tuple[int, bool]"})), None),
    (_edit("GEMMFwdOp", _set(("params", "p"), {"type": "list[float]"})), None),
    (
        _edit(
            "GEMMFwdOp",
            _set(("params", "mode"), {"type": "'full' | 'noop' | 'other'"}),
            _set(("let", "L"), "reduced((M,), None, False, mode) if mode != 'other' else (M,)"),
        ),
        None,
    ),
    (
        _edit(
            "GEMMFwdOp",
            _set(("params", "mode"), {"type": "'full' | 'noop' | 'other'"}),
            _set(("let", "L"), "reduced((M,), None, False, mode)"),
        ),
        "is not one of",
    ),
    (
        _edit("ConcatRowsFwdOp", _set(("inputs", "parts", "type"), "list[Tensor] | int")),
        "`list[Tensor]` is not yet a member of a union",
    ),
    (
        _edit("ConcatRowsFwdOp", _set(("inputs", "parts", "type"), "int")),
        "holds no tensor; a scalar is a parameter",
    ),
    (
        _edit("ConcatRowsFwdOp", _set(("inputs", "parts", "shape"), "[3, *A]")),
        "a `list[Tensor]` reads an element-wise (`Seq`) index",
    ),
    (
        _edit("ConcatRowsFwdOp", _set(("inputs", "parts", "mutated"), True)),
        "apply to one tensor",
    ),
    (
        _edit("ConcatRowsFwdOp", _set(("inputs", "parts", "type"), "Tensor | Strings")),
        "is not a union of distinct members",
    ),
    (
        _edit("ConcatRowsFwdOp", _set(("inputs", "head", "default"), 0)),
        "default 0 is not a value of 'Tensor'",
    ),
    (
        _edit("ConcatRowsFwdOp", _set(("params",), {"w": {"dtype": "T", "shape": "[2]"}})),
        "a tensor is a call-time input",
    ),
    # Inside the element type N is the element's item, a Dim, which has no length.
    (
        _edit("ConcatRowsFwdOp", _set(("inputs", "parts", "shape"), "[N, len(N), *A]")),
        "N has kind Dim, expected Seq",
    ),
]


@pytest.mark.parametrize(("case", "message"), _EDITS, ids=[m or "accepted" for _, m in _EDITS])
def test_judges_edit(case, message):
    name, entry = case
    errors, _ = check_entry(name, entry, _ADTS)
    errors += signature_schema_errors(entry["signature"])
    if message is None:
        assert errors == []
    else:
        assert any(message in e for e in errors), errors


_LAYOUT = SimpleNamespace(
    kind="contiguous", metadata_kind="per_row", packing="aligned", alignment=8
)
_PRIMITIVE_CALLS = [
    ("broadcast", ((8, 1, 16), (4, 16)), (8, 4, 16)),
    ("concat", ([(2, 3), (2, 5), (2, 0)], -1), (2, 8)),
    ("concat", ([(2, 3), (4, 3)], 0), (6, 3)),
    ("reduced", ((4, 8, 16), [0, 2], True, "full"), (1, 8, 1)),
    ("reduced", ((), [0], False, "full"), ()),
    ("reduced", ((4, 8), [], False, "noop"), (4, 8)),
    ("valid_axes", ([0, -1], 0), True),
    ("unique_axes", ([0, -2], 2), False),
    ("per_axis", (None, 0, 2, 3), 3),
    ("per_axis", ([5, 6], 1, 2), 6),
    ("ceil_div", (7, 2), 4),
    ("max", ([], 0), 0),
    ("promote_int_to_float", ("bool",), "bool"),
    ("promote_int_to_float", ("int8",), "float32"),
    ("promote_int_to_float", ("bfloat16",), "bfloat16"),
    ("coalesce_dtype", (None, "float32"), "float32"),
    ("coalesce_dtype", ("float16", "float32"), "float16"),
    ("conv.out", (32, 3, 2, 1, 1), 16),
    ("pool.out", (8, 3, 2, 0, 1, True), 4),
    ("pool.out", (8, 3, 2, 0, 1, False), 3),
    ("moe.capacity", (_LAYOUT, 10**20 + 1, 3), 10**20 + 24),
    ("moe.capacity", (SimpleNamespace(kind="masked", max_m=5), 7, 3), 15),
    (
        "moe.capacity",
        (
            SimpleNamespace(
                kind="contiguous", metadata_kind="physical_psum", packing="aligned", alignment=8
            ),
            10,
            3,
        ),
        32,
    ),
    ("mhc.expansion", (8,), 2),
    ("repeat", (512, 3), [512, 512, 512]),
]
_PRIMITIVE_DOMAIN_ERRORS = [
    ("broadcast", ((2,), (3,))),
    ("concat", ([(2, 3), (2, 4)], 0)),
    ("concat", ([(2, 3), (2, 3, 1)], 0)),
    ("concat", ([(), ()], 0)),
    ("concat", ([(2, 3)], 2)),
    ("concat", ([], 0)),
    ("promote", ([],)),
    ("promote", (["complex64"],)),
    ("promote", (["float8_e4m3fn", "float16"],)),
    ("reduced", ((4,), [], False, "reject")),
    ("per_axis", (None, 0, 2)),
    ("per_axis", ([1, 2, 3], 0, 2)),
    ("ceil_div", (1, 0)),
    ("max", ([],)),
    ("mhc.expansion", (7,)),
    ("repeat", (512, -1)),
]


@pytest.mark.parametrize(
    ("name", "args", "result"), _PRIMITIVE_CALLS, ids=[c[0] for c in _PRIMITIVE_CALLS]
)
def test_primitive_result(name, args, result):
    assert PRIMITIVES[name](*args) == result


@pytest.mark.parametrize(
    ("name", "args"), _PRIMITIVE_DOMAIN_ERRORS, ids=[c[0] for c in _PRIMITIVE_DOMAIN_ERRORS]
)
def test_primitive_domain(name, args):
    with pytest.raises(ValueError):
        PRIMITIVES[name](*args)


_SCALARS = (True, 0, 1, -1, 127, 128, -129, 255, 256, -255, -256, 2**31, -(2**31) - 1, 2**40)
_SCALARS += (1.5, -1.0, 255.9, 65504.0, 65520.0, 1e6, 3.4e38, 1e39, math.inf, -math.inf, math.nan)


@pytest.mark.parametrize(
    "dtype",
    ["bool", "uint8", "int8", "int16", "int32", "int64", "float16", "bfloat16", "float32"],
)
def test_category_and_representable_follow_torch(dtype):
    import torch

    t = torch.zeros(1, dtype=getattr(torch, dtype))
    torch_category = (
        "bool" if t.dtype == torch.bool else "float" if t.dtype.is_floating_point else "int"
    )
    assert PRIMITIVES["category"](dtype) == torch_category
    mask = torch.ones(1, dtype=torch.bool)
    for v in _SCALARS:
        try:
            t.masked_fill(mask, v)
            accepted = True
        except RuntimeError:
            accepted = False
        assert PRIMITIVES["representable"](v, dtype) == accepted, v
    assert [PRIMITIVES["category"](v) for v in (True, 1, 1.0, 1j)] == [
        "bool",
        "int",
        "float",
        "complex",
    ]


def test_adt_invariant_narrows_the_domain():
    adts = {
        "Flagged": {
            "sum": {"on": {"python": "a.On", "fields": {"flag": "bool"}, "invariant": "flag"}}
        }
    }
    entry = {
        "signature": {
            "forall": {"M": "Dim", "T": "DType[float16]"},
            "params": {"f": {"type": "Flagged"}},
            "types": {
                "G": {
                    "params": {"g": "Flagged", "M": "Dim"},
                    "match": "g",
                    "cases": [{"when": {"on": {"flag": True}}, "is": "[M]"}],
                }
            },
            "inputs": {"x": {"dtype": "T", "shape": "[M]"}},
            "outputs": {"y": {"dtype": "T", "shape": "G[f, M]"}},
        }
    }
    assert check_adts(adts) == (adts, [])
    assert check_entry("FlaggedFwdOp", entry, adts) == ([], [])
    # A payload's invariant narrows where it is present; `_` covers the absent payload.
    entry["signature"]["params"] = {"f": {"type": "Flagged | None"}}
    entry["signature"]["types"]["G"] = {
        "params": {"g": "Maybe[Flagged]", "M": "Dim"},
        "match": ["present(g)", "g"],
        "cases": [
            {"when": [False, "_"], "is": "[M]"},
            {"when": [True, {"on": {"flag": True}}], "is": "[M]"},
        ],
    }
    entry["signature"]["outputs"]["y"]["shape"] = "G[f, M]"
    assert check_entry("FlaggedFwdOp", entry, adts) == ([], [])


def test_adt_invariant_stays_in_the_language():
    adts = {
        "A": {
            "sum": {
                "c": {"python": "a.C", "fields": {"x": "Dim"}, "invariant": "isinstance(x, int)"}
            }
        }
    }
    assert any("isinstance is not a built-in" in e for e in check_adts(adts)[1])
    assert check_adts(_ADTS) == (_ADTS, [])


def test_validator_checks_every_level_of_an_entry(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "validate_manifest", Path(__file__).parents[1] / "scripts" / "validate_manifest.py"
    )
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    name, entry = _edit("GEMMFwdOp", _set(("outputs", "d", "shape"), "[M, Q]"))
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump({name: {"family": "missing", **entry}}))
    errors, _ = validator.validate_manifest(manifest_path=path)
    assert any("'Q' is not declared" in e for e in errors), errors
    assert any("missing required field 'status'" in e for e in errors), errors
    assert any("family 'missing' is not a tileops module" in e for e in errors), errors
    schema_only, _ = validator.validate_manifest(manifest_path=path, levels=frozenset({"schema"}))
    assert not any("'Q' is not declared" in e for e in schema_only), schema_only
    path.write_text(yaml.safe_dump({name: {"family": 1}}))
    errors, _ = validator.validate_manifest(manifest_path=path, levels=frozenset({"schema"}))
    assert any("'family' must be a str" in e for e in errors), errors


def test_validator_holds_an_implemented_key_to_an_exported_class(monkeypatch):
    import types

    spec = importlib.util.spec_from_file_location(
        "validate_manifest", Path(__file__).parents[1] / "scripts" / "validate_manifest.py"
    )
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)

    def DemoFwdOp():
        pass

    module = types.ModuleType("tileops.demo")
    module.__spec__ = importlib.machinery.ModuleSpec("tileops.demo", None)
    module.DemoFwdOp, module.__all__ = DemoFwdOp, ["DemoFwdOp"]
    monkeypatch.setitem(sys.modules, "tileops.demo", module)
    entry = {"family": "demo", "status": "implemented"}
    assert any(
        "does not export the class" in e for e in validator._family_errors("DemoFwdOp", entry)
    )
    assert validator._schema_errors(1, entry, {1: entry}) == [
        "[schema] 1: the key is not an op class name `<Name>FwdOp`"
    ]


# ---------------------------------------------------------------- tensor sequences


def _plan_of(shapes, sequences, known=()):
    from tileops.manifest._expr import parse
    from tileops.manifest._signature import unification

    nodes = {t: parse(text) for t, text in shapes.items()}
    return unification(nodes, set(known), {}, sequences)


def test_an_ordinary_tensor_solves_shared_names_before_any_element():
    steps = _plan_of({"head": "[*A]", "parts": "[N, *A]"}, {"parts": ("N",)})
    assert [s.action for s in steps] == ["bind", "each"]
    assert steps[-1].items == ("N",)


def test_element_zero_solves_what_waits_on_the_sequence():
    # `a` needs S, which the elements solve only once M is known; element 0 breaks the wait.
    steps = _plan_of({"a": "[M + len(S)]", "xs": "[M, *S]"}, {"xs": ("S",)})
    assert [(s.action, s.solved) for s in steps] == [
        ("first", ()),
        ("bind", ("M",)),
        ("bind", ("_item_S",)),
        ("each", ("S",)),
        ("check", ()),
    ]


def test_a_second_sequence_checks_the_items_the_first_collected():
    steps = _plan_of({"xs": "[N]", "ys": "[2 * N]"}, {"xs": ("N",), "ys": ("N",)})
    each = [s for s in steps if s.action == "each"]
    assert [(s.tensor, s.items) for s in each] == [("xs", ("N",)), ("ys", ())]


def test_a_roofline_reads_no_let_of_a_call_time_scalar():
    from tileops.manifest._plan import roofline_plan
    from tileops.manifest._signature import parse_signature

    name, entry = _edit(
        "ClampFwdOp",
        _set(("inputs", "min", "type"), "int | Tensor | None"),
        _set(("let", "Z"), "min if member(min) == 'int' else 0"),
    )
    sig = parse_signature(name, entry, _ADTS)
    errors, _ = roofline_plan(sig, {"flops": "Z"})
    assert any("let 'Z' reads a call-time scalar" in e for e in errors), errors


@pytest.mark.parametrize(
    ("value", "type_text", "accepted"),
    [
        ([[1, None], [2]], "list[list[int | None]]", True),
        ([["bad"]], "list[list[int | None]]", False),
        ((1, (2, 3)), "tuple[int, tuple[int, int]]", True),
        ([1.5, 2], "list[float]", True),
        ([1.5, "x"], "list[float]", False),
    ],
)
def test_a_list_parameter_checks_every_element_by_its_type(value, type_text, accepted):
    from tileops.manifest._values import convert

    if accepted:
        assert convert(value, type_text, {}) == value
    else:
        with pytest.raises(ValueError):
            convert(value, type_text, {})


def test_a_roofline_member_names_a_call_time_argument():
    from tileops.manifest._plan import roofline_plan
    from tileops.manifest._signature import parse_signature

    name, entry = _edit("ClampFwdOp")
    sig = parse_signature(name, entry, _ADTS)
    errors, _ = roofline_plan(sig, {"flops": "1 if member(typo) == 'Tensor' else 0"})
    assert any("member(typo) names no call-time argument" in e for e in errors), errors
    assert roofline_plan(sig, {"flops": "1 if member(min) == 'Tensor' else 0"})[0] == []


@pytest.mark.parametrize(
    ("value", "members", "expected"),
    [
        (2, ("float", "Tensor"), "float"),
        (2, ("int", "float"), "int"),
        (2, ("Number", "Tensor"), "Number"),
        (True, ("float", "Tensor"), None),
    ],
)
def test_an_int_takes_a_float_member_as_the_float_slot_does(value, members, expected):
    from tileops.manifest._signature import member_of

    assert member_of(value, members) == expected
