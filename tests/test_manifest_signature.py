"""Static signature checks (docs/design/manifest.md § Validation)."""

import copy
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tileops.manifest.plan import check_adts, check_entry, signature_schema_errors
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

_EDITS = [
    (
        _edit("ClampFwdOp", _set(("shape_rules",), ["min is not None or max is not None"])),
        "outside the expression language",
    ),
    (_edit("GemmFwdOp", _set(("shape_rules",), ["a.shape == (M, K)"])), "read as a value"),
    (
        _edit("SumFwdOp", _set(("shape_rules",), ["isinstance(dim, int)"])),
        "not a built-in primitive",
    ),
    (_edit("GemmFwdOp", _set(("outputs", "d", "shape"), "[*broadcast(M)]")), "expected Shape"),
    (_edit("GemmFwdOp", _set(("shape_rules",), ["M + 1"])), "expected Bool"),
    (_edit("GemmFwdOp", _set(("forall", "T"), "DType[banana]")), "unknown kind"),
    (_edit("GemmFwdOp", _set(("let", "L"), "(")), "cannot parse"),
    (_edit("MaxPool2dFwdOp", _set(("let", "kH"), "sH + 1")), "let cycle"),
    (_edit("ClampFwdOp", _set(("shape_rules",), [])), "0 cases match"),
    (
        _edit(
            "MoePrePermuteFwdOp",
            _without_masked,
            _set(("shape_rules",), ["K > 0 and layout.kind == 'contiguous'"]),
        ),
        "0 cases match",
    ),
    (
        _edit(
            "MoePrePermuteFwdOp",
            _without_masked,
            _set(("shape_rules",), ["layout.kind == 'contiguous'"]),
        ),
        None,
    ),
    (
        _edit("SiluAndMulFwdOp", _set(("inputs", "x", "shape"), "[M, N * N]")),
        "'N' cannot be solved",
    ),
    (
        _edit(
            "GemmFwdOp",
            _set(("forall", "U"), "DType[float16]"),
            _set(("outputs", "d", "dtype"), "U"),
        ),
        "'U' cannot be solved",
    ),
    (
        _edit(
            "FusedMoeSharedExpertFwdOp",
            _set(("outputs", "routed_output", "shape"), "[shared_ffn_size.value, hidden_size]"),
        ),
        "present(shared_ffn_size) is false",
    ),
    (
        _edit("GQAPrefillVarlenFwdOp", _set(("shape_rules",), ["sum(q_lens) == total_q"])),
        "value list",
    ),
    (
        _edit("RMSNormFwdOp", _set(("params", "normalized_shape", "type"), "float")),
        "expected Seq[Int]",
    ),
    (_edit("GemmFwdOp", _set(("inputs", "a", "shape"), "Mat[trans_a, M]")), "takes 3 arguments"),
    (_edit("GemmFwdOp", _set(("forall", "d"), "Dim")), "declared twice"),
    (_edit("GemmFwdOp", _set(("shape_rules",), ["present(trans_a)"])), "names no tensor"),
    (_edit("GemmFwdOp", _set(("outputs", "d", "shape"), "[-trans_a, N]")), "expected an integer"),
    (_edit("GemmFwdOp", _set(("let", "L"), "max(default=M)")), "misses argument 1"),
    (_edit("GemmFwdOp", _set(("types", "Mat", "cases", 0, "is"), "[M, C]")), "'M' is not declared"),
    (
        _edit("GemmFwdOp", _set(("inputs", "a", "mutated"), "M > 0")),
        "reads more than discriminants",
    ),
    (_edit("GemmFwdOp", _set(("shape_rules",), ["M == None"])), "compares with None"),
    (_edit("GemmFwdOp", _set(("types", "Mat", "match"), "len(())")), "not Bool, an enum or an ADT"),
    (_edit("GemmFwdOp", _set(("outputs", "d", "optional"), True)), "unknown key 'optional'"),
    (
        _edit(
            "MoePrePermuteFwdOp",
            _set(("types", "ExpertRows", "cases", 0, "when"), {"masked": {"max_m": 1}}),
        ),
        "not a finite field",
    ),
    (
        _edit(
            "MoePrePermuteFwdOp",
            _set(("types", "ExpertRows", "cases", 1, "when"), {"contiguous": {"bogus": 1}}),
        ),
        "has no field 'bogus'",
    ),
    (_edit("GemmFwdOp", _set(("outputs", "d", "shape"), "[M[0], N]")), "not a sequence"),
    (_edit("GemmFwdOp", _set(("dtype_combos",), [{"T": "float32"}])), "not in the set of 'T'"),
    (
        _edit(
            "MoePrePermuteFwdOp",
            _set(("types", "ExpertRows", "cases", 1, "is"), "[P, layout.max_m]"),
        ),
        "constructor 'contiguous' lacks",
    ),
    (
        _edit(
            "FusedMoeSharedExpertFwdOp",
            _set(
                ("outputs", "routed_output", "shape"),
                "[shared_ffn_size.value if present(shared_ffn_size) else num_tokens, hidden_size]",
            ),
        ),
        None,
    ),
    (
        _edit(
            "GemmFwdOp",
            _set(("let", "L"), "trans_a or M > 0"),
            _set(("shape_rules",), ["not L"]),
            _set(("types", "Mat", "cases"), [{"when": False, "is": "[R, C]"}]),
        ),
        "0 cases match",
    ),
    (
        _edit(
            "GemmFwdOp",
            _set(("params", "p"), {"type": "int | 'x'"}),
            _set(("shape_rules",), ["p < 2"]),
        ),
        "compares",
    ),
    (_edit("GemmFwdOp", _set(("let", "z"), "1 if trans_a else 'x'")), "has arms of kinds"),
    (
        _edit(
            "MoePrePermuteFwdOp",
            _set(("params", "q"), {"type": "MGroupedLayout | None"}),
            _set(("let", "z"), "q.value.max_m if present(q) else 0"),
        ),
        "reads q.value.max_m, which constructor 'contiguous' lacks",
    ),
    (
        _edit(
            "GemmFwdOp",
            _set(("params", "eps"), {"type": "float"}),
            _set(("outputs", "d", "shape"), "[M if eps > 0.5 else N, N]"),
        ),
        "takes no part in types",
    ),
    (_edit("GemmFwdOp", _set(("types", "Unused"), _UNUSED)), "Unused is applied by no shape"),
    (_edit("GemmFwdOp", _set(("forall", "out"), "Dim")), "'out' is reserved"),
    (
        _edit(
            "GemmFwdOp",
            _set(("params", "p"), {"type": "int | 'auto' | None"}),
            _set(
                ("outputs", "d", "shape"),
                "[M, N if not present(p) else (1 if p.value == 'auto' else p.value)]",
            ),
        ),
        None,
    ),
    (_edit("GemmFwdOp", _set(("shape_rules",), ["T == 'bfloat16' or M > 0"])), None),
    (_edit("GemmFwdOp", _set(("shape_rules",), ["T == 'bflaot16' or M > 0"])), "compares"),
    (_edit("GemmFwdOp", _set(("params", "p"), {"type": "list[int | None]"})), None),
    (_edit("GemmFwdOp", _set(("params", "p"), {"type": "tuple[int, bool]"})), "holds int or"),
    (
        _edit(
            "GemmFwdOp",
            _set(("params", "mode"), {"type": "'full' | 'noop' | 'other'"}),
            _set(("let", "L"), "reduced((M,), None, False, mode) if mode != 'other' else (M,)"),
        ),
        None,
    ),
    (
        _edit(
            "GemmFwdOp",
            _set(("params", "mode"), {"type": "'full' | 'noop' | 'other'"}),
            _set(("let", "L"), "reduced((M,), None, False, mode)"),
        ),
        "is not one of",
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
]
_PRIMITIVE_DOMAIN_ERRORS = [
    ("broadcast", ((2,), (3,))),
    ("reduced", ((4,), [], False, "reject")),
    ("per_axis", (None, 0, 2)),
    ("per_axis", ([1, 2, 3], 0, 2)),
    ("ceil_div", (1, 0)),
    ("max", ([],)),
    ("mhc.expansion", (7,)),
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
        "match": ["present(g)", "g.value"],
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


def test_validator_checks_converted_families(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "validate_manifest", Path(__file__).parents[1] / "scripts" / "validate_manifest.py"
    )
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    name, entry = _edit("GemmFwdOp", _set(("outputs", "d", "shape"), "[M, Q]"))
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump({name: {"family": "converted", **entry}}))
    errors, _ = validator.validate_manifest(manifest_path=path)
    assert any("'Q' is not declared" in e for e in errors), errors
    assert any("missing required field 'status'" in e for e in errors), errors
    assert any("family 'converted' is not a tileops module" in e for e in errors), errors
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

    def DemoFwdOp():  # noqa: N802 - a function under the op's name
        pass

    module = types.ModuleType("tileops.demo")
    module.__spec__ = importlib.machinery.ModuleSpec("tileops.demo", None)
    module.DemoFwdOp, module.__all__ = DemoFwdOp, ["DemoFwdOp"]
    monkeypatch.setitem(sys.modules, "tileops.demo", module)
    entry = {"family": "demo", "status": "implemented"}
    assert any(
        "does not export the class" in e for e in validator._family_errors("DemoFwdOp", entry)
    )
    assert validator._check_parametric_schema(1, entry, {1: entry}) == [
        "[schema] 1: the key is not an op class name `<Name>FwdOp`"
    ]
