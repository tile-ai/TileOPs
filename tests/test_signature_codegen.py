"""Methods generated from a parametric signature (docs/design/manifest.md § Call Semantics)."""

import copy
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils._python_dispatch import TorchDispatchMode

from tileops._ops._naming import operator_name
from tileops._ops._signature_codegen import (
    CheckError,
    _SignaturePlan,
    install,
    install_compile_boundary,
)
from tileops.manifest import load_adts, load_manifest
from tileops.manifest._expr import SignatureError
from tileops.manifest._plan import entry_plan
from tileops.manifest._workload import instantiate
from workloads.device import run_device, run_device_available

pytestmark = pytest.mark.smoke

_CASES = yaml.safe_load((Path(__file__).parent / "manifest_cases.yaml").read_text())
_ENTRIES = _CASES["entries"]


def _op(name, params):
    cls = type(name, (), {})
    install(cls, _ENTRIES[name], _CASES["adts"])
    op = cls()
    vars(op).update(params)
    op._construction_indices = op._check_construction()
    return op


def _calls(name):
    sig = entry_plan(name, _ENTRIES[name], _CASES["adts"])
    for row in _ENTRIES[name].get("workloads", []):
        if not row.get("expect_fail"):
            for case in row.get("dtype_cases") or [{}]:
                yield instantiate(sig, row, case)


@pytest.mark.parametrize("name", sorted(_ENTRIES))
def test_check_accepts_every_instantiated_row(name):
    for call in _calls(name):
        tensors = call.materialize(device="cpu")
        op = _op(name, call.arguments())
        checked = type(op)._signature.check(op, tensors)
        for t in call.specs:
            assert checked.tensors.get(t) == call.tensors.get(t)


def _gemm(**roofline):
    cls = type("GEMMFwdOp", (), {"__init__": lambda self, **p: vars(self).update(p)})
    install(cls, {**_ENTRIES["GEMMFwdOp"], "roofline": roofline}, _CASES["adts"])
    op = cls(trans_a=False, trans_b=True)
    op._construction_indices = op._check_construction()
    return op


_A, _B = torch.zeros(4, 8, dtype=torch.float16), torch.zeros(16, 8, dtype=torch.float16)


@pytest.mark.parametrize(
    ("tensors", "message"),
    [
        ({"a": _A, "b": _B.float()}, "b dtype differs from T"),
        ({"a": _A, "b": _B[:, :4]}, "does not match K"),
        ({"a": _A[None], "b": _B}, "a needs rank == 2"),
        ({"b": _B}, "'a' is required"),
    ],
    ids=["dtype", "axis", "rank", "missing"],
)
def test_check_rejects(tensors, message):
    op = _gemm(flops="1")
    with pytest.raises(ValueError, match=message):
        type(op)._signature.check(op, tensors)


@pytest.mark.parametrize(
    "name", ["ClampFwdOp", "ConcatRowsFwdOp", "GQAPrefillVarlenFwdOp", "SumFwdOp"]
)
def test_check_traces_on_symints(name):
    call = next(_calls(name))
    tensors = call.materialize(device="cpu")
    op = _op(name, call.arguments())
    plan = type(op)._signature
    torch._dynamo.reset()
    compiled = torch.compile(lambda ts: plan.check(op, ts).tensors, fullgraph=True, dynamic=True)
    assert compiled(tensors) == plan.check(op, tensors).tensors


def test_eval_roofline_prices_the_last_call(monkeypatch):
    derived = _gemm(flops="2 * M * N * K")
    derived._last_call = type(derived)._signature.check(derived, {"a": _A, "b": _B})
    assert derived.eval_roofline() == (2 * 4 * 16 * 8, (4 * 8 + 16 * 8 + 4 * 16) * 2)
    written = _gemm(flops="M", bytes="bytes(d) if present(a) else 0")
    written._last_call = derived._last_call
    assert written.eval_roofline() == (4, 4 * 16 * 2)
    import tileops._perf.formulas

    monkeypatch.setattr(
        tileops._perf.formulas,
        "probe_gemm",
        lambda call: (call.indices["M"], call.bytes("a")),
        raising=False,
    )
    func = _gemm(func="tileops._perf.formulas.probe_gemm")
    func._last_call = derived._last_call
    assert func.eval_roofline() == (4, 4 * 8 * 2)
    with pytest.raises(RuntimeError, match="needs a completed call"):
        _gemm(flops="1").eval_roofline()


def test_construction_checks_what_construction_decides():
    _op("DSADecodeFwdOp", {"dim_tail": -1})  # a signed offset carries no obligation of its own
    with pytest.raises(ValueError, match="axis 1 of hidden_states, hidden_size, is negative"):
        _op(
            "FusedMoESharedExpertFwdOp",
            {"num_tokens": 2, "hidden_size": -1, "shared_ffn_size": None},
        )


def _probe(name, signature, forward, *, boundary=False, roofline=None, status="implemented"):
    """An `Op` subclass whose entry is *signature* and whose `forward` is *forward*.

    Without *boundary* the entry carries a composition, which is what leaves an op with a
    call-time tensor input without a compile boundary.
    """
    from tileops._ops.op_base import Op

    entry = {
        "family": "probe",
        "status": status,
        "signature": signature,
        "roofline": roofline or {"flops": "1"},
    }
    if not boundary:
        entry["composition"] = {"kind": "composite", "stages": []}

    def construct(self, **params):
        vars(self).update(params)
        Op.__init__(self)

    # Inside tileops, as the in-tree ops these stand in for are: a target serves only those.
    body = {"__module__": "tileops._probe", "__init__": construct, "forward": forward}
    cls = type(name, (Op,), body)
    install(cls, entry)
    install_compile_boundary(cls, entry)
    return cls


_SILU = _ENTRIES["SiLUAndMulFwdOp"]["signature"]


def test_op_call_runs_the_check_and_keeps_its_binding():
    probe = _probe(
        "ProbeSiLUAndMulFwdOp",
        _SILU,
        lambda self, x: x[:, : x.shape[1] // 2].clone(),
        roofline={"flops": "M * N"},
    )
    op = probe()
    with pytest.raises(RuntimeError, match="no call has completed"):
        _ = op.last_call
    op(torch.zeros(3, 8, dtype=torch.float16))
    assert op.eval_roofline() == (12, (3 * 8 + 3 * 4) * 2)
    assert op.eval_roofline_read_bytes() == 3 * 8 * 2
    with pytest.raises(ValueError, match="axis 1 is not 2 \\* N"):
        op(torch.zeros(3, 7, dtype=torch.float16))
    assert op._infer_output_shapes((5, 8)) == {"output": (5, 4)}


def test_the_returned_value_is_checked_before_the_call_is_kept():
    op = _probe("ProbeWrongDtypeFwdOp", _SILU, lambda self, x: x[:, :4].float())()
    with pytest.raises(ValueError, match="output has dtype torch.float32"):
        op(torch.zeros(3, 8, dtype=torch.float16))
    with pytest.raises(RuntimeError, match="needs a completed call"):
        op.eval_roofline()


def test_an_out_buffer_is_held_to_the_call_device():
    signature = {
        "forall": {"M": "Dim", "T": "DType[float16]"},
        "inputs": {"x": {"dtype": "T", "shape": "[M]"}},
        "outputs": {"y": {"dtype": "T", "shape": "[M]", "buffer": "out"}},
    }
    op = _probe("ProbeOutFwdOp", signature, lambda self, x, out=None: out)()
    x = torch.zeros(3, dtype=torch.float16)
    with pytest.raises(ValueError, match="out is not on the call device"):
        op(x, torch.empty(3, dtype=torch.float16, device="meta"))
    out = torch.empty(3, dtype=torch.float16)
    assert op(x, out) is out


def test_each_effect_branch_registers_its_own_operator():
    signature = {
        "forall": {"M": "Dim", "T": "DType[float16]"},
        "params": {"inplace": {"type": "bool"}},
        "inputs": {"x": {"dtype": "T", "shape": "[M]", "mutated": "inplace"}},
        "outputs": {"y": {"dtype": "T", "shape": "[M]", "alias": "x"}},
    }
    probe = _probe(
        "ProbeInplaceFwdOp",
        signature,
        lambda self, x: x.add_(1) if self.inplace else x + 1,
        boundary=True,
    )
    assert probe.compile_op_names == (
        "tileops::probe_inplace_fwd",
        "tileops::probe_inplace_fwd_writes_x",
    )
    x = torch.zeros(2, dtype=torch.float16)
    assert probe(inplace=True)(x) is x and x.tolist() == [1, 1]
    y = probe(inplace=False)(x)
    assert y is not x and y.tolist() == [2, 2]


def test_a_traced_boundary_call_is_one_graph():
    signature = _UNARY
    op = _probe("ProbeTracedFwdOp", signature, lambda self, x: x + 1, boundary=True)()
    x = torch.zeros(4, dtype=torch.float16)
    torch._dynamo.reset()
    compiled = torch.compile(op, fullgraph=True)
    assert compiled(x).tolist() == [1] * 4
    assert compiled(x).tolist() == [1] * 4


def test_each_output_slot_is_returned_on_its_own():
    signature = {
        **_VEC,
        "params": {"with_aux": {"type": "bool"}},
        "outputs": {
            "y": {"dtype": "T", "shape": "[M]", "buffer": "out"},
            "aux": {"dtype": "T", "shape": "[M]", "nullable": "with_aux"},
        },
    }

    def eager(self, x, out=None):
        out.copy_(x + 1)
        return out, (x + 2 if self.with_aux else None)

    probe = _probe("ProbeMixedFwdOp", signature, eager, boundary=True)
    x, out = torch.zeros(2, dtype=torch.float16), torch.empty(2, dtype=torch.float16)
    y, aux = probe(with_aux=True)(x, out)
    assert y is out and aux.tolist() == [2, 2]
    y, aux = probe(with_aux=False)(x, out)
    assert y is out and aux is None


def test_the_fake_leaves_an_absent_output_absent():
    from torch._subclasses.fake_tensor import FakeTensorMode

    signature = {
        "forall": {"M": "Dim", "T": "DType[float16]"},
        "params": {"with_aux": {"type": "bool"}},
        "inputs": {"x": {"dtype": "T", "shape": "[M]"}},
        "outputs": {
            "y": {"dtype": "T", "shape": "[M]"},
            "aux": {"dtype": "T", "shape": "[M]", "nullable": "with_aux"},
        },
    }
    probe = _probe("ProbeNullableFwdOp", signature, lambda self, x: (x + 1, None), boundary=True)
    op = probe(with_aux=False)
    with FakeTensorMode() as mode:
        y, aux = op(mode.from_tensor(torch.zeros(4, dtype=torch.float16)))
    assert aux is None and tuple(y.shape) == (4,)


def test_construction_holds_parameters_to_their_type():
    signature = {
        "forall": {"M": "Dim", "T": "DType[float16]"},
        "params": {"flag": {"type": "bool"}},
        "inputs": {"x": {"dtype": "T", "shape": "[M]"}},
        "outputs": {"y": {"dtype": "T", "shape": "[M]"}},
    }
    probe = _probe("ProbeFlagFwdOp", signature, lambda self, x: x)
    with pytest.raises(ValueError, match="flag = 1 is not a bool"):
        probe(flag=1)


def test_roofline_expressions_read_only_ix():
    from tileops.manifest._plan import roofline_plan
    from tileops.manifest._signature import parse_signature

    sig = parse_signature("SiLUAndMulFwdOp", {"signature": _SILU}, {})
    assert roofline_plan(sig, {"flops": "M * N", "bytes": "bytes(x) + bytes(output)"})[0] == []
    errors = roofline_plan(sig, {"flops": "NOPE", "extra": 1})[0]
    assert any("unknown key 'extra'" in e for e in errors)
    assert any("'NOPE' is not in indices" in e for e in errors)
    assert roofline_plan(None, {"flops": "isinstance(M)", "extra": 1})[0] == [
        "roofline: unknown key 'extra'",
        "roofline.flops: isinstance is not a built-in primitive",
    ]


_VEC = {
    "forall": {"M": "Dim", "T": "DType[float16]"},
    "inputs": {"x": {"dtype": "T", "shape": "[M]"}},
}
# `x -> y` of one shape and dtype.
_UNARY = {**_VEC, "outputs": {"y": {"dtype": "T", "shape": "[M]"}}}


class _Passthrough(TorchDispatchMode):
    """Sends every call through the dispatcher, which runs it unchanged."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


def test_execution_parameters_cross_the_compile_boundary():
    def eager(self, x, scale: int = 1):
        return x * scale

    signature = _UNARY
    op = _probe("ProbeScaleFwdOp", signature, eager, boundary=True)()
    assert op(torch.ones(2, dtype=torch.float16), 3).tolist() == [3, 3]


def test_a_returned_out_buffer_is_checked_after_the_call():
    signature = {**_VEC, "outputs": {"y": {"dtype": "T", "shape": "[M]", "buffer": "out"}}}
    op = _probe("ProbeResizeFwdOp", signature, lambda self, x, out=None: out.resize_(4))()
    with pytest.raises(ValueError, match="y has shape"):
        op(torch.zeros(3, dtype=torch.float16), torch.empty(3, dtype=torch.float16))


def test_a_domain_restriction_on_parameters_is_checked_at_construction():
    signature = {
        **_VEC,
        "params": {"flag": {"type": "bool"}},
        "outputs": {"y": {"dtype": "T", "shape": "[M]"}},
        "shape_rules": ["flag"],
    }
    probe = _probe("ProbeFlaggedFwdOp", signature, lambda self, x=None: x)
    with pytest.raises(ValueError, match="refinement fails: flag"):
        probe(flag=False)


def test_a_contiguous_input_must_be_contiguous():
    signature = {
        "forall": {"M": "Dim", "N": "Dim", "T": "DType[float16]"},
        "inputs": {"x": {"dtype": "T", "shape": "[M, N]", "contiguous": True}},
        "outputs": {"y": {"dtype": "T", "shape": "[M, N]"}},
    }
    op = _probe("ProbeContiguousFwdOp", signature, lambda self, x: x)()
    with pytest.raises(ValueError, match="x must be contiguous"):
        op(torch.zeros(3, 2, dtype=torch.float16).t())


def test_an_alias_is_priced_as_a_fresh_output_where_its_input_is_not_written():
    signature = {
        **_VEC,
        "params": {"inplace": {"type": "bool"}},
        "inputs": {"x": {"dtype": "T", "shape": "[M]", "mutated": "inplace"}},
        "outputs": {"y": {"dtype": "T", "shape": "[M]", "alias": "x"}},
    }
    probe = _probe("ProbeAliasFwdOp", signature, lambda self, x: x + 1)
    op = probe(inplace=False)
    op(torch.zeros(4, dtype=torch.float16))
    assert op.eval_roofline()[1] == 4 * 2 + 4 * 2


def test_an_adt_parameter_must_be_its_declared_class():
    from tileops._ops.moe.contracts import MaskedLayoutSpec

    cls = type("MoEPrePermuteFwdOp", (), {})
    install(cls, _ENTRIES["MoEPrePermuteFwdOp"], _CASES["adts"])
    op = cls()
    vars(op).update(layout=MaskedLayoutSpec(max_m=4), num_local_experts=2)
    op._check_construction()
    op.layout = type("Impostor", (), {"kind": "masked", "max_m": 4})()
    with pytest.raises(ValueError, match="is not an object of a MGroupedLayout constructor"):
        op._check_construction()


def test_every_manifest_construction_point_emits():
    """A point's checks are emitted on the first construction there; emitting every point of
    every manifest entry keeps one that no other test constructs compiling."""
    adts = load_adts()
    for name, entry in load_manifest().items():
        try:
            plan = _SignaturePlan(entry_plan(name, entry, adts))
        except SignatureError:
            continue
        for key in list(plan._pending):
            plan._emit(key)
        assert not plan._pending, name


def test_axes_nothing_reads_emit_no_checks():
    signature = {
        **_VEC,
        "params": {f"unused{i}": {"type": "bool"} for i in range(12)},
        "outputs": {"y": {"dtype": "T", "shape": "[M]"}},
    }
    cls = type("ProbeUnusedFwdOp", (), {})
    install(cls, {"signature": signature}, {})
    assert len(cls._signature.effects) == 1


def test_an_output_present_with_out_has_its_own_operator():
    from torch._subclasses.fake_tensor import FakeTensorMode

    signature = {
        **_VEC,
        "outputs": {
            "y": {"dtype": "T", "shape": "[M]", "buffer": "out"},
            "aux": {"dtype": "T", "shape": "[M]", "nullable": "present(out)"},
        },
    }

    def eager(self, x, out=None):
        if out is None:
            return x + 1, None
        out.copy_(x + 1)
        return out, x + 2

    op = _probe("ProbeOutAuxFwdOp", signature, eager, boundary=True)()
    x, out = torch.zeros(2, dtype=torch.float16), torch.empty(2, dtype=torch.float16)
    y, aux = op(x, out)
    assert y is out and aux.tolist() == [2, 2]
    with FakeTensorMode() as mode:
        _, aux = op(mode.from_tensor(x), mode.from_tensor(out))
    assert tuple(aux.shape) == (2,)
    torch._dynamo.reset()
    y, aux = torch.compile(op, fullgraph=True)(x, out)
    assert aux.tolist() == [2, 2]


def test_one_instance_takes_its_own_branch_per_presence():
    """An effect branch is a function of what the call passed, not of the instance. One
    instance that first omits two optional inputs, then passes them, takes the branch that
    writes one and emits the output the other gates, and takes the first branch again when
    the next call omits them."""
    signature = {
        "forall": {"M": "Dim", "T": "DType[float16]"},
        "inputs": {
            "x": {"dtype": "T", "shape": "[M]"},
            "bias": {"dtype": "T", "shape": "[M]", "type": "Tensor | None", "default": None},
            "acc": {
                "dtype": "T",
                "shape": "[M]",
                "type": "Tensor | None",
                "default": None,
                "mutated": "present(acc)",
            },
        },
        "outputs": {
            "y": {"dtype": "T", "shape": "[M]"},
            "aux": {"dtype": "T", "shape": "[M]", "nullable": "present(bias)"},
        },
    }

    def eager(self, x, bias=None, acc=None):
        if acc is not None:
            acc.add_(x)
        return x + (1 if bias is None else bias), (None if bias is None else bias + 1)

    op = _probe("ProbeBranchPerPresenceFwdOp", signature, eager, boundary=True)()
    x = torch.ones(2, dtype=torch.float16)
    bias = torch.full((2,), 3, dtype=torch.float16)
    acc = torch.zeros(2, dtype=torch.float16)
    y, aux = op(x)
    assert y.tolist() == [2, 2] and aux is None and acc.tolist() == [0, 0]
    y, aux = op(x, bias, acc)
    assert y.tolist() == [4, 4] and aux.tolist() == [4, 4] and acc.tolist() == [1, 1]
    y, aux = op(x)
    assert y.tolist() == [2, 2] and aux is None and acc.tolist() == [1, 1]


def test_a_roofline_func_must_yield_two_ints(monkeypatch):
    import tileops._perf.formulas

    monkeypatch.setattr(tileops._perf.formulas, "probe_float", lambda call: (1.5, 2), raising=False)
    signature = _UNARY
    op = _probe(
        "ProbeFloatRoofFwdOp",
        signature,
        lambda self, x: x.clone(),
        roofline={"func": "tileops._perf.formulas.probe_float"},
    )()
    op(torch.zeros(2, dtype=torch.float16))
    with pytest.raises(TypeError, match="not \\(flops: int, bytes: int\\)"):
        op.eval_roofline()


def test_generated_methods_bind_inputs_by_name():
    signature = {
        **_VEC,
        "inputs": {
            **_VEC["inputs"],
            "w": {"dtype": "T", "shape": "[M]", "type": "Tensor | None", "default": None},
        },
        "outputs": {"y": {"dtype": "T", "shape": "[M]"}},
    }
    op = _probe("ProbeNamedFwdOp", signature, lambda self, x, w=None: x + 1)()
    x = torch.zeros(2, dtype=torch.float16)
    op(x, w=x)
    with pytest.raises(ValueError, match="'x' is not a tensor"):
        type(op)._signature.check(op, {"x": 3, "w": None})
    assert op._infer_output_shapes(x=(2,)) == {"y": (2,)}
    with pytest.raises(TypeError):
        op._infer_output_shapes((2,), (9,), (1,))


def test_a_kw_only_input_is_taken_by_name_after_out():
    """The boundary, the body and `_infer_output_shapes` take a `kw_only` input after `out`,
    keyword-only, so a positional `out` binds `out`."""
    signature = {
        **_VEC,
        "inputs": {
            **_VEC["inputs"],
            "w": {
                "dtype": "T",
                "shape": "[M]",
                "type": "Tensor | None",
                "default": None,
                "kw_only": True,
            },
        },
        "outputs": {"y": {"dtype": "T", "shape": "[M]", "buffer": "out"}},
    }

    def eager(self, x, out=None, *, w=None):
        y = x + 1 if w is None else x + w
        return y if out is None else out.copy_(y)

    op = _probe("ProbeKeywordFwdOp", signature, eager, boundary=True)()
    x, w = torch.ones(2, dtype=torch.float16), torch.full((2,), 5, dtype=torch.float16)
    out = torch.zeros(2, dtype=torch.float16)
    assert op(x, out) is out and out.tolist() == [2, 2]
    assert op(x, out, w=w) is out and out.tolist() == [6, 6]
    with pytest.raises(TypeError):
        op(x, out, w)
    assert op._infer_output_shapes((2,), w=(2,)) == {"y": (2,)}
    with pytest.raises(TypeError):
        op._infer_output_shapes((2,), (2,))


def test_a_roofline_reads_the_presence_of_out():
    signature = {**_VEC, "outputs": {"y": {"dtype": "T", "shape": "[M]", "buffer": "out"}}}

    def eager(self, x, out=None):
        return x + 1 if out is None else out.copy_(x + 1)

    op = _probe(
        "ProbeOutRoofFwdOp", signature, eager, roofline={"flops": "1 if present(out) else 2"}
    )()
    x = torch.zeros(2, dtype=torch.float16)
    op(x)
    assert op.eval_roofline()[0] == 2
    op(x, torch.empty(2, dtype=torch.float16))
    assert op.eval_roofline()[0] == 1


def test_one_output_at_most_takes_the_out_buffer():
    from tileops.manifest._plan import effect_errors
    from tileops.manifest._signature import parse_signature

    buffered = {"dtype": "T", "shape": "[M]", "buffer": "out"}
    signature = {**_VEC, "outputs": {"y": buffered, "z": buffered}}
    sig = parse_signature("ProbeTwoOutFwdOp", {"signature": signature}, {})
    assert effect_errors(sig) == [
        "outputs ['y', 'z']: one output takes the `out` buffer, not several"
    ]


def test_a_non_tensor_out_fails_by_name():
    signature = {**_VEC, "outputs": {"y": {"dtype": "T", "shape": "[M]", "buffer": "out"}}}
    op = _probe("ProbeIntOutFwdOp", signature, lambda self, x, out=None: x.clone())()
    with pytest.raises(ValueError, match="'out' is not a tensor"):
        op(torch.zeros(2, dtype=torch.float16), 3)


def test_a_failed_evaluation_names_its_declaration():
    signature = {
        "forall": {"A": "Shape", "B": "Shape", "T": "DType[float16]"},
        "inputs": {"x": {"dtype": "T", "shape": "[*A]"}, "y": {"dtype": "T", "shape": "[*B]"}},
        "outputs": {"z": {"dtype": "T", "shape": "[*broadcast(A, B)]"}},
    }
    op = _probe("ProbeBroadcastFwdOp", signature, lambda self, x, y: x + y)()
    x, y = torch.zeros(2, dtype=torch.float16), torch.zeros(3, dtype=torch.float16)
    with pytest.raises(
        CheckError, match="ProbeBroadcastFwdOp: tensor 'z' shape: .*not broadcastable"
    ):
        op(x, y)


def test_an_obligation_a_call_time_presence_activates_waits_for_the_call():
    signature = {
        **_VEC,
        "params": {"p": {"type": "int", "default": -1}},
        "inputs": {
            **_VEC["inputs"],
            "w": {"dtype": "T", "shape": "[p]", "type": "Tensor | None", "default": None},
        },
        "outputs": {"y": {"dtype": "T", "shape": "[M]"}},
        "shape_rules": ["not present(w) or p >= 0"],
    }
    op = _probe("ProbeDeferredFwdOp", signature, lambda self, x, w=None: x + 1)(p=-1)
    op(torch.zeros(2, dtype=torch.float16))
    with pytest.raises(ValueError, match="w shape does not match p"):
        op(torch.zeros(2, dtype=torch.float16), torch.zeros(0, dtype=torch.float16))


def test_shape_inference_binds_dtype_indices_from_the_dtypes_passed():
    signature = {
        "forall": {"M": "Dim", "T": "DType[float16 | float32]"},
        "inputs": {"x": {"dtype": "T", "shape": "[M]"}},
        "let": {"H": "M // 2 if T == 'float16' else M"},
        "outputs": {"y": {"dtype": "T", "shape": "[H]"}},
    }
    op = _probe("ProbePackedFwdOp", signature, lambda self, x: x[: x.shape[0] // 2].clone())()
    assert op._infer_output_shapes((4,), dtypes={"x": torch.float16}) == {"y": (2,)}
    with pytest.raises(ValueError, match="let H: .*'T'"):
        op._infer_output_shapes((4,))


def test_an_entry_with_a_call_time_input_and_no_composition_has_a_compile_boundary():
    source = {
        "forall": {"T": "DType[float16]"},
        "params": {"n": {"type": "int"}},
        "outputs": {"y": {"dtype": "T", "shape": "[n]"}},
    }
    # `boundary=True` leaves the composition out, so only the missing input decides here.
    assert (
        _probe("ProbeSourceFwdOp", source, lambda self: None, boundary=True).compile_op_names == ()
    )
    assert _probe("ProbeComposedFwdOp", _SILU, lambda self, x: None).compile_op_names == ()
    leaf = _probe("ProbeLeafFwdOp", _SILU, lambda self, x: None, boundary=True)
    assert leaf.compile_op_names == ("tileops::probe_leaf_fwd",)


@pytest.fixture
def acme():
    """Return ``serve(op_name, build)``, which has the target ``acme`` serve a probe op."""
    from tileops.backend import _registry as registry

    state = registry.snapshot()
    registry.DETECTORS.clear()
    registry.BUILDERS.clear()
    registry.default_target, registry._loaded = None, True
    registry.register_detector("acme", lambda device: True)

    def serve(op_name, build):
        # A probe op has no manifest entry, which `register_kernel_builder` requires.
        registry.BUILDERS[(op_name, "acme")] = build

    yield serve
    registry.restore(state)


def test_a_target_served_call_leaves_constructor_attributes_alone(acme):
    acme("ProbeTargetDtypeFwdOp", lambda *i, **p: lambda x: x[:, : x.shape[1] // 2].clone())
    op = _probe("ProbeTargetDtypeFwdOp", _SILU, lambda self, x: None)
    op = op(dtype=torch.float32)
    op(torch.zeros(3, 8, dtype=torch.float16))
    assert op.serving_target == "acme"
    assert op.dtype == torch.float32


def test_a_target_served_spec_only_call_is_checked_before_the_target_is_asked(acme):
    asked = []
    acme("ProbeSpecOnlyFwdOp", lambda *i, **p: asked.append(i) or (lambda x: x))
    op = _probe("ProbeSpecOnlyFwdOp", _SILU, lambda self, x: None, status="spec-only")()
    with pytest.raises(ValueError, match="x dtype"):
        op(torch.zeros(3, 8, dtype=torch.float32))
    assert asked == []


@pytest.mark.parametrize("boundary", [False, True])
def test_a_call_on_meta_tensors_completes_and_is_priced(boundary):
    forward = lambda self, x: x[:, : x.shape[1] // 2].clone()  # noqa: E731
    name = f"ProbeMeta{'Boundary' if boundary else ''}FwdOp"
    op = _probe(name, _SILU, forward, boundary=boundary, roofline={"flops": "M * N"})()
    op(torch.empty(3, 8, dtype=torch.float16, device="meta"))
    assert op.eval_roofline() == (12, (3 * 8 + 3 * 4) * 2)


def test_an_op_without_a_boundary_traces_inside_a_compiled_caller():
    op = _probe("ProbeInlineFwdOp", _SILU, lambda self, x: x[:, : x.shape[1] // 2] * 2)()
    torch._dynamo.reset()
    compiled = torch.compile(lambda x: op(x) + 1, fullgraph=True)
    assert compiled(torch.ones(3, 8, dtype=torch.float16)).tolist() == [[3] * 4] * 3


def test_a_traced_first_call_binds_the_target_and_passes_its_execution_arguments(acme, monkeypatch):
    flags = []

    def kernel(x, *, flag=False):
        flags.append(flag)
        return x[:, : x.shape[1] // 2].clone()

    def forward(self, x, flag: bool = False):
        raise AssertionError("the in-tree forward ran for a target-served op")

    acme("ProbeTracedTargetFwdOp", lambda *i, **p: kernel)
    op = _probe("ProbeTracedTargetFwdOp", _SILU, forward)()
    # The branch a call takes while Dynamo traces it, on an op without a compile boundary.
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    op(torch.zeros(3, 8, dtype=torch.float16), flag=True)
    assert op.serving_target == "acme" and flags == [True]


def test_a_meta_call_holds_no_metadata_values():
    from tileops._ops._signature_codegen import SignatureCall
    from tileops.backend import OpNotAvailableError

    ids = torch.empty(2, 2, dtype=torch.int32, device="meta")
    call = SignatureCall({}, {"ids": ((2, 2), "int32")}, (), metadata={"ids": ids})
    with pytest.raises(OpNotAvailableError, match="holds no values"):
        call.metadata_values("ids")
    assert SignatureCall({}, {}, (), metadata={"ids": torch.ones(2)}).metadata_values("ids") == [
        1.0,
        1.0,
    ]


def _staged_parent(name: str, forward, *, boundary=False):
    leaf_cls = _probe(f"{name}LeafFwdOp", _SILU, lambda self, x: x[:, : x.shape[1] // 2] * 1)
    parent_cls = _probe(f"{name}FwdOp", _SILU, forward, boundary=boundary)
    parent_cls.delegate_types = {"leaf": leaf_cls}
    return parent_cls()


def test_a_composite_call_carries_only_the_stage_calls_it_ran():
    # A boundary parent's meta call completes in its fake, which runs no sub-op.
    parent = _staged_parent(
        "ProbeStaged", lambda self, x: self.delegate_for("leaf", None)(x), boundary=True
    )
    parent(torch.ones(3, 8, dtype=torch.float16))
    (leaf_call,) = parent.last_call.stages["leaf"]
    assert leaf_call is parent._held_delegates()[0].last_call
    parent(torch.ones(2, 8, dtype=torch.float16, device="meta"))
    assert parent.last_call.stages == {"leaf": ()}


def test_a_composite_call_carries_every_call_of_a_stage():
    def forward(self, x):
        leaf = self.delegate_for("leaf", None)
        leaf(x[:1].contiguous())
        return leaf(x)

    parent = _staged_parent("ProbeTwice", forward)
    parent(torch.ones(3, 8, dtype=torch.float16))
    assert [c.indices["M"] for c in parent.last_call.stages["leaf"]] == [1, 3]


def test_a_meta_call_of_an_op_returning_nothing_completes_and_is_priced():
    signature = {
        "forall": {"M": "Dim", "N": "Dim", "T": "DType[float16 | float32]"},
        "inputs": {
            "result": {"dtype": "T", "shape": "[M, N]", "mutated": True, "write_only": True},
            "x": {"dtype": "T", "shape": "[M, N]"},
        },
        "outputs": {},
    }
    op = _probe(
        "ProbeWriteOnlyFwdOp",
        signature,
        lambda self, result, x: (result.copy_(x), None)[1],
        boundary=True,
        roofline={"flops": "M * N"},
    )()
    x = torch.empty(3, 8, dtype=torch.float16, device="meta")
    assert op(torch.empty_like(x), x) is None
    assert op.eval_roofline() == (24, 2 * 3 * 8 * 2)


@pytest.mark.parametrize(
    "family, class_name, expected",
    [
        ("norm", "RMSNormFwdOp", "norm_rms_norm_fwd"),
        ("convolution", "Conv2dFwdOp", "convolution_conv2d_fwd"),
        # A class whose name opens with the family names it once.
        ("moe", "MoEPrePermuteFwdOp", "moe_pre_permute_fwd"),
        # One capital inside the abbreviation, which the case split would otherwise take.
        ("moe", "FusedMoEExpertsFwdOp", "moe_fused_moe_experts_fwd"),
        # A digit on each side of the capital.
        ("gemm", "GEMMW4A16FwdOp", "gemm_w4a16_fwd"),
        ("gemm", "GEMMINT8W8A8FwdOp", "gemm_int8_w8a8_fwd"),
        # Two abbreviations, one of them the family's own name.
        ("fft", "FFTC2CFwdOp", "fft_c2c_fwd"),
        # Two abbreviations in a row.
        ("sampling", "TopKTopPMaskFwdOp", "sampling_topk_topp_mask_fwd"),
    ],
)
def test_the_operator_name_keeps_each_abbreviation_one_word(
    family: str, class_name: str, expected: str
) -> None:
    assert operator_name(family, class_name) == expected


def test_fresh_output_accepts_disjoint_dlpack_storage():
    signature = _UNARY
    buffer = torch.zeros(8, dtype=torch.float16)
    x = torch.from_dlpack(buffer[:4])
    separate = torch.from_dlpack(buffer[4:])
    op = _probe("ProbeImportedDisjointFwdOp", signature, lambda self, x: separate)()
    assert op(x) is separate


def test_storage_ownership_matches_enumerated_imported_bytes():
    """Imported storage ownership matches byte-set intersection on all small slices."""
    from tileops._ops._signature_codegen import _extent, _overlaps

    buffer = torch.zeros(4, dtype=torch.uint8)
    views = [
        (torch.from_dlpack(buffer[start:end]), set(range(start, end)))
        for start in range(buffer.numel() + 1)
        for end in range(start, buffer.numel() + 1)
    ]
    for first, first_bytes in views:
        for second, second_bytes in views:
            shared = _overlaps(_extent(first), [_extent(second)])
            assert shared is bool(first_bytes & second_bytes)


def _inplace_probe():
    signature = {
        "forall": {"M": "Dim", "T": "DType[float16]"},
        "params": {"inplace": {"type": "bool"}},
        "inputs": {"x": {"dtype": "T", "shape": "[M]", "mutated": "inplace"}},
        "outputs": {"y": {"dtype": "T", "shape": "[M]", "alias": "x"}},
    }
    return _probe("ProbeEagerInplaceFwdOp", signature, lambda self, x: x.add_(1), boundary=True)(
        inplace=True
    )


def test_an_eager_call_outside_any_mode_versions_its_written_input_as_the_dispatcher_does():
    op = _inplace_probe()
    x = torch.zeros(2, dtype=torch.float16)
    before = x._version
    with _Passthrough():
        op(x)
    dispatched = x._version - before
    boundary = type(op)._call_boundary.__globals__["_boundary"]
    saved = dict(boundary.branches)

    def undispatched(*args):
        raise AssertionError("a plain eager call reached the dispatcher")

    for effect, branch in saved.items():
        boundary.branches[effect] = (undispatched, *branch[1:])
    try:
        before = x._version
        op(x)
    finally:
        boundary.branches.update(saved)
    assert x._version - before == dispatched
    assert x.tolist() == [2, 2]


def test_an_eager_call_missing_a_written_input_raises_its_check():
    """A call that omits an input its branch writes fails on the signature check, as the
    dispatched call does, rather than on versioning the absent input."""
    signature = {
        "forall": {"M": "Dim", "T": "DType[float16]"},
        "inputs": {
            "x": {"dtype": "T", "shape": "[M]"},
            "mean": {
                "dtype": "T",
                "shape": "[M]",
                "type": "Tensor | None",
                "default": None,
                "mutated": "present(mean)",
            },
            "var": {
                "dtype": "T",
                "shape": "[M]",
                "type": "Tensor | None",
                "default": None,
                "mutated": "present(mean)",
            },
        },
        "outputs": {"y": {"dtype": "T", "shape": "[M]"}},
        "shape_rules": ["present(var) == present(mean)"],
    }

    def forward(self, x, mean=None, var=None):
        mean.add_(1)
        var.add_(1)
        return x + 1

    op = _probe("ProbeMissingWrittenFwdOp", signature, forward, boundary=True)()
    x, mean = torch.zeros(2, dtype=torch.float16), torch.zeros(2, dtype=torch.float16)
    with pytest.raises(ValueError, match="refinement fails: present\\(var\\) == present\\(mean\\)"):
        op(x, mean)


def test_a_param_changed_after_construction_is_checked_on_the_next_call():
    signature = {
        "forall": {"M": "Dim", "T": "DType[float16]"},
        "params": {"flag": {"type": "bool"}},
        "inputs": {"x": {"dtype": "T", "shape": "[M]"}},
        "outputs": {"y": {"dtype": "T", "shape": "[M]"}},
        "shape_rules": ["flag"],
    }
    op = _probe("ProbeParamRuleFwdOp", signature, lambda self, x: x.clone(), boundary=True)(
        flag=True
    )
    x = torch.zeros(4, dtype=torch.float16)
    op(x)
    op.flag = False
    with pytest.raises(ValueError, match="flag"):
        op(x)


def test_a_check_is_looked_up_once_per_presence():
    signature = _UNARY
    op = _probe("ProbeLookupOnceFwdOp", signature, lambda self, x: x.clone(), boundary=True)()
    op(torch.zeros(3, dtype=torch.float16))
    plan = type(op)._signature

    def lookup(*args):
        raise AssertionError("the check was looked up again")

    plan._lookup = lookup
    try:
        op(torch.zeros(5, dtype=torch.float16))
    finally:
        del plan._lookup
    assert op.last_call.indices["M"] == 5


def test_an_effect_param_changed_after_construction_takes_its_branch_on_the_next_call():
    signature = {
        "forall": {"M": "Dim", "T": "DType[float16]"},
        "params": {"inplace": {"type": "bool"}},
        "inputs": {"x": {"dtype": "T", "shape": "[M]", "mutated": "inplace"}},
        "outputs": {"y": {"dtype": "T", "shape": "[M]", "alias": "x"}},
    }
    op = _probe(
        "ProbeEffectParamFwdOp",
        signature,
        lambda self, x: x.add_(1) if self.inplace else x + 1,
        boundary=True,
    )(inplace=False)
    x = torch.zeros(2, dtype=torch.float16)
    assert op(x) is not x
    op.inplace = True
    before = x._version
    assert op(x) is x
    assert x._version - before == 1 and x.tolist() == [1, 1]


@pytest.mark.parametrize("emit", [False, True])
def test_a_repeated_call_returns_an_absent_nullable_output_absent(emit):
    signature = {
        "forall": {"M": "Dim", "T": "DType[float16]"},
        "params": {"emit": {"type": "bool"}},
        "inputs": {"x": {"dtype": "T", "shape": "[M]"}},
        "outputs": {"y": {"dtype": "T", "shape": "[M]", "nullable": "emit"}},
    }
    op = _probe(
        "ProbeNullableRepeatFwdOp",
        signature,
        lambda self, x: x.clone() if self.emit else None,
        boundary=True,
    )(emit=emit)
    x = torch.zeros(3, dtype=torch.float16)
    for _ in range(2):
        y = op(x)
        assert (y is not None) == emit


@pytest.mark.skipif(
    not run_device_available() or torch.device(run_device()).type == "cpu",
    reason="needs a device besides the CPU",
)
def test_a_repeat_places_a_call_without_device_bearing_inputs_on_the_current_device(monkeypatch):
    import tileops._ops._signature_codegen as codegen

    signature = {
        "forall": {"M": "Dim", "T": "DType[int32]"},
        "inputs": {"lengths": {"dtype": "T", "shape": "[M]", "device": "cpu"}},
        "outputs": {"y": {"dtype": "T", "shape": "[M]"}},
    }
    op = _probe(
        "ProbeCurrentDeviceRepeatFwdOp",
        signature,
        lambda self, lengths: lengths.to(codegen.current_device(), copy=True),
        boundary=True,
    )()
    lengths = torch.zeros(3, dtype=torch.int32)
    monkeypatch.setattr(codegen, "current_device", lambda: torch.device("cpu"))
    assert op(lengths).device.type == "cpu"
    device = torch.empty(0, device=run_device()).device
    monkeypatch.setattr(codegen, "current_device", lambda: device)
    assert op(lengths).device == device


def test_a_list_param_edited_in_place_is_checked_on_the_next_call():
    signature = {
        "forall": {"M": "Dim", "T": "DType[float16]"},
        "params": {"shape": {"type": "list[int]"}},
        "inputs": {"x": {"dtype": "T", "shape": "[M, *shape]"}},
        "outputs": {"y": {"dtype": "T", "shape": "[M, *shape]"}},
    }
    op = _probe("ProbeListParamFwdOp", signature, lambda self, x: x.clone(), boundary=True)(
        shape=[2, 3]
    )
    x = torch.zeros(4, 2, 3, dtype=torch.float16)
    op(x)
    op.shape[:] = [3, 2]
    with pytest.raises(ValueError):
        op(x)


def test_a_param_snapshot_differs_once_a_field_is_edited_in_place():
    from types import SimpleNamespace

    from tileops._ops._signature_codegen import param_state
    from tileops.manifest._values import ADTValue

    class Layout:
        def __init__(self, max_m):
            self.max_m = max_m

    class SlottedLayout:
        __slots__ = "max_m"

        def __init__(self, max_m):
            self.max_m = max_m

    op = SimpleNamespace(
        shape=[2, 3],
        adt=ADTValue("Layout", {"masked": {"max_m": 4}}),
        obj=Layout(4),
        slotted=SlottedLayout(4),
    )
    sig = SimpleNamespace(params={"shape": {}, "adt": {}, "obj": {}, "slotted": {}})
    before = param_state(op, sig)
    for edit in (
        lambda: op.shape.append(1),
        lambda: op.adt.fields.update(max_m=8),
        lambda: setattr(op.obj, "max_m", 8),
        lambda: setattr(op.slotted, "max_m", 8),
    ):
        edit()
        assert param_state(op, sig) != before
        before = param_state(op, sig)


def test_a_tensor_slot_holding_another_value_reaches_the_operator():
    signature = _UNARY
    op = _probe("ProbeNotTensorFwdOp", signature, lambda self, x: x.clone(), boundary=True)()
    op(torch.zeros(3, dtype=torch.float16))
    with pytest.raises((TypeError, RuntimeError), match="Tensor|tensor"):
        op([0.0, 0.0, 0.0])


def test_an_invalid_execution_argument_reaches_the_operator():
    signature = _UNARY

    def forward(self, x, scale: int = 1):
        return x * scale

    op = _probe("ProbeExecutionArgFwdOp", signature, forward, boundary=True)()
    x = torch.ones(3, dtype=torch.float16)
    assert op(x, scale=2).tolist() == [2, 2, 2]
    with pytest.raises((TypeError, RuntimeError)):
        op(x, scale=1.5)


def test_a_held_tensor_requiring_grad_leaves_the_result_untracked_as_the_dispatcher_does():
    signature = {
        "forall": {"M": "Dim", "T": "DType[float32]"},
        "inputs": {"x": {"dtype": "T", "shape": "[M]"}},
        "outputs": {"y": {"dtype": "T", "shape": "[M]"}},
    }
    op = _probe("ProbeHeldGradFwdOp", signature, lambda self, x: x * self.w, boundary=True)()
    op.w = torch.ones(1, requires_grad=True)
    with _Passthrough():
        dispatched = op(torch.ones(3))
    for m in (4, 4):
        assert op(torch.ones(m)).requires_grad == dispatched.requires_grad


def test_a_negative_view_is_resolved_as_the_dispatcher_does():
    signature = _UNARY
    seen = []

    def forward(self, x):
        seen.append(x.is_neg())
        return x.clone()

    op = _probe("ProbeNegViewFwdOp", signature, forward, boundary=True)()
    x = torch._neg_view(torch.ones(3, dtype=torch.float16))
    with _Passthrough():
        op(x)
    op(x)
    op(x)
    assert seen == [seen[0]] * 3


def test_a_target_served_op_repeats_on_the_target_kernel(acme):
    ran = []

    def kernel(x):
        ran.append(True)
        return x[:, : x.shape[1] // 2].clone()

    def forward(self, x):
        raise AssertionError("the in-tree forward ran for a target-served op")

    acme("ProbeTargetRepeatFwdOp", lambda *i, **p: kernel)
    op = _probe("ProbeTargetRepeatFwdOp", _SILU, forward, boundary=True)()
    x = torch.zeros(3, 8, dtype=torch.float16)
    op(x)
    op(x)
    assert len(ran) == 2 and op.serving_target == "acme"


def test_a_conjugate_view_is_left_to_the_dispatcher():
    signature = _UNARY
    op = _probe("ProbeConjugateFwdOp", signature, lambda self, x: x.clone(), boundary=True)()
    z = torch.ones(2, dtype=torch.complex64)
    assert op._observe((z,), {}, {}, direct=True)[0] is not None
    assert op._observe((z.conj(),), {}, {}, direct=True)[0] is None
    assert op._observe((torch._neg_view(z.real),), {}, {}, direct=True)[0] is None


def test_a_none_for_a_required_input_reaches_the_operator():
    signature = _UNARY
    op = _probe("ProbeRequiredNoneFwdOp", signature, lambda self, x: x.clone(), boundary=True)()
    op(torch.zeros(2, dtype=torch.float16))
    boundary = type(op)._call_boundary.__globals__["_boundary"]
    ((effect, branch),) = boundary.branches.items()
    reached = []

    def operator(*args):
        reached.append(True)
        return branch[0](*args)

    boundary.branches[effect] = (operator, *branch[1:])
    try:
        with pytest.raises((TypeError, RuntimeError, ValueError)):
            op(None)
    finally:
        boundary.branches[effect] = branch
    assert reached


@pytest.mark.parametrize("view", [False, True])
def test_eager_rejects_aliases_between_empty_outputs(view):
    signature = {
        "forall": {"M": "Dim", "T": "DType[float16]"},
        "inputs": {"x": {"dtype": "T", "shape": "[M]"}},
        "outputs": {
            "y": {"dtype": "T", "shape": "[M]"},
            "z": {"dtype": "T", "shape": "[0]"},
            "w": {"dtype": "T", "shape": "[0]"},
        },
    }

    def forward(self, x):
        empty = x.new_empty(0)
        return x.clone(), empty, empty.view(0) if view else empty

    op = _probe("ProbeEmptyOutputAliasFwdOp", signature, forward, boundary=True)()
    with pytest.raises(RuntimeError, match="alias"):
        op(torch.ones(3, dtype=torch.float16))


@pytest.mark.parametrize("mutated", [False, True], ids=["read-only", "written"])
def test_a_determined_input_is_derived_only_where_the_call_does_not_write_it(mutated):
    """A determined input moves nothing unless written, and then only its write; the
    manifest call and the checked call agree on it."""
    entry = copy.deepcopy(load_manifest()["GroupedGEMMFwdOp"])
    entry["signature"]["inputs"]["batch_offsets"]["mutated"] = mutated
    plan = entry_plan("GroupedGEMMFwdOp", entry, load_adts())
    row = entry["workloads"][0]
    call = instantiate(plan, row, row["dtype_cases"][0])
    tensors = call.materialize(device="cpu")
    cls = type("GroupedGEMMFwdOp", (), {"__init__": lambda self, **p: vars(self).update(p)})
    install(cls, entry, load_adts())
    op = cls(**call.arguments())
    op._construction_indices = op._check_construction()
    checked = cls._signature.check(op, {t: tensors[t] for t in plan.sig.inputs})
    traffic = {t: (r, w) for t, r, w in checked.traffic}
    assert traffic["batch_offsets"] == (0, int(mutated))
    assert traffic["batch_sizes"] == (1, 0)
    assert call.derived("batch_offsets") is checked.derived("batch_offsets") is (not mutated)
    assert not call.derived("batch_sizes") and not checked.derived("batch_sizes")


# ---------------------------------------------------------------- tensor sequences


_HEAD = torch.zeros(4, dtype=torch.float16)


@pytest.mark.parametrize(
    ("parts", "message"),
    [
        (torch.zeros(2, 4, dtype=torch.float16), "'parts' is not a list or tuple of tensors"),
        ([torch.zeros(2, 4, dtype=torch.float16), 1], "'parts' is not a list or tuple of tensors"),
        (
            [torch.zeros(2, 4, dtype=torch.float16), torch.zeros(1, 4)],
            "parts dtype differs from T",
        ),
        (
            [torch.zeros(2, 4, dtype=torch.float16), torch.zeros(1, 3, dtype=torch.float16)],
            "parts shape does not match",
        ),
        ([torch.zeros(4, dtype=torch.float16)], "parts has the wrong rank"),
    ],
    ids=["tensor", "element", "dtype", "later-element", "rank"],
)
def test_check_rejects_a_sequence_element_by_element(parts, message):
    op = _op("ConcatRowsFwdOp", {})
    with pytest.raises(ValueError, match=message):
        type(op)._signature.check(op, {"head": _HEAD, "parts": parts})


def test_an_empty_sequence_cannot_solve_what_only_it_names():
    signature = {
        "forall": {"S": "Seq[Shape]", "T": "DType[float16]"},
        "inputs": {"xs": {"dtype": "T", "shape": "[*S]", "type": "list[Tensor]"}},
        "outputs": {"y": {"dtype": "T", "shape": "[len(S)]"}},
    }
    op = _probe("ProbeEmptyFwdOp", signature, lambda self, xs: None)()
    op._construction_indices = op._check_construction()
    with pytest.raises(ValueError, match="xs is empty: T cannot be solved from it"):
        type(op)._signature.check(op, {"xs": []})


_ROWS = _ENTRIES["ConcatRowsFwdOp"]["signature"]


def test_a_sequence_crosses_the_compile_boundary_as_one_tensor_list():
    probe = _probe(
        "ProbeConcatRowsFwdOp",
        _ROWS,
        lambda self, head, parts: torch.cat(parts) + head,
        boundary=True,
        roofline={"flops": "sum(N) * prod(A)"},
    )
    (name,) = probe.compile_op_names
    schema = getattr(torch.ops.tileops, name.split("::")[1]).default._schema
    assert "Tensor head, Tensor[] parts" in str(schema)
    op = probe()
    parts = [torch.ones(2, 4, dtype=torch.float16), torch.full((3, 4), 2, dtype=torch.float16)]
    assert op(_HEAD, parts).tolist() == [[1] * 4] * 2 + [[2] * 4] * 3
    assert op.eval_roofline() == (20, (4 + 20 + 20) * 2)
    # The plan keys on the elements: another list of the same length but other extents
    # resolves its own.
    assert len(op._plans) == 1
    assert tuple(op(_HEAD, parts[:1] + [torch.ones(1, 4, dtype=torch.float16)]).shape) == (3, 4)
    assert len(op._plans) == 2
    with pytest.raises(ValueError, match="parts shape does not match"):
        op(_HEAD, [parts[0], torch.ones(1, 3, dtype=torch.float16)])
    torch._dynamo.reset()
    compiled = torch.compile(op, fullgraph=True)
    assert compiled(_HEAD, parts).tolist() == op(_HEAD, parts).tolist()


def test_a_misaligned_element_is_copied_and_an_aliasing_result_rejected():
    seen = []

    def forward(self, head, parts):
        seen.append([p.data_ptr() % 16 for p in parts])
        return parts[0] if self.alias else torch.cat(parts)

    signature = {**_ROWS, "params": {"alias": {"type": "bool"}}}
    probe = _probe("ProbeAlignedRowsFwdOp", signature, forward, boundary=True)
    base = torch.zeros(9, dtype=torch.float16)
    parts = [torch.zeros(1, 4, dtype=torch.float16), base[1:].view(2, 4)]
    probe(alias=False)(_HEAD, parts)
    assert seen == [[0, 0]]
    with pytest.raises((ValueError, RuntimeError), match="alias|shares storage"):
        probe(alias=True)(_HEAD, [torch.zeros(2, 4, dtype=torch.float16)])


def test_a_sequence_slot_holding_none_keeps_the_later_addresses_in_place():
    """`install` takes a `list[Tensor] | None` slot the manifest does not yet accept; its None
    is one leaf, so the overlap check reads every later tensor at its own address."""
    signature = {
        "forall": {"M": "Dim", "S": "Seq[Shape]", "T": "DType[float32]"},
        "inputs": {
            "xs": {"type": "list[Tensor] | None", "dtype": "T", "shape": "[*S]"},
            "y": {"dtype": "T", "shape": "[M]"},
            "x": {"dtype": "T", "shape": "[M]", "mutated": True},
        },
    }

    def forward(self, xs, y, x):
        x.add_(y)

    op = _probe("ProbeNoneSequenceFwdOp", signature, forward, boundary=True)()
    x, y = torch.zeros(3), torch.ones(3)
    op(None, y, x)
    op(None, y, x)
    assert x.tolist() == [2.0] * 3


# ---------------------------------------------------------------- argument types


_OFFSET = {
    "forall": {"M": "Dim", "T": "DType[float32]"},
    "inputs": {
        "x": {"dtype": "T", "shape": "[M]"},
        "off": {"type": "int | Tensor", "dtype": "T", "shape": "[M]", "default": 0},
    },
    "outputs": {"y": {"dtype": "T", "shape": "[M]"}},
    "shape_rules": ["member(off) != 'int' or off >= 0"],
}


def test_a_scalar_member_is_checked_on_every_call_from_one_plan():
    """A refinement reading a call-time scalar runs on every call; the plan keys on the
    member and the scalar's type, never its value."""
    op = _probe("ProbeOffsetFwdOp", _OFFSET, lambda self, x, off=0: x + off, boundary=True)()
    x = torch.zeros(3)
    checks, check = [], type(op)._signature.check
    type(op)._signature.check = lambda o, t: checks.append(True) or check(o, t)
    try:
        assert op(x, 2).tolist() == [2.0] * 3
        assert op(x, 5).tolist() == [5.0] * 3 and len(checks) == 1
        with pytest.raises(ValueError, match=r"member\(off\) != 'int' or off >= 0"):
            op(x, -1)
        assert op(x, torch.ones(3)).tolist() == [1.0] * 3 and len(checks) == 2
        assert op(x).tolist() == [0.0] * 3
    finally:
        del type(op)._signature.check
    with pytest.raises(ValueError, match=r"bool is not a member of int \| Tensor"):
        op(x, True)


def test_a_scalar_member_crosses_the_compile_boundary_in_its_own_slot():
    probe = _probe("ProbeOffsetSlotFwdOp", _OFFSET, lambda self, x, off=0: x + off, boundary=True)
    (name,) = probe.compile_op_names
    schema = getattr(torch.ops.tileops, name.split("::")[1]).default._schema
    assert "Tensor x, SymInt? off__int, Tensor? off" in str(schema)
    op = probe()
    torch._dynamo.reset()
    compiled = torch.compile(op, fullgraph=True, backend="eager")
    x = torch.zeros(3)
    assert compiled(x, 2).tolist() == [2.0] * 3
    assert compiled(x, torch.ones(3)).tolist() == [1.0] * 3
    torch._dynamo.reset()
    dynamic = torch.compile(lambda x: op(x, x.shape[0] - 1), fullgraph=True, dynamic=True)
    assert dynamic(torch.zeros(3)).tolist() == [2.0] * 3


def test_the_operator_refuses_two_slots_of_one_argument():
    probe = _probe(
        "ProbeOffsetTwoSlotsFwdOp", _OFFSET, lambda self, x, off=0: x + off, boundary=True
    )
    op = probe()
    op(torch.zeros(3), 1)
    (name,) = probe.compile_op_names
    operator = getattr(torch.ops.tileops, name.split("::")[1])
    with pytest.raises(ValueError, match="pass exactly one of the slots of off"):
        operator(torch.zeros(3), 2, torch.ones(3), op._instance_key)


def test_a_required_nullable_input_is_passed_as_none():
    signature = {
        **_VEC,
        "inputs": {
            **_VEC["inputs"],
            "bias": {"type": "Tensor | None", "dtype": "T", "shape": "[M]"},
        },
        "outputs": {"y": {"dtype": "T", "shape": "[M]"}},
    }
    op = _probe(
        "ProbeRequiredNullableFwdOp",
        signature,
        lambda self, x, bias: x.clone() if bias is None else x + bias,
    )()
    x = torch.ones(2, dtype=torch.float16)
    assert op(x, None).tolist() == [1.0, 1.0]
    assert op(x, x).tolist() == [2.0, 2.0]
    with pytest.raises(TypeError, match="bias"):
        op(x)


def test_a_range_requirement_is_checked_only_under_validate_inputs():
    signature = {
        "forall": {"M": "Dim", "N": "Dim", "T": "DType[float32]"},
        "inputs": {
            "table": {"dtype": "T", "shape": "[N]"},
            "index": {"dtype": "int64", "shape": "[M]", "requires": ["in_range(0, N)"]},
        },
        "params": {"validate_inputs": {"type": "bool", "default": False}},
        "outputs": {"y": {"dtype": "T", "shape": "[M]"}},
    }
    probe = _probe(
        "ProbeRangeFwdOp", signature, lambda self, table, index: table[index.clamp(0, 2)]
    )
    table = torch.arange(3.0)
    checked, unchecked = probe(validate_inputs=True), probe(validate_inputs=False)
    assert checked(table, torch.tensor([0, 2])).tolist() == [0.0, 2.0]
    with pytest.raises(ValueError, match=r"index must be in \[0, 3\)"):
        checked(table, torch.tensor([0, 3]))
    assert unchecked(table, torch.tensor([-1, 3])).tolist() == [0.0, 2.0]


def test_a_segment_bound_is_checked_with_its_offsets_only_under_validate_inputs():
    signature = {
        "forall": {"B": "Dim", "Tt": "Dim", "T": "DType[float32]"},
        "inputs": {
            "x": {"dtype": "T", "shape": "[Tt]"},
            "offsets": {
                "dtype": "int64",
                "shape": "[B + 1]",
                "requires": ["prefix_offsets(Tt)", "max_segment(longest)"],
            },
        },
        "params": {
            "longest": {"type": "int", "default": 2},
            "validate_inputs": {"type": "bool", "default": False},
        },
        "outputs": {"y": {"dtype": "T", "shape": "[Tt]"}},
    }
    probe = _probe("ProbeSegmentFwdOp", signature, lambda self, x, offsets: x.clone())
    x = torch.arange(4.0)
    checked = probe(longest=2, validate_inputs=True)
    assert checked(x, torch.tensor([0, 2, 4])).tolist() == [0.0, 1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="offsets must hold no segment longer than 2"):
        checked(x, torch.tensor([0, 1, 4]))
    unchecked = probe(longest=2, validate_inputs=False)
    assert unchecked(x, torch.tensor([0, 1, 4])).tolist() == [0.0, 1.0, 2.0, 3.0]


def test_a_misaligned_tensor_is_copied_and_a_scalar_member_kept():
    op = _probe("ProbeOffsetAlignFwdOp", _OFFSET, lambda self, x, off=0: x + off, boundary=True)()
    misaligned = torch.zeros(5)[1:4]
    assert op(misaligned, 2).tolist() == [2.0] * 3


def test_two_sequences_reading_one_element_wise_dtype_agree_on_it():
    signature = {
        "forall": {"S": "Seq[Shape]", "D": "Seq[DType[float16 | float32]]"},
        "inputs": {
            "xs": {"type": "list[Tensor]", "dtype": "D", "shape": "[*S]"},
            "ys": {"type": "list[Tensor]", "dtype": "D", "shape": "[*S]"},
        },
        "outputs": {"n": {"dtype": "float32", "shape": "[len(S)]"}},
    }
    op = _probe("ProbeTwoSequencesFwdOp", signature, lambda self, xs, ys: None)()
    op._construction_indices = op._check_construction()
    half, single = torch.zeros(2, dtype=torch.float16), torch.zeros(2)
    type(op)._signature.check(op, {"xs": [half, single], "ys": [half, single]})
    with pytest.raises(ValueError, match="ys dtypes differ from D"):
        type(op)._signature.check(op, {"xs": [half, single], "ys": [single, single]})


def test_an_empty_sequence_leaves_a_shared_dtype_to_another_sequence():
    signature = {
        "forall": {"A": "Seq[Shape]", "B": "Seq[Shape]", "T": "DType[float16 | float32]"},
        "inputs": {
            "xs": {"type": "list[Tensor]", "dtype": "T", "shape": "[*A]"},
            "ys": {"type": "list[Tensor]", "dtype": "T", "shape": "[*B]"},
        },
        "outputs": {"n": {"dtype": "T", "shape": "[len(A) + len(B)]"}},
    }
    op = _probe("ProbeSharedDtypeFwdOp", signature, lambda self, xs, ys: None)()
    op._construction_indices = op._check_construction()
    call = type(op)._signature.check(op, {"xs": [], "ys": [torch.zeros(2, dtype=torch.float16)]})
    assert call.indices["T"] == "float16"
    with pytest.raises(ValueError, match="xs and ys are empty: T cannot be solved from it"):
        type(op)._signature.check(op, {"xs": [], "ys": []})


def test_an_empty_sequence_leaves_a_shared_dimension_to_another_sequence():
    signature = {
        "forall": {"N": "Dim", "A": "Seq[Shape]", "B": "Seq[Shape]", "T": "DType[float16]"},
        "inputs": {
            "xs": {"type": "list[Tensor]", "dtype": "T", "shape": "[N, *A]"},
            "ys": {"type": "list[Tensor]", "dtype": "T", "shape": "[N, *B]"},
        },
        "outputs": {"n": {"dtype": "T", "shape": "[N]"}},
    }
    op = _probe("ProbeSharedDimFwdOp", signature, lambda self, xs, ys: None)()
    op._construction_indices = op._check_construction()
    y = torch.zeros(3, 4, dtype=torch.float16)
    call = type(op)._signature.check(op, {"xs": [], "ys": [y]})
    assert call.indices["N"] == 3
    # The sequence that solved the index is no index of the call.
    assert "ys" not in call.indices
    assert not any(isinstance(v, torch.Tensor) for v in call.indices.values())
    with pytest.raises(ValueError, match="xs and ys are empty"):
        type(op)._signature.check(op, {"xs": [], "ys": []})


@pytest.mark.parametrize(
    ("let", "rule"), [({}, "len(D) == len(S)"), ({"count": "len(D)"}, "count == len(S)")]
)
def test_shape_inference_without_dtypes_defers_a_rule_on_element_dtypes(let, rule):
    signature = {
        "forall": {"S": "Seq[Shape]", "D": "Seq[DType[float16 | float32]]"},
        "inputs": {"xs": {"type": "list[Tensor]", "dtype": "D", "shape": "[*S]"}},
        "outputs": {"n": {"dtype": "float32", "shape": "[len(S)]"}},
        "let": let,
        "shape_rules": [rule],
    }
    op = _probe("ProbeShapesOnlyFwdOp", signature, lambda self, xs: None)()
    assert op._infer_output_shapes(xs=[(2, 3), (4,)])["n"] == (2,)


def test_shape_inference_solves_through_a_dtype_coefficient_when_dtypes_are_passed():
    signature = {
        "forall": {"N": "Dim", "T": "DType[float16 | float32]"},
        "inputs": {"x": {"dtype": "T", "shape": "[a * N]"}},
        "let": {"a": "2 if T == 'float16' else 4"},
        "outputs": {"y": {"dtype": "T", "shape": "[N]"}},
    }
    op = _probe("ProbeCoefficientFwdOp", signature, lambda self, x: x[: x.shape[0] // 2])()
    assert op._infer_output_shapes((8,), dtypes={"x": torch.float16}) == {"y": (4,)}
