"""Methods generated from a parametric signature (docs/design/manifest.md § Call Semantics)."""

import functools
from pathlib import Path

import pytest
import torch
import yaml

from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate
from tileops.ops._signature_codegen import install

pytestmark = pytest.mark.smoke

_CASES = yaml.safe_load((Path(__file__).parent / "manifest_cases.yaml").read_text())
_ENTRIES = _CASES["entries"]


def _op(name, params):
    cls = type(name, (), {})
    install(cls, _ENTRIES[name], _CASES["adts"])
    op = cls()
    vars(op).update(params)
    op._check_construction()
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
        op = _op(name, call.arguments(tensors))
        checked = type(op)._signature.check(op, tensors)
        for t, spec in call.specs.items():
            assert checked.tensors.get(t) == (None if spec is None else (spec.shape, spec.dtype))


def _gemm(**roofline):
    cls = type("GemmFwdOp", (), {"__init__": lambda self, **p: vars(self).update(p)})
    install(cls, {**_ENTRIES["GemmFwdOp"], "roofline": roofline}, _CASES["adts"])
    op = cls(trans_a=False, trans_b=True)
    op._check_construction()
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


@pytest.mark.parametrize("name", ["ClampFwdOp", "GQAPrefillVarlenFwdOp", "SumFwdOp"])
def test_check_traces_on_symints(name):
    call = next(_calls(name))
    tensors = call.materialize(device="cpu")
    op = _op(name, call.arguments(tensors))
    plan = type(op)._signature
    torch._dynamo.reset()
    compiled = torch.compile(lambda ts: plan.check(op, ts).tensors, fullgraph=True, dynamic=True)
    assert compiled(tensors) == plan.check(op, tensors).tensors


def test_eval_roofline_prices_the_last_call(monkeypatch):
    derived = _gemm(flops="2 * M * N * K")
    derived._signature_call = type(derived)._signature.check(derived, {"a": _A, "b": _B})
    assert derived.eval_roofline() == (2 * 4 * 16 * 8, (4 * 8 + 16 * 8 + 4 * 16) * 2)
    written = _gemm(flops="M", bytes="bytes(d) if present(a) else 0")
    written._signature_call = derived._signature_call
    assert written.eval_roofline() == (4, 4 * 16 * 2)
    import tileops.perf.formulas

    monkeypatch.setattr(
        tileops.perf.formulas,
        "probe_gemm",
        lambda call: (call.ix["M"], call.bytes("a")),
        raising=False,
    )
    func = _gemm(func="tileops.perf.formulas.probe_gemm")
    func._signature_call = derived._signature_call
    assert func.eval_roofline() == (4, 4 * 8 * 2)
    with pytest.raises(RuntimeError, match="needs a completed call"):
        _gemm(flops="1").eval_roofline()


def test_construction_checks_what_construction_decides():
    _op("DSADecodeFwdOp", {"dim_tail": -1})  # a signed offset carries no obligation of its own
    with pytest.raises(ValueError, match="axis 1 of hidden_states, hidden_size, is negative"):
        _op(
            "FusedMoeSharedExpertFwdOp",
            {"num_tokens": 2, "hidden_size": -1, "shared_ffn_size": None},
        )


def _boundary_forward(eager):
    """A `forward` with *eager*'s parameters whose body is one call to the boundary."""

    @functools.wraps(eager)
    def forward(self, *args, **kwargs):
        return self._call_boundary(*args, **kwargs)

    return forward


def _probe(name, signature, forward, *, boundary=False, roofline=None):
    """An `Op` subclass whose converted entry is *signature* and whose `forward` is *forward*."""
    from tileops.ops.op_base import Op

    entry = {
        "family": "probe",
        "status": "implemented",
        "signature": signature,
        "roofline": roofline or {"flops": "1"},
    }

    def construct(self, **params):
        vars(self).update(params)
        self.dispatch_kernel(None)

    body = {
        "__init__": construct,
        "default_kernel_map": property(lambda self: {}),
        "forward": _boundary_forward(forward) if boundary else forward,
        "_eager_forward": forward,
    }
    if boundary:
        body["compile_boundary"] = True
    cls = type(name, (Op,), body)
    install(cls, entry)
    return cls


_SILU = _ENTRIES["SiluAndMulFwdOp"]["signature"]


def test_op_call_runs_the_check_and_keeps_its_binding():
    probe = _probe(
        "ProbeSiluAndMulFwdOp",
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


def test_a_spec_only_entry_gets_its_boundary_and_keeps_its_methods():
    """A spec-only class keeps its hand-written checks and shape inference; the signature
    still decides its operator and fake, so a traced call is one graph."""
    from tileops.ops.op_base import Op

    signature = {**_VEC, "outputs": {"y": {"dtype": "T", "shape": "[M]"}}}

    def construct(self):
        self.dispatch_kernel(None)

    def eager(self, x):
        return x + 1

    cls = type(
        "ProbeSpecOnlyFwdOp",
        (Op,),
        {
            "__init__": construct,
            "default_kernel_map": property(lambda self: {}),
            "forward": _boundary_forward(eager),
            "_eager_forward": eager,
            "_infer_output_shapes": lambda self, x_shape: {"y": x_shape},
            "_validate_dtypes": lambda self, x: None,
            "eval_roofline": lambda self: (0, 0),
            "compile_boundary": True,
        },
    )
    assert install(cls, {"family": "probe", "status": "spec-only", "signature": signature})
    assert cls.compile_op_names == ("tileops::probe_spec_only_fwd",)
    assert "_signature" not in vars(cls)
    torch._dynamo.reset()
    compiled = torch.compile(cls(), fullgraph=True)
    assert compiled(torch.zeros(4, dtype=torch.float16)).tolist() == [1] * 4


def test_a_traced_boundary_call_is_one_graph():
    signature = {**_VEC, "outputs": {"y": {"dtype": "T", "shape": "[M]"}}}
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
    from tileops.manifest.plan import roofline_plan
    from tileops.manifest.signature import parse_signature

    sig = parse_signature("SiluAndMulFwdOp", {"signature": _SILU}, {})
    assert roofline_plan(sig, {"flops": "M * N", "bytes": "bytes(x) + bytes(output)"})[0] == []
    errors = roofline_plan(sig, {"flops": "NOPE", "extra": 1})[0]
    assert any("unknown key 'extra'" in e for e in errors)
    assert any("'NOPE' is not in ix" in e for e in errors)
    assert roofline_plan(None, {"flops": "isinstance(M)", "extra": 1})[0] == [
        "roofline: unknown key 'extra'",
        "roofline.flops: isinstance is not a built-in primitive",
    ]


_VEC = {
    "forall": {"M": "Dim", "T": "DType[float16]"},
    "inputs": {"x": {"dtype": "T", "shape": "[M]"}},
}


def test_execution_parameters_cross_the_compile_boundary():
    def eager(self, x, scale: int = 1):
        return x * scale

    signature = {**_VEC, "outputs": {"y": {"dtype": "T", "shape": "[M]"}}}
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
    from tileops.ops.moe.contracts import MaskedLayoutSpec

    cls = type("MoePrePermuteFwdOp", (), {})
    install(cls, _ENTRIES["MoePrePermuteFwdOp"], _CASES["adts"])
    op = cls()
    vars(op).update(layout=MaskedLayoutSpec(max_m=4), num_local_experts=2)
    op._check_construction()
    op.layout = type("Impostor", (), {"kind": "masked", "max_m": 4})()
    with pytest.raises(ValueError, match="is not an object of a MGroupedLayout constructor"):
        op._check_construction()


def test_axes_nothing_reads_emit_no_checks():
    signature = {
        **_VEC,
        "params": {f"unused{i}": {"type": "bool"} for i in range(12)},
        "outputs": {"y": {"dtype": "T", "shape": "[M]"}},
    }
    cls = type("ProbeUnusedFwdOp", (), {})
    install(cls, {"signature": signature}, {})
    assert len(cls._signature.checks) == 1


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


def test_a_cpu_construction_tensor_takes_its_declared_dtype():
    signature = {
        **_VEC,
        "params": {"table": {"dtype": "float16", "shape": "[4]", "device": "cpu"}},
        "outputs": {"y": {"dtype": "T", "shape": "[M]"}},
    }
    op = _probe("ProbeTableFwdOp", signature, lambda self, x: x.clone())(table=torch.zeros(4))
    op(torch.zeros(2, dtype=torch.float16))
    assert op.table.dtype == torch.float16 and op.table.device.type == "cpu"


def test_a_roofline_func_must_yield_two_ints(monkeypatch):
    import tileops.perf.formulas

    monkeypatch.setattr(tileops.perf.formulas, "probe_float", lambda call: (1.5, 2), raising=False)
    signature = {**_VEC, "outputs": {"y": {"dtype": "T", "shape": "[M]"}}}
    op = _probe(
        "ProbeFloatRoofFwdOp",
        signature,
        lambda self, x: x.clone(),
        roofline={"func": "tileops.perf.formulas.probe_float"},
    )()
    op(torch.zeros(2, dtype=torch.float16))
    with pytest.raises(TypeError, match="not \\(flops: int, bytes: int\\)"):
        op.eval_roofline()


def test_a_fresh_output_owns_its_storage():
    signature = {**_VEC, "outputs": {"y": {"dtype": "T", "shape": "[M]"}}}
    op = _probe("ProbeReturnsInputFwdOp", signature, lambda self, x: x)()
    with pytest.raises(ValueError, match="y shares storage"):
        op(torch.zeros(2, dtype=torch.float16))
    held = {**signature, "params": {"table": {"dtype": "T", "shape": "[M]"}}}
    table = torch.zeros(2, dtype=torch.float16)
    op = _probe("ProbeHeldFwdOp", held, lambda self, x: self.table)(table=table)
    with pytest.raises(ValueError, match="y shares storage"):
        op(torch.zeros(2, dtype=torch.float16))


def test_generated_methods_bind_inputs_by_name():
    signature = {
        **_VEC,
        "inputs": {**_VEC["inputs"], "w": {"dtype": "T", "shape": "[M]", "optional": True}},
        "outputs": {"y": {"dtype": "T", "shape": "[M]"}},
    }
    op = _probe("ProbeNamedFwdOp", signature, lambda self, x, w=None: x + 1)()
    x = torch.zeros(2, dtype=torch.float16)
    op._validate_dtypes(x, w=x)
    with pytest.raises(ValueError, match="'x' is not a tensor"):
        op._validate_dtypes(3)
    assert op._infer_output_shapes(x=(2,)) == {"y": (2,)}
    with pytest.raises(TypeError):
        op._infer_output_shapes((2,), (9,), (1,))


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
    from tileops.manifest.plan import effect_errors
    from tileops.manifest.signature import parse_signature

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
        ValueError, match="ProbeBroadcastFwdOp: tensor 'z' shape: .*not broadcastable"
    ):
        op(x, y)


def test_a_construction_tensor_spliced_by_a_call_index_is_checked_per_call():
    signature = {
        "forall": {"S": "Shape", "T": "DType[float16]"},
        "inputs": {"x": {"dtype": "T", "shape": "[*S]"}},
        "params": {"table": {"dtype": "T", "shape": "[*broadcast(S, S)]"}},
        "outputs": {"y": {"dtype": "T", "shape": "[*S]"}},
    }
    probe = _probe("ProbeSplicedFwdOp", signature, lambda self, x: x.clone())
    op = probe(table=torch.zeros(2, 3, dtype=torch.float16))
    op(torch.zeros(2, 3, dtype=torch.float16))
    with pytest.raises(ValueError, match="table has the wrong rank|table shape does not match"):
        op(torch.zeros(2, 3, 1, dtype=torch.float16))


def test_an_obligation_a_call_time_presence_activates_waits_for_the_call():
    signature = {
        **_VEC,
        "params": {"p": {"type": "int", "default": -1}},
        "inputs": {**_VEC["inputs"], "w": {"dtype": "T", "shape": "[p]", "optional": True}},
        "outputs": {"y": {"dtype": "T", "shape": "[M]"}},
        "shape_rules": ["not present(w) or p >= 0"],
    }
    op = _probe("ProbeDeferredFwdOp", signature, lambda self, x, w=None: x + 1)(p=-1)
    op(torch.zeros(2, dtype=torch.float16))
    with pytest.raises(ValueError, match="w shape does not match p"):
        op(torch.zeros(2, dtype=torch.float16), torch.zeros(0, dtype=torch.float16))


def test_a_construction_time_tensor_is_given_exactly_where_its_presence_holds():
    signature = {
        **_VEC,
        "params": {
            "flag": {"type": "bool"},
            "table": {"dtype": "T", "shape": "[M]", "optional": "flag"},
        },
        "outputs": {"y": {"dtype": "T", "shape": "[M]"}},
    }
    probe = _probe("ProbeGatedFwdOp", signature, lambda self, x: x + 1)
    with pytest.raises(ValueError, match="'table' is given where its presence condition is false"):
        probe(flag=False, table=torch.zeros(2, dtype=torch.float16))
    with pytest.raises(ValueError, match="construction-time tensor 'table' is required"):
        probe(flag=True, table=None)


def test_the_checked_call_holds_generated_construction_time_tensors():
    signature = {
        **_VEC,
        "forall": {**_VEC["forall"], "L": "Seq[Int]", "N": "Dim"},
        "params": {"table": {"dtype": "int32", "shape": "[N]", "values": "as_tensor(L)"}},
        "outputs": {"y": {"dtype": "T", "shape": "[M]"}},
    }
    table = torch.tensor([3, 1], dtype=torch.int32)
    op = _probe("ProbeTableValuesFwdOp", signature, lambda self, x: x + 1)(table=table)
    op(torch.zeros(2, dtype=torch.float16))
    assert op.last_call.values("table") == [3, 1]


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


def test_a_compile_boundary_needs_a_call_time_tensor_input():
    signature = {
        "forall": {"T": "DType[float16]"},
        "params": {"n": {"type": "int"}},
        "outputs": {"y": {"dtype": "T", "shape": "[n]"}},
    }
    with pytest.raises(TypeError, match="compile_boundary needs a call-time tensor input"):
        _probe("ProbeSourceFwdOp", signature, lambda self: None, boundary=True)


def test_a_target_served_call_leaves_constructor_attributes_alone():
    from tileops.backend import registry

    state = registry.snapshot()
    try:
        registry.DETECTORS.clear()
        registry.BUILDERS.clear()
        registry.default_target = None
        registry._loaded = True
        registry.register_detector("acme", lambda device: True)
        registry.register_kernel_builder(
            "ProbeTargetDtypeFwdOp",
            "acme",
            lambda *i, **p: lambda x: x[:, : x.shape[1] // 2].clone(),
        )
        op = _probe("ProbeTargetDtypeFwdOp", _SILU, lambda self, x: None)
        op = op(dtype=torch.float32)
        op(torch.zeros(3, 8, dtype=torch.float16))
        assert op.settled_target == "acme"
        assert op.dtype == torch.float32
    finally:
        registry.restore(state)


@pytest.mark.parametrize("boundary", [False, True])
def test_a_call_on_meta_tensors_completes_and_is_priced(boundary):
    forward = lambda self, x: x[:, : x.shape[1] // 2].clone()  # noqa: E731
    name = f"ProbeMeta{'Boundary' if boundary else ''}FwdOp"
    op = _probe(name, _SILU, forward, boundary=boundary, roofline={"flops": "M * N"})()
    op(torch.empty(3, 8, dtype=torch.float16, device="meta"))
    assert op.eval_roofline() == (12, (3 * 8 + 3 * 4) * 2)


def test_a_converted_op_without_a_boundary_traces_inside_a_compiled_caller():
    op = _probe("ProbeInlineFwdOp", _SILU, lambda self, x: x[:, : x.shape[1] // 2] * 2)()
    torch._dynamo.reset()
    compiled = torch.compile(lambda x: op(x) + 1, fullgraph=True)
    assert compiled(torch.ones(3, 8, dtype=torch.float16)).tolist() == [[3] * 4] * 3


def test_a_meta_call_holds_no_metadata_values():
    from tileops.backend import OpNotAvailableError
    from tileops.ops._signature_codegen import SignatureCall

    ids = torch.empty(2, 2, dtype=torch.int32, device="meta")
    call = SignatureCall({}, {"ids": ((2, 2), "int32")}, (), metadata={"ids": ids})
    with pytest.raises(OpNotAvailableError, match="holds no values"):
        call.values("ids")
    assert SignatureCall({}, {}, (), metadata={"ids": torch.ones(2)}).values("ids") == [1.0, 1.0]


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
    assert leaf_call is parent.kernel_delegates()[0].last_call
    parent(torch.ones(2, 8, dtype=torch.float16, device="meta"))
    assert parent.last_call.stages == {"leaf": ()}


def test_a_composite_call_carries_every_call_of_a_stage():
    def forward(self, x):
        leaf = self.delegate_for("leaf", None)
        leaf(x[:1].contiguous())
        return leaf(x)

    parent = _staged_parent("ProbeTwice", forward)
    parent(torch.ones(3, 8, dtype=torch.float16))
    assert [c.ix["M"] for c in parent.last_call.stages["leaf"]] == [1, 3]


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
