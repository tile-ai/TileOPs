"""The eager call protocol (docs/design/ops-design.md § Call protocol).

Each test drives a cold call, a repeated one, a changed dependency and a return to an earlier
specialization where its guarantee has one. Probe ops run on the CPU; the tests that need a
compiled program or a CUDA stream carry ``cuda_only``.
"""

import dataclasses
import gc
import weakref
from abc import abstractmethod

import pytest
import torch

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.ops._signature_codegen import install, install_compile_boundary
from tileops.ops.op_base import Op

pytestmark = pytest.mark.smoke


@dataclasses.dataclass(frozen=True)
class _Call(CallSpec):
    rows: int = 0
    dtype: torch.dtype = None


class _Doubling(KernelInterface):
    """Double *x*."""

    request = _Call

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a new tensor."""


class _Doubler(Kernel, _Doubling):
    devices = frozenset({"cpu"})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2


def _op_class(name: str, signature: dict, forward, *, boundary: bool = True, **attributes):
    """An `Op` subclass whose entry is *signature*, whose `forward` is *forward*, holding the
    CPU `_Doubler` under the interface ``double``."""
    entry = {"family": "probe", "status": "implemented", "signature": signature}
    entry["roofline"] = {"flops": "1"}
    if not boundary:
        entry["composition"] = {"kind": "composite", "stages": []}

    def construct(self, **params):
        vars(self).update(params)
        Op.__init__(self)

    body = {
        "__init__": construct,
        "forward": forward,
        "kernel_types": {"double": _Doubler},
        "interfaces": {"double": _Doubling},
        **attributes,
    }
    cls = type(name, (Op,), body)
    install(cls, entry)
    install_compile_boundary(cls, entry)
    return cls


def _counted_checks(cls) -> list:
    """A list that grows by one each time *cls*'s generated call checks run."""
    ran = []
    plan = cls._signature
    check = plan.check

    def counting(op, tensors):
        ran.append(True)
        return check(op, tensors)

    plan.check = counting
    return ran


# ---------------------------------------------------------------- specialization

_SPECIALIZED = {
    "forall": {"M": "Dim", "N": "Dim", "T": "DType[float16 | float32]"},
    "params": {
        "scale": {"type": "float"},
        "dims": {"type": "list[int]"},
        "table": {"dtype": "float32", "shape": "[4]", "device": "cpu", "optional": True},
    },
    "inputs": {
        "x": {"dtype": "T", "shape": "[M, N]", "contiguous": True},
        "bias": {"dtype": "T", "shape": "[N]", "optional": True},
    },
    "outputs": {"y": {"dtype": "T", "shape": "[M, N]"}},
}


def _x(rows=3, dtype=torch.float32):
    return torch.ones(rows, 4, dtype=dtype)


# Each dependency of a specialization: how a call changes it after warming, given the op and
# the warming call's arguments, returning the changed call's arguments.
_CHANGES = {
    "shape": lambda op, x, bias, mode: (_x(5), bias, mode),
    "strides": lambda op, x, bias, mode: (torch.ones(3, 8)[:, :4], bias, mode),
    "dtype": lambda op, x, bias, mode: (_x(dtype=torch.float16), None, mode),
    "presence": lambda op, x, bias, mode: (x, torch.ones(4), mode),
    "param": lambda op, x, bias, mode: (setattr(op, "scale", 3.0), (x, bias, mode))[1],
    "nested param": lambda op, x, bias, mode: (op.dims.append(7), (x, bias, mode))[1],
    # Equal to the warmed value, of a type the checks refuse.
    "param type": lambda op, x, bias, mode: (setattr(op, "scale", True), (x, bias, mode))[1],
    "nested param type": lambda op, x, bias, mode: (op.dims.__setitem__(0, 1.0), (x, bias, mode))[
        1
    ],
    # Present either way: only its dtype, which the checks place it in, changes.
    "construction tensor": lambda op, x, bias, mode: (
        setattr(op, "table", torch.zeros(4, dtype=torch.float64)),
        (x, bias, mode),
    )[1],
    "execution argument": lambda op, x, bias, mode: (x, bias, mode + 1),
}

# The changes the checks refuse: what they raise, and whether the call checks run, or the
# construction checks refuse first.
_REFUSED = {
    # Same shape, dtype and device; the strides alone break the declared contiguity.
    "strides": ("x must be contiguous", True),
    "param type": ("is not a float", False),
    "nested param type": (r"is not a list\[int\]", False),
}


@pytest.mark.parametrize("dependency", sorted(_CHANGES))
def test_a_changed_dependency_resolves_its_own_plan_and_a_return_reuses_the_first(dependency):
    """The checks and each call spec run once per specialization: a dependency the key omits
    would reuse the first plan's verdict and call spec for the changed call."""
    built = []

    def forward(self, x, bias=None, mode: int = 0):
        def call():
            built.append((tuple(x.shape), x.dtype))
            return _Call(device=x.device, rows=x.shape[0], dtype=x.dtype)

        return self.kernel_for("double", call)(x.contiguous()) * self.scale + mode

    name = "ProbePlan" + "".join(word.title() for word in dependency.split()) + "FwdOp"
    cls = _op_class(name, _SPECIALIZED, forward)
    checks = _counted_checks(cls)
    table = torch.zeros(4)
    op = cls(scale=1.0, dims=[1], table=table)
    x, bias, mode = _x(), None, 0
    restore = {"scale": 1.0, "dims": [1], "table": table}

    op(x, bias, mode)
    op(x, bias, mode)
    assert (len(checks), len(built)) == (1, 1)

    changed = _CHANGES[dependency](op, x, bias, mode)
    refused, call_checked = _REFUSED.get(dependency, (None, True))
    if refused:
        with pytest.raises(ValueError, match=refused):
            op(*changed)
    else:
        y = op(*changed)
        assert y.shape == changed[0].shape and y.dtype == changed[0].dtype
        assert op.last_call.indices["M"] == changed[0].shape[0]
        assert op.table.dtype == torch.float32
    assert len(checks) == 1 + call_checked

    vars(op).update({k: (list(v) if isinstance(v, list) else v) for k, v in restore.items()})
    op(x, bias, mode)
    assert len(checks) == 1 + call_checked, "a return to the first specialization resolves it again"


def test_a_tensor_free_call_keys_its_plan_on_the_current_device(monkeypatch):
    """A call no input places runs on the current device, so that device is part of its
    specialization."""
    import tileops.ops._signature_codegen as codegen

    signature = {
        "forall": {"M": "Dim", "T": "DType[int32]"},
        "inputs": {"lengths": {"dtype": "T", "shape": "[M]", "device": "cpu"}},
        "outputs": {"y": {"dtype": "T", "shape": "[M]"}},
    }
    cls = _op_class(
        "ProbePlanCurrentDeviceFwdOp",
        signature,
        lambda self, lengths: lengths.to(codegen.current_device(), copy=True),
    )
    checks = _counted_checks(cls)
    op = cls()
    lengths = torch.zeros(3, dtype=torch.int32)
    monkeypatch.setattr(codegen, "current_device", lambda: torch.device("cpu"))
    op(lengths)
    op(lengths)
    meta = torch.device("meta")
    monkeypatch.setattr(codegen, "current_device", lambda: meta)
    assert op(lengths).device == meta
    assert len(checks) == 2


# ---------------------------------------------------------------- dispatcher state


def _modes():
    from torch.overrides import TorchFunctionMode
    from torch.utils._python_dispatch import TorchDispatchMode

    class Dispatching(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    class Functioning(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    def forward_ad(op, x):
        import torch.autograd.forward_ad as fwad

        with fwad.dual_level():
            return op(fwad.make_dual(x, torch.ones_like(x)))

    def grad(op, x):
        return op(x.clone().requires_grad_())

    def vmap(op, x):
        return torch.func.vmap(op)(x.unsqueeze(0))

    def under(context):
        def run(op, x):
            with context():
                return op(x)

        return run

    class Subclass(torch.Tensor):
        pass

    def fake(op, x):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode() as mode:
            return op(mode.from_tensor(x))

    def compiled(op, x):
        def backend(graph, inputs):
            _GRAPHED.extend(str(node.target) for node in graph.graph.nodes)
            return graph

        torch._dynamo.reset()
        return torch.compile(op, fullgraph=True, backend=backend)(x)

    return {
        "compile": compiled,
        "fake": fake,
        "subclass": lambda op, x: op(x.as_subclass(Subclass)),
        "dispatch mode": under(Dispatching),
        "function mode": under(Functioning),
        "autocast": under(lambda: torch.autocast("cpu", dtype=torch.bfloat16)),
        "profiler": under(torch.autograd.profiler.profile),
        "requires grad": grad,
        "forward AD": forward_ad,
        "vmap": vmap,
        "meta": lambda op, x: op(x.to("meta")),
        "negative view": lambda op, x: op(torch._neg_view(x)),
    }


# The targets of the graph the compile mode last traced.
_GRAPHED = []
_MODES = _modes()


@pytest.mark.parametrize("mode", sorted(_MODES))
def test_a_state_entered_after_warming_takes_the_registered_operator(mode):
    """A warm plan does not decide the route: a call the dispatcher would do more for reaches
    the operator, and returns what the same state returns for the plain function."""
    signature = {
        "forall": {"M": "Dim", "T": "DType[float32]"},
        "inputs": {"x": {"dtype": "T", "shape": "[M]"}},
        "outputs": {"y": {"dtype": "T", "shape": "[M]"}},
    }
    name = "ProbeRoute" + "".join(word.title() for word in mode.split()) + "FwdOp"
    cls = _op_class(name, signature, lambda self, x: x * 2)
    run = _MODES[mode]

    def outcome(op):
        result = run(op, torch.arange(3.0))
        if result.device.type == "meta" or type(result).__name__ == "FakeTensor":
            return type(result), result.shape, result.dtype
        return type(result), torch.Tensor.tolist(result), result.requires_grad

    expected = outcome(lambda x: x * 2)
    assert outcome(cls()) == expected

    warm = cls()
    warm(torch.arange(3.0))
    boundary = type(warm)._call_boundary.__globals__["_boundary"]
    ((effect, branch),) = boundary.branches.items()
    reached = []

    def operator(*args):
        reached.append(True)
        return branch[0](*args)

    boundary.branches[effect] = (operator, *branch[1:])
    _GRAPHED.clear()
    try:
        assert outcome(warm) == expected
    finally:
        boundary.branches[effect] = branch
    if mode == "compile":
        # A compiled graph calls the registered operator itself, not through the boundary.
        assert any(target.startswith("tileops.") for target in _GRAPHED), _GRAPHED
    else:
        assert reached


# ---------------------------------------------------------------- results

_RESULTS = {
    "count": (
        {"y": {"dtype": "T", "shape": "[M]"}, "z": {"dtype": "T", "shape": "[M]"}},
        lambda x: (x * 2,),
        "returns 1 outputs",
    ),
    "type": ({"y": {"dtype": "T", "shape": "[M]"}}, lambda x: [x * 2], "is not a tensor"),
    "presence": (
        {"y": {"dtype": "T", "shape": "[M]", "nullable": "emit"}},
        lambda x: None,
        "is not a tensor",
    ),
    "shape": ({"y": {"dtype": "T", "shape": "[M]"}}, lambda x: (x * 2)[:1], "has shape"),
    "dtype": ({"y": {"dtype": "T", "shape": "[M]"}}, lambda x: x.double(), "has dtype"),
    "device": ({"y": {"dtype": "T", "shape": "[M]"}}, lambda x: x.to("meta"), "call device"),
    "layout": (
        {"y": {"dtype": "T", "shape": "[M, M]", "contiguous": True}},
        lambda x: torch.ones(x.shape[0], x.shape[0]).t()[:, :],
        "must be contiguous",
    ),
    "storage": ({"y": {"dtype": "T", "shape": "[M]"}}, lambda x: x[:], "shares storage"),
}


@pytest.mark.parametrize("fault", sorted(_RESULTS))
def test_a_warm_call_returning_a_faulty_result_fails_and_keeps_the_last_record(fault):
    """Every call checks what its implementation returned: a fault on the call after a
    correct one is rejected before the call completes."""
    outputs, faulty, message = _RESULTS[fault]
    signature = {
        "forall": {"M": "Dim", "T": "DType[float32]"},
        "params": {"emit": {"type": "bool"}},
        "inputs": {"x": {"dtype": "T", "shape": "[M]"}},
        "outputs": outputs,
    }

    def correct(x):
        if len(outputs) == 2:
            return x * 2, x * 3
        shape = outputs["y"]["shape"]
        return torch.ones(x.shape[0], x.shape[0]) if shape == "[M, M]" else x * 2

    returns = [correct, faulty]
    name = "ProbeResult" + fault.title() + "FwdOp"
    op = _op_class(name, signature, lambda self, x: returns.pop(0)(x))(emit=True)
    x = torch.arange(3.0)
    op(x)
    kept = op.last_call
    with pytest.raises(ValueError, match=message):
        op(x)
    assert op.last_call is kept


@pytest.mark.parametrize("returned", ["the out passed", "the written input"])
def test_a_warm_call_returning_another_tensor_for_an_identity_slot_fails(returned):
    """An `out` slot and an alias of a written input are held to the tensor the call passed."""
    if returned == "the out passed":
        signature = {
            "forall": {"M": "Dim", "T": "DType[float32]"},
            "inputs": {"x": {"dtype": "T", "shape": "[M]"}},
            "outputs": {"y": {"dtype": "T", "shape": "[M]", "buffer": "out"}},
        }
        replies = [lambda x, out: out.copy_(x), lambda x, out: out.clone()]
        cls = _op_class(
            "ProbeIdentityOutFwdOp", signature, lambda self, x, out=None: replies.pop(0)(x, out)
        )
        call = lambda op: op(torch.arange(3.0), torch.empty(3))  # noqa: E731
        message = "must be the `out`"
    else:
        signature = {
            "forall": {"M": "Dim", "T": "DType[float32]"},
            "inputs": {"x": {"dtype": "T", "shape": "[M]", "mutated": True}},
            "outputs": {"y": {"dtype": "T", "shape": "[M]", "alias": "x"}},
        }
        replies = [lambda x: x.add_(1), lambda x: x + 1]
        cls = _op_class("ProbeIdentityInputFwdOp", signature, lambda self, x: replies.pop(0)(x))
        call = lambda op: op(torch.arange(3.0))  # noqa: E731
        message = "must be the input x"
    op = cls()
    call(op)
    with pytest.raises(ValueError, match=message):
        call(op)


# ---------------------------------------------------------------- aliasing


def _owner_probe(name: str, outputs: dict, reply, held: "str | None"):
    signature = {
        "forall": {"M": "Dim", "T": "DType[float32]"},
        "inputs": {"x": {"dtype": "T", "shape": "[M]"}, "e": {"dtype": "T", "shape": "[0]"}},
        "outputs": outputs,
    }
    if held is not None:
        signature["params"] = {"held": {"dtype": "float32", "shape": held}}
    replies = [lambda self, x, e: tuple(x * 2 for _ in outputs) if len(outputs) > 1 else x * 2]
    replies.append(reply)
    return _op_class(name, signature, lambda self, x, e: replies.pop(0)(self, x, e))


_ONE = {"y": {"dtype": "T", "shape": "[M]"}}
_TWO = {"y": {"dtype": "T", "shape": "[M]"}, "z": {"dtype": "T", "shape": "[M]"}}

# What a warm call returns that shares storage, and the error naming it.
_ALIASES = {
    "an input": (_ONE, lambda self, x, e: x, ValueError),
    # The bytes after the input's, in the storage the input views.
    "a disjoint view of an input's storage": (
        _ONE,
        lambda self, x, e: x.as_strided(x.shape, (1,), x.storage_offset() + x.shape[0]),
        ValueError,
    ),
    "an empty input's storage": (
        _ONE,
        lambda self, x, e: e.as_strided(x.shape, (1,)),
        RuntimeError,
    ),
    "a held tensor": (_ONE, lambda self, x, e: self.held, ValueError),
    "an empty held tensor's storage": (
        _ONE,
        lambda self, x, e: self.held.as_strided(x.shape, (1,)),
        RuntimeError,
    ),
    "another output": (_TWO, lambda self, x, e: (x * 2,) * 2, ValueError),
}


@pytest.mark.parametrize("shared", sorted(_ALIASES))
def test_a_warm_fresh_output_owns_its_storage(shared):
    """Storage is compared on the call that returned it: a fresh output may share no byte with
    a tensor passed, a tensor held or another output, after a call that returned its own."""
    outputs, reply, error = _ALIASES[shared]
    name = "ProbeOwner" + "".join(w.title() for w in shared.replace("'", "").split()) + "FwdOp"
    held = {
        "a held tensor": ("[M]", torch.ones(3)),
        "an empty held tensor's storage": ("[0]", torch.ones(3)[:0]),
    }.get(shared)
    params = {} if held is None else {"held": held[1]}
    op = _owner_probe(name, outputs, reply, None if held is None else held[0])(**params)
    storage = torch.zeros(6)
    x, e = storage[:3], torch.zeros(3)[:0]
    op(x, e)
    with pytest.raises(error, match="storage|alias"):
        op(x, e)


def test_a_warm_output_of_an_overlapping_imported_storage_is_rejected():
    """Two storage objects owning overlapping bytes are one storage to a fresh output."""
    base = torch.zeros(8)
    signature = {
        "forall": {"M": "Dim", "T": "DType[float32]"},
        "inputs": {"x": {"dtype": "T", "shape": "[M]"}},
        "outputs": {"y": {"dtype": "T", "shape": "[M]"}},
    }
    replies = [lambda x: x * 2, lambda x: torch.from_dlpack(base[2:6])]
    op = _op_class("ProbeImportedFwdOp", signature, lambda self, x: replies.pop(0)(x))()
    x = torch.from_dlpack(base[:4])
    op(x)
    with pytest.raises(ValueError, match="shares storage"):
        op(x)


# ---------------------------------------------------------------- allocation and effects


def test_a_fresh_output_copied_into_carries_no_version_on_either_route():
    """The body runs below the in-place bookkeeping: a fresh tensor it fills with ``copy_`` is
    returned unversioned, as the dispatcher returns it."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class Passthrough(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    signature = {
        "forall": {"M": "Dim", "T": "DType[float32]"},
        "inputs": {"x": {"dtype": "T", "shape": "[M]"}},
        "outputs": {"y": {"dtype": "T", "shape": "[M]"}},
    }
    op = _op_class("ProbeCopiedFwdOp", signature, lambda self, x: torch.empty_like(x).copy_(x))()
    x = torch.arange(3.0)
    with Passthrough():
        dispatched = op(x)._version
    op(x)
    assert op(x)._version == dispatched == 0


def test_a_warm_call_bumps_the_version_of_the_out_it_writes_once_also_when_it_raises():
    """A direct call bumps each written argument once, as the dispatcher does; a body that
    raises after writing has still written."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class Passthrough(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    signature = {
        "forall": {"M": "Dim", "T": "DType[float32]"},
        "inputs": {"x": {"dtype": "T", "shape": "[M]"}},
        "outputs": {"y": {"dtype": "T", "shape": "[M]", "buffer": "out"}},
    }

    def forward(self, x, out=None):
        out.copy_(x)
        if self.fail:
            raise RuntimeError("failed after writing")
        return out

    op = _op_class("ProbeBumpedOutFwdOp", signature, forward)(fail=False)
    x, out = torch.arange(3.0), torch.empty(3)
    with Passthrough():
        before = out._version
        op(x, out)
        dispatched = out._version - before
    op(x, out)
    before = out._version
    op(x, out)
    assert out._version - before == dispatched == 1
    op.fail = True
    before = out._version
    with pytest.raises(RuntimeError, match="after writing"):
        op(x, out)
    assert out._version - before == 1


def test_an_empty_call_returns_the_out_passed_and_allocates_nothing(monkeypatch):
    """A call writing no element runs no implementation; what it returns is what it was
    handed, so the executor builds nothing for it."""
    import tileops.ops.op_base as op_base

    signature = {
        "forall": {"M": "Dim", "T": "DType[float32]"},
        "inputs": {"x": {"dtype": "T", "shape": "[M]"}},
        "outputs": {"y": {"dtype": "T", "shape": "[M]", "buffer": "out"}},
    }

    def forward(self, x, out=None):
        raise AssertionError("an empty call ran its implementation")

    op = _op_class("ProbeEmptyOutFwdOp", signature, forward)()
    x, out = torch.zeros(0), torch.empty(0)
    op(x, out)

    def refuse(*args, **kwargs):
        raise AssertionError("the executor allocated for a call it was handed a buffer for")

    monkeypatch.setattr(op_base.torch, "empty", refuse)
    assert op(x, out) is out


# ---------------------------------------------------------------- records


def _metadata_probe(name: str):
    signature = {
        "forall": {"M": "Dim", "T": "DType[float32]"},
        "inputs": {
            "x": {"dtype": "T", "shape": "[M]"},
            "ids": {"dtype": "int64", "shape": "[M]", "values": "sample_indices(M, M)"},
        },
        "outputs": {"y": {"dtype": "T", "shape": "[M]"}},
    }
    return _op_class(name, signature, lambda self, x, ids: x * 2)


def test_a_record_holds_its_own_call_whenever_it_is_read():
    """A record is built from what its call held, read at once, read late or kept: a later
    call replaces it and changes nothing in it."""
    op = _metadata_probe("ProbeRecordFwdOp")()
    ids = torch.arange(3)
    op(torch.ones(3), ids)
    first = op.last_call
    assert op.last_call is first

    later = torch.zeros(4, dtype=torch.int64)
    op(torch.ones(4), later)
    op(torch.ones(4), torch.arange(4))
    assert first.indices["M"] == 3 and first.metadata["ids"] is ids
    second = op.last_call
    assert second is not first and second.indices["M"] == 4


def test_an_equal_layout_metadata_tensor_is_bound_by_its_own_call():
    """A plan holds no metadata tensor: a call passing another one of the same layout records
    that one, whose contents a formula reads when it reads them."""
    op = _metadata_probe("ProbeRebindFwdOp")()
    op(torch.ones(3), torch.arange(3))
    ids = torch.zeros(3, dtype=torch.int64)
    op(torch.ones(3), ids)
    assert op.last_call.metadata["ids"] is ids
    ids.fill_(2)
    assert op.last_call.metadata_values("ids") == [2, 2, 2]


def test_a_composite_record_holds_the_stage_calls_of_its_own_call():
    """A parent's record keeps the records its sub-ops completed inside it, so a sub-op's later
    call reaches neither the parent's stages nor their count or order."""
    leaf_signature = {
        "forall": {"M": "Dim", "T": "DType[float32]"},
        "inputs": {"x": {"dtype": "T", "shape": "[M]"}},
        "outputs": {"y": {"dtype": "T", "shape": "[M]"}},
    }
    leaf_cls = _op_class("ProbeStageLeafFwdOp", leaf_signature, lambda self, x: x * 2)

    def forward(self, x):
        leaf = self.delegate_for("leaf", None)
        for rows in range(1, self.calls + 1):
            leaf(x[:rows].contiguous())
        return x * 2

    parent = _op_class(
        "ProbeStageParentFwdOp",
        leaf_signature,
        forward,
        boundary=False,
        delegate_types={"leaf": leaf_cls},
    )(calls=1)
    x = torch.ones(4)
    parent(x)
    parent.calls = 3
    parent(x)
    leaf = parent.held_delegates()[0]
    leaf(torch.ones(2))
    assert [c.indices["M"] for c in parent.last_call.stages["leaf"]] == [1, 2, 3]
    parent.calls = 0
    parent(x)
    assert parent.last_call.stages == {"leaf": ()}


def test_a_composite_refuses_params_its_settled_sub_ops_were_not_built_from():
    """A sub-op held under no identity is built from the params of that time, so a call
    after they change fails rather than run it."""
    leaf_signature = {
        "forall": {"M": "Dim", "T": "DType[float32]"},
        "inputs": {"x": {"dtype": "T", "shape": "[M]"}},
        "outputs": {"y": {"dtype": "T", "shape": "[M]"}},
    }
    leaf_cls = _op_class("ProbeSettledLeafFwdOp", leaf_signature, lambda self, x: x * 2)
    parent = _op_class(
        "ProbeSettledParentFwdOp",
        {**leaf_signature, "params": {"scale": {"type": "float"}}},
        lambda self, x: self.delegate_for("leaf", None)(x) * self.scale,
        boundary=False,
        delegate_types={"leaf": leaf_cls},
    )(scale=2.0)
    x = torch.ones(2)
    parent(x)
    parent.scale = 3.0
    with pytest.raises(ValueError, match="built its sub-ops"):
        parent(x)
    parent.scale = 2.0
    assert parent(x).tolist() == [4.0, 4.0]


# ---------------------------------------------------------------- lifetime


def test_a_plan_holds_no_tensor_its_call_passed_or_returned():
    """What a plan holds outlives the call; a tensor of the call it held would outlive it too."""
    op = _metadata_probe("ProbeLifetimeFwdOp")()
    x, ids = torch.ones(3), torch.arange(3)
    y = op(x, ids)
    op(torch.ones(3), torch.arange(3))
    refs = [weakref.ref(t) for t in (x, ids, y)]
    del x, ids, y
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_a_construction_tensor_replaced_after_placement_is_placed_again():
    """The checks place a construction-time tensor in its declared dtype; a plan is held
    only under what a call observes after placement, so a replaced tensor is placed too."""
    cls = _op_class("ProbePlacedTableFwdOp", _SPECIALIZED, lambda self, x, bias=None: x * 2)
    op = cls(scale=1.0, dims=[1], table=torch.zeros(4, dtype=torch.float64))
    op(_x())
    assert op.table.dtype == torch.float32
    op.table = torch.zeros(4, dtype=torch.float64)
    op(_x())
    assert op.table.dtype == torch.float32


def test_a_record_read_and_changed_leaves_later_records_alone():
    """A record's call is its own: changing what one read returns reaches no later call."""
    op = _metadata_probe("ProbeOwnRecordFwdOp")()
    op(torch.ones(3), torch.arange(3))
    op.last_call.indices["M"] = 99
    op(torch.ones(3), torch.arange(3))
    assert op.last_call.indices["M"] == 3


def test_a_dispatched_call_returning_an_alias_keeps_the_last_record():
    """The alias check runs before the call completes on the dispatcher's route too."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class Passthrough(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    alias = lambda self, x, e: e.as_strided(x.shape, (1,))  # noqa: E731
    op = _owner_probe("ProbeDispatchedAliasFwdOp", _ONE, alias, held=None)()
    x, e = torch.zeros(3), torch.zeros(3)[:0]
    op(x, e)
    kept = op.last_call
    with Passthrough(), pytest.raises(RuntimeError, match="alias"):
        op(x, e)
    assert op.last_call is kept


def test_a_param_changed_after_construction_is_solved_again():
    """What construction solved from a param is solved again for the new specialization,
    and for a meta call after a return to the first one."""
    signature = {
        "forall": {"M": "Dim", "T": "DType[float32]"},
        "params": {"n": {"type": "int"}},
        "inputs": {"x": {"dtype": "T", "shape": "[M]"}},
        "outputs": {"y": {"dtype": "T", "shape": "[n]"}},
    }
    cls = _op_class("ProbeSolvedParamFwdOp", signature, lambda self, x: x.new_zeros(self.n))
    op = cls(n=2)
    op(torch.ones(3))
    op.n = 4
    assert op(torch.ones(3)).shape == (4,)
    assert op.last_call.indices["n"] == 4
    op.n = 2
    op(torch.ones(3))
    assert op(torch.ones(3, device="meta")).shape == (2,)
    assert op.last_call.indices["n"] == 2
    op.n = 5
    assert op._infer_output_shapes((3,))["y"] == (5,)


def test_a_record_holds_a_copy_of_the_params_it_read():
    """A list param edited after a call reaches neither that call's record nor, through an
    edited record, a later one."""
    op = _op_class("ProbeListRecordFwdOp", _SPECIALIZED, lambda self, x, bias=None: x * 2)(
        scale=1.0, dims=[1], table=None
    )
    op(_x())
    op.dims.append(2)
    assert op.last_call.indices["dims"] == [1]
    op.dims.pop()
    op.last_call.indices["dims"].append(5)
    op(_x())
    assert op.last_call.indices["dims"] == [1]


def test_a_target_kernel_is_built_again_for_a_changed_param():
    """A target builds its kernel from the op's params, so a changed param, or a construction
    tensor replaced by one of the same layout, builds another."""
    from tileops.backend import registry
    from tileops.ops._params_codegen import PARAM_NAMES_ATTRIBUTE

    state = registry.snapshot()
    try:
        registry.DETECTORS.clear()
        registry.BUILDERS.clear()
        registry.default_target = None
        registry._loaded = True
        registry.register_detector("acme", lambda device: True)
        built = []

        def build(x, *, scale, table):
            built.append(scale)
            return lambda x: x * scale * table

        registry.register_kernel_builder("ProbeTargetParamFwdOp", "acme", build)
        signature = {
            "forall": {"M": "Dim", "T": "DType[float32]"},
            "params": {
                "scale": {"type": "float"},
                "table": {"dtype": "float32", "shape": "[2]", "device": "cpu"},
            },
            "inputs": {"x": {"dtype": "T", "shape": "[M]"}},
            "outputs": {"y": {"dtype": "T", "shape": "[M]"}},
        }
        cls = _op_class("ProbeTargetParamFwdOp", signature, lambda self, x: x)
        setattr(cls, PARAM_NAMES_ATTRIBUTE, ("scale", "table"))
        op = cls(scale=2.0, table=torch.ones(2))
        x = torch.ones(2)
        op(x)
        op.scale = 3.0
        assert op(x).tolist() == [3.0, 3.0]
        op.table = torch.full((2,), 2.0)
        assert op(x).tolist() == [6.0, 6.0]
        assert built == [2.0, 3.0, 3.0]
    finally:
        registry.restore(state)
