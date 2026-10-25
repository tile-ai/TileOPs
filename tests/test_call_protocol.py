"""The eager call protocol (docs/design/ops-design.md § Call protocol).

Probe ops run on the CPU. A test that walks several cases runs each under ``_case``, so a
failure names its case.
"""

import contextlib
import gc
import itertools
import types
import weakref

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tileops._ops._signature_codegen import install, install_compile_boundary
from tileops._ops.op_base import Op

pytestmark = pytest.mark.smoke

_T = {"dtype": "T", "shape": "[M]"}
_ONE = {"y": _T}
_TWO = {"y": _T, "z": _T}
_OUT = {"y": {**_T, "buffer": "out"}}
_IDS = itertools.count()


class _Passthrough(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


def _probe(forward, *, name=None, boundary=True, attributes=None, **signature):
    """An `Op` class running *forward*, whose signature is a unary float32 one updated by
    *signature*; with *boundary* False, a composite."""
    signature = {"forall": {"M": "Dim", "T": "DType[float32]"}, "inputs": {"x": _T}, **signature}
    signature.setdefault("outputs", _ONE)
    entry = {"family": "probe", "status": "implemented", "signature": signature}
    entry["roofline"] = {"flops": "1"}
    if not boundary:
        entry["composition"] = {"kind": "composite", "stages": []}

    def construct(self, **params):
        vars(self).update(params)
        Op.__init__(self)

    body = {"__init__": construct, "forward": forward, **(attributes or {})}
    cls = type(name or f"Probe{next(_IDS)}FwdOp", (Op,), body)
    install(cls, entry)
    install_compile_boundary(cls, entry)
    return cls


def _counted_checks(cls) -> list:
    """A list that grows by one each time *cls*'s generated call checks run."""
    ran, check = [], cls._signature.check
    cls._signature.check = lambda op, tensors: ran.append(True) or check(op, tensors)
    return ran


@contextlib.contextmanager
def _case(name: str):
    """Name the case *name* in the failure of whatever the block raises."""
    try:
        yield
    except (Exception, pytest.fail.Exception) as exc:
        raise AssertionError(f"case {name!r} failed: {exc}") from exc


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


def _scaled(self, x, bias=None, mode: int = 0):
    return x * self.scale + mode


def _x(rows=3, dtype=torch.float32):
    return torch.ones(rows, 4, dtype=dtype)


def _after(effect, args):
    """*args*, once *effect*, the change a call makes before it, has run."""
    return args


# Each dependency of a specialization: how a call changes it after warming, given the op and
# the warming call's arguments (x, bias, mode), returning the changed call's arguments.
_CHANGES = {
    "shape": lambda op, a: (_x(5), *a[1:]),
    "strides": lambda op, a: (torch.ones(3, 8)[:, :4], *a[1:]),
    "dtype": lambda op, a: (_x(dtype=torch.float16), None, a[2]),
    "presence": lambda op, a: (a[0], torch.ones(4), a[2]),
    "param": lambda op, a: _after(setattr(op, "scale", 3.0), a),
    "nested param": lambda op, a: _after(op.dims.append(7), a),
    # Equal to the warmed value, of a type the checks refuse.
    "param type": lambda op, a: _after(setattr(op, "scale", True), a),
    "nested param type": lambda op, a: _after(op.dims.__setitem__(0, 1.0), a),
    # Present either way: only its dtype, which the checks place it in, changes.
    "construction tensor": lambda op, a: _after(setattr(op, "table", _x(1)[0].double()), a),
    "execution argument": lambda op, a: (*a[:2], a[2] + 1),
}

# The changes the checks refuse: what they raise, and whether the call checks run, or the
# construction checks refuse first.
_REFUSED = {
    "strides": ("x must be contiguous", True),
    "param type": ("is not a float", False),
    "nested param type": (r"is not a list\[int\]", False),
}


def test_a_changed_dependency_resolves_its_own_plan_and_a_return_reuses_the_first(monkeypatch):
    """The checks run once per specialization, keyed on every dependency of the call: each
    tensor's layout and presence, each param, each construction tensor after placement, each
    execution argument and, for a call no tensor places, the current device."""
    for dependency, change in _CHANGES.items():
        with _case(dependency):
            cls = _probe(_scaled, **_SPECIALIZED)
            checks = _counted_checks(cls)
            table, args = torch.zeros(4), (_x(), None, 0)
            op = cls(scale=1.0, dims=[1], table=table)
            op(*args)
            op(*args)
            changed = change(op, args)
            refused, call_checked = _REFUSED.get(dependency, (None, True))
            if refused:
                with pytest.raises(ValueError, match=refused):
                    op(*changed)
            else:
                y = op(*changed)
                assert (y.shape, y.dtype) == (changed[0].shape, changed[0].dtype)
                assert op.last_call.indices["M"] == changed[0].shape[0]
            assert len(checks) == 1 + call_checked
            vars(op).update(scale=1.0, dims=[1], table=table)
            op(*args)
            assert len(checks) == 1 + call_checked, "a return resolved the first plan again"

    with _case("construction tensor placed again"):
        op = _probe(_scaled, **_SPECIALIZED)(scale=1.0, dims=[1], table=_x(1)[0].double())
        op(_x())
        op.table = _x(1)[0].double()
        op(_x())
        assert op.table.dtype == torch.float32

    with _case("current device"):
        import tileops._ops._signature_codegen as codegen

        def forward(self, lengths):
            return lengths.to(codegen.current_device(), copy=True)

        lengths = {"dtype": "T", "shape": "[M]", "device": "cpu"}
        cls = _probe(forward, forall={"M": "Dim", "T": "DType[int32]"}, inputs={"lengths": lengths})
        checks = _counted_checks(cls)
        op, lengths = cls(), torch.zeros(3, dtype=torch.int32)
        monkeypatch.setattr(codegen, "current_device", lambda: torch.device("cpu"))
        op(lengths)
        op(lengths)
        monkeypatch.setattr(codegen, "current_device", lambda: torch.device("meta"))
        assert op(lengths).device.type == "meta" and len(checks) == 2


def test_a_specialization_evicted_by_newer_ones_is_resolved_again():
    """An op holds a bounded number of plans: past it the oldest is dropped, and its next
    call is checked again rather than served stale."""
    cls = _probe(_scaled, **_SPECIALIZED)
    checks = _counted_checks(cls)
    op = cls(scale=2.0, dims=[1], table=None)
    for rows in range(1, 300):
        op(_x(rows))
    resolved = len(checks)
    assert op(_x(299)).shape == (299, 4) and len(checks) == resolved
    assert op(_x(1)).tolist() == [[2.0] * 4] and len(checks) == resolved + 1


def test_a_param_changed_after_construction_is_solved_again():
    """What construction solved from a param is solved again for the new specialization,
    and for a meta call after a return to the first one."""
    op = _probe(
        lambda self, x: x.new_zeros(self.n),
        params={"n": {"type": "int"}},
        outputs={"y": {"dtype": "T", "shape": "[n]"}},
    )(n=2)
    op(torch.ones(3))
    op.n = 4
    assert op(torch.ones(3)).shape == (4,) and op.last_call.indices["n"] == 4
    op.n = 2
    op(torch.ones(3))
    assert op(torch.ones(3, device="meta")).shape == (2,) and op.last_call.indices["n"] == 2
    op.n = 5
    assert op._infer_output_shapes((3,))["y"] == (5,)


def test_a_target_kernel_is_built_again_for_a_changed_param():
    """A target builds its kernel from the op's params, so a changed param, or a construction
    tensor replaced by one of the same layout, builds another."""
    from tileops._ops._params_codegen import PARAM_NAMES_ATTRIBUTE
    from tileops.backend import _registry as registry

    state, built = registry.snapshot(), []

    def build(x, *, scale, table):
        built.append(scale)
        return lambda x: x * scale * table

    try:
        registry.DETECTORS.clear()
        registry.BUILDERS.clear()
        registry.default_target, registry._loaded = None, True
        registry.register_detector("acme", lambda device: True)
        registry.register_kernel_builder("ProbeTargetParamFwdOp", "acme", build)
        table = {"dtype": "float32", "shape": "[2]", "device": "cpu"}
        params = {"scale": {"type": "float"}, "table": table}
        cls = _probe(lambda self, x: x, name="ProbeTargetParamFwdOp", params=params)
        setattr(cls, PARAM_NAMES_ATTRIBUTE, ("scale", "table"))
        op, x = cls(scale=2.0, table=torch.ones(2)), torch.ones(2)
        op(x)
        op.scale = 3.0
        assert op(x).tolist() == [3.0, 3.0]
        op.table = torch.full((2,), 2.0)
        assert op(x).tolist() == [6.0, 6.0] and built == [2.0, 3.0, 3.0]
    finally:
        registry.restore(state)


class _Functioning(torch.overrides.TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


class _Subclass(torch.Tensor):
    pass


def _within(context):
    """A run that calls the op inside *context*."""

    def run(op, x):
        with context():
            return op(x)

    return run


def _fake(op, x):
    with torch._subclasses.fake_tensor.FakeTensorMode() as mode:
        return op(mode.from_tensor(x))


def _dual(op, x):
    import torch.autograd.forward_ad as fwad

    with fwad.dual_level():
        return op(fwad.make_dual(x, torch.ones_like(x)))


# Each state the dispatcher does more for than run the body: how a call enters it.
_STATES = {
    "fake": _fake,
    "subclass": lambda op, x: op(x.as_subclass(_Subclass)),
    "dispatch mode": _within(_Passthrough),
    "function mode": _within(_Functioning),
    "autocast": _within(lambda: torch.autocast("cpu", dtype=torch.bfloat16)),
    "profiler": _within(torch.autograd.profiler.profile),
    "requires grad": lambda op, x: op(x.clone().requires_grad_()),
    "forward AD": _dual,
    "vmap": lambda op, x: torch.func.vmap(op)(x.unsqueeze(0)),
    "meta": lambda op, x: op(x.to("meta")),
    "negative view": lambda op, x: op(torch._neg_view(x)),
}


def _outcome(run, op):
    """What *run* returns for *op* on a fixed input, comparable across ops."""
    result = run(op, torch.arange(3.0))
    if result.device.type == "meta" or type(result).__name__ == "FakeTensor":
        return type(result), result.shape, result.dtype
    return type(result), torch.Tensor.tolist(result), result.requires_grad


def test_a_state_entered_after_warming_takes_the_registered_operator():
    """A warm plan does not decide the route: a call the dispatcher would do more for reaches
    the operator, and returns what the same state returns for the plain function."""
    for state, run in _STATES.items():
        with _case(state):
            cls = _probe(lambda self, x: x * 2)
            expected = _outcome(run, lambda x: x * 2)
            assert _outcome(run, cls()) == expected
            warm = cls()
            warm(torch.arange(3.0))
            boundary = type(warm)._call_boundary.__globals__["_boundary"]
            ((effect, branch),) = boundary.branches.items()
            reached = []
            operator = lambda *args, b=branch[0], r=reached: r.append(1) or b(*args)  # noqa: E731
            boundary.branches[effect] = (operator, *branch[1:])
            try:
                assert _outcome(run, warm) == expected and reached
            finally:
                boundary.branches[effect] = branch


def test_a_compiled_call_after_warming_traces_the_registered_operator():
    """A compiled graph of a warm op calls its registered operator and returns what the plain
    function returns."""
    graphed = []

    def backend(gm, inputs):
        graphed.extend(str(node.target) for node in gm.graph.nodes)
        return gm

    def compiled(op, x):
        torch._dynamo.reset()
        return torch.compile(op, fullgraph=True, backend=backend)(x)

    cls = _probe(lambda self, x: x * 2)
    expected = _outcome(compiled, lambda x: x * 2)
    assert _outcome(compiled, cls()) == expected
    warm = cls()
    warm(torch.arange(3.0))
    graphed.clear()
    assert _outcome(compiled, warm) == expected
    assert any(target.startswith("tileops.") for target in graphed), graphed


def _held(shape: str) -> dict:
    """The params of a probe holding a float32 tensor of *shape*."""
    return {"emit": {"type": "bool"}, "held": {"dtype": "float32", "shape": shape}}


def _replying(replies: list):
    """A ``forward(x, e, out=None)`` answering each call with the next of *replies*, given the
    call as ``c`` with ``c.op``, ``c.x``, ``c.e`` and ``c.out``."""

    def forward(self, x, e, out=None):
        return replies.pop(0)(types.SimpleNamespace(op=self, x=x, e=e, out=out))

    return forward


def _spread(t: torch.Tensor) -> torch.Tensor:
    """Three elements read from where *t*'s storage starts, an empty *t*'s included."""
    return t.as_strided((3,), (1,))


_DOUBLED = lambda c: c.x * 2  # noqa: E731
_TWICE = lambda c: (c.x * 2, c.x * 3)  # noqa: E731
_E = {"dtype": "T", "shape": "[0]"}
_NULLABLE = {"outputs": {"y": {**_T, "nullable": "emit"}}}
_CONTIGUOUS = {"outputs": {"y": {**_T, "contiguous": True}}}
_WRITTEN = {
    "inputs": {"x": {**_T, "mutated": True}, "e": _E},
    "outputs": {"y": {**_T, "alias": "x"}},
}

# Each fault a warm call may return: the probe's signature overrides, a correct reply, the
# faulty one, and the error naming it. x views the first half of a storage; e is empty.
_FAULTS = {
    "count": ({"outputs": _TWO}, _TWICE, lambda c: (c.x,), "returns 1 outputs"),
    "type": ({}, _DOUBLED, lambda c: [c.x * 2], "is not a tensor"),
    "presence": (_NULLABLE, _DOUBLED, lambda c: None, "is not a tensor"),
    "shape": ({}, _DOUBLED, lambda c: (c.x * 2)[:1], "has shape"),
    "dtype": ({}, _DOUBLED, lambda c: c.x.double(), "has dtype"),
    "device": ({}, _DOUBLED, lambda c: c.x.to("meta"), "call device"),
    "layout": (_CONTIGUOUS, _DOUBLED, lambda c: torch.ones(6)[::2], "must be contiguous"),
    "another out": ({"outputs": _OUT}, lambda c: c.out.copy_(c.x), lambda c: c.x * 2, "`out`"),
    "another written input": (_WRITTEN, lambda c: c.x.add_(1), lambda c: c.x + 1, "input x"),
    "an input": ({}, _DOUBLED, lambda c: c.x, "storage"),
    "a disjoint view of an input's storage": ({}, _DOUBLED, lambda c: c.x._base[3:], "storage"),
    # Another storage object owning bytes of the input's storage.
    "an imported input": ({}, _DOUBLED, lambda c: torch.from_dlpack(c.x._base[1:4]), "storage"),
    "an empty input": ({}, _DOUBLED, lambda c: _spread(c.e), "alias"),
    "a held tensor": ({"params": _held("[M]")}, _DOUBLED, lambda c: c.op.held, "storage"),
    "an empty held tensor": (
        {"params": _held("[0]")},
        _DOUBLED,
        lambda c: _spread(c.op.held),
        "alias",
    ),
    "another output": ({"outputs": _TWO}, _TWICE, lambda c: (c.x * 2,) * 2, "storage"),
}


def test_a_warm_call_returning_a_faulty_result_fails_and_keeps_the_last_record():
    """Every call checks what its implementation returned, on the direct route and the
    dispatcher's: a fault after a correct call, a fresh output sharing a byte with a tensor
    passed, held or returned included, is rejected before the call completes."""
    held = {"[M]": torch.ones(3), "[0]": torch.ones(3)[:0]}
    for fault, (overrides, correct, faulty, message) in _FAULTS.items():
        with _case(fault):
            signature = {"inputs": {"x": _T, "e": _E}, "params": {"emit": {"type": "bool"}}}
            signature.update(overrides)
            shape = signature["params"].get("held", {}).get("shape")
            cls = _probe(_replying([correct, faulty, faulty]), **signature)
            op = cls(emit=True, **({"held": held[shape]} if shape else {}))
            args = (torch.zeros(6)[:3], torch.zeros(3)[:0])
            args += (torch.empty(3),) if overrides.get("outputs") is _OUT else ()
            op(*args)
            kept = op.last_call
            with pytest.raises((ValueError, RuntimeError), match=message):
                op(*args)
            with _Passthrough(), pytest.raises((ValueError, RuntimeError), match=message):
                op(*args)
            assert op.last_call is kept


def _bumped(op, x, out, raises=None) -> int:
    """How many versions a call of *op* writing *out* adds to it, raising *raises* if given."""
    before = out._version
    with pytest.raises(RuntimeError, match=raises) if raises else contextlib.nullcontext():
        op(x, out)
    return out._version - before


def test_a_direct_call_versions_what_it_writes_as_the_dispatcher_does():
    """A direct call runs below the in-place bookkeeping: a fresh output filled with ``copy_``
    is returned unversioned, and a written `out` is bumped once, also when the body raises
    after writing."""
    copied = _probe(lambda self, x: torch.empty_like(x).copy_(x))()
    x, out = torch.arange(3.0), torch.empty(3)
    with _Passthrough():
        dispatched = copied(x)._version
    copied(x)
    assert copied(x)._version == dispatched == 0

    def forward(self, x, out=None):
        out.copy_(x)
        if self.fail:
            raise RuntimeError("failed after writing")
        return out

    op = _probe(forward, outputs=_OUT)(fail=False)
    with _Passthrough():
        dispatched = _bumped(op, x, out)
    op(x, out)
    assert _bumped(op, x, out) == dispatched == 1
    op.fail = True
    assert _bumped(op, x, out, raises="after writing") == 1


def test_an_empty_call_returns_the_out_passed_and_allocates_nothing(monkeypatch):
    """A call writing no element runs no implementation and returns the `out` it was
    handed, so the executor builds nothing for it."""
    import tileops._ops.op_base as op_base

    def refuse(*args, **kwargs):
        raise AssertionError("an empty call ran its implementation or allocated")

    op = _probe(lambda self, x, out=None: refuse(), outputs=_OUT)()
    x, out = torch.zeros(0), torch.empty(0)
    op(x, out)
    monkeypatch.setattr(op_base.torch, "empty", refuse)
    assert op(x, out) is out


def test_a_record_holds_its_own_call():
    """A record holds the params, indices and metadata tensors of its own call: a later call,
    a param edited after the call or an edit to the record reaches no other record, and the
    plan holds no tensor of the call."""
    ids = {"dtype": "int64", "shape": "[M]", "values": "sample_indices(M, M)"}
    op = _probe(
        lambda self, x, ids: x * 2,
        params={"dims": {"type": "list[int]"}},
        inputs={"x": _T, "ids": ids},
    )(dims=[1])
    ids = torch.arange(3)
    op(torch.ones(3), ids)
    op.dims.append(2)  # before the record is first read
    first = op.last_call
    assert first.indices["dims"] == [1] and op.last_call is first
    op.dims.pop()
    first.indices["M"] = 99
    first.indices["dims"].append(5)
    other = torch.zeros(3, dtype=torch.int64)
    op(torch.ones(3), other)
    second = op.last_call
    assert second is not first and (second.indices["M"], second.indices["dims"]) == (3, [1])
    assert first.metadata["ids"] is ids and second.metadata["ids"] is other
    other.fill_(2)
    assert second.metadata_values("ids") == [2, 2, 2]

    # A meta call completes with no plan; its record holds its own params too.
    op(torch.ones(3, device="meta"), torch.arange(3, device="meta"))
    op.dims.append(2)
    assert op.last_call.indices["dims"] == [1]
    op.dims.pop()

    # A new specialization, whose plan the first of these calls resolves.
    x, ids = torch.ones(5), torch.arange(5)
    y = op(x, ids)
    op(torch.ones(5), torch.arange(5))
    refs = [weakref.ref(t) for t in (x, ids, y)]
    del x, ids, y
    gc.collect()
    assert all(ref() is None for ref in refs), "the plan holds a tensor of its call"


def test_a_composite_record_holds_the_stage_calls_of_its_own_call():
    """A parent's record keeps the records its sub-ops completed inside it, so a sub-op's later
    call reaches neither the parent's stages nor their count or order."""

    def forward(self, x):
        leaf = self.delegate_for("leaf", None)
        for rows in range(1, self.calls + 1):
            leaf(x[:rows].contiguous())
        return x * 2

    leaf = {"leaf": _probe(lambda self, x: x * 2)}
    parent = _probe(forward, boundary=False, attributes={"delegate_types": leaf})(calls=1)
    parent(torch.ones(4))
    parent.calls = 3
    parent(torch.ones(4))
    parent._held_delegates()[0](torch.ones(2))
    assert [c.indices["M"] for c in parent.last_call.stages["leaf"]] == [1, 2, 3]
    parent.calls = 0
    parent(torch.ones(4))
    assert parent.last_call.stages == {"leaf": ()}


def test_a_composite_runs_the_sub_op_built_from_its_params_as_they_are():
    """A sub-op held under no identity is held by the params it was built from: a call after
    they change runs one built from the new ones, a return runs the first again, and a
    param it is not built from leaves it."""
    scale = {"scale": {"type": "float"}}
    leaf = {"leaf": _probe(lambda self, x: x * self.scale, params=scale)}
    parent = _probe(
        lambda self, x: self.delegate_for("leaf", None, scale=self.scale)(x) + self.offset,
        params={**scale, "offset": {"type": "float"}},
        boundary=False,
        attributes={"delegate_types": leaf},
    )(scale=2.0, offset=0.0)
    x = torch.ones(2)
    assert parent(x).tolist() == [2.0, 2.0]
    (first,) = parent._held_delegates()
    parent.offset = 1.0
    assert parent(x).tolist() == [3.0, 3.0] and parent._held_delegates() == (first,)
    parent.offset, parent.scale = 0.0, 3.0
    assert parent(x).tolist() == [3.0, 3.0]
    assert [stage.indices["scale"] for stage in parent.last_call.stages["leaf"]] == [3.0]
    parent.scale = 2.0
    assert parent(x).tolist() == [2.0, 2.0]
    assert len(parent._held_delegates()) == 2 and parent._held_delegates()[0] is first
    # A tensor argument replaced by one of the same layout builds a sub-op holding it.
    table = {"table": {"dtype": "float32", "shape": "[2]", "device": "cpu"}}
    leaf_table = {"leaf": _probe(lambda self, x: x * self.table, params=table)}
    holder = _probe(
        lambda self, x: self.delegate_for("leaf", None, table=self.table)(x),
        params=table,
        boundary=False,
        attributes={"delegate_types": leaf_table},
    )(table=torch.full((2,), 2.0))
    first_table = holder.table
    for value in (2.0, 3.0):
        holder(x)
        holder.table = torch.full((2,), value)
        assert holder(x).tolist() == [value, value]
    holder.table = first_table
    assert holder(x).tolist() == [2.0, 2.0]
    # One call requesting the stage with two argument sets runs both, warm as cold.
    both = _probe(
        lambda self, x: sum(self.delegate_for("leaf", None, scale=s)(x) for s in (2.0, 3.0)),
        params=scale,
        boundary=False,
        attributes={"delegate_types": leaf},
    )(scale=1.0)
    assert both(x).tolist() == both(x).tolist() == [5.0, 5.0]


def test_calls_resolving_one_plan_together_hold_each_request_once():
    """Threads making the first calls of one specialization together resolve its plan once,
    and leave it one request per position, so each request keeps its own call spec."""
    import dataclasses
    import threading
    import time

    from tileops._kernels.call_spec import CallSpec

    @dataclasses.dataclass(frozen=True)
    class Scale(CallSpec):
        scale: float = 1.0

    built = []

    def slowly(scale):
        time.sleep(0.01)
        built.append(scale)
        return Scale(scale=scale)

    def forward(self, x):
        return sum(x * self.call_spec("scale", lambda s=s: slowly(s)).scale for s in (2.0, 3.0))

    for _ in range(3):
        cls = _probe(forward)
        checks, check = _counted_checks(cls), cls._signature.check
        # Slow checks, so cold calls overlap while the plan is resolved.
        cls._signature.check = lambda op, tensors, check=check: time.sleep(0.01) or check(
            op, tensors
        )
        op = cls()
        start, results = threading.Barrier(4), []
        built.clear()

        def call(op=op, start=start, results=results):
            start.wait()
            results.append(op(torch.ones(2)).tolist())

        threads = [threading.Thread(target=call) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert results == [[5.0, 5.0]] * 4
        assert (len(checks), built) == (1, [2.0, 3.0]), "the plan was resolved more than once"


def test_calls_missing_one_entry_together_build_the_entry_tuning_reaches():
    """Threads whose first calls miss one entry together build it once, so the entry every
    call launches is the one the op holds and tuning reaches."""
    import dataclasses
    import threading
    import time
    from abc import abstractmethod

    from tileops._kernels.call_spec import CallSpec
    from tileops._kernels.kernel_base import Kernel, KernelInterface

    @dataclasses.dataclass(frozen=True)
    class Rows(CallSpec):
        rows: int = 0

    class Doubling(KernelInterface):
        """Double *x*."""

        request = Rows

        @abstractmethod
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return a new tensor."""

    built = []

    class Doubler(Kernel, Doubling):
        devices = frozenset({"cpu"})

        @classmethod
        def entry_for(cls, call):
            def build():
                time.sleep(0.01)
                built.append(cls())
                return built[-1]

            return call.rows, build

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 2

    def forward(self, x):
        kernel = self.kernel_for("double", lambda: Rows(device=x.device, rows=x.shape[0]))
        launched.append(kernel)
        return kernel(x)

    launched = []
    attributes = {"kernel_types": {"double": Doubler}, "interfaces": {"double": Doubling}}
    op = _probe(forward, attributes=attributes)()
    start = threading.Barrier(4)

    def call():
        start.wait()
        op(torch.ones(2))

    threads = [threading.Thread(target=call) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    op(torch.ones(2))
    assert len(built) == 1 and all(kernel is built[0] for kernel in launched)
    assert list(op.iter_kernels()) == built
