"""The kernel dispatch contract: selection, the dispatch cache, and the three extension points.

The mechanism is driven through a fake family whose implementations run on the CPU, so
selection, caching and installation checks need no device. The contract tests use a shipped
kernel interface and write their implementations against it only.
"""

import dataclasses
import importlib
import inspect
import pkgutil
from abc import abstractmethod

import pytest
import torch
import torch.nn.functional as F

import tileops.ops
from tileops.backend import BUILTIN, register_implementation, registry
from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.norm.call_spec import LayerNormCall, LayerNormFwdInterface
from tileops.ops import LayerNormFwdOp
from tileops.ops.op_base import Op
from workloads.device import run_device

pytestmark = pytest.mark.smoke


@dataclasses.dataclass(frozen=True)
class _Call(CallSpec):
    n: int = 0


class _Scaling(KernelInterface):
    """Scale *x*."""

    request = _Call

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a new tensor."""


class _Implementation(Kernel, _Scaling):
    """Built per decade of ``n``, so two call specs can share one build identity."""

    devices = frozenset({"cpu"})

    def __init__(self, decade: int) -> None:
        super().__init__()
        self.decade = decade

    @classmethod
    def entry_for(cls, call: _Call):
        return call.n // 10, lambda: cls(call.n // 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def autotune(self, warmup: int = 25, rep: int = 50) -> None:
        self.config = {"tuned": True}


def _implementation(name: str, applies=lambda call: True, **attrs) -> type:
    """An implementation of ``_Scaling`` that serves the calls *applies* accepts."""
    namespace = {"applies": classmethod(lambda cls, call: applies(call)), **attrs}
    return type(name, (_Implementation,), namespace)


_GENERAL = _implementation("General", general=True)
_POSITIVE = _implementation("Positive", lambda c: c.n > 0)
_BAND = _implementation("Band", lambda c: 10 < c.n <= 50, preferred_over=frozenset({"positive"}))
# Never applies with the band; it wins over the positive through it.
_HUNDREDS = _implementation("Hundreds", lambda c: c.n > 100, preferred_over=frozenset({"band"}))
_NEGATIVE = _implementation("Negative", lambda c: c.n < 0)


class _ScaleOp(Op):
    kernel_types = {
        "general": _GENERAL,
        "positive": _POSITIVE,
        "band": _BAND,
        "hundreds": _HUNDREDS,
        "negative": _NEGATIVE,
    }
    interfaces = {"scale": _Scaling}

    def __init__(self, kernel_map=None, tune: bool = False) -> None:
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def _infer_output_shapes(self, *shapes, dtypes=None):
        return {}

    def _validate_dtypes(self, *args):
        return None

    def eval_roofline(self):
        return (0, 0)

    def forward(self, x):
        return x

    def entry(self, n: int):
        return self.kernel_for("scale", (), _Call(device=torch.device("cpu"), n=n))


def _selected(op: Op, *ns: int) -> dict:
    """The key selected for each ``n``, or the error it raises."""
    selected = {}
    for n in ns:
        try:
            selected[n] = op.select_implementation("scale", _Call(device=torch.device("cpu"), n=n))
        except ValueError as exc:
            selected[n] = str(exc).split(":")[0]
    return selected


@pytest.fixture(autouse=True)
def isolated_registry():
    """Registrations made by a test do not outlive it."""
    state = registry.snapshot()
    yield
    registry.restore(state)


def test_selection_takes_the_implementation_no_other_is_preferred_over() -> None:
    """The general one is below every other; ``preferred_over`` orders the rest, transitively."""
    assert _selected(_ScaleOp(), 0, 5, 20, 200, -1) == {
        0: "general",
        5: "positive",
        20: "band",
        200: "hundreds",
        -1: "negative",
    }


def test_availability_filters_before_precedence() -> None:
    """An implementation that cannot run on the call's device takes nothing it is preferred over."""
    register_implementation(
        "_ScaleOp",
        "meta_positive",
        _implementation(
            "MetaPositive",
            lambda c: c.n > 0,
            devices=frozenset({"meta"}),
            preferred_over=frozenset({"positive"}),
        ),
    )
    assert _selected(_ScaleOp(), 5) == {5: "positive"}


def test_undeclared_overlap_and_an_uncovered_call_are_errors() -> None:
    register_implementation(
        "_ScaleOp", "also_positive", _implementation("AlsoPositive", lambda c: c.n > 0)
    )
    assert _selected(_ScaleOp(), 5) == {5: "dispatch is ambiguous"}

    class _NoGeneralOp(_ScaleOp):
        kernel_types = {k: v for k, v in _ScaleOp.kernel_types.items() if k != "general"}

    assert _selected(_NoGeneralOp(), 0) == {0: "no implementation serves this call"}


def test_a_replacement_keeps_the_rule_of_the_key_it_replaces() -> None:
    """``kernel_map=`` changes what runs under a key, never which calls select the key.

    The replacement serves every call the key is selected for, and one it does not serve is
    an error; a call selecting another key never asks it.
    """
    narrow = _implementation("NarrowBand", lambda c: 10 < c.n <= 30)
    op = _ScaleOp(kernel_map={"band": narrow})
    assert _selected(op, 0, 5, 20, 200, -1) == {
        0: "general",
        5: "positive",
        20: "band",
        200: "hundreds",
        -1: "negative",
    }
    assert type(op.entry(20)).__name__ == "NarrowBand"
    with pytest.raises(ValueError, match="the kernel supplied for band"):
        op.entry(40)


def test_a_hit_is_one_lookup_and_reads_no_device_fact(monkeypatch: pytest.MonkeyPatch) -> None:
    op = _ScaleOp()
    first = op.entry(5)

    def unreachable(*args, **kwargs):
        raise AssertionError("a hit resolved something")

    monkeypatch.setattr(op, "_resolve_entry", unreachable)
    call = _Call(device=torch.device("cpu"), n=5)
    assert op.kernel_for("scale", (), call) is first
    assert not {"_arch", "_calibration", "_sm_count", "_smem_budget"} & set(vars(call))


def test_call_specs_sharing_a_build_identity_share_one_entry() -> None:
    op = _ScaleOp()
    assert op.entry(1) is op.entry(9)
    assert op.entry(1) is not op.entry(20)
    assert len(op.built_kernels("scale")) == 2


def test_tuning_acts_on_the_resolved_entry_not_the_builder() -> None:
    """An equal call spec after ``autotune`` serves the tuned entry; builders take no tune."""
    op = _ScaleOp()
    entry = op.entry(5)
    assert entry.config == {}
    op.autotune()
    assert op.entry(5) is entry and entry.config == {"tuned": True}
    assert op.entry(7) is entry
    assert op.entry(25).config == {"tuned": True}


class _NotScaling(Kernel):
    def forward(self, x):
        return x


@pytest.mark.parametrize(
    ("kernel_map", "error", "match"),
    [
        ({"positive": _NotScaling}, TypeError, "does not implement _Scaling; .* inherits _Scaling"),
        (
            {"positive": _implementation("TwoArgs", forward=lambda self, x, y: x)},
            TypeError,
            "does not take _Scaling's arguments",
        ),
        (
            {"positive": _implementation("StaticEntry", entry_for=staticmethod(lambda call: 0))},
            TypeError,
            "entry_for is not a classmethod",
        ),
    ],
)
def test_what_runs_under_a_key_implements_its_interface(kernel_map, error, match) -> None:
    with pytest.raises(error, match=match):
        _ScaleOp(kernel_map=kernel_map)


@pytest.mark.parametrize(
    ("added", "match"),
    [
        ({"also_general": _implementation("AlsoGeneral", general=True)}, "more than one"),
        (
            {"stray": _implementation("Stray", preferred_over=frozenset({"elsewhere"}))},
            "preferred over \\['elsewhere'\\]",
        ),
        (
            {
                "ping": _implementation("Ping", preferred_over=frozenset({"pong"})),
                "pong": _implementation("Pong", preferred_over=frozenset({"ping"})),
            },
            "cycle",
        ),
        ({"positive": _implementation("Clash")}, "reuse keys it has"),
        ({"loose": _NotScaling}, "implement none of its kernel interfaces"),
    ],
)
def test_installation_refuses_a_malformed_registration(added, match) -> None:
    for key, cls in added.items():
        register_implementation("_ScaleOp", key, cls)
    with pytest.raises(ValueError, match=match):
        _ScaleOp()


def test_kernel_for_refuses_a_call_spec_it_cannot_key() -> None:
    """A call spec of another type, with an unhashable field, or stating a device fact,
    whether or not an equal call spec was served before."""
    op = _ScaleOp()
    cpu = torch.device("cpu")
    op.entry(5)
    for call, match in (
        (CallSpec(device=cpu), "takes a _Call call spec"),
        (_Call(device=cpu, n=[5]), "cannot key a dispatch cache"),
        (_Call(device=cpu, n=5, arch=90), "states \\['arch'\\]"),
        (_Call(device=cpu, n=5, smem_budget=1), "states \\['smem_budget'\\]"),
    ):
        with pytest.raises(TypeError, match=match):
            op.kernel_for("scale", (), call)


def test_a_record_reads_no_device_fact_where_the_process_has_no_cuda_device() -> None:
    """A record resolves no CUDA fact, and copying it — which reads every field — still works."""
    call = _Call(n=8)
    if torch.cuda.is_available():
        pytest.skip("needs a host with no CUDA device")
    assert (call.arch, call.sm_count, call.smem_budget) == (-1, 0, 0)
    assert dataclasses.replace(call, n=9) == _Call(n=9)


def test_an_installed_implementation_set_cannot_change() -> None:
    op = _ScaleOp()
    op.entry(5)
    with pytest.raises(TypeError):
        op.kernel_map["positive"] = _NEGATIVE
    op.dispatch_kernel({"positive": _implementation("Reinstalled")})
    assert type(op.entry(5)).__name__ == "Reinstalled"


@pytest.mark.cuda_only
def test_device_facts_come_from_the_calls_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """With two devices, a call on the second one resolves the second one's facts.

    The current device stays the first throughout, which is what a record without a
    device would have read.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices")
    import tileops.utils

    asked = []
    real = tileops.utils.device_facts

    def recording(index=None):
        asked.append(index)
        return real(index)

    monkeypatch.setattr(tileops.utils, "device_facts", recording)
    seen = []
    cuda = _implementation(
        "OnCuda",
        lambda c: seen.append((c.device, c.sm_count)) or True,
        devices=frozenset({"cuda"}),
        general=True,
    )

    class _CudaOp(_ScaleOp):
        kernel_types = {"on_cuda": cuda}

    op = _CudaOp()
    torch.cuda.set_device(0)
    for index in (1, 0, 1):
        op.kernel_for("scale", (), _Call(device=torch.device("cuda", index), n=1))
    assert asked == [1, 0]
    assert [device.index for device, _ in seen] == [1, 0]
    assert seen[0][1] == torch.cuda.get_device_properties(1).multi_processor_count


class _TorchLayerNorm(Kernel, LayerNormFwdInterface):
    """A replacement written against ``LayerNormFwdInterface`` alone."""

    devices = frozenset({torch.device(run_device()).type})

    def __init__(self, n: int, eps: float) -> None:
        super().__init__()
        self.n, self.eps = n, eps

    @classmethod
    def entry_for(cls, call: LayerNormCall):
        return (call.n, call.eps), lambda: cls(call.n, call.eps)

    def forward(self, x, weight, bias):
        return F.layer_norm(x.float(), (self.n,), weight.float(), bias.float(), self.eps).to(
            x.dtype
        )


class _NarrowTorchLayerNorm(_TorchLayerNorm):
    """An added implementation for short rows, which wins over the in-tree one there."""

    preferred_over = frozenset({"layer_norm"})

    @classmethod
    def applies(cls, call: LayerNormCall) -> bool:
        return call.n <= 64


def _layer_norm(op, n: int):
    x = torch.randn(8, n, device=run_device())
    weight, bias = torch.randn(n, device=run_device()), torch.randn(n, device=run_device())
    expected = F.layer_norm(x, (n,), weight, bias)
    torch.testing.assert_close(op(x, weight, bias), expected, atol=1e-4, rtol=1e-4)
    return [type(k).__name__ for k in op.built_kernels("layer_norm").values()]


def test_a_replacement_needs_only_the_published_contract() -> None:
    op = LayerNormFwdOp((32,), kernel_map={"layer_norm": _TorchLayerNorm}, target=BUILTIN)
    assert _layer_norm(op, 32) == ["_TorchLayerNorm"]


@pytest.mark.cuda_only
def test_an_added_implementation_serves_its_calls_and_the_in_tree_one_the_rest() -> None:
    register_implementation("LayerNormFwdOp", "torch_short_rows", _NarrowTorchLayerNorm)
    assert _layer_norm(LayerNormFwdOp((32,)), 32) == ["_NarrowTorchLayerNorm"]
    assert _layer_norm(LayerNormFwdOp((1024,)), 1024) == ["LayerNormKernel"]


def test_every_op_reaches_its_kernels_through_an_interface() -> None:
    """An op that builds a kernel of its own declares the interface it calls it through."""
    for module in pkgutil.walk_packages(tileops.ops.__path__, "tileops.ops."):
        importlib.import_module(module.name)
    ops, pending = set(), [Op]
    while pending:
        for cls in pending.pop().__subclasses__():
            pending.append(cls)
            ops.add(cls)
    legacy = {
        cls.__name__
        for cls in ops
        if cls.__module__.startswith("tileops.ops")
        and not cls.__name__.startswith("_")
        and not inspect.isabstract(cls)
        and not cls.interfaces
        and (cls.kernel_types or cls.default_kernel_map is not Op.default_kernel_map)
    }
    assert sorted(legacy) == [], "declare interfaces instead"
