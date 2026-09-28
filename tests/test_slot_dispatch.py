"""The slot dispatch contract: selection, the dispatch cache, and the three extension points.

The mechanism is driven through a fake family whose candidates run on the CPU, so selection,
caching and installation checks need no device. The contract tests use a shipped slot and
write their candidates against its published interface only.
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
from tileops.backend import BUILTIN, register_candidate, registry
from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import Kernel, Slot
from tileops.kernels.norm.call_spec import LayerNormCall, LayerNormFwdSlot
from tileops.ops import LayerNormFwdOp
from tileops.ops.op_base import Op
from workloads.device import run_device

pytestmark = pytest.mark.smoke


@dataclasses.dataclass(frozen=True)
class _Call(CallSpec):
    n: int = 0


class _Scaling(Slot):
    """Scale *x*."""

    request = _Call

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a new tensor."""


class _Candidate(Kernel, _Scaling):
    """Built per decade of ``n``, so two request keys can share one build identity."""

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


def _candidate(name: str, region=lambda call: True, **attrs) -> type:
    """A candidate class whose region is *region*."""
    namespace = {"applies": classmethod(lambda cls, call: region(call)), **attrs}
    return type(name, (_Candidate,), namespace)


_GENERAL = _candidate("General", general=True)
_POSITIVE = _candidate("Positive", lambda c: c.n > 0)
_BAND = _candidate("Band", lambda c: 10 < c.n <= 50, refines=frozenset({"positive"}))
_HUNDREDS = _candidate("Hundreds", lambda c: c.n > 100, refines=frozenset({"band"}))
_NEGATIVE = _candidate("Negative", lambda c: c.n < 0)


class _ScaleOp(Op):
    kernel_types = {
        "general": _GENERAL,
        "positive": _POSITIVE,
        "band": _BAND,
        "hundreds": _HUNDREDS,
        "negative": _NEGATIVE,
    }
    slots = {"scale": _Scaling}

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


@pytest.fixture(autouse=True)
def isolated_registry():
    """Registrations made by a test do not outlive it."""
    state = registry.snapshot()
    yield
    registry.restore(state)


def test_selection_takes_the_most_specific_applicable_candidate() -> None:
    """The general one is below every other; ``refines`` orders the rest, transitively.

    At 200 the band does not apply, and the hundreds still beat the positive through it.
    """
    op = _ScaleOp()
    selected = {
        n: op.select_candidate("scale", _Call(device=torch.device("cpu"), n=n))
        for n in (0, 5, 20, 200, -1)
    }
    assert selected == {0: "general", 5: "positive", 20: "band", 200: "hundreds", -1: "negative"}


def test_overlap_without_refinement_and_an_uncovered_call_are_errors() -> None:
    also_positive = _candidate("AlsoPositive", lambda c: c.n > 0)
    with pytest.raises(ValueError, match="ambiguous"):
        _ScaleOp(kernel_map={"negative": also_positive}).entry(5)
    not_general = _candidate("NotGeneral", lambda c: c.n > 0)
    with pytest.raises(ValueError, match="no implementation serves"):
        _ScaleOp(kernel_map={"general": not_general}).entry(0)


def test_a_hit_is_one_lookup_and_reads_no_device_fact(monkeypatch: pytest.MonkeyPatch) -> None:
    op = _ScaleOp()
    first = op.entry(5)

    def unreachable(*args, **kwargs):
        raise AssertionError("a hit resolved something")

    monkeypatch.setattr(op, "_resolve_entry", unreachable)
    call = _Call(device=torch.device("cpu"), n=5)
    assert op.kernel_for("scale", (), call) is first
    assert "_arch" not in vars(call) and "_sm_count" not in vars(call)


def test_request_keys_sharing_a_build_identity_share_one_entry() -> None:
    op = _ScaleOp()
    assert op.entry(1) is op.entry(9)
    assert op.entry(1) is not op.entry(20)
    assert len(op.built_kernels("scale")) == 2


def test_tuning_acts_on_the_resolved_entry_not_the_builder() -> None:
    """An equal request key after ``autotune`` serves the tuned entry; builders take no tune."""
    op = _ScaleOp()
    entry = op.entry(5)
    assert entry.config == {}
    op.autotune()
    assert op.entry(5) is entry and entry.config == {"tuned": True}
    assert op.entry(7) is entry
    assert op.entry(25).config == {"tuned": True}
    assert "tune" not in inspect.signature(_Candidate.__init__).parameters


class _NotScaling(Kernel):
    def forward(self, x):
        return x


@pytest.mark.parametrize(
    ("kernel_map", "error", "match"),
    [
        ({"positive": _NotScaling}, TypeError, "does not implement _Scaling"),
        (
            {"positive": _candidate("TwoArgs", forward=lambda self, x, y: x)},
            TypeError,
            "does not take _Scaling's arguments",
        ),
        (
            {"positive": _candidate("Renamed", forward=lambda self, y: y)},
            TypeError,
            "takes \\['y'\\]",
        ),
        ({"positive": _candidate("AlsoGeneral", general=True)}, ValueError, "more than one"),
        (
            {"positive": _candidate("Stray", refines=frozenset({"elsewhere"}))},
            ValueError,
            "refines \\['elsewhere'\\]",
        ),
        (
            {"positive": _candidate("Cycle", refines=frozenset({"hundreds"}))},
            ValueError,
            "cycle",
        ),
    ],
)
def test_installation_checks_each_candidate_against_its_slot(kernel_map, error, match) -> None:
    with pytest.raises(error, match=match):
        _ScaleOp(kernel_map=kernel_map)


def test_an_installed_candidate_set_cannot_change() -> None:
    op = _ScaleOp()
    op.entry(5)
    with pytest.raises(TypeError):
        op.kernel_map["positive"] = _NEGATIVE
    op.dispatch_kernel({"positive": _candidate("Reinstalled", lambda c: c.n > 0)})
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
    cuda = _candidate(
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


class _TorchLayerNorm(Kernel, LayerNormFwdSlot):
    """A replacement written against ``LayerNormFwdSlot`` alone."""

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
    """An added candidate for short rows, nested in the in-tree one's region."""

    refines = frozenset({"layer_norm"})

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
def test_an_added_candidate_serves_its_region_and_the_in_tree_one_the_rest() -> None:
    register_candidate("LayerNormFwdOp", "layer_norm", "torch_short_rows", _NarrowTorchLayerNorm)
    assert _layer_norm(LayerNormFwdOp((32,)), 32) == ["_NarrowTorchLayerNorm"]
    assert _layer_norm(LayerNormFwdOp((1024,)), 1024) == ["LayerNormKernel"]


# FIXME(staged-rollout): the ops that still reach kernels outside a declared slot.
#
# Broken invariant: every op declares ``slots`` (ops-design.md § Kernel selection).
# Why: ops migrate one family per PR.
# Cleanup: a migration PR deletes the names it migrates; delete this list with the last one.
_LEGACY_OPS = frozenset(
    [
        "AbsFwdOp",
        "AdaLayerNormFwdOp",
        "AdaLayerNormZeroFwdOp",
        "AdaptiveAvgPool2dFwdOp",
        "AdaptiveMaxPool2dFwdOp",
        "AdaptiveMaxPool2dIndicesFwdOp",
        "AddFwdOp",
        "AlibiFwdOp",
        "AllFwdOp",
        "AmaxFwdOp",
        "AminFwdOp",
        "AnyFwdOp",
        "ArgmaxFwdOp",
        "ArgminFwdOp",
        "AvgPool1dFwdOp",
        "AvgPool2dFwdOp",
        "AvgPool3dFwdOp",
        "BitwiseAndFwdOp",
        "BitwiseNotFwdOp",
        "BitwiseOrFwdOp",
        "BitwiseXorFwdOp",
        "BmmFp8FwdOp",
        "BmmFwdOp",
        "CBProducerFwdOp",
        "CeilFwdOp",
        "ClampFwdOp",
        "ClampScalarFwdOp",
        "Conv1dFwdOp",
        "Conv2dFwdOp",
        "Conv3dFwdOp",
        "CosFwdOp",
        "CountNonzeroFwdOp",
        "CumprodFwdOp",
        "CumsumFwdOp",
        "DaCumsumFwdOp",
        "DeepSeekSparseAttentionDecodeWithKVCacheFwdOp",
        "DeltaNetBwdOp",
        "DeltaNetDecodeFwdOp",
        "DeltaNetFwdOp",
        "DeltaNetInferenceFwdOp",
        "DivFwdOp",
        "DropoutFwdOp",
        "EluFwdOp",
        "EngramDecodeFwdOp",
        "EngramGateConvBwdOp",
        "EngramGateConvFwdOp",
        "EqFwdOp",
        "ErfFwdOp",
        "ExpFwdOp",
        "Expm1FwdOp",
        "FFTC2CFwdOp",
        "FP8LightningIndexerFwdOp",
        "FP8QuantFwdOp",
        "FloorDivideFwdOp",
        "FloorFwdOp",
        "FusedAddLayerNormFwdOp",
        "FusedAddRMSNormFwdOp",
        "FusedTopKFwdOp",
        "GLABwdOp",
        "GLADecodeFwdOp",
        "GLAFwdOp",
        "GatedDeltaNetFwdOp",
        "GeFwdOp",
        "GeluAndMulFwdOp",
        "GeluFwdOp",
        "GeluTanhAndMulFwdOp",
        "GemmFp8FwdOp",
        "GemmFwdOp",
        "GemmW4A16FwdOp",
        "GroupNormFwdOp",
        "GroupedGemmFwdOp",
        "GroupedQueryAttentionBwdOp",
        "GroupedQueryAttentionPagedFwdOp",
        "GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp",
        "GroupedQueryAttentionVarlenFwdOp",
        "GtFwdOp",
        "HardsigmoidFwdOp",
        "HardswishFwdOp",
        "HardtanhFwdOp",
        "IndexedExpertMLPFwdOp",
        "InfNormFwdOp",
        "IsfiniteFwdOp",
        "IsinfFwdOp",
        "IsnanFwdOp",
        "L1NormFwdOp",
        "L2NormFwdOp",
        "LeFwdOp",
        "LeakyReluFwdOp",
        "LerpFwdOp",
        "LerpTensorFwdOp",
        "Log1pFwdOp",
        "LogFwdOp",
        "LogSoftmaxFwdOp",
        "LogSumExpFwdOp",
        "LogicalAndFwdOp",
        "LogicalNotFwdOp",
        "LogicalOrFwdOp",
        "LtFwdOp",
        "MHCPostFwdOp",
        "MHCPreFwdOp",
        "MaskedFillFwdOp",
        "MaskedFillScalarFwdOp",
        "MaxPool1dFwdOp",
        "MaxPool1dIndicesFwdOp",
        "MaxPool2dFwdOp",
        "MaxPool2dIndicesFwdOp",
        "MaxPool3dFwdOp",
        "MaxPool3dIndicesFwdOp",
        "MaximumFwdOp",
        "MeanFwdOp",
        "MeanPoolingFwdOp",
        "MinimumFwdOp",
        "MishFwdOp",
        "MoeGroupedGemmFwdOp",
        "MoePermuteAlignFwdOp",
        "MoePostPermuteFwdOp",
        "MoePrePermuteFwdOp",
        "MulFwdOp",
        "MultiHeadAttentionDecodePagedWithKVCacheFwdOp",
        "MultiHeadLatentAttentionDecodeWithKVCacheFwdOp",
        "NSACmpVarlenFwdOp",
        "NSATopkVarlenFwdOp",
        "NSAVarlenFwdOp",
        "NanToNumFwdOp",
        "NeFwdOp",
        "NegFwdOp",
        "PowFwdOp",
        "PreluFwdOp",
        "ProdFwdOp",
        "RMSNormFwdOp",
        "ReciprocalFwdOp",
        "ReluFwdOp",
        "RemainderFwdOp",
        "RopeLlama31FwdOp",
        "RopeLongRopeFwdOp",
        "RopeNeoxFwdOp",
        "RopeNeoxPositionIdsFwdOp",
        "RopeNonNeoxFwdOp",
        "RopeYarnFwdOp",
        "RoundFwdOp",
        "RsqrtFwdOp",
        "SSDChunkScanFwdOp",
        "SSDChunkStateFwdOp",
        "SSDDecodeFwdOp",
        "SSDStatePassingFwdOp",
        "SeluFwdOp",
        "SharedExpertMLPFwdOp",
        "SigmoidFwdOp",
        "SignFwdOp",
        "SiluAndMulFwdOp",
        "SiluFwdOp",
        "SinFwdOp",
        "SinusoidalFwdOp",
        "SoftmaxFwdOp",
        "SoftplusFwdOp",
        "SqrtFwdOp",
        "StdFwdOp",
        "SubFwdOp",
        "SumFwdOp",
        "TanhFwdOp",
        "TopkSelectorFwdOp",
        "TruncFwdOp",
        "VarFwdOp",
        "VarMeanFwdOp",
        "WhereFwdOp",
    ]
)


def test_no_op_reaches_kernels_outside_a_slot_unless_listed() -> None:
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
        and not cls.slots
        and (
            cls.kernel_types
            or cls.default_kernel_map is not Op.default_kernel_map
            or cls.entry_for is not Op.entry_for
        )
    }
    assert sorted(legacy - _LEGACY_OPS) == [], "declare slots instead"
    assert sorted(_LEGACY_OPS - legacy) == [], "migrated: remove these from the list"
