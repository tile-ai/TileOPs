"""Tests for tileops.ops.op_base.

Covers composite kernel-map overrides, the ``kernel_for`` path, and the explicit kernel
enumeration ``Op.autotune`` runs over.
"""

import dataclasses
import types
from abc import abstractmethod
from pathlib import Path

import pytest
import torch
import yaml

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.ops import op_base
from tileops.ops.op_base import Op

pytestmark = pytest.mark.smoke

_CPU = torch.device("cpu")


@dataclasses.dataclass(frozen=True)
class _Call(CallSpec):
    """What the doubles are built from: one build per ``key``, named by ``name``."""

    key: torch.dtype = torch.float16
    name: str = ""


class _FwdRecording(KernelInterface):
    """The place the doubles call their main kernel."""

    request = _Call

    @abstractmethod
    def forward(self) -> None:
        """Run nothing; the double exists to be built, enumerated and tuned."""


class _AuxRecording(KernelInterface):
    """A second place, so one ``key`` can name two entries that do not collide."""

    request = _Call

    @abstractmethod
    def forward(self) -> None:
        """Run nothing; the double exists to be built, enumerated and tuned."""


class _RecordingKernel(Kernel):
    """Kernel that records its name when tuned, so autotune order is visible."""

    devices = frozenset({"cpu"})
    # Per-test sinks, installed by ``_recording_kernels``.
    tuned: list = []
    builds: list = []
    role: str = "fwd"

    @classmethod
    def entry_for(cls, call: _Call):
        def factory():
            cls.builds.append((cls.role, call.key))
            return cls(call.name)

        return call.key, factory

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def forward(self):
        return None

    def autotune(self, warmup=25, rep=50):
        type(self).tuned.append(self.name)


def _recording_kernels(tuned: list, builds: list):
    """The two implementations one ``_SlottedOp`` instance runs, recording into *tuned*."""

    class Fwd(_RecordingKernel, _FwdRecording):
        role = "fwd"

    class Aux(_RecordingKernel, _AuxRecording):
        role = "aux"

    for cls in (Fwd, Aux):
        cls.tuned = tuned
        cls.builds = builds
    return Fwd, Aux


def _make_op_subclass():
    """Build a minimal concrete Op subclass for testing."""
    attrs = {
        "forward": lambda self, *a, **kw: None,
        # The three manifest-driven methods are abstract on Op; these doubles
        # exercise the get-or-build plumbing, so a minimal body is the contract.
        "_infer_output_shapes": lambda self, *shapes: {},
        "eval_roofline": lambda self: (0, 0),
    }
    return type("TestOp", (Op,), attrs)


_GATED = yaml.safe_load((Path(__file__).parent / "manifest_cases.yaml").read_text())["entries"][
    "SiluAndMulFwdOp"
]["signature"]


def _gated_op(name: str, forward, delegate_types=None) -> type:
    """An op on the gated activation's signature whose ``forward`` is *forward*."""
    from tileops.ops._signature_codegen import install

    def construct(self, *, target=None, kernel_map=None, tune=False):
        self.target = target
        self.dispatch_kernel(kernel_map)

    cls = type(
        name,
        (Op,),
        {"__init__": construct, "forward": forward, "delegate_types": delegate_types or {}},
    )
    install(cls, {"family": "probe", "signature": _GATED, "roofline": {"flops": "1"}})
    return cls


class TestCompositeKernelMapOverride:
    """Composite ops (empty ``kernel_types``) accept a non-empty override and store it verbatim."""

    def test_empty_default_with_empty_override_yields_empty_map(self):
        Cls = _make_op_subclass()
        op = Cls()
        op.dispatch_kernel(None)
        assert op.kernel_map == {}

    def test_empty_default_with_non_empty_override_stores_override(self):
        Cls = _make_op_subclass()
        op = Cls()
        override = {"first": object(), "second": object()}
        op.dispatch_kernel(override)
        assert op.kernel_map == override

    def test_empty_default_override_is_copied_not_aliased(self):
        Cls = _make_op_subclass()
        op = Cls()
        override = {"first": object()}
        op.dispatch_kernel(override)
        override["extra"] = object()
        assert "extra" not in op.kernel_map


class _SlottedOp(Op):
    """Op whose forward-built kernels all go through ``kernel_for``."""

    interfaces = {"fwd": _FwdRecording, "aux": _AuxRecording}

    def __init__(self, tuned: list):
        self._tuned = tuned
        self.builds: list[tuple[str, object]] = []
        self.tune = False
        fwd, aux = _recording_kernels(tuned, self.builds)
        self.kernel_types = {"fwd": fwd, "aux": aux}
        self._install_kernel_map(None)

    def _infer_output_shapes(self, *shapes):
        return {}

    def eval_roofline(self):
        return (0, 0)

    def forward(self, *a, **kw):
        return None

    def build(self, role: str, key, name: str):
        return self.kernel_for(role, _Call(device=_CPU, key=key, name=name))


class TestGetOrBuildKernel:
    """``Op.kernel_for`` is the single get-or-build in the Op layer."""

    def test_factory_runs_once_per_key(self):
        op = _SlottedOp([])
        first = op.build("fwd", torch.float16, "fp16")
        again = op.build("fwd", torch.float16, "fp16")
        assert again is first
        assert op.builds == [("fwd", torch.float16)]

    def test_distinct_keys_build_distinct_entries(self):
        op = _SlottedOp([])
        fp16 = op.build("fwd", torch.float16, "fp16")
        bf16 = op.build("fwd", torch.bfloat16, "bf16")
        assert fp16 is not bf16
        assert {key for _, key in op.built_kernels("fwd")} == {torch.float16, torch.bfloat16}

    def test_same_key_in_distinct_roles_does_not_collide(self):
        """An auxiliary kernel keyed by the same dtype is a second role."""
        op = _SlottedOp([])
        main = op.build("fwd", torch.float16, "main")
        aux = op.build("aux", torch.float16, "aux")
        assert main is not aux
        assert [key for _, key in op.built_kernels("aux")] == [torch.float16]

    def test_built_kernels_is_empty_before_the_first_build(self):
        assert dict(_SlottedOp([]).built_kernels("fwd")) == {}

    def test_built_kernels_view_rejects_mutation(self):
        op = _SlottedOp([])
        op.build("fwd", torch.float16, "fp16")
        with pytest.raises(TypeError):
            op.built_kernels("fwd")[torch.bfloat16] = object()


class TestIterKernels:
    """``Op.iter_kernels`` is the explicit enumeration ``autotune`` runs over."""

    def test_yields_role_entries_including_bundles(self):
        tuned: list[str] = []

        @dataclasses.dataclass(frozen=True)
        class Entry:
            kernel: Kernel
            compute_dtype: torch.dtype

        op = _SlottedOp(tuned)
        fwd, aux = op.kernel_map["fwd"], op.kernel_map["aux"]
        fwd.entry_for = classmethod(lambda cls, call: (call.key, lambda: (cls("pre"), cls("bwd"))))
        aux.entry_for = classmethod(
            lambda cls, call: (call.key, lambda: Entry(cls("record"), torch.float32))
        )
        op.build("fwd", torch.float16, "pair")
        op.build("aux", torch.bfloat16, "entry")
        assert sorted(k.name for k in op.iter_kernels()) == ["bwd", "pre", "record"]

    def test_yields_the_directly_bound_kernel(self):
        tuned: list[str] = []
        op = _SlottedOp(tuned)
        op.kernel = op.kernel_map["fwd"]("bound")
        assert [k.name for k in op.iter_kernels()] == ["bound"]

    def test_ignores_kernels_bound_to_other_attributes(self):
        """Enumeration is explicit: an unregistered attribute is not searched."""
        tuned: list[str] = []
        op = _SlottedOp(tuned)
        op.some_other_attribute = op.kernel_map["fwd"]("hidden")
        assert list(op.iter_kernels()) == []

    def test_ignores_a_kernel_dict_the_op_owns(self):
        """A private dict of kernels is unreachable — the miss reflection used to hide.

        The old ``dir(self)`` walk descended dict values, so an op could keep its
        own cache and still be tuned. Enumeration does not, which is the point:
        the kernels have to be built through a role to be seen at all.
        """
        tuned: list[str] = []
        op = _SlottedOp(tuned)
        op.private_cache = {torch.float16: op.kernel_map["fwd"]("private")}
        assert list(op.iter_kernels()) == []
        op.autotune()
        assert tuned == []

    def test_deduplicates_a_kernel_reachable_twice(self):
        tuned: list[str] = []
        op = _SlottedOp(tuned)
        op.kernel = op.build("fwd", torch.float16, "fp16")
        assert [k.name for k in op.iter_kernels()] == ["fp16"]

    def test_descends_into_delegates(self):
        tuned: list[str] = []
        delegate = _SlottedOp(tuned)
        delegate.build("fwd", torch.float16, "delegate")

        class CompositeOp(_SlottedOp):
            delegate_types = {"stage": _SlottedOp}

        composite = CompositeOp(tuned)
        composite.delegate_for("stage", None, delegate)
        composite.build("fwd", torch.float16, "own")
        assert sorted(k.name for k in composite.iter_kernels()) == ["delegate", "own"]

    def test_deduplicates_a_kernel_shared_with_a_delegate(self):
        """A composite that caches the kernel its delegate built tunes it once."""
        tuned: list[str] = []
        delegate = _SlottedOp(tuned)
        shared = delegate.build("fwd", torch.float16, "shared")

        class CompositeOp(_SlottedOp):
            delegate_types = {"stage": _SlottedOp}

        composite = CompositeOp(tuned)
        composite.kernel_map["fwd"].entry_for = classmethod(
            lambda cls, call: (call.key, lambda: shared)
        )
        composite.delegate_for("stage", None, delegate)
        composite.build("fwd", torch.float16, "shared")
        assert [k.name for k in composite.iter_kernels()] == ["shared"]


class TestAutotune:
    """``Op.autotune`` tunes exactly what ``iter_kernels`` yields."""

    def test_autotune_tunes_the_bound_kernel_and_every_role_entry(self):
        tuned: list[str] = []
        op = _SlottedOp(tuned)
        op.kernel = op.kernel_map["fwd"]("bound")
        op.build("fwd", torch.float16, "fp16")
        op.build("fwd", torch.bfloat16, "bf16")
        op.build("aux", torch.float16, "aux")

        op.autotune()
        assert sorted(tuned) == ["aux", "bf16", "bound", "fp16"]

    def test_autotune_reaches_a_delegates_kernels(self):
        """A composite tunes through ``kernel_delegates``, not an override."""
        tuned: list[str] = []
        delegate = _SlottedOp(tuned)
        delegate.build("fwd", torch.float16, "delegate")

        class CompositeOp(_SlottedOp):
            delegate_types = {"stage": _SlottedOp}

        composite = CompositeOp(tuned)
        composite.delegate_for("stage", None, delegate)
        composite.autotune()
        assert tuned == ["delegate"]


class _TunableOp(_SlottedOp):
    """Op whose builds the dispatcher puts in tuned mode from ``self.tune``."""

    def __init__(self, tuned: list, *, tune: bool = False):
        super().__init__(tuned)
        self.tune = tune

    def build(self, dtype):
        return super().build("fwd", dtype, str(dtype))


class TestTunedMode:
    """``autotune()`` is a lifecycle decision, so it governs later builds too."""

    def test_a_kernel_built_after_autotune_is_tuned(self):
        tuned: list[str] = []
        op = _TunableOp(tuned)
        op.autotune()  # nothing built yet
        assert tuned == []
        op.build(torch.float16)
        assert tuned == ["torch.float16"]

    def test_every_later_specialization_is_tuned_not_just_the_next(self):
        """The decision persists: a second dtype arriving later is tuned too."""
        tuned: list[str] = []
        op = _TunableOp(tuned)
        op.autotune()
        op.build(torch.float16)
        op.build(torch.bfloat16)
        assert sorted(tuned) == ["torch.bfloat16", "torch.float16"]

    def test_an_untuned_op_leaves_later_builds_alone(self):
        tuned: list[str] = []
        op = _TunableOp(tuned)
        op.build(torch.float16)
        assert tuned == []

    def test_a_kernel_built_after_autotune_is_tuned_once_without_its_factory_reading_tune(self):
        """The build path tunes what a factory returns, and a second request changes nothing."""
        tuned: list[str] = []
        op = _SlottedOp(tuned)
        op.autotune()
        op.build("fwd", torch.float16, "fp16")
        op.autotune()
        assert tuned == ["fp16"]

    def test_a_kernel_whose_program_is_built_at_launch_tunes_at_that_launch_once(self):
        tuned: list[str] = []

        class LaunchBuiltKernel(Kernel):
            autotune_configs = [{"threads": 128}]

            def forward(self):
                self.kernel = "program"

            def tune_jit_kernel(self, kernel, configs, warmup, rep):
                tuned.append(kernel)
                return types.SimpleNamespace(config={"threads": 128})

        kernel = LaunchBuiltKernel()
        kernel.request_tune()
        assert tuned == []
        kernel()
        kernel()
        assert tuned == ["program"]

    def test_a_delegate_built_after_autotune_inherits_tuned_mode(self):
        """``delegate_for`` hands the composite's flag on, so the decision carries."""
        tuned: list[str] = []

        class DelegateOp(_TunableOp):
            def __init__(self, rec, *, target=None, kernel_map=None, tune=False):
                super().__init__(rec, tune=tune)

        class CompositeOp(_TunableOp):
            delegate_types = {"stage": DelegateOp}

        op = CompositeOp(tuned)
        op.autotune()
        op.delegate_for("stage", None, rec=tuned).build(torch.float16)
        assert tuned == ["torch.float16"]


class TestDelegateFor:
    """``Op.delegate_for`` is the single get-or-build for sub-ops."""

    def test_builds_the_declared_class_once_per_key_with_the_parents_policy(self):
        seen: list[dict] = []

        class DelegateOp(_SlottedOp):
            def __init__(self, *, width, target=None, kernel_map=None, tune=False):
                super().__init__([])
                seen.append({"width": width, "target": target, "kernel_map": kernel_map})

        class CompositeOp(_SlottedOp):
            delegate_types = {"stage": DelegateOp}

        op = CompositeOp([])
        op.target = "acme"
        first = op.delegate_for("stage", 1, width=1)
        assert op.delegate_for("stage", 1, width=1) is first
        second = op.delegate_for("stage", 2, width=2)
        assert type(first) is DelegateOp and second is not first
        assert seen == [
            {"width": 1, "target": "acme", "kernel_map": None},
            {"width": 2, "target": "acme", "kernel_map": None},
        ]

    def test_enumerates_held_sub_ops_in_stage_order(self):
        class CompositeOp(_SlottedOp):
            delegate_types = {"first": _SlottedOp, "second": _SlottedOp}

        op = CompositeOp([])
        late, early = _SlottedOp([]), _SlottedOp([])
        op.delegate_for("second", None, late)
        op.delegate_for("first", None, early)
        assert op.kernel_delegates() == (early, late)
        assert list(op._walk_ops()) == [op, early, late]

    def test_a_failed_settling_call_unsettles_the_sub_ops(self):
        class CompositeOp(_SlottedOp):
            delegate_types = {"stage": _SlottedOp}

        op = CompositeOp([])
        delegate = op.delegate_for("stage", None, _SlottedOp([]))
        delegate._builder = None
        delegate.build("fwd", torch.float16, "fp16")
        op._unsettle()
        assert delegate.settled_target is None and dict(delegate.built_kernels("fwd")) == {}

    def test_a_call_that_fails_before_selecting_a_target_leaves_the_sub_ops_bound(self):
        class CompositeOp(_SlottedOp):
            delegate_types = {"stage": _SlottedOp}

        op = CompositeOp([])
        delegate = op.delegate_for("stage", None, _SlottedOp([]))
        delegate._builder = None
        entry = delegate.build("fwd", torch.float16, "fp16")

        def refuse(tensors):
            raise ValueError("outside the signature")

        # The probe has no manifest signature, so its tensors and its check are stubbed.
        op._named_tensors = lambda inputs, writes: {}
        op._check_signature = refuse
        with pytest.raises(ValueError, match="outside the signature"):
            op._serve((), op.forward)
        assert delegate._builder is None
        assert list(delegate.built_kernels("fwd").values()) == [entry]


class TestStages:
    """Every call a sub-op completes inside its parent's call is filed under one stage."""

    def test_a_sub_op_not_held_through_delegate_for_fails_the_parent_call(self):
        leaf_cls = _gated_op("ProbeStrayLeafFwdOp", lambda self, x: x[:, : x.shape[1] // 2] * 1)
        stray = leaf_cls()

        def forward(self, x):
            held = self.delegate_for("leaf", None)
            return stray(x) if self.stray else held(x)

        parent = _gated_op("ProbeStrayFwdOp", forward, {"leaf": leaf_cls})()
        parent.stray = False
        x = torch.ones(3, 8, dtype=torch.float16)
        parent(x)
        kept = parent.last_call
        parent.stray = True
        with pytest.raises(RuntimeError, match="not held through delegate_for"):
            parent(x)
        assert parent.last_call is kept

    def test_one_sub_op_is_held_under_one_stage_and_identity(self):
        class CompositeOp(_SlottedOp):
            delegate_types = {"first": _SlottedOp, "second": _SlottedOp}

        op = CompositeOp([])
        shared = _SlottedOp([])
        assert op.delegate_for("first", None, shared) is shared
        assert op.delegate_for("first", None, shared) is shared
        with pytest.raises(ValueError, match="already holds"):
            op.delegate_for("second", None, shared)
        with pytest.raises(ValueError, match="already holds"):
            op.delegate_for("first", 1, shared)


class TestInstanceKeys:
    def test_a_collected_instances_key_is_never_handed_out_again(self):
        """An op reaching a used key inherits that op's compiled shapes."""
        import weakref

        class _Dummy:
            pass

        keys = set()
        for _ in range(50):
            op = _Dummy()
            keys.add(op_base.register_instance(op))
            ref = weakref.ref(op)
            del op
            assert ref() is None

        assert len(keys) == 50

    def test_a_key_names_the_class_it_belongs_to(self):
        """Graph dumps and guard failures show the key, not the instance."""

        class _Dummy:
            pass

        assert op_base.register_instance(_Dummy()).startswith("_Dummy")


def test_no_abstract_op_class_is_instantiated_anywhere():
    """An abstract Op cannot be constructed, so nothing in the tree may try.

    A class is abstract when it does not answer the manifest-driven contract:
    ``_infer_output_shapes``, ``eval_roofline``. Those are
    the family bases and the modular interfaces; a call site naming one is a call
    site that wanted a concrete op.
    """
    import importlib
    import inspect
    import pkgutil
    import re
    from pathlib import Path

    import tileops.ops as ops_pkg

    abstract = set()
    for module in pkgutil.walk_packages(ops_pkg.__path__, ops_pkg.__name__ + "."):
        try:
            mod = importlib.import_module(module.name)
        except Exception:  # a family whose kernels need a GPU-only import
            continue
        for obj in vars(mod).values():
            if (
                inspect.isclass(obj)
                and issubclass(obj, Op)
                and getattr(obj, "__abstractmethods__", None)
            ):
                abstract.add(obj.__name__)
    assert abstract, "no abstract Op classes resolved — the scan is not looking at the tree"

    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in (
        list((root / "src").rglob("*.py"))
        + list((root / "tests").rglob("*.py"))
        + list((root / "benchmarks").rglob("*.py"))
    ):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if (
                stripped.startswith(("class ", "#", "*", '"'))
                or "import" in stripped
                or "``" in stripped  # prose naming a class, not a call
            ):
                continue
            for name in abstract:
                if re.search(rf"(?<![\w.]){name}\(", line):
                    offenders.append(f"{path.relative_to(root)}:{lineno} {name}")
    assert offenders == [], offenders


_RENAMED_KEYS = [
    "gemm_kernel",
    "gemm_basic_kernel",
    "small_batch_kernel",
    "gemm_fp8_epilogue_kernel",
    "gemm_fp8_block_scaled_kernel",
    "gemm_w4a16_decode_kernel",
    "bmm_template_kernel",
]


@pytest.mark.parametrize("stale", _RENAMED_KEYS)
def test_a_key_no_op_declares_is_refused(stale: str) -> None:
    """A name nothing in the library has replaces nothing, so construction refuses it.

    Every key this rename retired is one: dropping it silently would hand the caller the
    shipped implementation under the name it asked to replace.
    """
    from tileops.kernels.gemm import GemmTMAKernel
    from tileops.ops import GemmFwdOp

    with pytest.raises(ValueError, match="no op has"):
        GemmFwdOp(kernel_map={stale: GemmTMAKernel})


def test_a_key_another_op_declares_passes_through() -> None:
    """A composite hands every sub-op the whole set, so a sibling's key is not an error."""
    from tileops.kernels.gemm import GemmTMAKernel
    from tileops.ops import GemmFwdOp

    op = GemmFwdOp(kernel_map={"shared_expert_mlp": GemmTMAKernel})
    assert "shared_expert_mlp" not in op.kernel_map


def test_kernel_types_declare_the_keys_an_override_may_name() -> None:
    """An override may name only a key some
    created op class declares."""
    from tileops.kernels.gemm import GemmTMAKernel
    from tileops.kernels.gemm.call_spec import GemmFwdInterface

    attrs = {
        "kernel_types": {"probe_kernel": GemmTMAKernel},
        "interfaces": {"gemm": GemmFwdInterface},
        "forward": lambda self, *a, **kw: None,
        "_infer_output_shapes": lambda self, *shapes: {},
        "eval_roofline": lambda self: (0, 0),
    }
    keyed = type("KeyedOp", (Op,), attrs)
    op = keyed()
    op.dispatch_kernel({"probe_kernel": GemmTMAKernel})
    assert op.kernel_map == {"probe_kernel": GemmTMAKernel}
    with pytest.raises(ValueError, match="no op has"):
        keyed().dispatch_kernel({"stale_kernel": GemmTMAKernel})
