"""Tests for tileops.ops.op_base.

Covers ``Op._cache_key`` default behavior, the runtime warning fired when
a subclass with empty ``_static_axes`` does not override ``_cache_key``,
composite kernel-map overrides, the ``get_or_build_kernel`` primitive, and
the explicit kernel enumeration ``Op.autotune`` runs over.
"""

import dataclasses
import warnings

import pytest
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.ops import op_base
from tileops.ops.op_base import Op

pytestmark = pytest.mark.smoke


class _RecordingKernel(Kernel):
    """Kernel that records its name when tuned, so autotune order is visible."""

    def __init__(self, name: str, tuned: list):
        super().__init__()
        self.name = name
        self._tuned = tuned

    def forward(self):
        return None

    def autotune(self, warmup=25, rep=50):
        self._tuned.append(self.name)


@pytest.fixture(autouse=True)
def _reset_warned_types():
    """Clear the module-level dedup set so each test sees a fresh warn state."""
    op_base._EMPTY_STATIC_DIMS_WARNED.clear()
    yield
    op_base._EMPTY_STATIC_DIMS_WARNED.clear()


def _make_op_subclass(*, static_axes=frozenset(), override_cache_key=False):
    """Build a minimal concrete Op subclass for testing.

    ``static_axes`` populates ``_static_axes``.
    ``override_cache_key=True`` attaches a subclass-level override.
    """
    attrs = {
        "_static_axes": static_axes,
        "default_kernel_map": property(lambda self: {}),
        "forward": lambda self, *a, **kw: None,
        # The three manifest-driven methods are abstract on Op; these doubles
        # exercise the get-or-build plumbing, so a minimal body is the contract.
        "_infer_output_shapes": lambda self, *shapes: {},
        "_validate_dtypes": lambda self, *args: None,
        "eval_roofline": lambda self: (0, 0),
    }
    if override_cache_key:
        attrs["_cache_key"] = lambda self, *shapes: ("overridden",)
    return type("TestOp", (Op,), attrs)


class TestCacheKeyDefault:
    def test_static_axes_exclude_single_input(self):
        """_static_axes=[(0,1)] on a 3D input excludes axis 1 from the key."""
        Cls = _make_op_subclass(static_axes=frozenset({(0, 1)}))
        op = Cls()
        key = op._cache_key((2, 4, 8))
        assert key == (2, 8)

    def test_static_axes_across_multiple_inputs(self):
        """_static_axes can reference axes in different input positions."""
        Cls = _make_op_subclass(static_axes=frozenset({(0, 1), (1, 0)}))
        op = Cls()
        key = op._cache_key((2, 4, 8), (16, 32))
        # Input 0: exclude axis 1 -> (2, 8); Input 1: exclude axis 0 -> (32,)
        assert key == (2, 8, 32)

    def test_empty_static_axes_returns_full_shape(self):
        """With no static axes, the key concatenates all input shape values."""
        Cls = _make_op_subclass(static_axes=frozenset())
        op = Cls()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # warning tested separately
            key = op._cache_key((2, 4, 8), (3, 5))
        assert key == (2, 4, 8, 3, 5)


class TestCacheKeyWarning:
    def test_empty_static_axes_warns_once_per_type(self):
        """Default path with empty _static_axes warns exactly once per subclass,
        even across multiple instances and repeated calls."""
        Cls = _make_op_subclass(static_axes=frozenset())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Cls()._cache_key((2, 4))
            Cls()._cache_key((3, 5))
            Cls()._cache_key((7, 9))
            Cls()._cache_key((11, 13))

        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "TestOp" in str(user_warnings[0].message)
        assert "_cache_key" in str(user_warnings[0].message)

    def test_override_suppresses_warning(self):
        """When the subclass overrides _cache_key, no warning fires."""
        Cls = _make_op_subclass(static_axes=frozenset(), override_cache_key=True)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = Cls()._cache_key((2, 4))

        assert result == ("overridden",)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert user_warnings == []

    def test_populated_static_axes_suppresses_warning(self):
        """Non-empty _static_axes means the user committed at ctor; no warning
        fires regardless of whether _cache_key was overridden."""
        Cls = _make_op_subclass(static_axes=frozenset({(0, 0)}))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Cls()._cache_key((2, 4))

        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert user_warnings == []

    def test_distinct_subclasses_each_warn_once(self):
        """Two different subclasses each warn once; the dedup set is keyed by
        type, not globally suppressed after the first warning."""
        ClsA = _make_op_subclass(static_axes=frozenset())
        ClsB = _make_op_subclass(static_axes=frozenset())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ClsA()._cache_key((1,))
            ClsA()._cache_key((2,))  # no re-warn for A
            ClsB()._cache_key((3,))  # fresh warn for B
            ClsB()._cache_key((4,))  # no re-warn for B

        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 2


class TestCompositeKernelMapOverride:
    """Composite ops (empty ``default_kernel_map``) accept a non-empty override and store it verbatim."""

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
    """Op whose forward-built kernels all go through ``get_or_build_kernel``."""

    def __init__(self, tuned: list):
        self._tuned = tuned
        self.builds: list[tuple[str, object]] = []

    @property
    def default_kernel_map(self):
        return {}

    def _infer_output_shapes(self, *shapes):
        return {}

    def _validate_dtypes(self, *args):
        return None

    def eval_roofline(self):
        return (0, 0)

    def forward(self, *a, **kw):
        return None

    def build(self, role: str, key, name: str):
        def factory():
            self.builds.append((role, key))
            return _RecordingKernel(name, self._tuned)

        return self.get_or_build_kernel(role, (), key=key, build=factory)


class TestGetOrBuildKernel:
    """``Op.get_or_build_kernel`` is the single get-or-build in the Op layer."""

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
        assert set(op.built_kernels("fwd")) == {torch.float16, torch.bfloat16}

    def test_same_key_in_distinct_roles_does_not_collide(self):
        """An auxiliary kernel keyed by the same dtype is a second role."""
        op = _SlottedOp([])
        attention = op.build("attention", torch.float16, "attention")
        append = op.build("append", torch.float16, "append")
        assert attention is not append
        assert list(op.built_kernels("append")) == [torch.float16]

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

        class BundleOp(_SlottedOp):
            def populate(self):
                self.get_or_build_kernel(
                    "pair",
                    (),
                    key=torch.float16,
                    build=lambda: (_RecordingKernel("pre", tuned), _RecordingKernel("bwd", tuned)),
                )
                self.get_or_build_kernel(
                    "entry",
                    (),
                    key=torch.bfloat16,
                    build=lambda: Entry(_RecordingKernel("record", tuned), torch.float32),
                )

        op = BundleOp(tuned)
        op.populate()
        assert sorted(k.name for k in op.iter_kernels()) == ["bwd", "pre", "record"]

    def test_yields_the_directly_bound_kernel(self):
        tuned: list[str] = []
        op = _SlottedOp(tuned)
        op.kernel = _RecordingKernel("bound", tuned)
        assert [k.name for k in op.iter_kernels()] == ["bound"]

    def test_ignores_kernels_bound_to_other_attributes(self):
        """Enumeration is explicit: an unregistered attribute is not searched."""
        tuned: list[str] = []
        op = _SlottedOp(tuned)
        op.some_other_attribute = _RecordingKernel("hidden", tuned)
        assert list(op.iter_kernels()) == []

    def test_ignores_a_kernel_dict_the_op_owns(self):
        """A private dict of kernels is unreachable — the miss reflection used to hide.

        The old ``dir(self)`` walk descended dict values, so an op could keep its
        own cache and still be tuned. Enumeration does not, which is the point:
        the kernels have to be built through a role to be seen at all.
        """
        tuned: list[str] = []
        op = _SlottedOp(tuned)
        op.private_cache = {torch.float16: _RecordingKernel("private", tuned)}
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
            def kernel_delegates(self):
                return (delegate,)

        composite = CompositeOp(tuned)
        composite.build("own", torch.float16, "own")
        assert sorted(k.name for k in composite.iter_kernels()) == ["delegate", "own"]

    def test_deduplicates_a_kernel_shared_with_a_delegate(self):
        """A composite that caches the kernel its delegate built tunes it once."""
        tuned: list[str] = []
        delegate = _SlottedOp(tuned)
        shared = delegate.build("fwd", torch.float16, "shared")

        class CompositeOp(_SlottedOp):
            def kernel_delegates(self):
                return (delegate,)

        composite = CompositeOp(tuned)
        composite.get_or_build_kernel("fwd", (), key=torch.float16, build=lambda: shared)
        assert [k.name for k in composite.iter_kernels()] == ["shared"]


class TestAutotune:
    """``Op.autotune`` tunes exactly what ``iter_kernels`` yields."""

    def test_autotune_tunes_the_bound_kernel_and_every_role_entry(self):
        tuned: list[str] = []
        op = _SlottedOp(tuned)
        op.kernel = _RecordingKernel("bound", tuned)
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
            def kernel_delegates(self):
                return (delegate,)

        CompositeOp(tuned).autotune()
        assert tuned == ["delegate"]


class _TunableOp(Op):
    """Op whose factory honours ``self.tune``, the way a shipped op's does.

    Mirrors the call sites: the flag is read when the factory runs and handed
    to the kernel, which tunes itself at construction.
    """

    def __init__(self, tuned: list, *, tune: bool = False):
        self._tuned = tuned
        self.tune = tune

    @property
    def default_kernel_map(self):
        return {}

    def forward(self, *a, **kw):
        return None

    def _infer_output_shapes(self, *shapes):
        return {}

    def _validate_dtypes(self, *args):
        return None

    def eval_roofline(self):
        return (0, 0)

    def build(self, dtype):
        def factory():
            kernel = _RecordingKernel(str(dtype), self._tuned)
            if self.tune:
                kernel.autotune()
            return kernel

        return self.get_or_build_kernel("fwd", (), key=dtype, build=factory)


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

    def test_a_delegate_built_after_autotune_inherits_tuned_mode(self):
        """The composite passes its own flag on, so the decision carries."""
        tuned: list[str] = []

        class CompositeOp(_TunableOp):
            def __init__(self, rec):
                super().__init__(rec)
                self.delegate = None

            def kernel_delegates(self):
                return (self.delegate,) if self.delegate else ()

            def make_delegate(self):
                self.delegate = _TunableOp(self._tuned, tune=self.tune)
                return self.delegate

        op = CompositeOp(tuned)
        op.autotune()
        op.make_delegate().build(torch.float16)
        assert tuned == ["torch.float16"]


class TestInstanceKeys:
    def test_a_collected_instances_key_is_never_handed_out_again(self):
        """An op reaching a used key inherits that op's compiled shapes."""
        import gc

        class _Dummy:
            pass

        keys = set()
        for _ in range(50):
            op = _Dummy()
            keys.add(op_base.register_instance(op))
            del op
            gc.collect()

        assert len(keys) == 50

    def test_a_key_names_the_class_it_belongs_to(self):
        """Graph dumps and guard failures show the key, not the instance."""

        class _Dummy:
            pass

        assert op_base.register_instance(_Dummy()).startswith("_Dummy")


def test_no_abstract_op_class_is_instantiated_anywhere():
    """An abstract Op cannot be constructed, so nothing in the tree may try.

    A class is abstract when it does not answer the manifest-driven contract:
    ``_infer_output_shapes``, ``_validate_dtypes``, ``eval_roofline``. Those are
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
