"""What a backend distribution can rely on. No GPU, no tilelang.

Scoped to behaviour a backend author or caller can observe: what registration accepts and
refuses, which target serves a call, what the errors say, and what a broken or half-broken
distribution does to the rest. Internal mechanism — the lock, the loading flag, the memo
dicts — is exercised through that behaviour rather than asserted directly, so it stays free
to change.
"""

import subprocess
import sys
import warnings

import pytest
import torch

from tileops import backend
from tileops.backend import (
    AmbiguousTargetError,
    BackendError,
    InputSpec,
    OpNotAvailableError,
    UnknownTargetError,
    registry,
)

pytestmark = pytest.mark.smoke


@pytest.fixture(autouse=True)
def isolated_registry():
    """Give each test an empty registry and hand the real one back afterwards."""
    state = registry.snapshot()
    registry.DETECTORS.clear()
    registry.KERNELS.clear()
    registry.RESOLVED.clear()
    registry.LOAD_ERRORS.clear()
    registry.default_target = None
    registry._loaded = True  # no test may import the backends actually installed
    yield
    registry.restore(state)


def fake_get_kernel(*_inputs, **_params):
    """Stand in for callback two. Returns a kernel that only writes into its inputs."""
    return lambda *_tensors: None


class _EntryPoint:
    """An installed backend, as importlib.metadata would report it."""

    def __init__(self, name, load):
        self.name = name
        self.value = f"tileops_{name}"
        self.load = load


@pytest.fixture
def installed(monkeypatch):
    """Declare which backends this test should discover."""

    def declare(**loaders):
        eps = [_EntryPoint(name, load) for name, load in loaders.items()]
        monkeypatch.setattr(registry, "entry_points", lambda group: eps)
        registry._loaded = False

    return declare


def _raise(exc):
    """Raise from inside a lambda, for backends and detectors that misbehave."""
    raise exc


# --------------------------------------------------------------------------------------
# What crosses the boundary
# --------------------------------------------------------------------------------------


def test_input_spec_describes_a_tensor_and_compares_by_its_properties():
    """It is handed to get_kernel and used as the op layer's memo key, so both matter."""
    spec = InputSpec.of(torch.zeros(4, 8, dtype=torch.bfloat16))

    assert spec == (torch.device("cpu"), torch.bfloat16, (4, 8))
    assert not any(isinstance(field, torch.Tensor) for field in spec)
    assert spec == InputSpec.of(torch.ones(4, 8, dtype=torch.bfloat16))
    assert spec != InputSpec.of(torch.zeros(4, 9, dtype=torch.bfloat16))
    assert spec != InputSpec.of(torch.zeros(4, 8, dtype=torch.float32))


# --------------------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------------------


def test_register_then_look_up():
    backend.register("RMSNormFwdOp", "fake", fake_get_kernel)

    assert backend.get_kernel_for("RMSNormFwdOp", "fake") is fake_get_kernel
    assert backend.registered() == frozenset({("RMSNormFwdOp", "fake")})
    assert backend.registered_targets("RMSNormFwdOp") == ["fake"]


def test_the_same_callable_may_register_twice_but_a_rival_may_not():
    """Two distributions claiming one cell is a conflict; one module seen twice is not."""
    backend.register("Op", "fake", fake_get_kernel)
    backend.register("Op", "fake", fake_get_kernel)

    with pytest.raises(BackendError, match="already registered"):
        backend.register("Op", "fake", lambda *a, **k: None)


def test_a_target_may_not_have_two_detectors():
    def detect(device):
        return device.type == "cpu"

    backend.register_detector("fake", detect)
    backend.register_detector("fake", detect)

    with pytest.raises(BackendError, match="already has detector"):
        backend.register_detector("fake", lambda device: True)


# --------------------------------------------------------------------------------------
# Which target serves a call
# --------------------------------------------------------------------------------------


def test_the_sole_claimant_wins_and_is_asked_once_per_device():
    asked = []
    backend.register_detector("fake", lambda device: asked.append(device) or True)

    assert backend.resolve_target(torch.device("cpu")) == "fake"
    assert backend.resolve_target(torch.device("cpu")) == "fake"
    assert asked == [torch.device("cpu")], "detection is memoized"


def test_a_backend_installed_later_can_change_the_answer():
    """A memo must not outlive the table it was computed from."""
    backend.register_detector("first", lambda device: True)
    assert backend.resolve_target(torch.device("cpu")) == "first"

    backend.register_detector("second", lambda device: True)
    with pytest.raises(AmbiguousTargetError, match="pass target="):
        backend.resolve_target(torch.device("cpu"))


def test_no_claimant_names_the_targets_that_do_exist():
    backend.register_detector("fake", lambda device: device.type == "xpu")

    with pytest.raises(UnknownTargetError, match=r"no registered target claims.*'fake'"):
        backend.resolve_target(torch.device("cpu"))


def test_the_device_reaches_the_detector_untouched():
    """This layer must not interpret the device, so it does not normalize it either."""
    seen = []
    backend.register_detector("fake", lambda device: seen.append(device) or True)

    backend.resolve_target(torch.device("cpu"))
    backend.resolve_target(torch.device("cpu", 0))
    assert seen == [torch.device("cpu"), torch.device("cpu", 0)]


@pytest.mark.parametrize(
    ("explicit", "default", "expected"),
    [
        ("named", "configured", "named"),
        (None, "configured", "configured"),
        (None, None, "detected"),
    ],
    ids=["explicit wins", "default beats detection", "detection is the fallback"],
)
def test_target_precedence(explicit, default, expected):
    for target in ("named", "configured"):
        backend.register("Op", target, fake_get_kernel)
    backend.register_detector("detected", lambda device: True)
    backend.set_default_target(default)

    assert backend.select_target(explicit, torch.device("cpu")) == expected


def test_a_named_target_is_taken_at_its_word():
    """Naming a target is how a caller overrides detection, so nothing is re-checked."""
    backend.register_detector("detected", lambda device: pytest.fail("must not detect"))

    assert backend.select_target("named", torch.device("cpu")) == "named"
    assert backend.select_target("named", None) == "named"


def test_a_call_with_no_tensor_input_must_name_its_target():
    """With no tensor there is no device, so detection has nothing to work from."""
    backend.register_detector("detected", lambda device: True)

    with pytest.raises(UnknownTargetError, match="no tensor input"):
        backend.select_target(None, None)


def test_the_default_target_starts_unset_and_must_name_a_real_target():
    assert backend.default_target() is None
    with pytest.raises(UnknownTargetError, match="no backend registered target"):
        backend.set_default_target("nope")

    backend.register("Op", "fake", fake_get_kernel)
    backend.set_default_target("fake")
    assert backend.default_target() == "fake"


def test_a_target_with_kernels_but_no_detector_is_still_reachable():
    """Such a target never wins by detection, which does not make it unusable."""
    backend.register("Op", "explicit_only", fake_get_kernel)

    backend.set_default_target("explicit_only")
    assert backend.get_kernel_for("Op", "explicit_only") is fake_get_kernel


def test_a_missing_op_never_falls_back_and_says_who_does_have_it():
    backend.register("GQAPrefillFwdOp", "nv", fake_get_kernel)

    with pytest.raises(OpNotAvailableError, match=r"'other'.*this op: \['nv'\]"):
        backend.get_kernel_for("GQAPrefillFwdOp", "other")


# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------


def test_installing_a_backend_is_enough_to_be_dispatched_to(installed):
    """The entry point names a module; importing it registers. No protocol object."""

    def acme():
        backend.register_detector("acme", lambda device: device.type == "cpu")
        backend.register("RMSNormFwdOp", "acme", fake_get_kernel)

    installed(acme=acme)

    assert backend.select_target(None, torch.device("cpu")) == "acme"
    assert backend.get_kernel_for("RMSNormFwdOp", "acme") is fake_get_kernel


def test_a_backend_is_imported_once_however_dispatch_is_entered(installed):
    loads = []
    installed(once=lambda: loads.append(1))

    backend.registered()
    backend.registered_targets()
    backend.load_errors()
    with pytest.raises(UnknownTargetError):
        backend.resolve_target(torch.device("cpu"))

    assert loads == [1]


def test_a_broken_backend_is_skipped_and_stays_visible(installed):
    installed(
        broken=lambda: _raise(ImportError("libacme.so not found")),
        working=lambda: backend.register("Op", "working", fake_get_kernel),
    )

    with pytest.warns(RuntimeWarning, match="failed to load"):
        assert backend.registered_targets("Op") == ["working"]
    assert [str(e) for e in backend.load_errors()] == [
        "broken (tileops_broken): ImportError: libacme.so not found"
    ]


def test_errors_point_at_a_broken_backend_rather_than_blame_the_device(installed):
    """Otherwise a wheel that failed to load looks like a device nobody supports."""
    installed(broken=lambda: _raise(OSError("boom")))

    with pytest.warns(RuntimeWarning), pytest.raises(UnknownTargetError, match="failed"):
        backend.resolve_target(torch.device("cpu"))


def test_a_backend_that_fails_midway_registers_nothing(installed):
    """A half-registered backend would advertise ops its distribution never finished."""

    def half():
        backend.register_detector("half", lambda device: True)
        backend.register("First", "half", fake_get_kernel)
        raise RuntimeError("failed after registering two things")

    installed(
        half=half,
        intact=lambda: backend.register("Kept", "intact", fake_get_kernel),
    )

    with pytest.warns(RuntimeWarning):
        assert backend.registered_targets("Kept") == ["intact"]
    assert backend.registered_targets("First") == []
    assert "half" not in backend.registered_targets()


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_an_interrupt_reaches_the_caller_and_leaves_discovery_retryable(installed, interrupt):
    """Ctrl-C on the first dispatch must not strand the registry half-populated."""
    attempts = []

    def rude():
        attempts.append(1)
        backend.register("Op", f"attempt{len(attempts)}", fake_get_kernel)
        if len(attempts) == 1:
            raise interrupt

    installed(rude=rude)

    with pytest.raises(interrupt):
        backend.registered()
    assert backend.registered_targets("Op") == ["attempt2"], "asking again tries again"


def test_warnings_as_errors_cannot_truncate_discovery(installed):
    """CI runs with -W error, where warning mid-loop would strand every later backend."""
    installed(
        broken=lambda: _raise(ZeroDivisionError()),
        working=lambda: backend.register("Op", "working", fake_get_kernel),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning):
            backend.registered()

    assert backend.registered_targets("Op") == ["working"]
    assert [e.name for e in backend.load_errors()] == ["broken"]


def test_a_backend_may_read_the_registry_while_it_is_registering(installed):
    """A backend module imports tileops at its top level, so this must not recurse."""

    def introspective():
        backend.register("Op", "introspective", fake_get_kernel)
        assert backend.registered_targets("Op") == ["introspective"]

    installed(introspective=introspective)

    assert backend.registered_targets("Op") == ["introspective"]


def test_a_detector_that_raises_names_the_target_that_owns_it(installed):
    """A detector must answer. Guessing past a broken one would hand the device away."""
    installed(
        broken=lambda: backend.register_detector(
            "broken", lambda device: _raise(RuntimeError("vendor runtime missing"))
        )
    )

    with pytest.raises(BackendError, match="detector for target 'broken'") as excinfo:
        backend.resolve_target(torch.device("cpu"))
    assert "vendor runtime missing" in str(excinfo.value)


# --------------------------------------------------------------------------------------
# The packaging boundary
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "why"),
    [
        (
            "import sys, tileops.backend; print('tilelang' in sys.modules)",
            "the module a backend imports must not drag the nv toolchain in with it",
        ),
        (
            "import sys, tileops; print('torch' in sys.modules)",
            "the manifest tooling reads YAML where torch is not installed",
        ),
        (
            "import tileops.backend as b; print(b.registry._loaded)",
            "installing a backend costs nothing until something asks for it",
        ),
    ],
    ids=["no tilelang", "no torch", "no discovery"],
)
def test_importing_costs_nothing(source, why):
    """Run in a fresh interpreter, since pytest has already imported everything."""
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", why


def test_the_caller_api_is_reachable_from_the_package_root():
    """Lazy re-exports must not mean absent ones."""
    import tileops

    assert tileops.set_default_target is backend.set_default_target
    assert "registered_targets" in dir(tileops)
    with pytest.raises(AttributeError, match="has no attribute 'nope'"):
        _ = tileops.nope
