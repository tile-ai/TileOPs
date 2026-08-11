"""What a backend distribution can rely on. No GPU, no tilelang.

Scoped to what a backend author or caller can observe. Internal mechanism — the lock, the
loading flag, the memo dicts — is exercised through that behaviour rather than asserted, so
it stays free to change.
"""

import subprocess
import sys
import warnings

import pytest
import torch

from tileops import backend
from tileops.backend import (
    BUILTIN,
    AmbiguousTargetError,
    BackendError,
    TensorSpec,
    UnknownTargetError,
    registry,
)
from tileops.backend.dispatch import (
    detect_target,
    registered_kernel_builder,
    select_target,
)

pytestmark = pytest.mark.smoke


@pytest.fixture(autouse=True)
def isolated_registry():
    """Give each test an empty registry and hand the real one back afterwards."""
    state = registry.snapshot()
    registry.DETECTORS.clear()
    registry.BUILDERS.clear()
    registry.LOAD_FAILURES.clear()
    registry.default_target = None
    registry._loaded = True  # no test may import the backends actually installed
    yield
    registry.restore(state)


def fake_build_kernel(*_inputs, **_params):
    """Stand in for a backend's builder. Returns a kernel that computes nothing."""
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


def test_tensor_spec_describes_a_tensor_and_compares_by_its_properties():
    """A builder is handed these instead of tensors: no data to read, none to keep."""
    spec = TensorSpec.of(torch.zeros(4, 8, dtype=torch.bfloat16))

    assert spec == (torch.device("cpu"), torch.bfloat16, (4, 8))
    assert not any(isinstance(field, torch.Tensor) for field in spec)
    assert spec == TensorSpec.of(torch.ones(4, 8, dtype=torch.bfloat16))
    assert spec != TensorSpec.of(torch.zeros(4, 9, dtype=torch.bfloat16))
    assert spec != TensorSpec.of(torch.zeros(4, 8, dtype=torch.float32))


def test_builtin_is_a_sentinel_that_outranks_a_target_claiming_the_device():
    """It says "run what ships with TileOPs", which no target name can say."""
    backend.register_kernel_builder("Op", "acme", fake_build_kernel)
    backend.register_detector("acme", lambda device: True)

    assert BUILTIN not in backend.registered_targets()
    assert select_target(BUILTIN, torch.device("cpu")) is BUILTIN

    backend.set_default_target(BUILTIN)
    assert backend.default_target() is BUILTIN
    assert select_target(None, torch.device("cpu")) is BUILTIN


# --------------------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------------------


def test_register_then_look_up():
    backend.register_kernel_builder("RMSNormFwdOp", "fake", fake_build_kernel)

    assert registered_kernel_builder("RMSNormFwdOp", "fake") is fake_build_kernel
    assert backend.registered_targets("RMSNormFwdOp") == ["fake"]
    assert backend.registered_targets() == ["fake"]


def test_a_cell_is_claimed_once():
    """A target belongs to one distribution; a second claim is a misinstall."""
    backend.register_kernel_builder("Op", "fake", fake_build_kernel)

    with pytest.raises(BackendError, match="already registered"):
        backend.register_kernel_builder("Op", "fake", lambda *a, **k: None)
    with pytest.raises(BackendError, match="misinstall"):
        backend.register_kernel_builder("Op", "fake", fake_build_kernel)


def test_a_target_has_one_detector():
    backend.register_detector("fake", lambda device: device.type == "cpu")

    with pytest.raises(BackendError, match="already has detector"):
        backend.register_detector("fake", lambda device: True)


# --------------------------------------------------------------------------------------
# Which target serves a call
# --------------------------------------------------------------------------------------


def test_the_sole_claimant_wins_and_the_device_reaches_it_untouched():
    """This layer must not interpret the device, so it does not normalize or cache it."""
    asked = []
    backend.register_detector("fake", lambda device: asked.append(device) or True)

    assert detect_target(torch.device("cpu")) == "fake"
    assert detect_target(torch.device("cpu", 0)) == "fake"
    assert asked == [torch.device("cpu"), torch.device("cpu", 0)]


def test_a_backend_installed_later_can_change_the_answer():
    """Nothing remembers an earlier answer, so a later registration simply takes effect."""
    backend.register_detector("first", lambda device: True)
    assert detect_target(torch.device("cpu")) == "first"

    backend.register_detector("second", lambda device: True)
    with pytest.raises(AmbiguousTargetError, match="pass target="):
        detect_target(torch.device("cpu"))


def test_nobody_claiming_the_device_is_the_normal_case_and_not_a_final_one():
    """It means no backend serves this hardware yet, so the in-tree implementation does."""
    backend.register_detector("early", lambda device: device.type == "xpu")

    assert detect_target(torch.device("cpu")) is None
    assert select_target(None, torch.device("cpu")) is None

    backend.register_detector("late", lambda device: device.type == "cpu")
    assert detect_target(torch.device("cpu")) == "late", "nothing cached the earlier answer"


def test_ambiguity_names_both_targets_in_a_fixed_order():
    for name in ("zeta", "alpha"):
        backend.register_detector(name, lambda device: True)

    with pytest.raises(AmbiguousTargetError, match=r"\['alpha', 'zeta'\]"):
        detect_target(torch.device("cpu"))


@pytest.mark.parametrize(
    ("requested", "default", "expected"),
    [
        ("named", "configured", "named"),
        (None, "configured", "configured"),
        (None, None, "detected"),
    ],
    ids=["explicit wins", "default beats detection", "detection is the fallback"],
)
def test_target_precedence(requested, default, expected):
    for target in ("named", "configured"):
        backend.register_kernel_builder("Op", target, fake_build_kernel)
    backend.register_detector("detected", lambda device: True)
    backend.set_default_target(default)

    assert select_target(requested, torch.device("cpu")) == expected


def test_a_named_target_is_taken_at_its_word():
    """Naming a target is how a caller overrides detection, so nothing is re-checked."""
    backend.register_kernel_builder("Op", "named", fake_build_kernel)
    backend.register_detector("detected", lambda device: pytest.fail("must not detect"))

    assert select_target("named", torch.device("cpu")) == "named"
    assert select_target("named", None) == "named"


def test_a_named_target_nobody_registered_is_an_error():
    with pytest.raises(UnknownTargetError, match="no backend registered target 'nope'"):
        select_target("nope", torch.device("cpu"))


def test_a_call_with_no_tensor_input_runs_the_in_tree_implementation():
    """With no tensor there is no device, and the in-tree implementation is always there."""
    backend.register_detector("detected", lambda device: True)

    assert select_target(None, None) is None


def test_the_default_target_starts_unset_and_must_name_a_real_target():
    assert backend.default_target() is None
    with pytest.raises(UnknownTargetError, match="no backend registered target"):
        backend.set_default_target("nope")

    backend.register_kernel_builder("Op", "fake", fake_build_kernel)
    backend.set_default_target("fake")
    assert backend.default_target() == "fake"

    backend.set_default_target(None)
    assert backend.default_target() is None, "None restores detection"


def test_a_target_with_builders_but_no_detector_is_still_reachable():
    """Such a target never wins by detection, which does not make it unusable."""
    backend.register_kernel_builder("Op", "explicit_only", fake_build_kernel)

    backend.set_default_target("explicit_only")
    assert registered_kernel_builder("Op", "explicit_only") is fake_build_kernel


def test_a_missing_op_reads_as_missing_and_says_who_does_have_it():
    """The op layer turns this into an error; this layer only reports the absence."""
    backend.register_kernel_builder("GQAPrefillFwdOp", "nv", fake_build_kernel)

    assert registered_kernel_builder("GQAPrefillFwdOp", "other") is None
    assert backend.registered_targets("GQAPrefillFwdOp") == ["nv"]


# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------


def test_installing_a_backend_is_enough_to_be_dispatched_to(installed):
    """The entry point names a module; importing it registers."""

    def acme():
        backend.register_detector("acme", lambda device: device.type == "cpu")
        backend.register_kernel_builder("RMSNormFwdOp", "acme", fake_build_kernel)

    installed(acme=acme)

    assert select_target(None, torch.device("cpu")) == "acme"
    assert registered_kernel_builder("RMSNormFwdOp", "acme") is fake_build_kernel


def test_a_backend_is_imported_once_however_dispatch_is_entered(installed):
    loads = []
    installed(once=lambda: loads.append(1))

    backend.registered_targets()
    backend.load_failures()
    detect_target(torch.device("cpu"))
    select_target(None, None)

    assert loads == [1]


def test_a_broken_backend_is_skipped_and_stays_visible(installed):
    installed(
        broken=lambda: _raise(ImportError("libacme.so not found")),
        working=lambda: backend.register_kernel_builder("Op", "working", fake_build_kernel),
    )

    with pytest.warns(RuntimeWarning, match="failed to load"):
        assert backend.registered_targets("Op") == ["working"]
    assert backend.load_failures() == (
        "broken (tileops_broken): ImportError: libacme.so not found",
    )


def test_backends_load_in_a_fixed_order(installed):
    """Same installed packages, same records and same warnings, every run."""
    installed(
        zeta=lambda: _raise(ImportError("z")),
        alpha=lambda: _raise(ImportError("a")),
    )

    with pytest.warns(RuntimeWarning):
        failures = backend.load_failures()

    assert [line.split()[0] for line in failures] == ["alpha", "zeta"]


def test_a_backend_that_fails_midway_registers_nothing(installed):
    """A half-registered backend would advertise ops its distribution never finished."""

    def half():
        backend.register_detector("half", lambda device: True)
        backend.register_kernel_builder("First", "half", fake_build_kernel)
        raise RuntimeError("failed after registering two things")

    installed(
        half=half,
        intact=lambda: backend.register_kernel_builder("Kept", "intact", fake_build_kernel),
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
        # The same cell twice: only a real rollback lets the retry claim it.
        backend.register_kernel_builder("Op", "rude", fake_build_kernel)
        if len(attempts) == 1:
            raise interrupt

    installed(rude=rude)

    with pytest.raises(interrupt):
        backend.registered_targets()
    assert backend.registered_targets("Op") == ["rude"], "asking again tries again"
    assert attempts == [1, 1]


def test_warnings_as_errors_cannot_truncate_discovery(installed):
    """CI runs with -W error, where warning mid-loop would strand every later backend."""
    installed(
        broken=lambda: _raise(ZeroDivisionError()),
        working=lambda: backend.register_kernel_builder("Op", "working", fake_build_kernel),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning):
            backend.registered_targets()

    assert backend.registered_targets("Op") == ["working"]
    assert [line.split()[0] for line in backend.load_failures()] == ["broken"]


def test_a_backend_may_read_the_registry_while_it_is_registering(installed):
    """A backend module imports tileops at its top level, so this must not recurse."""

    def introspective():
        backend.register_kernel_builder("Op", "introspective", fake_build_kernel)
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
        detect_target(torch.device("cpu"))
    assert "vendor runtime missing" in str(excinfo.value)


def test_every_error_can_still_blame_a_wheel_that_failed_to_load(installed):
    """Otherwise a broken install presents itself as a detector or a target misbehaving."""
    installed(
        wont_import=lambda: _raise(ImportError("libacme.so not found")),
        rude_detector=lambda: backend.register_detector(
            "rude", lambda device: _raise(RuntimeError("vendor runtime missing"))
        ),
    )

    with (pytest.warns(RuntimeWarning),
          pytest.raises(BackendError, match="detector for target 'rude'") as excinfo):
        detect_target(torch.device("cpu"))
    assert "1 backend(s) failed to load" in str(excinfo.value)


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
    assert tileops.BUILTIN is BUILTIN
    assert "registered_targets" in dir(tileops)
    with pytest.raises(AttributeError, match="has no attribute 'nope'"):
        _ = tileops.nope
