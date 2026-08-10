"""The backend registry's rules, verified without tilelang or a GPU.

Every test registers fake targets, so the fixture below puts the registry back
afterwards: without it one test's fake backend changes how every later test behaves and
the suite's outcome depends on its order.
"""

import itertools
import subprocess
import sys
import threading
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
)

pytestmark = pytest.mark.smoke


@pytest.fixture(autouse=True)
def isolated_registry():
    """Give each test the registry it started with."""
    state = backend._snapshot()
    # Pretend discovery already happened: no test should import installed backends.
    backend._DETECTORS.clear()
    backend._KERNELS.clear()
    backend._RESOLVED.clear()
    backend._LOAD_ERRORS.clear()
    backend._DEFAULT_TARGET = None
    backend._loaded = True
    yield backend
    backend._restore(state)


def _kernel(*_tensors):
    """A kernel that returns nothing, standing in for a mutating op."""
    return None


def fake_get_kernel(*_inputs, **_params):
    return _kernel


# --------------------------------------------------------------------------------------
# The description that crosses the boundary
# --------------------------------------------------------------------------------------


def test_input_spec_describes_a_tensor_without_holding_it():
    t = torch.zeros(2, 3, dtype=torch.float16)
    spec = InputSpec.of(t)
    assert (spec.device, spec.dtype, spec.shape) == (t.device, torch.float16, (2, 3))
    assert not any(isinstance(field, torch.Tensor) for field in spec)


def test_input_spec_equality_is_the_memo_key():
    a = torch.zeros(4, 8, dtype=torch.bfloat16)
    assert InputSpec.of(a) == InputSpec.of(torch.ones(4, 8, dtype=torch.bfloat16))
    assert InputSpec.of(a) != InputSpec.of(torch.zeros(4, 9, dtype=torch.bfloat16))
    assert InputSpec.of(a) != InputSpec.of(torch.zeros(4, 8, dtype=torch.float32))
    assert hash(InputSpec.of(a))  # usable as a dict key


# --------------------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------------------


def test_register_then_look_up():
    backend.register("RMSNormFwdOp", "fake", fake_get_kernel)
    assert backend.get_kernel_for("RMSNormFwdOp", "fake") is fake_get_kernel
    assert ("RMSNormFwdOp", "fake") in backend.registered()


def test_registering_the_same_callable_twice_is_a_no_op():
    backend.register("Op", "fake", fake_get_kernel)
    backend.register("Op", "fake", fake_get_kernel)  # module imported twice
    assert backend.registered_targets("Op") == ["fake"]


def test_two_distributions_claiming_one_cell_is_an_error():
    backend.register("Op", "fake", fake_get_kernel)
    with pytest.raises(BackendError, match="already registered"):
        backend.register("Op", "fake", lambda *a, **k: _kernel)


def test_detector_conflict_is_an_error_but_re_registration_is_not():
    detect = lambda device: device.type == "cpu"  # noqa: E731
    backend.register_detector("fake", detect)
    backend.register_detector("fake", detect)
    with pytest.raises(BackendError, match="already has detector"):
        backend.register_detector("fake", lambda device: True)


def test_registered_exposes_keys_but_not_the_callbacks():
    """One lookup path only: reaching a get_kernel goes through get_kernel_for."""
    backend.register("Op", "fake", fake_get_kernel)
    assert backend.registered() == frozenset({("Op", "fake")})
    assert not hasattr(backend.registered(), "values")


def test_registered_targets_lists_per_op():
    backend.register("Op", "a", fake_get_kernel)
    backend.register("Op", "b", fake_get_kernel)
    backend.register("Other", "c", fake_get_kernel)
    assert backend.registered_targets("Op") == ["a", "b"]
    assert backend.registered_targets("Missing") == []


# --------------------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------------------


def test_sole_claimant_wins_and_is_cached():
    calls = []

    def detect(device):
        calls.append(device)
        return device.type == "cpu"

    backend.register_detector("fake", detect)
    assert backend.resolve_target(torch.device("cpu")) == "fake"
    assert backend.resolve_target(torch.device("cpu")) == "fake"
    assert len(calls) == 1, "detection must be paid once per device"


def test_nothing_claims_the_device():
    backend.register_detector("fake", lambda device: device.type == "xpu")
    with pytest.raises(UnknownTargetError, match="no registered target claims"):
        backend.resolve_target(torch.device("cpu"))


def test_two_detectors_claiming_one_device():
    backend.register_detector("a", lambda device: True)
    backend.register_detector("b", lambda device: True)
    with pytest.raises(AmbiguousTargetError, match="pass target="):
        backend.resolve_target(torch.device("cpu"))


def test_a_new_detector_invalidates_a_cached_answer():
    backend.register_detector("a", lambda device: True)
    assert backend.resolve_target(torch.device("cpu")) == "a"
    backend.register_detector("b", lambda device: True)
    with pytest.raises(AmbiguousTargetError):
        backend.resolve_target(torch.device("cpu"))


def test_the_detector_sees_the_device_untouched():
    """This layer must not interpret the device, so it does not normalize it either."""
    calls = []

    def detect(device):
        calls.append(device)
        return device.type == "cpu"

    backend.register_detector("fake", detect)
    assert backend.resolve_target(torch.device("cpu")) == "fake"
    assert backend.resolve_target(torch.device("cpu", 0)) == "fake"
    assert calls == [torch.device("cpu"), torch.device("cpu", 0)]


# --------------------------------------------------------------------------------------
# Lookup and its errors
# --------------------------------------------------------------------------------------


def test_missing_cell_names_the_targets_that_do_have_it():
    backend.register("GQAPrefillFwdOp", "nv", fake_get_kernel)
    with pytest.raises(OpNotAvailableError) as excinfo:
        backend.get_kernel_for("GQAPrefillFwdOp", "ascend")
    assert "'nv'" in str(excinfo.value)


def test_lookup_never_falls_back_to_another_target():
    backend.register("Op", "nv", fake_get_kernel)
    with pytest.raises(OpNotAvailableError):
        backend.get_kernel_for("Op", "musa")


def test_every_error_is_a_backend_error():
    assert issubclass(UnknownTargetError, BackendError)
    assert issubclass(AmbiguousTargetError, BackendError)
    assert issubclass(OpNotAvailableError, BackendError)


# --------------------------------------------------------------------------------------
# The process-wide default
# --------------------------------------------------------------------------------------


def test_default_target_starts_unset_so_detection_is_the_normal_path():
    assert backend.default_target() is None


def test_set_default_target_round_trips_and_clears():
    backend.register("Op", "fake", fake_get_kernel)
    backend.set_default_target("fake")
    assert backend.default_target() == "fake"
    backend.set_default_target(None)
    assert backend.default_target() is None


def test_set_default_target_rejects_an_unregistered_target():
    with pytest.raises(UnknownTargetError, match="no backend registered target"):
        backend.set_default_target("nope")


def test_a_target_with_kernels_but_no_detector_is_still_selectable():
    backend.register("Op", "explicit_only", fake_get_kernel)
    backend.set_default_target("explicit_only")
    assert backend.get_kernel_for("Op", "explicit_only") is fake_get_kernel


def test_explicit_target_beats_the_default_and_detection():
    backend.register("Op", "explicit", fake_get_kernel)
    backend.register("Op", "configured", fake_get_kernel)
    backend.register_detector("detected", lambda device: True)
    backend.set_default_target("configured")
    assert backend.select_target("explicit", torch.device("cpu")) == "explicit"


def test_the_default_beats_detection():
    backend.register("Op", "configured", fake_get_kernel)
    backend.register_detector("detected", lambda device: True)
    backend.set_default_target("configured")
    assert backend.select_target(None, torch.device("cpu")) == "configured"


def test_detection_decides_when_nothing_was_named():
    backend.register_detector("detected", lambda device: True)
    assert backend.select_target(None, torch.device("cpu")) == "detected"


def test_no_tensor_input_means_target_must_be_named():
    """With no tensor there is no device, so detection has nothing to work from."""
    backend.register_detector("detected", lambda device: True)
    with pytest.raises(UnknownTargetError, match="no tensor input"):
        backend.select_target(None, None)
    assert backend.select_target("named", None) == "named"


def test_an_explicit_target_is_not_checked_against_the_device():
    """Naming a target is how a caller overrides detection, so detection is skipped."""
    backend.register_detector("detected", lambda device: pytest.fail("must not detect"))
    assert backend.select_target("named", torch.device("cpu")) == "named"


# --------------------------------------------------------------------------------------
# Discovery: a broken backend must stay visible without taking TileOPs down
# --------------------------------------------------------------------------------------


class _FakeEntryPoint:
    def __init__(self, name, value, loader):
        self.name = name
        self.value = value
        self._loader = loader

    def load(self):
        return self._loader()


@pytest.fixture
def entry_points(monkeypatch):
    """Let a test declare the installed backends."""

    def declare(*eps):
        monkeypatch.setattr(backend, "entry_points", lambda group: list(eps))
        backend._loaded = False

    return declare


def test_a_backend_module_registers_when_imported(entry_points):
    def loader():
        backend.register_detector("fake", lambda device: device.type == "cpu")
        backend.register("Op", "fake", fake_get_kernel)

    entry_points(_FakeEntryPoint("fake", "tileops_fake", loader))
    assert backend.resolve_target(torch.device("cpu")) == "fake"


def test_enumeration_happens_once_across_every_entry_path(entry_points):
    loads = []
    entry_points(_FakeEntryPoint("fake", "tileops_fake", lambda: loads.append(1)))
    backend.registered()
    backend.registered_targets()
    backend.load_errors()
    with pytest.raises(UnknownTargetError):
        backend.resolve_target(torch.device("cpu"))
    assert len(loads) == 1


def test_a_broken_backend_is_skipped_warned_and_recorded(entry_points):
    def good():
        backend.register("Op", "good", fake_get_kernel)

    def bad():
        raise ImportError("libmusa.so not found")

    entry_points(
        _FakeEntryPoint("bad", "tileops_bad", bad),
        _FakeEntryPoint("good", "tileops_good", good),
    )
    with pytest.warns(RuntimeWarning, match="failed to load"):
        assert backend.get_kernel_for("Op", "good") is fake_get_kernel
    assert [e.name for e in backend.load_errors()] == ["bad"]
    assert "libmusa.so" in str(backend.load_errors()[0])


def test_a_backend_that_raises_midway_registers_nothing(entry_points):
    """One entry point's load is all-or-nothing.

    A half-registered backend is worse than an absent one: the registry would advertise
    ops belonging to a distribution that never finished initializing.
    """

    def half_broken():
        backend.register_detector("half", lambda device: True)
        backend.register("First", "half", fake_get_kernel)
        backend.register("Second", "half", fake_get_kernel)
        raise RuntimeError("failed after registering three things")

    def intact():
        backend.register("Kept", "intact", fake_get_kernel)

    entry_points(
        _FakeEntryPoint("half", "tileops_half", half_broken),
        _FakeEntryPoint("intact", "tileops_intact", intact),
    )
    with pytest.warns(RuntimeWarning, match="failed to load"):
        assert backend.registered_targets("Kept") == ["intact"]
    assert backend.registered_targets("First") == []
    assert backend.registered_targets("Second") == []
    assert "half" not in backend.registered_targets()


def test_errors_point_at_the_broken_backend(entry_points):
    entry_points(
        _FakeEntryPoint("bad", "tileops_bad", lambda: (_ for _ in ()).throw(OSError("boom")))
    )
    with pytest.warns(RuntimeWarning), pytest.raises(UnknownTargetError, match="failed to load"):
        backend.resolve_target(torch.device("cpu"))


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_an_interrupt_propagates_rolls_back_and_leaves_discovery_retryable(entry_points, interrupt):
    """Ctrl-C on the first dispatch must not strand the registry half-populated.

    Three things at once, because they are one sequence: the interrupt reaches the
    caller, the partial registration is undone, and the next call tries again instead of
    believing discovery already happened.
    """
    attempts = []

    def rude():
        attempts.append(1)
        backend.register("Op", f"attempt{len(attempts)}", fake_get_kernel)
        if len(attempts) == 1:
            raise interrupt

    entry_points(_FakeEntryPoint("rude", "tileops_rude", rude))

    with pytest.raises(interrupt):
        backend.registered()
    assert backend._KERNELS == {}, "the interrupted load registered nothing"

    assert backend.registered_targets("Op") == ["attempt2"], "the next call retries"


def test_one_failing_backend_does_not_erase_an_earlier_failure(entry_points):
    entry_points(
        _FakeEntryPoint("first", "tileops_first", lambda: 1 / 0),
        _FakeEntryPoint("second", "tileops_second", lambda: 1 / 0),
    )
    with pytest.warns(RuntimeWarning):
        assert [e.name for e in backend.load_errors()] == ["first", "second"]


def test_a_backend_importing_tileops_does_not_recurse(entry_points):
    """A real backend module does ``from tileops.backend import register`` at top level."""

    def loader():
        backend.registered()  # re-enters the loader guard
        backend.register("Op", "fake", fake_get_kernel)

    entry_points(_FakeEntryPoint("fake", "tileops_fake", loader))
    assert backend.registered_targets("Op") == ["fake"]


def test_a_detector_that_raises_is_reported_against_its_target():
    """A detector must answer. Guessing past a broken one would hand the device away."""

    def broken(device):
        raise RuntimeError("vendor runtime missing")

    backend.register_detector("broken", broken)
    backend.register_detector("fine", lambda device: True)
    with pytest.raises(BackendError, match="detector for target 'broken'") as excinfo:
        backend.resolve_target(torch.device("cpu"))
    assert "must return False" in str(excinfo.value)


def test_warnings_as_errors_cannot_truncate_discovery(entry_points, recwarn):
    """Under -W error, warning inside the loop would stop every later backend loading."""
    entry_points(
        _FakeEntryPoint("bad", "tileops_bad", lambda: 1 / 0),
        _FakeEntryPoint(
            "good", "tileops_good", lambda: backend.register("Op", "good", fake_get_kernel)
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning):
            backend.registered()
    assert backend.registered_targets("Op") == ["good"], "the later backend still loaded"
    assert [e.name for e in backend.load_errors()] == ["bad"]


def test_rollback_restores_state_a_failing_backend_touched(entry_points):
    """Everything the snapshot covers, not just the two tables."""
    backend.register("Kept", "pre", fake_get_kernel)
    backend.register_detector("pre", lambda device: device.type == "cpu")
    backend.set_default_target("pre")
    backend.resolve_target(torch.device("cpu"))
    before = (dict(backend._RESOLVED), backend.default_target())

    def greedy():
        backend.register("Added", "greedy", fake_get_kernel)
        backend.register_detector("greedy", lambda device: True)
        backend.set_default_target("greedy")
        backend.resolve_target(torch.device("cpu"))
        raise RuntimeError("too late")

    entry_points(_FakeEntryPoint("greedy", "tileops_greedy", greedy))
    with pytest.warns(RuntimeWarning):
        assert backend.registered_targets("Added") == []
    assert backend.registered_targets("Kept") == ["pre"]
    assert (dict(backend._RESOLVED), backend.default_target()) == before


def test_another_thread_blocks_rather_than_see_a_half_built_registry(entry_points, monkeypatch):
    """Discovery is published only when done, so a partial registry is unobservable.

    Asserted as blocking rather than as ordering: the reader is held mid-discovery and
    checked to be still waiting, which does not depend on winning a race.
    """
    seen = []
    mid_discovery = threading.Event()
    may_finish = threading.Event()

    def slow():
        backend.register("First", "slow", fake_get_kernel)
        mid_discovery.set()
        assert may_finish.wait(timeout=5), "test did not release the loader"
        backend.register("Second", "slow", fake_get_kernel)

    entry_points(_FakeEntryPoint("slow", "tileops_slow", slow))
    loader = threading.Thread(target=backend.registered)
    reader = threading.Thread(target=lambda: seen.append(backend.registered()))

    loader.start()
    assert mid_discovery.wait(timeout=5)

    # Watch the lock rather than the clock: the reader is known to be waiting because it
    # asked for the lock, not because a timeout expired without it finishing.
    reader_reached_lock = threading.Event()
    loader_thread = loader

    class _WatchedLock:
        def __init__(self, inner):
            self._inner = inner

        def __enter__(self):
            if threading.current_thread() is not loader_thread:
                reader_reached_lock.set()
            return self._inner.__enter__()

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)

    monkeypatch.setattr(backend, "_LOCK", _WatchedLock(backend._LOCK))
    reader.start()
    assert reader_reached_lock.wait(timeout=5), "the reader never contended for the lock"
    assert seen == [], "and it did not get past it"

    may_finish.set()
    loader.join(timeout=5)
    reader.join(timeout=5)
    assert seen == [frozenset({("First", "slow"), ("Second", "slow")})]


def test_a_detector_that_keeps_registering_while_probed_is_named_not_waited_out():
    """Unbounded churn is a contract violation, so it must fail rather than hang."""
    count = itertools.count()

    def rude(device):
        backend.register_detector(f"late{next(count)}", lambda d: False)
        return True

    backend.register_detector("rude", rude)
    with pytest.raises(BackendError, match="kept changing"):
        backend.resolve_target(torch.device("cpu"))


def test_rollback_invalidates_a_probe_that_was_in_flight():
    """A restore replaces the table, so a result computed against the old one is void."""
    backend.register_detector("a", lambda device: True)
    before = backend._DETECTOR_VERSION
    backend._restore(backend._snapshot())
    assert before < backend._DETECTOR_VERSION


def test_a_detector_arriving_mid_probe_does_not_get_a_stale_answer_cached():
    """The probe runs outside the lock, so its result is only trusted if nothing moved."""
    probes = []

    def slow_detect(device):
        probes.append(device)
        if len(probes) == 1:
            # Stand in for another thread registering while this probe is in flight.
            backend.register_detector("late", lambda d: True)
        return True

    backend.register_detector("first", slow_detect)
    with pytest.raises(AmbiguousTargetError):
        backend.resolve_target(torch.device("cpu"))
    assert torch.device("cpu") not in backend._RESOLVED


def test_enumeration_failing_outright_leaves_no_loading_flag_behind(monkeypatch):
    """The reentrancy flag is per-thread state; a stuck one would silence discovery."""

    def explode(group):
        raise RuntimeError("importlib.metadata is unhappy")

    monkeypatch.setattr(backend, "entry_points", explode)
    backend._loaded = False
    with pytest.raises(RuntimeError, match="unhappy"):
        backend.registered()
    assert not getattr(backend._LOADING, "active", False)
    assert not backend._loaded, "discovery is retryable"


# --------------------------------------------------------------------------------------
# The packaging boundary
# --------------------------------------------------------------------------------------


def _in_fresh_interpreter(source: str) -> str:
    """Run *source* in a new interpreter and return its stdout.

    Both guarantees below are about what an import pulls in, so they cannot be checked
    in this process — pytest has already imported everything.
    """
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_the_protocol_module_does_not_pull_in_tilelang():
    """The one module a backend imports must not drag the nv toolchain in with it."""
    assert (
        _in_fresh_interpreter("import sys, tileops.backend; print('tilelang' in sys.modules)")
        == "False"
    )


def test_importing_tileops_does_not_enumerate_backends():
    """Discovery is deferred, so installing a backend costs nothing until it is asked for."""
    assert (
        _in_fresh_interpreter("import tileops.backend; print(tileops.backend._loaded)") == "False"
    )


def test_importing_tileops_does_not_even_need_torch():
    """The manifest tooling reads YAML and runs where torch is not installed."""
    assert _in_fresh_interpreter("import sys, tileops; print('torch' in sys.modules)") == "False"


def test_the_caller_api_still_resolves_on_access():
    """Lazy must not mean absent: the documented names work and dir() lists them."""
    import tileops

    assert tileops.set_default_target is backend.set_default_target
    assert "registered_targets" in dir(tileops)
    with pytest.raises(AttributeError, match="has no attribute 'nope'"):
        _ = tileops.nope


def test_the_protocol_surface_is_small():
    """Guard against the protocol growing kernel-level vocabulary."""
    assert set(backend.__all__) == {
        "select_target",
        "ENTRY_POINT_GROUP",
        "AmbiguousTargetError",
        "BackendError",
        "BackendLoadFailure",
        "DetectFn",
        "GetKernelFn",
        "InputSpec",
        "Kernel",
        "KernelResult",
        "OpNotAvailableError",
        "UnknownTargetError",
        "default_target",
        "get_kernel_for",
        "load_errors",
        "register",
        "register_detector",
        "registered",
        "registered_targets",
        "resolve_target",
        "set_default_target",
    }
