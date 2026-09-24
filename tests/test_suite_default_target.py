"""The suite runs the in-tree kernels, whatever backend the environment has installed."""

import pytest

from tests.conftest import _pin_default_target
from tests.test_op_backend_seam import _inputs, _Recorder, _register, _stub_op
from tileops.backend import BUILTIN, registry, set_default_target

pytestmark = pytest.mark.smoke


@pytest.fixture
def empty_registry():
    """An empty registry for the test, the pinned one back afterwards."""
    state = registry.snapshot()
    registry.DETECTORS.clear()
    registry.BUILDERS.clear()
    registry.LOAD_FAILURES.clear()
    yield
    registry.restore(state)


def test_an_installed_backend_does_not_serve_the_suite(request, empty_registry) -> None:
    if request.config.getoption("--tileops-target") != "builtin":
        pytest.skip("this run was asked to serve the suite from a backend")
    recorder = _Recorder()
    _register(recorder, op="StubOp")  # claims every device, CPU included
    op = _stub_op()

    # The stub's in-tree builder returns None; the target's path would refuse the call.
    assert op(*_inputs()) is None
    assert op._settled_target is BUILTIN
    assert recorder.calls == []


def test_a_backend_setting_a_default_on_import_does_not_replace_the_pin(
    monkeypatch, empty_registry
) -> None:
    class _EntryPoint:
        name, value = "acme", "acme_backend"

        def load(self):
            registry.register_detector("acme", lambda device: True)
            set_default_target("acme")

    monkeypatch.setattr(registry, "entry_points", lambda group: [_EntryPoint()])
    registry.default_target = None
    registry._loaded = False

    _pin_default_target("builtin")
    registry.ensure_loaded()  # what the first op constructed afterwards would do

    assert registry.default_target is BUILTIN
