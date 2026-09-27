"""The target and device every pytest run under this repository uses: tests and benchmarks.

Both suites load this file once, so the two options are registered here and nowhere else.
"""

import pytest

from tileops.backend import BUILTIN, UnknownTargetError, registry, set_default_target
from workloads.device import run_device, set_run_device


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the target and the device a run uses."""
    parser.addoption(
        "--tileops-target",
        default="builtin",
        help="Which kernels serve the ops: 'builtin' (default) for the in-tree kernels, "
        "'detect' to let installed backends claim their devices, or a target name.",
    )
    parser.addoption(
        "--tileops-device",
        default="cuda",
        help="The device the run places its tensors on (default 'cuda'). A test run on "
        "another device deselects the tests marked cuda_only; a benchmark run refuses it.",
    )


# The process default target and run device in force before configure, put back at unconfigure.
_OUTER_DEFAULT_TARGET = pytest.StashKey[object]()
_OUTER_RUN_DEVICE = pytest.StashKey[object]()


def pytest_configure(config: pytest.Config) -> None:
    """Pin the target and the device before collection.

    Settled here, so no fixture of any scope and no collected module calls an op before it.
    """
    config.stash[_OUTER_DEFAULT_TARGET] = registry.default_target
    _pin_default_target(config.getoption("--tileops-target"))
    config.stash[_OUTER_RUN_DEVICE] = run_device()
    set_run_device(config.getoption("--tileops-device"))


def _pin_default_target(choice: str) -> None:
    """Make the in-tree kernels serve the run, unless it names another target.

    Tests and benchmarks measure the in-tree implementation, so a backend installed in the
    environment must not claim its devices. ``set_default_target`` loads the installed
    backends before it sets the default, so one that sets its own while being imported
    cannot replace this one later. A test of target dispatch isolates the registry or names
    its target, and either overrides this.
    """
    target = {"builtin": BUILTIN, "detect": None}.get(choice, choice)
    try:
        set_default_target(target)
    except UnknownTargetError as exc:
        raise pytest.UsageError(f"--tileops-target: {exc}") from None


def pytest_unconfigure(config: pytest.Config) -> None:
    """Put back the process default target and run device this file replaced."""
    if _OUTER_DEFAULT_TARGET in config.stash:
        registry.default_target = config.stash[_OUTER_DEFAULT_TARGET]
    if _OUTER_RUN_DEVICE in config.stash:
        set_run_device(config.stash[_OUTER_RUN_DEVICE])
