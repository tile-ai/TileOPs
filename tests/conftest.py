import gc
from collections import defaultdict

import pytest
import torch

from tests.workload_test_base import _check_result
from tileops.backend import BUILTIN, default_target
from workloads.device import run_device, run_device_is_cuda


def _under_repo_tests(item: pytest.Item) -> bool:
    path = str(item.path)
    return "tests/" in path and "benchmarks/tests/" not in path


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the opt-in in-kernel timeline-trace flag.

    Off by default: when ``--trace-kernel`` is absent the process-local trace
    switch stays off, so trace-dump tests no-op and normal runs are zero cost.
    (``--trace`` itself is reserved by pytest for its pdb tracer.)
    """
    parser.addoption(
        "--trace-kernel",
        action="store_true",
        default=False,
        help="Build instrumented kernels with in-kernel tracing and dump their "
        "timeline (HTML + Chrome JSON) for the trace-dump tests.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Flip the in-process trace switch on when ``--trace-kernel`` is passed.

    Runs once at startup, before any kernel is built, so the traced build is the
    one that gets cached. No environment variable is involved — the switch lives
    in this pytest process only.
    """
    if config.getoption("--trace-kernel"):
        from tileops.trace import trace

        trace.enable()  # dumps to debug/ (gitignored)


@pytest.fixture(autouse=True)
def setup() -> None:
    torch.manual_seed(1235)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1235)


def pytest_runtest_teardown() -> None:
    """Return a worker's cached CUDA blocks to the driver, after its fixtures have torn down."""
    # Eight xdist workers share one GPU in CI, and the caching allocator holds every block a worker
    # ever took until it exits, so the card must fit the sum of eight high-water marks. Releasing
    # bounds a worker to its current case; it does not bound eight concurrent cases, a few of which
    # are tens of GiB alone. The threshold is a cost gate, not a limit: zero is equally correct.
    cache_release_bytes = 2 << 30
    if not torch.cuda.is_available():
        return
    if torch.cuda.memory_reserved() < cache_release_bytes:
        return
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture
def isolated_dynamo():
    """Reset torch._dynamo state around a test that calls ``torch.compile``.

    Dynamo's recompile cache is keyed per code object, and every
    ``torch.compile``-d plain callable (any non-``nn.Module``, e.g. a TileOps
    ``Op`` instance) shares torch's single wrapper frame. Each compiled op
    instance therefore consumes one slot of that frame's shared
    ``cache_size_limit`` (default 8) for the whole pytest process, so compile
    tests pollute each other's cache and later ``fullgraph=True`` tests fail
    with ``FailOnRecompileLimitHit``. Request this fixture from every test
    that calls ``torch.compile``.
    """
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _get_callspec_params(item: pytest.Item) -> dict | None:
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return None
    return getattr(callspec, "params", None)


def _freeze_value(value: object) -> object:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_value(val)) for key, val in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze_value(item) for item in value), key=str))
    return value


def _without_dtype(params: dict) -> tuple[tuple[str, object], ...]:
    return tuple(
        sorted((key, _freeze_value(value)) for key, value in params.items() if key != "dtype")
    )


# Each architecture marker and the compute capability (major, minor) it needs. The
# collection skip, the skip record and the session-end check all read this one table.
_ARCH_MARKERS = {
    "sm90": ("compute capability 9.x", lambda capability: capability[0] == 9),
    "sm89": ("compute capability 8.9", lambda capability: capability == (8, 9)),
}


def _run_capability() -> tuple[int, int] | None:
    """Compute capability of the device this run places tensors on; ``None`` off CUDA."""
    if not run_device_is_cuda() or not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability(torch.device(run_device()))


def _serves(marker: str, capability: tuple[int, int] | None) -> bool:
    return capability is not None and _ARCH_MARKERS[marker][1](capability)


_arch_skipped: dict[str, list[str]] = defaultdict(list)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> None:
    """Carry the test's architecture markers on its report, which xdist ships to the controller."""
    outcome = yield
    outcome.get_result().arch_markers = [
        name for name in _ARCH_MARKERS if item.get_closest_marker(name) is not None
    ]


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Record an architecture-marked test that was skipped, so a run on that architecture can refuse it."""
    if report.skipped and not hasattr(report, "wasxfail"):
        for marker in getattr(report, "arch_markers", ()):
            _arch_skipped[marker].append(report.nodeid)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """On the architecture a marker names, a skipped test with that marker is a failed run.

    The mark exists because the path is selected on that architecture. On the hardware it
    needs, skipping it would leave the only evidence for that path unexercised while the
    run still reported green. A skip at collection, in a fixture or in the test body all
    count. Deselecting the mark outright (``-m "not sm90"``) is not covered, and is not
    meant to be — that is the operator saying which tests to run, not a run losing its
    evidence.
    """
    capability = _run_capability()
    for marker, skipped in _arch_skipped.items():
        if skipped and _serves(marker, capability):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
            raise pytest.UsageError(
                f"{marker}-marked tests were skipped on a {_ARCH_MARKERS[marker][0]} device: "
                + ", ".join(skipped)
            )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Validate explicit test tier assignments, then drop the tests this run cannot serve."""
    marker_errors: list[str] = []
    tier_names = ("smoke", "full", "nightly")
    capability = _run_capability()

    for item in items:
        if not _under_repo_tests(item):
            continue
        archs = [name for name in _ARCH_MARKERS if item.get_closest_marker(name) is not None]
        if len(archs) > 1:
            marker_errors.append(
                f"{item.nodeid}: expected at most one architecture marker, found {archs}"
            )
        for marker in archs:
            if not _serves(marker, capability):
                item.add_marker(pytest.mark.skip(reason=f"needs {_ARCH_MARKERS[marker][0]}"))

        tiers = [name for name in tier_names if item.get_closest_marker(name) is not None]
        if len(tiers) != 1:
            marker_errors.append(
                f"{item.nodeid}: expected exactly one tier marker, found {tiers or 'none'}"
            )

    ops_groups: dict[tuple[str, str], list[pytest.Item]] = defaultdict(list)
    for item in items:
        path = str(item.path)
        if "tests/ops/" not in path or "benchmarks/tests/" in path:
            continue
        test_name = getattr(item, "originalname", item.name)
        ops_groups[(path, test_name)].append(item)

    for (_path, _test_name), group in ops_groups.items():
        non_xfail_items = [item for item in group if item.get_closest_marker("xfail") is None]
        smoke_items = [item for item in group if item.get_closest_marker("smoke") is not None]

        # Smoke cases must never be xfail (checked before tune gate)
        for item in smoke_items:
            if item.get_closest_marker("xfail") is not None:
                marker_errors.append(f"{item.nodeid}: smoke cases must not be xfail")

        # For count and ordering checks, only consider non-xfail smoke cases
        valid_smoke_items = [
            item for item in smoke_items if item.get_closest_marker("xfail") is None
        ]

        if non_xfail_items:
            if len(valid_smoke_items) < 1:
                marker_errors.append(
                    f"{non_xfail_items[0].nodeid}: each test must have at least one smoke case"
                )
            else:
                # All smoke cases must appear as the first N non-xfail items
                expected_smoke = non_xfail_items[: len(valid_smoke_items)]
                if valid_smoke_items != expected_smoke:
                    marker_errors.append(
                        f"{non_xfail_items[0].nodeid}: all smoke cases must appear "
                        f"as the first {len(valid_smoke_items)} non-xfail cases of each test"
                    )

        dtype_supported: set[object] = set()
        dtype_smoke: set[object] = set()
        smoke_signatures: set[tuple[tuple[str, object], ...]] = set()
        dtype_cases_present = False

        for item in non_xfail_items:
            params = _get_callspec_params(item)
            if not params or "dtype" not in params:
                continue

            dtype_cases_present = True
            dtype_supported.add(params["dtype"])

            if item.get_closest_marker("smoke") is not None:
                dtype_smoke.add(params["dtype"])
                smoke_signatures.add(_without_dtype(params))

        if dtype_cases_present:
            missing_smoke_dtypes = dtype_supported - dtype_smoke
            if missing_smoke_dtypes:
                marker_errors.append(
                    f"{non_xfail_items[0].nodeid}: each dtype must have at least one smoke case; "
                    f"missing smoke for {sorted(str(dtype) for dtype in missing_smoke_dtypes)}"
                )

            for item in non_xfail_items:
                if item.get_closest_marker("full") is None:
                    continue

                params = _get_callspec_params(item)
                if not params or "dtype" not in params:
                    continue

                if _without_dtype(params) in smoke_signatures:
                    marker_errors.append(
                        f"{item.nodeid}: full cases must not differ from a smoke case only by dtype"
                    )

        first_tuned_item: pytest.Item | None = None
        full_tuned_items: list[pytest.Item] = []
        for item in group:
            params = _get_callspec_params(item)
            if params is None or "tune" not in params:
                continue

            tune = params["tune"]
            is_smoke = item.get_closest_marker("smoke") is not None
            if is_smoke and tune is True:
                marker_errors.append(f"{item.nodeid}: smoke cases must use tune=False")
            if tune is True:
                if first_tuned_item is None:
                    first_tuned_item = item
                if item.get_closest_marker("full") is not None:
                    full_tuned_items.append(item)
        if first_tuned_item is not None:
            if not full_tuned_items:
                marker_errors.append(
                    f"{first_tuned_item.nodeid}: the first tune=True case must be marked full"
                )
            elif len(full_tuned_items) > 1:
                marker_errors.append(
                    f"{group[0].path}::{group[0].originalname}: at most one tune=True case may be full"
                )
            elif full_tuned_items[0] is not first_tuned_item:
                marker_errors.append(
                    f"{first_tuned_item.nodeid}: the first tune=True case must be the only full tuned case"
                )

    if marker_errors:
        raise pytest.UsageError(
            "Invalid explicit test marker assignments detected:\n" + "\n".join(marker_errors)
        )

    # A run on another device drops what needs CUDA; a run on another target drops what
    # reads the in-tree kernels.
    marks = []
    if not run_device_is_cuda():
        marks.append("cuda_only")
    if default_target() is not BUILTIN:
        marks.append("in_tree_kernels")
    dropped = [
        item
        for item in items
        if _under_repo_tests(item) and any(item.get_closest_marker(m) for m in marks)
    ]
    if dropped:
        dropped[0].config.hook.pytest_deselected(items=dropped)
        kept = set(map(id, items)) - set(map(id, dropped))
        items[:] = [item for item in items if id(item) in kept]


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Record what check() ran and what it measured.

    The op a test establishes something about is read off the Op that check()
    ran. A test declares it instead only where that object cannot name one: a
    kernel several ops register, or a compiled callable, which keeps no handle
    on the Op it wraps.
    """
    yield
    op_name = getattr(_check_result, "op_name", None)
    if op_name:
        item.user_properties.append(("op", op_name))
        op_module = getattr(_check_result, "op_module", None)
        if op_module:
            item.user_properties.append(("op_module", op_module))
        checked = getattr(_check_result, "checked_outputs", 0)
        item.user_properties.append(("checked_outputs", str(checked)))
        max_err = getattr(_check_result, "max_abs_err", None)
        if max_err is not None:
            item.user_properties.append(("max_abs_err", f"{max_err:.2e}"))
        _check_result.checked_outputs = 0
        _check_result.op_name = None
        _check_result.op_module = None
        _check_result.max_abs_err = None
