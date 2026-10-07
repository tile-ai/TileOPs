"""The benchmark a bench file writes: ``bench.Runner(op, case).compare(implementations)``.

``cases(Op)`` turns the op's manifest workload rows into :class:`Case` objects; a
:class:`Runner` verifies the TileOps op and every implementation against the case's
reference, times them, and records one report row each.
"""

import dataclasses
import functools
import gc
import statistics
from collections.abc import Mapping
from typing import Any, Callable, Optional

import torch

from benchmarks.report import BenchmarkReport
from benchmarks.timing import (
    _MAX_ITERS,
    _MIN_ITERS,
    DRY_RUN_MS,
    REPEAT_MS,
    Sample,
    _capture_bench_meta,
    _sample_spread_ms,
    bench_kernel,
    median_busy_ms,
)
from tileops.manifest import load_adts, load_manifest, load_workloads, manifest_key
from tileops.manifest.plan import entry_plan
from tileops.manifest.values import ADTValue
from tileops.manifest.workload import Call, instantiate
from workloads.numerics import CheckResult, Request, check_inputs_preserved, verify

__all__ = [
    "Case",
    "Implementation",
    "Runner",
    "Sample",
    "bench_kernel",
    "cases",
]

# The name the TileOps op is recorded under; no implementation passed to compare() may take it.
TILEOPS = "tileops"
# The name a single implementation passed to compare() is recorded under.
BASELINE = "baseline"

# --tileops-verify runs the correctness check only, omitting timing.
_verifying = False


def verifying() -> bool:
    """Whether this run omits timing after the correctness check."""
    return _verifying


def set_verifying(value: bool) -> None:
    """Put the run into verification mode. The benchmark conftest owns this."""
    global _verifying
    _verifying = value


@dataclasses.dataclass(frozen=True)
class Implementation:
    """One complete call taking part in a comparison.

    Every round runs ``reset``, flushes L2, then times ``run(*args)``. ``args=None`` calls
    ``run`` with ``case.inputs``, which it must leave unchanged; ``args=()`` calls a closure
    that takes none. An implementation that overwrites an argument takes a private copy of
    it in ``args`` and restores it in ``reset``. ``noncomparable_reason`` records an external
    implementation whose semantics differ from the case: it is timed, not verified, and
    gets no ratio.
    """

    run: Callable
    args: Optional[tuple] = None
    reset: Optional[Callable[[], None]] = None
    noncomparable_reason: Optional[str] = None


def _report_value(value: Any) -> Any:
    """A manifest parameter as the report records it: an ADT as the literal the manifest writes."""
    if isinstance(value, ADTValue):
        return {value.kind: dict(value.fields)}
    return value


@dataclasses.dataclass(frozen=True, eq=False)
class Case:
    """One benchmark case of a manifest op: its inputs, constructor arguments, reference,
    verification and timing policy.

    Data is created on first access and kept until the test ends, so collecting cases needs
    no GPU and the verification and timing of one case read the same tensors.
    """

    id: str
    _call: Call = dataclasses.field(repr=False)
    _entry: Any = dataclasses.field(repr=False)

    @property
    def count_copies(self) -> bool:
        """Whether device-to-device copies inside ``run`` count, for every implementation alike."""
        return self._entry.count_copies

    @functools.cached_property
    def workload(self) -> Any:
        """The workload the inputs come from, for an implementation that needs what it derives."""
        return self._entry.workload(self._call)

    @functools.cached_property
    def inputs(self) -> tuple:
        """The call-time inputs, in the order the op takes them."""
        if self._entry.inputs is not None:
            return tuple(self._entry.inputs(self.workload))
        return tuple(self.workload.gen_inputs())

    @functools.cached_property
    def arguments(self) -> dict:
        """The op's constructor arguments."""
        arguments = getattr(self.workload, "arguments", None)
        return arguments() if arguments is not None else self._call.arguments({})

    @functools.cached_property
    def reference(self) -> Optional[Callable]:
        """The callable whose result every implementation is checked against."""
        return getattr(self.workload, "ref_program", None)

    @functools.cached_property
    def verification(self) -> Any:
        """The workload's numerical declaration for these inputs."""
        return self.workload.verification(*self.inputs)

    @functools.cached_property
    def params(self) -> dict:
        """What the report records for the case: each tensor's shape and dtype, and the
        manifest parameters as the call states them."""
        tensors = {
            name: (tuple(shape), dtype) for name, (shape, dtype) in self._call.tensors.items()
        }
        return tensors | {name: _report_value(v) for name, v in self._call.params.items()}

    def _release(self) -> None:
        """Drop the data this case created; the next access creates it again."""
        for name in ("workload", "inputs", "arguments", "reference", "verification"):
            self.__dict__.pop(name, None)


def cases(op: "str | type") -> list[Case]:
    """One :class:`Case` per manifest call of *op*, an Op class or its manifest key.

    Each workload row is instantiated once per dtype case. An op with no registered case
    factory fails here, at collection.
    """
    from benchmarks._cases import entry

    name = manifest_key(op)
    factory = entry(name)
    plan = entry_plan(name, load_manifest()[name], load_adts(), resolve=False)
    return [
        Case(call.case_id, call, factory)
        for call in (
            instantiate(plan, row, dtype_case)
            for row in load_workloads(name)
            for dtype_case in row.get("dtype_cases") or [{}]
        )
    ]


def _as_implementation(value: Any) -> Implementation:
    return value if isinstance(value, Implementation) else Implementation(run=value)


class Runner:
    """Verifies, times and records the TileOps op and the implementations it is compared with.

    The op is the TileOps implementation, recorded as ``tileops``; the case's reference is
    the correctness baseline. A row of the report is this op's measurement, so the op must
    be a manifest entry.
    """

    def __init__(self, op: Any, case: Case):
        name = type(op).__name__
        if name not in load_manifest():
            raise KeyError(
                f"{name} is not a manifest op; a benchmark reports under the name the "
                "manifest declares, so a wrapper or a subclass cannot stand in for one"
            )
        if case._call.signature.name != name:
            raise ValueError(f"{name} cannot run case {case.id!r} of {case._call.signature.name}")
        self.op = op
        self.case = case
        self._roofline: Optional[tuple[float, float]] = None

    def compare(self, implementations: Any) -> dict[str, dict]:
        """Verify every implementation, then time all of them in both orders and record them.

        *implementations* is one implementation, recorded as ``baseline``, or a mapping of
        report name to implementation. A value is an :class:`Implementation` or a callable
        on ``case.inputs``. Every implementation is checked before any is timed; a failure
        stops the comparison. ``--tileops-verify`` returns after the check.
        """
        named = (
            dict(implementations)
            if isinstance(implementations, Mapping)
            else {BASELINE: implementations}
        )
        if TILEOPS in named:
            raise ValueError(
                f"{TILEOPS!r} names the TileOps op; give the implementation another name"
            )
        plan = {TILEOPS: self.case._entry.binder(self.op, self.case)}
        plan |= {name: _as_implementation(value) for name, value in named.items()}
        for name, impl in plan.items():
            reason = impl.noncomparable_reason
            if reason is not None and (name == TILEOPS or not reason.strip()):
                raise ValueError(
                    f"{name}: noncomparable_reason needs a reason, and the TileOps op cannot carry one"
                )
        checks = self._verify(plan)
        if verifying():
            return {}
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        results = self._time(plan)
        for name, result in results.items():
            BenchmarkReport.record(
                self.op,
                self.case.params,
                result,
                tag=name,
                unverified=checks[name].unchecked_reason or "",
                ratio=bool(checks[name].checked_outputs),
            )
        return results

    def _verify(self, plan: dict[str, Implementation]) -> dict[str, CheckResult]:
        case = self.case
        evidence = case.verification
        checks: dict[str, CheckResult] = {}
        requests = {}
        itself = []
        for name, impl in plan.items():
            if impl.noncomparable_reason is not None:
                try:
                    check_inputs_preserved(
                        Request(
                            impl.run,
                            case.inputs if impl.args is None else impl.args,
                            impl.reset,
                        ),
                        case.inputs,
                    )
                except AssertionError as exc:
                    raise AssertionError(f"{name}: {exc}") from exc
                checks[name] = CheckResult(
                    0, None, None, f"noncomparable: {impl.noncomparable_reason}"
                )
            elif impl.run is case.reference and impl.args is None and impl.reset is None:
                itself.append(name)
            else:
                requests[name] = Request(
                    impl.run,
                    case.inputs if impl.args is None else impl.args,
                    impl.reset,
                    preserve_inputs=True,
                )
        checks |= verify(
            case.reference,
            case.inputs,
            evidence=evidence,
            requests=requests,
            preserve_reference_inputs=bool(itself),
        )
        # The reference timed as an implementation is as established as the op's check
        # shows the reference to be: unchecked where it could not run or nothing is declared.
        op_check = checks[TILEOPS]
        for name in itself:
            checks[name] = CheckResult(
                min(op_check.checked_outputs, 1), None, None, op_check.unchecked_reason
            )
        return {name: checks[name] for name in plan}

    def _time(self, plan: dict[str, Implementation]) -> dict[str, dict]:
        names = list(plan)
        # Split the budget across the two passes rather than spending it twice:
        # the point is symmetry, not more samples.
        passes = 2
        samples: dict[str, list[Sample]] = {name: [] for name in names}
        meta: dict[str, dict] = {}
        for name in names + names[::-1]:
            impl = plan[name]
            with torch.no_grad():
                samples[name].extend(
                    bench_kernel(
                        impl.run,
                        args=self.case.inputs if impl.args is None else impl.args,
                        dry_run_ms=DRY_RUN_MS / passes,
                        repeat_ms=REPEAT_MS / passes,
                        max_iters=_MAX_ITERS // passes,
                        min_iters=max(1, _MIN_ITERS // passes),
                        count_copies=self.case.count_copies,
                        reset=impl.reset,
                    )
                )
            pass_meta = _capture_bench_meta()
            previous = meta.get(name)
            if previous is not None and previous["timing"] != pass_meta["timing"]:
                raise RuntimeError(
                    f"{name}: the two passes timed with different methods "
                    f"({previous['timing']} then {pass_meta['timing']}); pooling "
                    "them would report one median over two kinds of measurement. "
                    "Only reachable with --tileops-allow-events-fallback."
                )
            meta[name] = pass_meta
        return {name: self._result(samples[name], meta[name]) for name in names}

    def _roofline_counts(self) -> tuple[float, float]:
        """The op's own count of the work, read once.

        Read after timing, because an op whose shapes are dynamic binds its roofline
        variables during ``forward()``.
        """
        if self._roofline is None:
            flops, mem_bytes = self.op.eval_roofline()
            self._roofline = (float(flops), float(mem_bytes))
        return self._roofline

    def _roofline_inputs(self) -> dict:
        """The op's diagnostic mapping, or nothing when it declares none."""
        reader = getattr(self.op, "roofline_inputs", None)
        if reader is None:
            return {}
        try:
            return dict(reader())
        except Exception:
            # A diagnostic never fails a measurement.
            return {}

    def _result(self, samples: list[Sample], meta: dict) -> dict:
        """Turn per-iteration samples into the row a report records.

        ``device_busy_ms`` carries the conclusions; the rest are diagnostics. See
        :class:`~benchmarks.timing.Sample` for what each one counts.
        """
        if not samples:
            raise ValueError("bench_kernel returned no samples")
        busy = median_busy_ms(samples)
        latency = statistics.median(s.latency_ms for s in samples)
        result = {
            "device_busy_ms": busy,
            "latency_ms": latency,
            "gap_ms": latency - busy,
            "n_samples": len(samples),
        }
        copy_ms = statistics.median(s.uncounted_copy_ms for s in samples)
        if copy_ms > 0:
            # A row whose reading omits work says so in the row, next to the number.
            result["uncounted_copy_ms"] = copy_ms
        counts = [s.n_kernels for s in samples]
        if all(c is not None for c in counts):
            # The largest count observed: a median would round a call that varies
            # between one and two kernels to a number it never launched.
            result["n_kernels"] = max(counts)
        p10, p90 = _sample_spread_ms([s.device_busy_ms for s in samples])
        if p10 is not None:
            result["device_busy_p10_ms"], result["device_busy_p90_ms"] = p10, p90
        # How the number was measured must travel with it: a run that fell back
        # to CUDA events is not comparable with a CUPTI-timed one.
        result.update(meta)
        # Roofline describes throughput reached while the device was executing, so the
        # denominator excludes the gaps.
        flops, memory = self._roofline_counts()
        result["flops"] = flops
        result["tflops"] = flops / busy * 1e-9
        result["bytes"] = memory
        result["bandwidth_tbs"] = memory / busy * 1e-9
        roof = self.op.compute_roof()
        if roof is not None:
            result["compute_roof"] = roof
        # What decided this call's bytes, where its inputs' values decided it.
        # Nothing judges it; it explains a number that moved.
        decided_by = self._roofline_inputs()
        if decided_by:
            result["roofline_inputs"] = decided_by
        return result
