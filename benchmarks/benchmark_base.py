"""The benchmark a bench file writes: a workload in, a recorded result out.

Timing lives in :mod:`benchmarks.timing`, reporting in :mod:`benchmarks.report`. Both
are re-exported here, so a bench file keeps importing what it always did.
"""

import gc
import statistics
from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

import pytest
import torch

from benchmarks.baselines import assert_matches_reference, reference_tolerance
from benchmarks.report import BenchmarkReport
from benchmarks.timing import (
    _MAX_ITERS,
    _MIN_ITERS,
    DRY_RUN_MS,
    REPEAT_MS,
    CUPTIError,
    Sample,
    _capture_bench_meta,
    _sample_spread_ms,
    bench_kernel,
    median_busy_ms,
)
from tileops.manifest import load_adts, load_manifest, load_workloads, manifest_key
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate
from workloads.numerics import Evidence, Exact, Unestablished, describe

# --tileops-verify runs correctness warmup only, omitting timing.
_verifying = False


def verifying() -> bool:
    """Whether this run omits timing after the correctness warmup."""
    return _verifying


def set_verifying(value: bool) -> None:
    """Put the run into verification mode. The benchmark conftest owns this."""
    global _verifying
    _verifying = value


def ratio_allowed(evidence: Evidence) -> bool:
    """Whether *evidence* establishes enough to publish a ratio against the op.

    A publication policy, which is the benchmark's to make: the same evidence
    tells a test what was established with no ratio involved.
    """
    return evidence.kind in ("exact", "custom", "partial")


__all__ = [
    "BenchmarkBase",
    "BenchmarkReport",
    "CUPTIError",
    "ManifestBenchmark",
    "OpBenchmark",
    "backward_of",
    "bench_kernel",
    "manifest_calls",
]

W = TypeVar("W")


def backward_of(output: torch.Tensor) -> Any:
    """Return a callable running *output*'s backward on the thread that calls it.

    How a baseline reaches its gradients: ``Tensor.backward`` hands the graph to
    autograd's engine thread, whose kernels carry no iteration id for the timer to
    attribute them to, and charges the baseline for engine overhead a tileops backward
    op never pays. Takes one gradient per output of the op that produced *output*, so
    one returning ``(out, lse)`` is driven with ``(grad, None)``.
    """
    node = output.grad_fn
    if node is None:
        raise ValueError(
            f"{type(output).__name__} has no grad_fn; build the graph under "
            "enable_grad on inputs that require grad before timing its backward."
        )
    # A Python autograd.Function's node exposes apply() and is not callable; a node
    # built in C++ is callable and has no apply(). Neither offers the other's form.
    return getattr(node, "apply", None) or node


def _flatten_tensors(value: Any) -> list:
    """Every tensor reachable in *value* through tuples, lists and dicts."""
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (tuple, list)):
        return [found for item in value for found in _flatten_tensors(item)]
    if isinstance(value, dict):
        return [found for item in value.values() for found in _flatten_tensors(item)]
    return []


def _detached_copy(value: Any) -> Any:
    """*value* with every tensor replaced by a copy, so a later write cannot reach it."""
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_detached_copy(item) for item in value)
    if isinstance(value, list):
        return [_detached_copy(item) for item in value]
    if isinstance(value, dict):
        return {key: _detached_copy(item) for key, item in value.items()}
    return value


def _same_callable(one: Any, other: Any) -> bool:
    """Whether both run the same code on the same object."""
    if one is other:
        return True
    function = getattr(one, "__func__", None)
    return (
        function is not None
        and function is getattr(other, "__func__", None)
        and getattr(one, "__self__", None) is getattr(other, "__self__", None)
    )


def _result_dtype(value: Any) -> Optional[torch.dtype]:
    """The least precise floating dtype in *value*, which is the one a tolerance must admit."""
    dtypes = [t.dtype for t in _flatten_tensors(value) if t.is_floating_point() or t.is_complex()]
    return max(dtypes, key=lambda dtype: torch.finfo(dtype).eps) if dtypes else None


class BenchmarkBase(Generic[W], ABC):
    """Turns measured latency into roofline-relative metrics.

    It times and reports; it does not publish. A row of the report is one op's
    measurement, and this class holds no op — a benchmark that publishes is an
    :class:`OpBenchmark`.
    """

    def __init__(self, workload: W):
        self.workload = workload

    @abstractmethod
    def calculate_flops(self) -> Optional[float]:
        """Total FLOPs of one call, or ``None`` to leave TFLOPS out of the row."""
        raise NotImplementedError

    @abstractmethod
    def calculate_memory(self) -> Optional[float]:
        """Bytes moved by one call, or ``None`` to leave bandwidth out of the row."""
        raise NotImplementedError

    def compute_roof(self) -> Optional[str]:
        """GPU-profile key pricing the FLOPs, or ``None`` to leave it out of the row.

        ``ManifestBenchmark`` reads it off ``op.compute_roof()``; a benchmark
        without an Op instance has no roof to declare.
        """
        return None

    def case_params(self) -> dict:
        """What distinguishes this case, read off the workload it was built for.

        The workload holds the case: the shapes and the dtype it was built with.
        Reading them here keeps the row's columns a property of the workload
        class rather than of which locals a bench function happened to have in
        scope. Stored fields only — a workload's computed attributes build data
        (a dequantized weight, say), and a row must not pay to print a column.
        """
        return {
            name: value for name, value in vars(self.workload).items() if not name.startswith("_")
        }

    def profile(self, functor: Any, *inputs: Any) -> dict:
        """Profile a callable and return its structured result."""
        with torch.no_grad():
            return self._build_result(bench_kernel(functor, args=inputs))

    def _build_result(self, samples: list[Sample], meta: Optional[dict] = None) -> dict:
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
        result.update(meta if meta is not None else _capture_bench_meta())
        # Roofline describes throughput reached while the device was executing, so the
        # denominator excludes the gaps.
        flops = self.calculate_flops()
        if flops is not None:
            result["flops"] = flops
            result["tflops"] = flops / busy * 1e-9
        memory = self.calculate_memory()
        if memory is not None:
            result["bytes"] = memory
            result["bandwidth_tbs"] = memory / busy * 1e-9
        roof = self.compute_roof()
        if roof is not None:
            result["compute_roof"] = roof
        # What decided this call's bytes, where its inputs' values decided it.
        # Nothing judges it; it explains a number that moved.
        decided_by = self._roofline_inputs()
        if decided_by:
            result["roofline_inputs"] = decided_by
        return result

    def _roofline_inputs(self) -> dict:
        """The op's diagnostic mapping, or nothing when it declares none."""
        reader = getattr(getattr(self, "op", None), "roofline_inputs", None)
        if reader is None:
            return {}
        try:
            return dict(reader())
        except Exception:
            # A diagnostic never fails a measurement.
            return {}


def manifest_calls(op: "str | type") -> list:
    """One ``pytest.param(call)`` per manifest call of *op*, an Op class or its manifest key.

    Each workload row of the op's entry is instantiated once per dtype case, and the
    case is ided by its case id (docs/design/manifest.md § Rows). ``CallWorkload(call)`` builds
    the call's inputs.
    """
    name = manifest_key(op)
    plan = entry_plan(name, load_manifest()[name], load_adts(), resolve=False)
    calls = [
        instantiate(plan, row, case)
        for row in load_workloads(name)
        for case in row.get("dtype_cases") or [{}]
    ]
    return [pytest.param(call, id=call.case_id) for call in calls]


class OpBenchmark(BenchmarkBase[W]):
    """A benchmark of one op, which is what a row of the report is.

    The op is named once, here, and every row the benchmark publishes carries
    it. A comparison whose subject is not an op — a kernel strategy, a field of
    library implementations — decides something rather than tracking it, and
    stays on :class:`BenchmarkBase`, which cannot record.
    """

    def __init__(self, op: Any, workload: W):
        super().__init__(workload)
        self.op = op
        self._roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        """The op's own count of the work, read once.

        Read lazily, because an op whose shapes are dynamic binds its roofline
        variables during ``forward()``, and every tag is timed before any result
        is built. An op that models no roofline says so, and its benchmark
        overrides the two methods below.
        """
        if self._roofline_cache is None:
            flops, mem_bytes = self.op.eval_roofline()
            self._roofline_cache = (float(flops), float(mem_bytes))
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]

    def _resolve_evidence(self, plan: dict, declared: Optional[dict]) -> dict:
        """Resolve each tag's evidence; missing oracles disable ratios."""
        declared = declared or {}
        stray = set(declared) - set(plan)
        if stray:
            raise ValueError(
                f"evidence names {sorted(stray)}, which this call does not time; "
                "a declaration matching no tag protects nothing"
            )
        resolved = {tag: declared.get(tag, Exact()) for tag in plan}
        if getattr(self.workload, "ref_program", None) is not None:
            return resolved
        # Ratio-bearing evidence requires an oracle.
        return {
            tag: Unestablished()
            if mark.kind in ("exact", "partial", "custom")
            and getattr(mark, "reference", None) is None
            else mark
            for tag, mark in resolved.items()
        }

    def _verify(self, plan: dict, evidence: dict, inputs: tuple) -> None:
        """Validate timed callables on their arguments against oracles on canonical inputs."""
        reference = getattr(self.workload, "ref_program", None)
        active = []
        for tag, (functor, args) in plan.items():
            mark = evidence[tag]
            oracle = getattr(mark, "reference", None) or reference
            if oracle is None or mark.kind not in ("exact", "partial", "custom"):
                continue
            if args is inputs and _same_callable(functor, oracle):
                continue
            active.append((tag, functor, args, mark, oracle))
        if not active:
            return

        # Include tag-specific arguments: one tag must not mutate the next tag's input.
        live = {id(t): t for t in _flatten_tensors(inputs)}
        for _, args in plan.values():
            live.update({id(t): t for t in _flatten_tensors(args)})
        pristine = {key: tensor.detach().clone() for key, tensor in live.items()}

        def restore():
            with torch.no_grad():
                for key, tensor in live.items():
                    tensor.copy_(pristine[key])

        # Share each oracle result across tags.
        references = []
        control_checks = {}
        try:
            for tag, functor, args, mark, oracle in active:
                restore()
                # Match the no-grad mode used by the timer, including compiled baselines.
                with torch.no_grad():
                    produced = _detached_copy(functor(*args))
                restore()
                for cached_oracle, cached_result in references:
                    if _same_callable(oracle, cached_oracle):
                        expected = cached_result
                        break
                else:
                    expected = _detached_copy(oracle(*inputs))
                    references.append((oracle, expected))
                restore()
                OpBenchmark._check_result(tag, produced, expected, mark)
                outputs = produced if isinstance(produced, (tuple, list)) else (produced,)
                dtypes = tuple(_result_dtype(output) for output in outputs)
                reference_index = next(
                    i
                    for i, (candidate, _) in enumerate(references)
                    if _same_callable(oracle, candidate)
                )
                for control in getattr(mark, "controls", ()):
                    key = (reference_index, id(control))
                    control_checks.setdefault(key, (control, []))[1].append((tag, mark, dtypes))
                del produced

            # Share each fault across tag comparators, retaining one fault output at a time.
            for (reference_index, _), (control, checks) in control_checks.items():
                oracle, expected = references[reference_index]
                restore()
                faulty = _detached_copy(control.run(oracle, inputs))
                restore()
                try:
                    for tag, mark, dtypes in checks:
                        try:
                            OpBenchmark._check_result(
                                tag, faulty, expected, mark, tolerance_dtypes=dtypes
                            )
                        except AssertionError:
                            continue
                        raise ValueError(
                            f"{tag}: negative control {control.name!r} was accepted; "
                            "the inputs or tolerance do not distinguish this fault"
                        )
                finally:
                    del faulty

        finally:
            restore()

    @staticmethod
    def _check_result(
        tag: str,
        produced: Any,
        expected: Any,
        mark: Evidence,
        *,
        tolerance_dtypes: Optional[tuple] = None,
    ) -> None:
        if mark.kind == "custom":
            mark.validator(produced, expected)
            return
        outputs = produced if isinstance(produced, (tuple, list)) else (produced,)
        targets = expected if isinstance(expected, (tuple, list)) else (expected,)
        width, covered = len(outputs), len(targets)
        claimed = getattr(mark, "outputs", width)
        if covered < 1 or width < 1:
            raise ValueError(f"{tag}: the reference or implementation returned no outputs")
        if mark.kind == "exact" and covered < width:
            raise ValueError(
                f"{tag}: the reference establishes {covered} of {width} outputs; declare "
                f"Partial(outputs={covered}, reason=...) naming what the rest leaves open"
            )
        if claimed > min(width, covered):
            raise ValueError(
                f"{tag}: claims {claimed} outputs, implementation returns {width} "
                f"and reference establishes {covered}"
            )
        dtypes = tolerance_dtypes or tuple(_result_dtype(output) for output in outputs)
        for output, target, dtype in zip(
            outputs[:claimed], targets[:claimed], dtypes[:claimed], strict=True
        ):
            tolerance = reference_tolerance(dtype) if dtype is not None else {}
            try:
                assert_matches_reference(
                    lambda _o=output: _o, lambda _t=target: _t, **mark.tolerance(tolerance)
                )
            except AssertionError as exc:
                raise AssertionError(f"{tag}: {exc}") from exc

    def compare(
        self,
        functors: dict[str, Any],
        *inputs: Any,
        count_copies: bool = False,
        evidence: Optional[dict[str, Evidence]] = None,
    ) -> dict[str, dict]:
        """Verify, time in both tag orders, and record results under this op.

        Values are callables on ``inputs`` or ``(callable, args)`` pairs. Timing runs
        under ``no_grad``; backward baselines must invoke their autograd node directly.
        ``count_copies`` includes device copies consistently across all tags.

        ``evidence`` overrides the default reference check. Verification temporaries
        are released before sampling; ``--tileops-verify`` omits timing.

        Raises:
            ValueError: Evidence names a tag absent from the timing plan."""
        plan = {
            tag: value if isinstance(value, tuple) else (value, inputs)
            for tag, value in functors.items()
        }
        evidence = self._resolve_evidence(plan, evidence)
        # Release verification temporaries before allocating timer buffers.
        self._verify(plan, evidence, inputs)
        if verifying():
            return {}
        if any(ratio_allowed(mark) for mark in evidence.values()):
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
        tags = list(plan)
        order = tags + tags[::-1]
        # Split the budget across the two passes rather than spending it twice:
        # the point is symmetry, not more samples.
        passes = 2
        samples: dict[str, list[Sample]] = {tag: [] for tag in tags}
        meta: dict[str, dict] = {}
        for tag in order:
            functor, args = plan[tag]
            with torch.no_grad():
                samples[tag].extend(
                    bench_kernel(
                        functor,
                        args=args,
                        dry_run_ms=DRY_RUN_MS / passes,
                        repeat_ms=REPEAT_MS / passes,
                        max_iters=_MAX_ITERS // passes,
                        min_iters=max(1, _MIN_ITERS // passes),
                        count_copies=count_copies,
                    )
                )
            pass_meta = _capture_bench_meta()
            previous = meta.get(tag)
            if previous is not None and previous["timing"] != pass_meta["timing"]:
                raise RuntimeError(
                    f"{tag}: the two passes timed with different methods "
                    f"({previous['timing']} then {pass_meta['timing']}); pooling "
                    "them would report one median over two kinds of measurement. "
                    "Only reachable with --tileops-allow-events-fallback."
                )
            meta[tag] = pass_meta
        results = {tag: self._build_result(samples[tag], meta[tag]) for tag in tags}
        params = self.case_params()
        for tag in tags:
            BenchmarkReport.record(
                self.op,
                params,
                results[tag],
                tag=tag,
                unverified=describe(evidence[tag]) or "",
                ratio=ratio_allowed(evidence[tag]),
            )
        return results


class ManifestBenchmark(OpBenchmark[Any]):
    """Reads the roofline off ``op.eval_roofline()``, never off the workload.

    Called lazily while building a result, because a dynamic-shape op binds its
    roofline variables during ``forward()``.

    The benchmark reads its op's name off the instance: ``self.op_name`` is the
    wrapped class's name, which every manifest key equals by validator rule.
    """

    def __init__(self, op: Any, workload: Any):
        super().__init__(op, workload)
        self.op_name = type(op).__name__
        if self.op_name not in load_manifest():
            raise KeyError(
                f"{self.op_name} is not a manifest op; a benchmark reports under the name the "
                "manifest declares, so a wrapper or a subclass cannot stand in for one"
            )

    def case_params(self) -> dict:
        """The workload's fields, and the parameters the manifest declares for the op.

        A reduce over ``dim=0`` and one over ``dim=-1`` share a workload and are
        different cases; the manifest says which parameters an op has, and the op
        holds their values.
        """
        params = super().case_params()
        declared = (load_manifest()[self.op_name].get("signature") or {}).get("params") or {}
        for name in declared:
            if hasattr(self.op, name):
                params[name] = getattr(self.op, name)
        return params

    def compute_roof(self) -> Optional[str]:
        return self.op.compute_roof()
