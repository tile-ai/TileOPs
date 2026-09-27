"""The benchmark a bench file writes: a workload in, a recorded result out.

Timing lives in :mod:`benchmarks.timing`, reporting in :mod:`benchmarks.report`. Both
are re-exported here, so a bench file keeps importing what it always did.
"""

import statistics
from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

import pytest
import torch

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

    def compare(
        self,
        functors: dict[str, Any],
        *inputs: Any,
        count_copies: bool = False,
    ) -> dict[str, dict]:
        """Time several implementations forward then reversed, and record them.

        Every tag is recorded under the op this benchmark was built for: the row
        names what ran, so no call site can put one op's numbers under another's
        name.

        Timing each one twice in opposite orders keeps drift across the case
        from landing on whichever ran last. A value is a callable timed on
        *inputs*, or a ``(callable, args)`` pair. Every callable runs under
        ``no_grad``: a graph built inside the timed region is host work, and a
        backward reached through autograd runs where the timer cannot attribute
        it. A backward baseline is timed by applying its node directly.

        ``count_copies`` puts device-to-device copies into every tag's reading, for a
        case where an implementation computes part of the result with one. It belongs to
        the case rather than the tag: reading one side with copies and the other without
        compares two instruments.
        """
        plan = {
            tag: value if isinstance(value, tuple) else (value, inputs)
            for tag, value in functors.items()
        }
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
                    "Only reachable with TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK=1."
                )
            meta[tag] = pass_meta
        results = {tag: self._build_result(samples[tag], meta[tag]) for tag in tags}
        params = self.case_params()
        for tag in tags:
            BenchmarkReport.record(self.op, params, results[tag], tag=tag)
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
