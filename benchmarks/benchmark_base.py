"""The benchmark a bench file writes: a workload in, a recorded result out.

Timing lives in :mod:`benchmarks.timing`, reporting in :mod:`benchmarks.report`. Both
are re-exported here, so a bench file keeps importing what it always did.
"""

import statistics
from abc import ABC, abstractmethod
from collections.abc import Callable
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
from tileops.manifest import (
    WORKLOAD_RESERVED_KEYS,
    load_manifest,
    load_workloads,
    single_input_workload_contract,
)

__all__ = [
    "BenchmarkBase",
    "BenchmarkReport",
    "CUPTIError",
    "ManifestBenchmark",
    "backward_of",
    "bench_kernel",
    "fields",
    "then_dtype",
    "workload_params",
    "workloads_to_params",
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


def _workload_contract(op_name: str) -> tuple[str, frozenset[str]]:
    """Resolve the shared workload contract for an op known to exist."""
    sig = load_manifest()[op_name].get("signature") or {}
    contract = single_input_workload_contract(sig)
    if contract is None:
        raise KeyError(
            f"workloads_to_params({op_name!r}) needs exactly one manifest "
            "tensor input; multi-input ops use their own bench files."
        )
    return contract


class BenchmarkBase(Generic[W], ABC):
    """Turns measured latency into the op's roofline-relative metrics."""

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

    def profile(self, functor: Any, *inputs: Any) -> dict:
        """Profile a callable and return its structured result."""
        with torch.no_grad():
            return self._build_result(bench_kernel(functor, args=inputs))

    def compare(
        self,
        functors: dict[str, Any],
        *inputs: Any,
        record_as: Any = None,
        params: Optional[dict] = None,
        count_copies: bool = False,
    ) -> dict[str, dict]:
        """Time several implementations forward then reversed, and record them.

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
        if record_as is not None:
            for tag in tags:
                BenchmarkReport.record(record_as, params or {}, results[tag], tag=tag)
        return results

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
        return result


# Manifest-driven benchmark helpers


def _workload_extra_params(w: dict, shape_key: str) -> dict[str, Any]:
    """Return op-call params on a workload entry, stripping reserved keys."""
    reserved = WORKLOAD_RESERVED_KEYS | {shape_key}
    return {
        k: v
        for k, v in w.items()
        if isinstance(k, str) and k not in reserved and not k.startswith("__")
    }


def workload_params(
    workloads: list,
    build_args: "Callable[[dict, torch.dtype], tuple]",
    *,
    smoke_first: bool = False,
    marks: "Callable[[dict, torch.dtype, int], tuple] | None" = None,
) -> list:
    """The one place a manifest workload row becomes a pytest case.

    One case per (row, dtype in its ``dtypes``), ided ``f"{label}-{dtype}"`` so a
    label never has to spell a dtype. *build_args* maps a row and that dtype to
    the case's positional args — the one thing that differs per family; compose
    it with :func:`fields` and :func:`then_dtype`. *marks* takes
    ``(row, dtype, row_index)`` for marks that depend on the dtype, and
    *smoke_first* marks the first row ``smoke`` and the rest ``full``.
    """
    params: list = []
    for index, w in enumerate(workloads):
        # A row the manifest says to skip is skipped whichever else applies.
        skip = (
            (pytest.mark.skip(reason=w["bench_skip_reason"]),) if w.get("bench_skip_reason") else ()
        )
        smoke = (pytest.mark.smoke if index == 0 else pytest.mark.full,) if smoke_first else ()
        label = w.get("label", "manifest")
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            case_marks = skip or (tuple(marks(w, dtype, index)) if marks else smoke)
            params.append(
                pytest.param(
                    *build_args(w, dtype),
                    id=f"{label}-{dtype_str}",
                    marks=case_marks,
                )
            )
    return params


def fields(*keys: str, dtype_last: bool = False) -> "Callable[[dict, torch.dtype], tuple]":
    """Row values under *keys*, positionally; a key ending in ``dtype`` resolves to
    the ``torch.dtype`` it names, and *dtype_last* appends the case's dtype for a
    test whose signature ends in one the row does not name."""

    def build(w: dict, dtype: torch.dtype) -> tuple:
        values = tuple(getattr(torch, w[k]) if k.endswith("dtype") else w[k] for k in keys)
        return (*values, dtype) if dtype_last else values

    return build


def then_dtype(
    row_args: "Callable[[dict], tuple]",
    *,
    tune: bool | None = None,
) -> "Callable[[dict, torch.dtype], tuple]":
    """Append the case's dtype to what *row_args* returns, and *tune* after it when
    the test takes one. *row_args* reads the row only."""

    def build(w: dict, dtype: torch.dtype) -> tuple:
        tail = (dtype,) if tune is None else (dtype, tune)
        return (*row_args(w), *tail)

    return build


def workloads_to_params(op_name: str, include_extra: bool = False) -> list:
    """Single-tensor-input wrapper over :func:`workload_params`: reads the manifest,
    checks each row against the signature, and yields ``pytest.param(shape,
    dtype)``; with *include_extra* a third element carries the row's op-call
    params (e.g. ``{"dim": 0}``)."""
    workloads = load_workloads(op_name)  # canonical not-found error
    shape_key, allowed = _workload_contract(op_name)
    for w in workloads:
        if shape_key not in w:
            raise KeyError(
                f"workload {w.get('label', w)!r} of {op_name!r} is missing "
                f"{shape_key!r} (derived from the signature's input name)."
            )
        unknown = sorted(
            repr(k)
            for k in w
            if not isinstance(k, str) or (k not in allowed and not k.startswith("__"))
        )
        if unknown:
            raise KeyError(
                f"workload {w.get('label', w)!r} of {op_name!r} has unknown "
                f"keys {unknown}; allowed: {sorted(allowed)}."
            )

    def build(w: dict, dtype: torch.dtype) -> tuple:
        shape = tuple(w[shape_key])
        if not include_extra:
            return (shape, dtype)
        # Copy the extras per case so mutation in one cannot leak into a later
        # case sharing the row.
        return (shape, dtype, dict(_workload_extra_params(w, shape_key)))

    # Name a copy: an unlabelled row is named by its shape, as the reports did
    # before, and the rows are the manifest's own dicts.
    rows = [
        w if "label" in w else {**w, "label": "x".join(str(s) for s in w[shape_key])}
        for w in workloads
    ]
    return workload_params(rows, build)


class ManifestBenchmark(BenchmarkBase[Any]):
    """Reads the roofline off ``op.eval_roofline()``, never off the workload.

    Called lazily while building a result, because a dynamic-shape op binds its
    roofline variables during ``forward()``.
    """

    def __init__(
        self,
        # Nothing reads it at run time; validate_manifest parses this argument to
        # tie the benchmark file to a manifest entry, so it must stay a literal.
        op_name: str,
        op: Any,
        workload: Any,
    ):
        super().__init__(workload)
        self._op = op
        self._roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            flops, mem_bytes = self._op.eval_roofline()
            self._roofline_cache = (float(flops), float(mem_bytes))
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]

    def compute_roof(self) -> Optional[str]:
        return self._op.compute_roof()
