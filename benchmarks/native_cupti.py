"""CUPTI activity collection, via NVIDIA's cupti-python binding."""

from __future__ import annotations

import contextlib
from typing import Any, Callable

import torch

_CUPTI = None
_COLLECTOR_ACTIVE = False
_CALLBACKS_REGISTERED = False
_BUFFER_BYTES = 8 * 1024 * 1024
_BUFFER_ALIGN = 8
_RECORDS: list[dict[str, Any]] = []

# CUPTI reports drops per (context, stream) through pointers this layer does not
# hold; a drop instead shows up as an activity count attribution fails closed on.
_UNKNOWN_DROPS = 0


class NativeCUPTIError(RuntimeError):
    """The CUPTI collector is unavailable or could not be operated."""


def load_extension():
    global _CUPTI
    if _CUPTI is not None:
        return _CUPTI
    try:
        from cupti import cupti
    except Exception as exc:  # noqa: BLE001
        raise NativeCUPTIError(
            "cupti-python is unavailable. Install it with "
            "`pip install --no-deps cupti-python==12.8.0`; --no-deps is required "
            "or it downgrades torch's cuda-bindings pin."
        ) from exc
    _CUPTI = cupti
    return _CUPTI


def _buffer_requested():
    return _BUFFER_BYTES, _BUFFER_ALIGN


def _buffer_completed(records) -> None:
    # Copy the fields out and keep no record alive: the binding's other
    # accessors misread a newer libcupti's struct and raise, including from
    # __del__ at shutdown.
    for record in records:
        _RECORDS.append({
            "kind": "kernel",
            "name": str(record.name),
            "start_ns": int(record.start),
            "end_ns": int(record.end),
        })


@contextlib.contextmanager
def _phase_session():
    """Own one session, so a discovery mismatch leaves nothing for timing."""
    global _COLLECTOR_ACTIVE, _CALLBACKS_REGISTERED
    if _COLLECTOR_ACTIVE:
        raise RuntimeError("CUPTI collector is already active")
    cupti = load_extension()
    try:
        if not _CALLBACKS_REGISTERED:
            cupti.activity_register_callbacks(_buffer_requested, _buffer_completed)
            _CALLBACKS_REGISTERED = True
        _RECORDS.clear()
        cupti.activity_enable(cupti.ActivityKind.CONCURRENT_KERNEL)
    except Exception as exc:  # noqa: BLE001
        raise NativeCUPTIError(f"CUPTI collector failed to start: {exc}") from exc
    _COLLECTOR_ACTIVE = True
    try:
        yield
    finally:
        _COLLECTOR_ACTIVE = False
        try:
            cupti.activity_disable(cupti.ActivityKind.CONCURRENT_KERNEL)
        except Exception as exc:  # noqa: BLE001
            raise NativeCUPTIError(f"CUPTI collector failed to stop: {exc}") from exc


def _flush() -> list[dict[str, Any]]:
    """Return the records completed since the previous flush."""
    cupti = load_extension()
    torch.cuda.synchronize()
    try:
        cupti.activity_flush_all(1)  # CUPTI_ACTIVITY_FLAG_FLUSH_FORCED
    except Exception as exc:  # noqa: BLE001
        raise NativeCUPTIError(f"CUPTI flush failed: {exc}") from exc
    drained = list(_RECORDS)
    _RECORDS.clear()
    return drained


def collect_discovery(
    run_one: Callable[[int], None],
    n_repeat: int,
    prepare_one: Callable[[int], None],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Capture prepare and operator activity separately, untimed."""
    prepare_traces, operator_traces = [], []
    with _phase_session():
        for i in range(n_repeat):
            prepare_one(i)
            prepare_traces.append({"kernels": _flush(), "dropped": _UNKNOWN_DROPS})
            run_one(i)
            operator_traces.append({"kernels": _flush(), "dropped": _UNKNOWN_DROPS})
    return prepare_traces, operator_traces


def collect_repeats(
    run_one: Callable[[int], None],
    n_repeat: int,
    prepare_one: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Capture a complete timed trial as one ordered activity-record range."""
    with _phase_session():
        for i in range(n_repeat):
            if prepare_one is not None:
                prepare_one(i)
            run_one(i)
        return {"kernels": _flush(), "dropped": _UNKNOWN_DROPS}
