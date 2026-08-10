import gzip
import json

import pytest

from benchmarks import benchmark_base, native_cupti


class _FakeExtension:
    def __init__(self):
        self.started = 0
        self.stopped = 0
        self.checkpoints = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def checkpoint(self):
        self.checkpoints += 1
        return {
            "kernel_index": self.checkpoints,
            "dropped": 0,
        }

    def results_range(self, begin, end, dropped_begin):
        return {"kernels": [], "dropped": 0}

    def results(self):
        return {"kernels": [], "dropped": 0}


def _fresh_session() -> dict:
    return {
        "active": False,
        "collector_active": False,
        "segments": [],
        "captures": [],
        "ops": [],
        "case_id": None,
        "current_op_index": None,
        "current_capture_index": None,
    }


def test_extension_runtime_errors_are_typed():
    def fail():
        raise RuntimeError("cuptiActivityFlushAll failed")

    with pytest.raises(native_cupti.NativeCUPTIError, match="cuptiActivityFlushAll"):
        native_cupti._call_extension(fail)


def test_sol_phase_isolation_is_the_only_protocol():
    assert native_cupti.BENCHMARK_PROTOCOL == "native-cupti-sol-phase-isolated-v3"


def test_case_session_contains_multiple_op_ledgers_and_persists_one_trace(
    monkeypatch,
    tmp_path,
):
    extension = _FakeExtension()
    trace_dir = tmp_path / "traces"
    monkeypatch.setattr(native_cupti, "_EXT", extension)
    monkeypatch.setattr(native_cupti, "_SESSION", _fresh_session())
    monkeypatch.setattr(native_cupti, "_CASE_COUNTER", 0)
    monkeypatch.setattr(native_cupti.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setenv("TILEOPS_CUPTI_BENCH_FILE", "bench_example.py")
    monkeypatch.setenv("TILEOPS_CUPTI_CASE_TRACE_DIR", str(trace_dir))

    native_cupti.start_case_session("bench_example.py::test_case[param]")
    for _ in range(2):
        op_index = native_cupti.begin_op()
        native_cupti.collect_repeats(lambda _: None, 2, prepare_one=lambda _: None)
        native_cupti.finish_op(op_index, status="passed")
    native_cupti.stop_case_session()
    native_cupti.stop_case_session()

    assert extension.started == 2
    assert extension.stopped == 2
    assert extension.checkpoints == 4
    trace_path = next(trace_dir.glob("*.json.gz"))
    with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    assert payload["case_window"]["benchmark_file"] == "bench_example.py"
    assert payload["case_window"]["case_id"] == "bench_example.py::test_case[param]"
    assert payload["case_window"]["session_scope"] == "phase"
    assert "kernels" not in payload
    assert [segment["op_index"] for segment in payload["segments"]] == [1, 2]
    assert [op["status"] for op in payload["ops"]] == ["passed", "passed"]


def test_phase_isolation_uses_fresh_discovery_and_timing_collectors(
    monkeypatch,
    tmp_path,
):
    extension = _FakeExtension()
    trace_dir = tmp_path / "traces"
    monkeypatch.setattr(native_cupti, "_EXT", extension)
    monkeypatch.setattr(native_cupti, "_SESSION", _fresh_session())
    monkeypatch.setattr(native_cupti, "_CASE_COUNTER", 0)
    monkeypatch.setattr(native_cupti.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setenv("TILEOPS_CUPTI_CASE_TRACE_DIR", str(trace_dir))

    native_cupti.start_case_session("case")
    op_index = native_cupti.begin_op()
    native_cupti.collect_discovery(lambda _: None, 1, lambda _: None)
    native_cupti.collect_repeats(lambda _: None, 2, prepare_one=lambda _: None)
    native_cupti.finish_op(op_index, status="passed")
    native_cupti.stop_case_session()

    assert extension.started == 2
    assert extension.stopped == 2
    trace_path = next(trace_dir.glob("*.json.gz"))
    with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    assert [capture["capture_phase"] for capture in payload["captures"]] == [
        "discovery",
        "timing",
    ]
    assert [segment["capture_index"] for segment in payload["segments"]] == [
        0,
        0,
        1,
    ]
    assert payload["case_window"]["session_scope"] == "phase"


def test_failed_op_segment_does_not_prevent_later_op_in_same_case(monkeypatch):
    extension = _FakeExtension()
    monkeypatch.setattr(native_cupti, "_EXT", extension)
    monkeypatch.setattr(native_cupti, "_SESSION", _fresh_session())
    monkeypatch.setattr(native_cupti, "_CASE_COUNTER", 0)
    monkeypatch.setattr(native_cupti.torch.cuda, "synchronize", lambda: None)
    monkeypatch.delenv("TILEOPS_CUPTI_CASE_TRACE_DIR", raising=False)

    native_cupti.start_case_session("case")
    first_op = native_cupti.begin_op()

    def fail(_):
        raise AssertionError("numeric mismatch")

    with pytest.raises(AssertionError, match="numeric mismatch"):
        native_cupti.collect_repeats(fail, 1)
    native_cupti.finish_op(first_op, status="failed", error="numeric mismatch")

    second_op = native_cupti.begin_op()
    trace = native_cupti.collect_repeats(lambda _: None, 1)
    native_cupti.finish_op(second_op, status="passed")
    native_cupti.stop_case_session()

    assert trace["begin_checkpoint"]["kernel_index"] == 2
    assert trace["end_checkpoint"]["kernel_index"] == 3
    assert extension.started == 2
    assert extension.stopped == 2


def test_bench_kernel_failure_closes_op_but_keeps_case_session(monkeypatch):
    calls = []
    finished = []
    monkeypatch.setattr(native_cupti, "case_session_active", lambda: True)
    monkeypatch.setattr(native_cupti, "begin_op", lambda: len(calls) + 1)
    monkeypatch.setattr(
        native_cupti,
        "finish_op",
        lambda op_index, **kwargs: finished.append((op_index, kwargs["status"])),
    )

    def fake_impl(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise AssertionError("numeric mismatch")
        return 0.125

    monkeypatch.setattr(benchmark_base, "_bench_kernel_impl", fake_impl)

    with pytest.raises(AssertionError, match="numeric mismatch"):
        benchmark_base.bench_kernel(lambda: None)
    assert benchmark_base.bench_kernel(lambda: None) == 0.125
    assert finished == [(1, "failed"), (2, "passed")]
