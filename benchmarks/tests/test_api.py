"""What ``bench.Runner.compare`` promises a bench file, checked without timing a kernel."""

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from benchmarks import _cases
from benchmarks import api as bench
from benchmarks._cases import Entry, default_binder
from benchmarks.baselines import private_inputs
from benchmarks.report import BenchmarkReport
from benchmarks.timing import Sample
from tileops.manifest import load_manifest
from workloads.numerics import Exact, Unestablished

pytestmark = pytest.mark.smoke


class SumFwdOp:
    """Stands in for the manifest op of that name: the runner reads its class name."""

    def __call__(self, x):
        return x * 2

    def eval_roofline(self):
        return 1.0, 1.0

    def roof_key(self):
        return None


class NotAManifestOp(SumFwdOp):
    """A wrapper of the kind a benchmark must not report under."""


def _case(
    *,
    reference=lambda x: x * 2,
    binder=default_binder,
    count_copies=False,
    calls=None,
    evidence=None,
):
    def reference_counted(*inputs):
        if calls is not None:
            calls.append("reference")
        return reference(*inputs)

    workload = SimpleNamespace(
        gen_inputs=lambda: (torch.ones(4),),
        ref_program=reference_counted,
        verification=lambda *inputs: evidence or Exact(atol=0, rtol=0),
        arguments=dict,
    )
    call = SimpleNamespace(
        tensors={"x": ((4,), "float32")},
        params={"dim": [0]},
        signature=SimpleNamespace(name="SumFwdOp"),
    )
    return bench.Case("probe", call, Entry(lambda _call: workload, count_copies, binder))


def _out_of_memory(*_):
    raise torch.OutOfMemoryError("out of memory")


@pytest.fixture
def timed(monkeypatch):
    """Record every bench_kernel call, answered with one sample, and every recorded row."""
    runs, rows = [], []

    def bench_kernel(run, args=(), reset=None, count_copies=False, **_):
        runs.append(SimpleNamespace(run=run, args=args, reset=reset, count_copies=count_copies))
        return [Sample(device_busy_ms=1.0, latency_ms=1.0, n_kernels=1)]

    monkeypatch.setattr(bench, "bench_kernel", bench_kernel)
    monkeypatch.setattr(bench, "_capture_bench_meta", lambda: {"timing": "cupti"})
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        BenchmarkReport, "record", lambda *a, **k: rows.append(k | {"params": a[1]})
    )
    return SimpleNamespace(runs=runs, rows=rows)


def test_collecting_cases_creates_no_data():
    for case in bench.cases("RMSNormFwdOp"):
        assert not {"workload", "inputs", "arguments", "reference"} & set(vars(case))


def test_every_implemented_op_has_one_case_factory(monkeypatch):
    manifest = load_manifest()
    implemented = {name for name, e in manifest.items() if e.get("status") == "implemented"}
    assert implemented <= _cases.registered() <= set(manifest)

    monkeypatch.setattr(_cases._registry, "entries", None)
    monkeypatch.setattr(_cases, "_FAMILIES", ("norm", "norm"))
    with pytest.raises(ValueError, match="registered twice"):
        _cases.registered()


def test_an_op_without_a_factory_fails_at_collection(monkeypatch):
    monkeypatch.setattr(_cases._registry, "entries", {})
    with pytest.raises(KeyError, match="no benchmark case factory"):
        bench.cases("RMSNormFwdOp")


def test_a_released_case_creates_its_data_again():
    case = _case()
    first = case.inputs
    case._release()
    assert "inputs" not in vars(case) and case.inputs is not first


def test_the_runner_refuses_an_op_the_manifest_does_not_declare():
    with pytest.raises(KeyError, match="NotAManifestOp"):
        bench.Runner(NotAManifestOp(), _case())


def test_the_runner_refuses_a_case_for_another_op():
    case = _case()
    with pytest.raises(ValueError, match="cannot run case"):
        other_call = SimpleNamespace(signature=SimpleNamespace(name="AbsFwdOp"))
        bench.Runner(SumFwdOp(), dataclasses.replace(case, _call=other_call))


def test_the_reference_runs_once_for_every_implementation(timed):
    calls = []
    case = _case(calls=calls)
    bench.Runner(SumFwdOp(), case).compare(
        {"a": lambda x: x * 2, "b": lambda x: x + x, "torch": case.reference}
    )
    assert calls == ["reference"]
    assert [row["tag"] for row in timed.rows] == ["tileops", "a", "b", "torch"]


def test_the_reference_row_is_unverified_where_the_reference_could_not_run(timed):
    case = _case(reference=_out_of_memory)
    bench.Runner(SumFwdOp(), case).compare({"torch": case.reference})
    assert all(row["ratio"] is False and "memory" in row["unverified"] for row in timed.rows)


def test_a_callable_that_overwrites_the_shared_inputs_fails_before_timing(timed):
    with pytest.raises(AssertionError, match="overwrote the shared inputs"):
        bench.Runner(SumFwdOp(), _case()).compare({"in-place": lambda x: x.mul_(2)})
    assert not timed.runs


@pytest.mark.parametrize(
    "noncomparable, reference, evidence",
    [
        (False, None, None),
        (False, None, Unestablished()),
        (False, _out_of_memory, None),
        (True, None, None),
    ],
    ids=["checked", "unestablished", "reference-oom", "noncomparable"],
)
def test_explicit_args_cannot_overwrite_shared_inputs(timed, noncomparable, reference, evidence):
    case = _case(evidence=evidence, **({"reference": reference} if reference else {}))
    impl = bench.Implementation(
        run=lambda x: x.mul_(2),
        args=case.inputs,
        noncomparable_reason="different semantics" if noncomparable else None,
    )
    with pytest.raises(AssertionError, match="overwrote the shared inputs"):
        bench.Runner(SumFwdOp(), case).compare({"in-place": impl})
    assert not timed.runs


def _overwrite_then_out_of_memory(x):
    x.mul_(2)
    raise torch.OutOfMemoryError("out of memory")


@pytest.mark.parametrize("reference", [lambda x: x.mul_(2), _overwrite_then_out_of_memory])
def test_reference_timed_on_shared_inputs_must_preserve_them(timed, reference):
    case = _case(reference=reference)
    with pytest.raises(AssertionError, match="reference: overwrote the shared inputs"):
        bench.Runner(SumFwdOp(), case).compare({"torch": case.reference})
    assert not timed.runs


def test_an_in_place_implementation_runs_on_private_inputs_reset_every_round(timed):
    case = _case()
    bench.Runner(SumFwdOp(), case).compare(
        {"in-place": private_inputs(lambda x: x.mul_(2), case.inputs, 0)}
    )
    in_place = [r for r in timed.runs if r.args is not case.inputs]
    assert in_place and all(r.reset is not None for r in in_place)
    in_place[0].run(*in_place[0].args)
    in_place[0].reset()
    torch.testing.assert_close(in_place[0].args[0], case.inputs[0])
    torch.testing.assert_close(case.inputs[0], torch.ones(4))


def test_the_registered_binder_decides_how_the_op_is_called(timed):
    def binder(op, case):
        return bench.Implementation(run=lambda: op(*case.inputs), args=())

    bench.Runner(SumFwdOp(), _case(binder=binder)).compare({})
    assert timed.runs[0].args == ()


def test_every_implementation_is_timed_with_the_cases_copy_policy(timed):
    bench.Runner(SumFwdOp(), _case(count_copies=True)).compare({"a": lambda x: x * 2})
    assert [r.count_copies for r in timed.runs] == [True] * 4


def test_a_noncomparable_implementation_is_timed_without_a_ratio(timed):
    differs = bench.Implementation(run=lambda x: x * 3, noncomparable_reason="other rounding")
    bench.Runner(SumFwdOp(), _case()).compare({"vendor": differs})
    vendor = next(row for row in timed.rows if row["tag"] == "vendor")
    assert vendor["ratio"] is False and "other rounding" in vendor["unverified"]


@pytest.mark.parametrize(
    "implementations, binder",
    [
        ({"tileops": lambda x: x * 2}, default_binder),
        (
            {"vendor": bench.Implementation(run=lambda x: x, noncomparable_reason=" ")},
            default_binder,
        ),
        ({}, lambda op, case: bench.Implementation(run=op, noncomparable_reason="exempt")),
    ],
    ids=["reserved-name", "empty-reason", "tileops-exempt"],
)
def test_a_malformed_comparison_fails_before_anything_runs(timed, implementations, binder):
    calls = []
    with pytest.raises(ValueError):
        bench.Runner(SumFwdOp(), _case(binder=binder, calls=calls)).compare(implementations)
    assert not calls and not timed.runs


def test_a_single_implementation_is_reported_as_baseline(timed):
    bench.Runner(SumFwdOp(), _case()).compare(lambda x: x * 2)
    assert [row["tag"] for row in timed.rows] == ["tileops", "baseline"]


def test_a_row_records_the_case_params(timed):
    bench.Runner(SumFwdOp(), _case()).compare({})
    assert timed.rows[0]["params"] == {"x": ((4,), "float32"), "dim": [0]}


@pytest.mark.parametrize(
    "reader, recorded", [(lambda: {"experts": 96}, {"experts": 96}), (lambda: 1 / 0, {})]
)
def test_what_decided_the_bytes_travels_with_the_reading(timed, reader, recorded):
    op = SumFwdOp()
    op.roofline_data_terms = reader
    results = bench.Runner(op, _case()).compare({})
    assert results["tileops"].get("roofline_data_terms", {}) == recorded
