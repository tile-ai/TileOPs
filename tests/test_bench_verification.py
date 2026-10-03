"""The verification step must reject a tag whose result is wrong.

One case per way a check could pass something it should not, each calling the verification
on the plan ``compare()`` would build.
"""

import types

import pytest
import torch

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def bench():
    """What the benchmark layer exposes, imported here because tests may not at module level."""
    from benchmarks import benchmark_base, verification

    return types.SimpleNamespace(
        verify=benchmark_base.OpBenchmark._verify,
        resolve=benchmark_base.OpBenchmark._resolve_evidence,
        Exact=verification.Exact,
        Partial=verification.Partial,
        Custom=verification.Custom,
        zeroed_input=verification.zeroed_input,
        Control=verification.NegativeControl,
        OpBenchmark=benchmark_base.OpBenchmark,
    )


def _workload(ref=None):
    """A benchmark carrying nothing but the workload the verification reads."""
    return types.SimpleNamespace(workload=types.SimpleNamespace(ref_program=ref, dtype=None))


def test_in_place_tag_cannot_hide_behind_the_restore(bench):
    """A tag writing its input returns an alias; restoring the input must not fix it."""
    inputs = (torch.tensor([1.0]),)
    plan = {"tileops": (lambda x: x.add_(9), inputs)}
    with pytest.raises(AssertionError):
        bench.verify(_workload(ref=lambda x: x), plan, {"tileops": bench.Exact()}, inputs)


def test_one_tag_cannot_write_another_tags_arguments(bench):
    """A tag's own arguments are restored too, so the next tag reads what the call made."""
    shared = torch.tensor([1.0])
    inputs = (torch.tensor([1.0]),)
    # The writer returns the right answer and leaves the argument one too high.
    plan = {
        "writer": (lambda y: y.add_(1) - 1, (shared,)),
        "reader": (lambda y: y - 1, (shared,)),
    }
    evidence = {"writer": bench.Exact(), "reader": bench.Exact()}
    with pytest.raises(AssertionError):
        bench.verify(_workload(ref=lambda x: x), plan, evidence, inputs)


def test_the_reference_is_checked_when_it_runs_on_other_arguments(bench):
    """Skipping the oracle against itself holds only where it is called the same way."""
    reference = lambda x: 2 * x  # noqa: E731
    inputs = (torch.tensor([1.0]),)
    plan = {"torch": (reference, (torch.tensor([2.0]),))}
    with pytest.raises(AssertionError):
        bench.verify(_workload(ref=reference), plan, {"torch": bench.Exact()}, inputs)


def test_a_list_result_counts_its_outputs(bench):
    """A result returned as a list is a sequence of outputs, as the assertion reads it."""
    inputs = (torch.tensor([1.0]),)
    plan = {"tileops": (lambda x: [x, torch.tensor([999.0])], inputs)}
    with pytest.raises(ValueError, match="establishes 1 of 2 outputs"):
        bench.verify(_workload(ref=lambda x: x), plan, {"tileops": bench.Exact()}, inputs)


def test_partial_must_establish_an_output(bench):
    """`Partial(outputs=0)` would keep a ratio while nothing was compared."""
    with pytest.raises(ValueError, match="establishes nothing"):
        bench.Partial(outputs=0, reason="nothing")


@pytest.mark.parametrize("declared", ["partial", "custom"])
def test_a_declaration_needing_an_oracle_falls_to_unestablished(bench, declared):
    """With no reference, a declaration claims a check that nothing can run."""
    mark = (
        bench.Partial(outputs=1, reason="the state is unchecked")
        if declared == "partial"
        else bench.Custom(lambda got, want: None, "a draw")
    )
    plan = {"tileops": (lambda x: x, ())}
    resolved = bench.resolve(_workload(), plan, {"tileops": mark})
    assert resolved["tileops"].kind == "unestablished"


def test_the_tolerance_follows_the_result_not_an_unrelated_input(bench):
    """A bf16 input the op does not compute in must not loosen an fp32 comparison."""
    inputs = (torch.tensor([0], dtype=torch.bfloat16), torch.tensor([1.0]))
    plan = {"tileops": (lambda unused, x: x + 0.01, inputs)}
    with pytest.raises(AssertionError):
        bench.verify(_workload(ref=lambda unused, x: x), plan, {"tileops": bench.Exact()}, inputs)


def test_available_reference_is_checked_without_an_explicit_declaration(bench):
    plan = {"tileops": (lambda x: x, ())}
    resolved = bench.resolve(_workload(ref=lambda x: x), plan, None)
    assert resolved["tileops"].kind == "exact"


def test_local_reference_checks_compiled_tag_without_workload_reference(bench):
    inputs = (torch.tensor([1.0]),)
    plan = {"torch-compile": (lambda x: x + 1, inputs)}
    declared = {"torch-compile": bench.Exact(reference=lambda x: x)}
    resolved = bench.resolve(_workload(), plan, declared)
    assert resolved["torch-compile"].kind == "exact"
    with pytest.raises(AssertionError):
        bench.verify(_workload(), plan, resolved, inputs)


def test_reference_runs_once_and_input_is_restored_after_failure(bench):
    inputs = (torch.tensor([1.0]),)
    calls = []

    def reference(x):
        calls.append(True)
        return x.add_(1)

    plan = {"first": (lambda x: x + 1, inputs), "wrong": (lambda x: x.add_(9), inputs)}
    evidence = dict.fromkeys(plan, bench.Exact())
    with pytest.raises(AssertionError):
        bench.verify(_workload(ref=reference), plan, evidence, inputs)
    assert len(calls) == 1
    torch.testing.assert_close(inputs[0], torch.tensor([1.0]))


def test_partial_checks_only_the_claimed_outputs(bench):
    inputs = (torch.tensor([1.0]),)
    plan = {"tileops": (lambda x: [x, x + 9], inputs)}
    evidence = {"tileops": bench.Partial(outputs=1, reason="only the first output is established")}
    bench.verify(_workload(ref=lambda x: [x, x]), plan, evidence, inputs)


def test_low_precision_output_cannot_loosen_other_outputs(bench):
    inputs = (torch.tensor([1.0]),)
    plan = {"tileops": (lambda x: (x.bfloat16(), x + 0.01), inputs)}
    with pytest.raises(AssertionError):
        bench.verify(
            _workload(ref=lambda x: (x.bfloat16(), x)), plan, {"tileops": bench.Exact()}, inputs
        )


def test_mapping_output_is_copied_before_restoration(bench):
    inputs = (torch.tensor([1.0]),)
    plan = {"tileops": (lambda x: {"out": x.add_(9)}, inputs)}
    with pytest.raises(AssertionError):
        bench.verify(_workload(ref=lambda x: {"out": x}), plan, {"tileops": bench.Exact()}, inputs)


@pytest.mark.parametrize("value, atol", [(0.001, 0.01), (1.0, 2.0)])
def test_negative_control_detects_weak_draw_or_tolerance(bench, value, atol):
    inputs = (torch.tensor([value]),)
    mark = bench.Exact(atol=atol, rtol=0, controls=(bench.zeroed_input(0, "dropped-input"),))
    with pytest.raises(ValueError, match="negative control.*was accepted"):
        bench.verify(
            _workload(ref=lambda x: x), {"tag": (lambda x: x, inputs)}, {"tag": mark}, inputs
        )


def test_negative_control_reuses_reference_and_runs_once_per_comparator(bench):
    inputs = (torch.tensor([1.0]),)
    calls = []

    def reference(x):
        calls.append(True)
        return x + 1

    mark = bench.Exact(controls=(bench.zeroed_input(0, "dropped-input"),))
    plan = {"one": (lambda x: x + 1, inputs), "two": (lambda x: x + 1, inputs)}
    bench.verify(_workload(ref=reference), plan, dict.fromkeys(plan, mark), inputs)
    assert len(calls) == 2  # One oracle result and one fault, shared by both implementations.


def test_partial_control_must_change_the_checked_prefix(bench):
    inputs = (torch.tensor([1.0]),)
    control = bench.Control("only-unchecked-output", lambda reference, args: (args[0], args[0] + 9))
    mark = bench.Partial(outputs=1, reason="first output only", controls=(control,))
    with pytest.raises(ValueError, match="was accepted"):
        bench.verify(
            _workload(ref=lambda x: (x, x)),
            {"tag": (lambda x: (x, x), inputs)},
            {"tag": mark},
            inputs,
        )


def test_normal_benchmark_rejects_bad_warmup_before_sampling(bench, monkeypatch):
    from benchmarks import benchmark_base, verification

    monkeypatch.setattr(verification, "_VERIFYING", False)
    monkeypatch.setattr(
        benchmark_base, "bench_kernel", lambda *_a, **_k: pytest.fail("timed a wrong result")
    )
    benchmark = bench.OpBenchmark(object(), types.SimpleNamespace(ref_program=lambda x: x))
    with pytest.raises(AssertionError):
        benchmark.compare({"wrong": lambda x: x + 9}, torch.tensor([1.0]))


@pytest.mark.parametrize("fault", ["key-for-query", "mask-bypassed"])
def test_attention_faults_are_rejected(bench, fault):
    q = torch.tensor([[3.0, 0.0], [0.0, 1.0]])
    k = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    mask = torch.tensor([[True, False], [True, True]])
    inputs = (q, k, mask)

    def reference(q, k, mask):
        return (q @ k.T).masked_fill(~mask, float("-inf")).softmax(-1)

    def faulty(ref, args):
        q, k, mask = args
        return ref(
            k if fault == "key-for-query" else q,
            k,
            torch.ones_like(mask) if fault == "mask-bypassed" else mask,
        )

    mark = bench.Exact(controls=(bench.Control(fault, faulty),))
    bench.verify(
        _workload(ref=reference), {"op": (lambda *a: reference(*a), inputs)}, {"op": mark}, inputs
    )


def test_control_execution_failure_is_not_a_successful_rejection(bench):
    def broken(_reference, _inputs):
        raise RuntimeError("invalid fault")

    inputs = (torch.ones(1),)
    mark = bench.Exact(controls=(bench.Control("broken", broken),))
    with pytest.raises(RuntimeError, match="invalid fault"):
        bench.verify(
            _workload(ref=lambda x: x), {"op": (lambda x: x, inputs)}, {"op": mark}, inputs
        )


def test_control_is_computed_once_but_checked_with_each_tags_tolerance(bench):
    calls = []

    def fault(reference, inputs):
        calls.append(True)
        return reference(*inputs) + 0.1

    inputs = (torch.ones(1),)
    control = bench.Control("offset", fault)
    marks = {
        "strict": bench.Exact(atol=0.01, rtol=0, controls=(control,)),
        "loose": bench.Exact(atol=0.2, rtol=0, controls=(control,)),
    }
    with pytest.raises(ValueError, match="loose.*was accepted"):
        bench.verify(
            _workload(ref=lambda x: x), {tag: (lambda x: x, inputs) for tag in marks}, marks, inputs
        )
    assert len(calls) == 1


def test_custom_control_uses_the_declared_validator(bench):
    inputs = (torch.ones(1),)
    mark = bench.Custom(
        lambda a, b: torch.testing.assert_close(a, b),
        "numeric contract",
        controls=(bench.zeroed_input(0, "dropped"),),
    )
    bench.verify(_workload(ref=lambda x: x), {"op": (lambda x: x, inputs)}, {"op": mark}, inputs)


def test_quantization_validator_rejects_scale_and_code_faults():
    from benchmarks.verification import assert_quantized

    scale = torch.ones(1)
    q = torch.tensor([1.0, 2.0, -1.0]).to(torch.float8_e4m3fn)
    assert_quantized((q, scale), (q, scale))
    for actual in [(q, scale * 2), (torch.zeros_like(q), scale)]:
        with pytest.raises(AssertionError):
            assert_quantized(actual, (q, scale))


def test_normalized_validator_rejects_wrong_nonfinite_values_and_scale():
    from benchmarks.verification import assert_normalized_error

    expected = torch.tensor([1.0, 2.0, float("-inf")])
    assert_normalized_error(expected, expected)
    for actual in [torch.tensor([2.0, 4.0, float("-inf")]), torch.tensor([1.0, 2.0, float("inf")])]:
        with pytest.raises(AssertionError):
            assert_normalized_error(actual, expected)


def test_mask_validator_checks_boundary_values_and_distant_mask_faults():
    from benchmarks.verification import logit_mask_validator

    logits = torch.tensor([[1.0, 2.0, 3.0]])
    expected = torch.tensor([[float("-inf"), 2.0, 3.0]])
    validate = logit_mask_validator(logits, torch.tensor([[True, False, False]]))
    validate(logits, expected)  # A boundary token may be retained.
    for actual in [torch.tensor([[9.0, 2.0, 3.0]]), torch.tensor([[1.0, float("-inf"), 3.0]])]:
        with pytest.raises(AssertionError):
            validate(actual, expected)


def test_int8_quantization_rejects_more_than_one_code_step():
    from benchmarks.verification import assert_quantized

    expected = (torch.tensor([-127, 0, 127], dtype=torch.int8), torch.ones(1))
    assert_quantized((torch.tensor([-126, 1, 126], dtype=torch.int8), torch.ones(1)), expected)
    with pytest.raises(AssertionError):
        assert_quantized((torch.zeros(3, dtype=torch.int8), torch.ones(1)), expected)
