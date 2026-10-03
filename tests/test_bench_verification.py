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
