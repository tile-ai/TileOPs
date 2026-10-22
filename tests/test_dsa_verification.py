"""The paged attention verifier preserves the empty selection contract."""

import pytest

from workloads.attention.dsa import DSAPagedCall
from workloads.numerics import Request, compare_outputs, verify
from workloads.workload_base import manifest_call


@pytest.fixture
def empty_selection(request):
    call = manifest_call(
        "DSAPagedFwdOp",
        S_q=2,
        H=64,
        K=64,
        NP=8,
        PS=64,
        W=2,
        cache_lens="repeat(100, 2)",
    )
    workload = DSAPagedCall(call, device="cpu")
    inputs = workload.gen_inputs()
    if request.param == "all":
        inputs[-1].fill_(-1)
    else:
        inputs[-1][0, 0].fill_(-1)
    return workload, inputs


@pytest.mark.smoke
@pytest.mark.parametrize("empty_selection", ["all", "mixed"], indirect=True)
def test_empty_selection_accepts_reference(empty_selection):
    workload, inputs = empty_selection
    verify(
        workload.ref_program,
        inputs,
        evidence=workload.verification(*inputs),
        requests={"reference": Request(workload.ref_program, inputs)},
    )


@pytest.mark.smoke
@pytest.mark.parametrize("empty_selection", ["mixed"], indirect=True)
@pytest.mark.parametrize("field,value", [(0, 1.0), (1, 0.0), (1, float("inf")), (1, float("nan"))])
def test_empty_selection_rejects_wrong_result(empty_selection, field, value):
    workload, inputs = empty_selection
    expected = workload.ref_program(*inputs)
    evidence = workload.verification(*inputs)
    changed = [tensor.clone() for tensor in expected]
    changed[field][0, 0, 0] = value
    with pytest.raises(AssertionError):
        compare_outputs(tuple(changed), expected, evidence)
