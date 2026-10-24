"""Routed MoE verification accepts the output of every admissible routing of a near-tied token."""

import math

import pytest
import torch

from workloads import moe
from workloads.moe import (
    FusedMoESharedExpertWorkload,
    FusedMoEWorkload,
    FusedTopKWorkload,
    _admissible_reference,
    ref_routed_experts,
)
from workloads.numerics import compare_outputs, verify
from workloads.workload_base import manifest_call

pytestmark = pytest.mark.smoke

H, F = 16, 8
# Groups of two experts; each line is one group's selection scores. Group scores are the sum
# of the two, as a correction bias makes them.
GROUP_TIE = [[0.9, 0.85], [0.6, 0.3], [0.6 - 2e-6, 0.3], [0.1, 0.1]]
EXPERT_TIE = [[0.9, 0.5], [0.5 + 2e-6, 0.1], [0.3, 0.2], [0.1, 0.1]]
GROUP_THEN_EXPERT_TIE = [[0.9, 0.85], [0.6, 0.3], [0.45, 0.45], [0.1, 0.1]]


def _case(selection, top_k, shared=False):
    """A one-token biased grouped routing over *selection* with weights large enough that
    any change of routed expert moves the output far beyond the verification bound."""
    scores = torch.tensor([sum(selection, [])])
    num_experts = scores.shape[-1]
    some = ["correction_bias"] + (["shared_w_gate_up", "shared_w_down"] if shared else [])
    call = manifest_call(
        "FusedMoESharedExpertFwdOp" if shared else "FusedMoEFwdOp",
        {"D": "bfloat16"},
        T=1,
        E=num_experts,
        H=H,
        F=F,
        top_k=top_k,
        num_expert_group=len(selection),
        topk_group=2,
        scoring_func="sigmoid",
        renormalize=True,
        some=some,
        **({"S": F} if shared else {}),
    )
    workload = (FusedMoESharedExpertWorkload if shared else FusedMoEWorkload)(call, device="cpu")
    g = torch.Generator().manual_seed(0)
    hidden = torch.randn(1, H, generator=g).to(torch.bfloat16)
    w_gate_up = torch.randn(num_experts, 2 * F, H, generator=g).to(torch.bfloat16)
    w_down = (torch.randn(num_experts, H, F, generator=g) * 0.5).to(torch.bfloat16)
    inputs = [hidden, torch.logit(scores), w_gate_up, w_down, torch.zeros(num_experts)]
    if shared:
        inputs += [torch.randn(2 * F, H, generator=g).to(torch.bfloat16)]
        inputs += [torch.randn(H, F, generator=g).to(torch.bfloat16)]
    return workload, tuple(inputs), scores


def _routed_output(inputs, scores, ids):
    weights = scores[:, ids]
    weights = weights / weights.sum(-1, keepdim=True)
    ids = torch.tensor([ids], dtype=torch.int32)
    return ref_routed_experts(inputs[0], inputs[2], inputs[3], weights, ids)


def _accepts(workload, inputs, got):
    try:
        compare_outputs(got, workload.ref_program(*inputs), workload.verification(*inputs))
    except AssertionError:
        return False
    return True


@pytest.mark.parametrize(
    "selection, top_k, admissible, inadmissible",
    [
        pytest.param(GROUP_TIE, 3, [[0, 1, 2], [0, 1, 4]], [0, 1, 6], id="group-tie"),
        pytest.param(EXPERT_TIE, 2, [[0, 2], [0, 1]], [0, 4], id="expert-tie"),
        pytest.param(
            GROUP_THEN_EXPERT_TIE,
            3,
            [[0, 1, 2], [0, 1, 4], [0, 1, 5]],
            [0, 1, 6],
            id="expert-tie-under-the-other-group-set",
        ),
    ],
)
def test_routed_verification_accepts_exactly_the_admissible_routings(
    selection, top_k, admissible, inadmissible
):
    workload, inputs, scores = _case(selection, top_k)
    outputs = [_routed_output(inputs, scores, ids) for ids in admissible]
    wrong = _routed_output(inputs, scores, inadmissible)
    # Each routing moves the output past the elementwise bound, so the verdict is the routing's.
    for other in outputs[1:] + [wrong]:
        assert ((other.float() - outputs[0].float()).abs() > 0.1).any()
    assert all(_accepts(workload, inputs, got) for got in outputs)
    assert not _accepts(workload, inputs, wrong)


def test_routed_verification_keeps_its_gate_up_control():
    workload, inputs, _ = _case(GROUP_TIE, 3)
    verify(workload.ref_program, inputs, evidence=workload.verification(*inputs), requests={})


def test_shared_expert_verification_routes_and_checks_the_shared_half_on_its_own():
    workload, inputs, scores = _case(GROUP_TIE, 3, shared=True)
    shared, _ = workload.ref_program(*inputs)
    routed = _routed_output(inputs, scores, [0, 1, 4])
    assert _accepts(workload, inputs, (shared, routed))
    assert not _accepts(workload, inputs, (shared * 1.1, routed))


@pytest.mark.parametrize(
    "gap, admissible",
    [pytest.param(0.5e-6, True, id="inside"), pytest.param(1.5e-6, False, id="just-outside")],
)
def test_topk_tie_tolerance_keeps_its_absolute_floor(gap, admissible):
    """A correction bias cancels the selection scores to near zero, where only the 1e-6
    absolute term of the tolerance can admit the runner-up."""
    call = manifest_call(
        "FusedTopKFwdOp",
        {"G": "float32"},
        T=1,
        E=4,
        top_k=1,
        scoring_func="sigmoid",
        renormalize=False,
        some=["correction_bias"],
    )
    workload = FusedTopKWorkload(call, device="cpu")
    inputs = (torch.zeros(1, 4), torch.tensor([-0.5, -0.5 - gap, -1.0, -1.0]))
    runner_up = (torch.tensor([[0.5]]), torch.tensor([[1]], dtype=torch.int32))
    assert _accepts(workload, inputs, runner_up) == admissible


@pytest.mark.parametrize("op", ["FusedTopKFwdOp", "FusedMoEFwdOp"])
@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize(
    "routing_case",
    [
        pytest.param((64, 6, 1, 1), id="ungrouped"),
        pytest.param((12, 2, 12, 6), id="grouped-union"),
    ],
)
def test_a_token_admitting_too_many_routings_is_refused(op, routing_case, monkeypatch):
    """Refuse excessive local choices or their union across otherwise admissible group sets.

    The grouped case has only 15 pairs inside each kept set, but 66 across all sets.
    """
    experts, top_k, groups, kept_groups = routing_case
    g = torch.Generator().manual_seed(0)
    gating = torch.zeros(1, experts)
    if op == "FusedTopKFwdOp":
        call = manifest_call(
            op,
            {"G": "float32"},
            T=1,
            E=experts,
            top_k=top_k,
            scoring_func="sigmoid",
            renormalize=True,
            num_expert_group=groups,
            topk_group=kept_groups,
        )
        workload, inputs = FusedTopKWorkload(call, device="cpu"), (gating,)
    else:
        call = manifest_call(
            op,
            {"D": "bfloat16"},
            T=1,
            E=experts,
            H=H,
            F=F,
            top_k=top_k,
            scoring_func="sigmoid",
            renormalize=True,
            num_expert_group=groups,
            topk_group=kept_groups,
        )
        workload = FusedMoEWorkload(call, device="cpu")
        inputs = (
            torch.randn(1, H, generator=g).to(torch.bfloat16),
            gating,
            torch.randn(experts, 2 * F, H, generator=g).to(torch.bfloat16),
            torch.randn(experts, H, F, generator=g).to(torch.bfloat16),
            None,
        )
    visited = 0
    combinations = moe.itertools.combinations

    def counted_combinations(pool, count):
        nonlocal visited
        for selection in combinations(pool, count):
            if len(pool) == groups and count == kept_groups:
                visited += 1
            yield selection

    monkeypatch.setattr(moe.itertools, "combinations", counted_combinations)
    reference = workload.ref_program(*inputs)
    with pytest.raises(ValueError, match="candidate routings|admits more than"):
        compare_outputs(reference, reference, workload.verification(*inputs))
    if groups > 1:
        assert visited < math.comb(groups, kept_groups), "refusal exhausted every group set"


def test_reference_choice_prefers_the_candidate_inside_the_elementwise_bound():
    """The candidate nearest in squared error can fail the elementwise bound that another
    candidate meets; the choice takes the one that can pass."""
    out = torch.ones(64)
    inside = torch.ones(64) + 5e-3
    nearest = torch.ones(64)
    nearest[0] += 0.03
    assert ((out - nearest) ** 2).sum() < ((out - inside) ** 2).sum()
    assert torch.equal(_admissible_reference(out, [nearest, inside], 1e-2, 1e-4), inside)


def test_reference_choice_takes_the_smallest_share_of_the_normalized_error():
    """A larger-energy candidate can leave more normalized-error room than the one nearest in
    squared error; the choice weighs both, as the bound over the whole output does."""
    out = torch.ones(64)
    nearest = torch.full((64,), 1 - 1e-3)
    roomier = torch.full((64,), 1 + 1.1e-3)
    assert ((out - nearest) ** 2).sum() < ((out - roomier) ** 2).sum()
    assert torch.equal(_admissible_reference(out, [nearest, roomier], 1e-2, 1e-4), roomier)
