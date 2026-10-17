"""Which outputs the MoE verifications accept.

Each op's contract leaves some decisions open: which intermediates are held in the operand
dtype, the order of float32 sums, how closely the activation's sigmoid or Gaussian CDF is
evaluated, which expert fills a top-k tie, and which groups fill a group tie. An output any
admissible decision produces is accepted; one that takes a decision the contract does not
admit, or that takes an admissible one and then computes a later step wrongly, is rejected.
Every case feeds a constructed output on the CPU; no kernel runs.
"""

import itertools
import math

import pytest
import torch

from workloads.moe import (
    FusedMoESharedExpertWorkload,
    FusedMoEWorkload,
    FusedTopKWorkload,
    MoEExpertMLPWorkload,
    MoEGroupedGemmWorkload,
    post_permute_verification,
    routed_experts_verification,
)
from workloads.numerics import compare_outputs
from workloads.workload_base import manifest_call

pytestmark = pytest.mark.smoke

_BF16 = torch.bfloat16
_ALPHA, _LIMIT = 1.702, 7.0


def _hold(x: torch.Tensor, dtype: torch.dtype, mantissa_bits: int | None = None) -> torch.Tensor:
    """*x* stored in *dtype* and read back in float32, or rounded to *mantissa_bits*."""
    if mantissa_bits is None:
        return x.to(dtype).float()
    exponent = torch.floor(torch.log2(x.abs().clamp(min=1e-30)))
    step = torch.exp2(exponent - mantissa_bits)
    return torch.round(x / step) * step


def _routed_case(activation="swigluoai", seed=0, tokens=8, experts=3, top_k=2, hidden=8, ffn=4):
    """Small routed inputs on the CPU: unit-scale results, a swigluoai gate/up result at std 4,
    each bias half its product's std, every token's experts distinct."""
    g = torch.Generator().manual_seed(seed)
    spread = 4.0 if activation == "swigluoai" else 1.0
    hidden_states = torch.randn(tokens, hidden, generator=g).to(_BF16)
    w_gate_up = (torch.randn(experts, 2 * ffn, hidden, generator=g) * spread / hidden**0.5).to(
        _BF16
    )
    w_down = (torch.randn(experts, hidden, ffn, generator=g) / ffn**0.5).to(_BF16)
    b_gate_up = (torch.randn(experts, 2 * ffn, generator=g) * spread / 2).to(_BF16)
    b_down = (torch.randn(experts, hidden, generator=g) / 2).to(_BF16)
    ids = torch.stack([torch.randperm(experts, generator=g)[:top_k] for _ in range(tokens)])
    weights = torch.rand(tokens, top_k, generator=g) + 0.1
    weights = weights / weights.sum(-1, keepdim=True)
    return dict(
        hidden=hidden_states,
        w_gate_up=w_gate_up,
        w_down=w_down,
        topk_weights=weights,
        topk_ids=ids.int(),
        b_gate_up=b_gate_up,
        b_down=b_down,
        activation=activation,
    )


def _activate(gu: torch.Tensor, activation: str, fault, hold_factor, dtype) -> torch.Tensor:
    """The gated activation by definition, with an activation fault or a held gate factor."""
    if activation == "swigluoai":
        if fault == "stacked-read":
            gate, up = gu.chunk(2, dim=-1)
        elif fault == "pair-swap":
            gate, up = gu[..., 1::2], gu[..., ::2]
        else:
            gate, up = gu[..., ::2], gu[..., 1::2]
        if fault != "no-clamp":
            gate = gate.clamp(max=_LIMIT)
            if fault == "up-lower-clamp-missing":
                up = up.clamp(max=_LIMIT)
            elif fault != "up-clamp-missing":
                up = up.clamp(-_LIMIT, _LIMIT)
        alpha = 1.0 if fault == "alpha-1" else _ALPHA
        if fault == "tanh-sigmoid":
            sig = 0.5 + 0.5 * torch.tanh(alpha * gate / 2)
        elif fault == "factor-off-bound":
            sig = torch.sigmoid(alpha * gate) + 2.0**-7
        else:
            sig = torch.sigmoid(alpha * gate)
        factor = gate * sig
        if hold_factor:
            factor = _hold(factor, dtype)
        return factor * (up if fault == "no-plus-one" else up + 1)
    gate, up = gu.chunk(2, dim=-1)
    act = (
        torch.nn.functional.silu(gate)
        if activation == "silu_and_mul"
        else torch.nn.functional.gelu(gate)
    )
    if hold_factor:
        act = _hold(act, dtype)
    return act * up


def _routed(case, scale=1.0, hold=frozenset(), fault=None, ids=None, weights=None):
    """The routed experts' output by definition in float32, holding the *hold* points in the
    operand dtype, with at most one *fault*; *ids* and *weights* override the routing."""
    h, w1, w2 = case["hidden"], case["w_gate_up"], case["w_down"]
    b1, b2 = case["b_gate_up"], case["b_down"]
    ids = case["topk_ids"] if ids is None else ids
    weights = case["topk_weights"] if weights is None else weights
    dtype, act = h.dtype, case["activation"]
    tokens, top_k = ids.shape
    terms = torch.zeros(tokens, top_k, h.shape[-1])
    for t, k in itertools.product(range(tokens), range(top_k)):
        e = int(ids[t, k])
        product = h[t].float() @ w1[e].float().T
        if fault == "gate-up-held-before-bias":
            product = _hold(product, dtype)
        bias = b1[e].float()
        if fault == "b-gate-up-stacked":
            bias = torch.stack(bias.chunk(2), dim=-1).flatten()
        gu = product if fault == "drop-b-gate-up" else product + bias
        if fault == "gate-up-held-coarser":
            gu = _hold(gu, dtype, 3)
        elif "gate_up" in hold:
            gu = _hold(gu, dtype)
        a = _activate(gu, act, fault, "factor" in hold, dtype)
        if "activated" in hold:
            a = _hold(a, dtype)
        d = torch.zeros(h.shape[-1]) if fault == "drop-down-gemm" else a @ w2[e].float().T
        if fault != "drop-b-down" and fault != "b-down-after-weight":
            d = d + b2[e].float()
        if fault == "down-held-coarser":
            d = _hold(d, dtype, 3)
        elif "down" in hold:
            d = _hold(d, dtype)
        wd = d * weights[t, k]
        if fault == "b-down-after-weight":
            wd = wd + b2[e].float()
        if "weighted" in hold:
            wd = _hold(wd, dtype)
        terms[t, k] = wd
    order = range(top_k - 1, -1, -1) if fault == "reverse-combine" else range(top_k)
    total = torch.zeros(tokens, h.shape[-1])
    for k in order:
        total = total + terms[:, k]
        if "partial" in hold:
            total = _hold(total, dtype)
    if "sum" in hold:
        total = _hold(total, dtype)
    out = total * (1.0 if fault == "no-scale" else scale)
    if fault == "zero":
        out = torch.zeros_like(out)
    if fault == "one-token-zeroed":
        out[3] = 0
    return out.to(dtype)


def _accepts(verification, got, expected) -> bool:
    try:
        compare_outputs(got, expected, verification)
    except AssertionError:
        return False
    return True


def _routed_verification(case, scale=1.0):
    return routed_experts_verification(
        case["hidden"],
        case["w_gate_up"],
        case["w_down"],
        case["topk_weights"],
        case["topk_ids"],
        case["activation"],
        scale,
        b_gate_up=case["b_gate_up"],
        b_down=case["b_down"],
    )


# Decision: where the routed computation holds an intermediate in the operand dtype, in what
# order it sums, and how closely it evaluates the activation's sigmoid or Gaussian CDF.


@pytest.mark.parametrize(
    ("activation", "hold", "variant", "scale"),
    [
        pytest.param("swigluoai", frozenset(), None, 1.0, id="nothing-held"),
        pytest.param(
            "swigluoai", frozenset({"activated", "down"}), None, 1.0, id="activated-and-down-held"
        ),
        pytest.param(
            "swigluoai",
            frozenset({"gate_up", "factor", "activated", "weighted", "partial"}),
            None,
            1.0,
            id="gate-up-factor-weighted-partials-held",
        ),
        pytest.param(
            "swigluoai",
            frozenset({"gate_up", "factor", "activated", "down", "weighted", "partial", "sum"}),
            None,
            2.5,
            id="every-point-held-scaled",
        ),
        pytest.param(
            "swigluoai", frozenset({"down"}), "reverse-combine", 1.0, id="routes-summed-in-reverse"
        ),
        pytest.param("swigluoai", frozenset(), "tanh-sigmoid", 1.0, id="sigmoid-as-tanh"),
        pytest.param(
            "silu_and_mul",
            frozenset({"gate_up", "factor", "activated"}),
            None,
            1.0,
            id="silu-gate-up-and-factor-held",
        ),
        pytest.param(
            "gelu_and_mul",
            frozenset({"gate_up", "factor", "activated"}),
            None,
            1.0,
            id="gelu-gate-up-and-factor-held",
        ),
    ],
)
def test_routed_accepts_every_admissible_rounding(activation, hold, variant, scale):
    """An output that holds any of the named intermediates in the operand dtype, sums the routes
    in another order or takes the sigmoid through tanh is accepted."""
    case = _routed_case(activation)
    reference = _routed(case, scale)
    got = _routed(case, scale, hold, variant)
    assert _accepts(_routed_verification(case, scale), got, reference)


@pytest.mark.parametrize(
    "fault",
    [
        "gate-up-held-before-bias",
        "gate-up-held-coarser",
        "down-held-coarser",
        "factor-off-bound",
    ],
)
def test_routed_rejects_a_rounding_the_contract_does_not_admit(fault):
    """Holding the unbiased gate/up product, holding a value coarser than the operand dtype, or
    evaluating the sigmoid past its error bound is rejected."""
    case = _routed_case()
    if fault == "gate-up-held-before-bias":
        # Token 0's first gate is a large product its bias cancels, so holding the product
        # before the bias moves the gate by far more than the gate's own rounding.
        e = int(case["topk_ids"][0, 0])
        case["w_gate_up"][e, 0] = (case["hidden"][0].float().sign() * 8).to(_BF16)
        _place_gate_up(case, 0, 0, 0.25)
        _place_gate_up(case, 0, 1, 6.0)
    got = _routed(case, hold=frozenset({"activated", "down"}), fault=fault)
    assert not _accepts(_routed_verification(case), got, _routed(case))


def _place_gate_up(case, token, index, target):
    """Set the bias of token's first expert so its gate/up element *index* lands on *target*."""
    e = int(case["topk_ids"][token, 0])
    product = case["hidden"][token].float() @ case["w_gate_up"][e, index].float()
    case["b_gate_up"][e, index] = (target - product).to(_BF16)


@pytest.mark.parametrize(
    "fault",
    [
        "stacked-read",
        "pair-swap",
        "no-clamp",
        "up-clamp-missing",
        "up-lower-clamp-missing",
        "alpha-1",
        "no-plus-one",
        "drop-b-gate-up",
        "b-gate-up-stacked",
        "drop-b-down",
        "b-down-after-weight",
        "drop-down-gemm",
        "no-scale",
        "zero",
        "one-token-zeroed",
    ],
)
def test_routed_rejects_an_inexact_step_after_an_admissible_rounding(fault):
    """With the gate/up product, the gate factor, the activated rows, the weighted products and
    the partial sums held, each computation the contract fixes, computed wrongly alone, is
    rejected."""
    case = _routed_case()
    # Token 0's first expert sees a gate above the limit, and an open gate with an up above
    # and below the limit.
    for index, target in ((0, 10.0), (2, 2.5), (3, 10.0), (4, 2.5), (5, -10.0)):
        _place_gate_up(case, 0, index, target)
    hold = frozenset({"gate_up", "factor", "activated", "weighted", "partial"})
    got = _routed(case, 2.5, hold, fault)
    with pytest.raises(AssertionError, match="no admitted routing and single activation precision"):
        compare_outputs(got, _routed(case, 2.5), _routed_verification(case, 2.5))


@pytest.mark.parametrize("fault", ["one-element-past-budget", "nan"])
def test_routed_rejects_a_single_element_outside_its_budget(fault):
    """One element more than two rounding budgets off, or not finite, is rejected while every
    other element is exact."""
    case = _routed_case()
    got = _routed(case)
    exact = _routed(case).float()
    if fault == "nan":
        got[5, 2] = math.nan
    else:
        got[5, 2] = (exact[5, 2] * 1.25 + 0.5).to(_BF16)
    assert not _accepts(_routed_verification(case), got, _routed(case))


# Decision: which expert fills a top-k tie.


def _tied_case(top_k=2, renormalize=True):
    """Fused MoE inputs whose token 0 ties experts 1 and 2 for its second place, expert 0 first,
    and whose token 1 has no tie."""
    case = _routed_case(experts=4, tokens=2)
    gating = torch.tensor([[3.0, 1.0, 1.0, -2.0], [0.5, 2.0, -1.0, 1.0]])
    call = manifest_call(
        "FusedMoEFwdOp",
        {"D": "bfloat16"},
        T=2,
        E=4,
        H=8,
        F=4,
        top_k=top_k,
        scoring_func="softmax",
        renormalize=renormalize,
        activation="swigluoai",
        some=["b_gate_up", "b_down"],
    )
    workload = FusedMoEWorkload(call, device="cpu")
    inputs = (
        case["hidden"],
        gating,
        case["w_gate_up"],
        case["w_down"],
        None,
        case["b_gate_up"],
        case["b_down"],
    )
    return case, workload, inputs


def _routing(gating, ids, renormalize=True):
    scores = gating.softmax(-1)
    weights = scores.gather(1, ids.long())
    return weights / weights.sum(-1, keepdim=True) if renormalize else weights


@pytest.mark.parametrize(
    "ids",
    [
        pytest.param([[0, 1], [1, 3]], id="first-tied-expert"),
        pytest.param([[0, 2], [1, 3]], id="second-tied-expert"),
        pytest.param([[2, 0], [3, 1]], id="selection-order-permuted"),
    ],
)
def test_fused_moe_accepts_every_selection_a_tie_admits(ids):
    """Either tied expert may fill the tied place, in any order of the selection."""
    case, workload, inputs = _tied_case()
    ids = torch.tensor(ids)
    got = _routed(case, ids=ids, weights=_routing(inputs[1], ids))
    assert _accepts(workload.verification(*inputs), got, workload.ref_program(*inputs))


@pytest.mark.parametrize(
    "ids",
    [
        pytest.param([[0, 3], [1, 3]], id="expert-outside-the-tie"),
        pytest.param([[1, 2], [1, 3]], id="expert-above-the-tie-dropped"),
        pytest.param([[0, 1], [3, 0]], id="untied-token-reselected"),
        pytest.param([[0], [1]], id="one-route-fewer"),
    ],
)
def test_fused_moe_rejects_a_selection_no_tie_admits(ids):
    """Choosing an expert outside the tie, dropping one ranked above it, reselecting a token
    without a tie, or routing to fewer experts than top_k is rejected."""
    case, workload, inputs = _tied_case()
    ids = torch.tensor(ids)
    got = _routed(case, ids=ids, weights=_routing(inputs[1], ids))
    assert not _accepts(workload.verification(*inputs), got, workload.ref_program(*inputs))


@pytest.mark.parametrize("fault", ["weights-not-renormalized", "value-off"])
def test_fused_moe_rejects_an_inexact_step_after_a_tied_selection(fault):
    """Taking the other tied expert and then leaving its weights unrenormalized, or missing one
    element by more than its budget, is rejected; the untied token stays exact."""
    case, workload, inputs = _tied_case()
    ids = torch.tensor([[0, 2], [1, 3]])
    weights = _routing(inputs[1], ids)
    if fault == "weights-not-renormalized":
        weights[0] = _routing(inputs[1], ids, renormalize=False)[0]
    got = _routed(case, ids=ids, weights=weights)
    if fault == "value-off":
        got[0, 4] = (got[0, 4].float() * 1.25 + 0.5).to(_BF16)
    assert not _accepts(workload.verification(*inputs), got, workload.ref_program(*inputs))


# Decision: which groups grouped routing keeps.


def _logit(p: list) -> torch.Tensor:
    p = torch.tensor(p, dtype=torch.float64)
    return (p / (1 - p)).log().float()


def _grouped_tied_case(biased, gap=0.0):
    """Grouped fused MoE inputs, E = 8 in 4 groups of 2, 2 groups kept, top_k = 3; token 0
    ties groups 1 and 2 for the second kept place, group 0 first; token 1 has no group tie.

    Unbiased, a group scores its best expert, and token 1's groups 1 and 2 tie by the sum of
    their two best but not by the best. Biased (sigmoid with a zero correction bias), a group
    scores the sum of its two best, and token 1's groups 1 and 2 tie by their best expert but
    not by that sum. Unbiased, *gap* lowers token 0's group 2 below group 1 by that many group
    tie bands (1e-5 of group 1's score, plus 1e-6)."""
    case = _routed_case(experts=8, tokens=2)
    if biased:
        gating = torch.stack(
            [
                _logit([0.9, 0.3, 0.35, 0.25, 0.5, 0.1, 0.05, 0.05]),
                _logit([0.9, 0.3, 0.5, 0.4, 0.5, 0.05, 0.05, 0.05]),
            ]
        )
        correction_bias = torch.zeros(8)
    else:
        gating = torch.stack(
            [
                torch.tensor([3.0, 0.95, 1.0, 0.9, 1.0, -2.0, 0.1, 0.1]),
                torch.tensor([0.05, 0.05, 0.5, 0.1, 0.4, 0.2, 0.9, 0.3]).log(),
            ]
        )
        if gap:
            last = gating[0].softmax(-1)[2].item()
            band = 1e-5 * last + 1e-6
            gating[0, 4] = 1.0 + math.log1p(-gap * band / last)
        correction_bias = None
    call = manifest_call(
        "FusedMoEFwdOp",
        {"D": "bfloat16"},
        T=2,
        E=8,
        H=8,
        F=4,
        top_k=3,
        num_expert_group=4,
        topk_group=2,
        scoring_func="sigmoid" if biased else "softmax",
        renormalize=True,
        activation="swigluoai",
        some=["b_gate_up", "b_down"] + (["correction_bias"] if biased else []),
    )
    workload = FusedMoEWorkload(call, device="cpu")
    inputs = (
        case["hidden"],
        gating,
        case["w_gate_up"],
        case["w_down"],
        correction_bias,
        case["b_gate_up"],
        case["b_down"],
    )
    return case, workload, inputs


def _grouped_routing(inputs, ids, renormalize=True):
    gating, biased = inputs[1], inputs[4] is not None
    scores = gating.sigmoid() if biased else gating.softmax(-1)
    weights = scores.gather(1, ids.long())
    return weights / weights.sum(-1, keepdim=True) if renormalize else weights


@pytest.mark.parametrize(
    "biased, ids",
    [
        pytest.param(False, [[0, 2, 1], [6, 2, 7]], id="best-expert-lower-tied-group"),
        pytest.param(False, [[1, 4, 0], [6, 2, 7]], id="best-expert-higher-tied-group"),
        pytest.param(True, [[0, 2, 1], [0, 2, 3]], id="two-best-sum-lower-tied-group"),
        pytest.param(True, [[1, 4, 0], [0, 2, 3]], id="two-best-sum-higher-tied-group"),
    ],
)
def test_fused_moe_accepts_every_group_set_a_group_tie_admits(biased, ids):
    """Either tied group may fill the second kept place, under either group score; the top-k
    is then taken inside the kept groups, in any order."""
    case, workload, inputs = _grouped_tied_case(biased)
    ids = torch.tensor(ids)
    got = _routed(case, ids=ids, weights=_grouped_routing(inputs, ids))
    assert _accepts(workload.verification(*inputs), got, workload.ref_program(*inputs))


@pytest.mark.parametrize(
    "biased, ids",
    [
        pytest.param(False, [[0, 2, 4], [6, 2, 7]], id="both-tied-groups-kept"),
        pytest.param(False, [[0, 6, 1], [6, 2, 7]], id="group-outside-the-tie"),
        pytest.param(False, [[2, 4, 3], [6, 2, 7]], id="group-above-the-tie-dropped"),
        pytest.param(False, [[0, 4, 5], [6, 2, 7]], id="top-k-not-inside-the-kept-groups"),
        pytest.param(False, [[0, 4, 1], [6, 4, 7]], id="tie-by-two-best-sum-not-by-best"),
        pytest.param(True, [[0, 2, 1], [0, 4, 1]], id="tie-by-best-expert-not-by-sum"),
    ],
)
def test_fused_moe_rejects_a_group_set_no_group_tie_admits(biased, ids):
    """Keeping more groups than topk_group, a group outside the tie, dropping a group ranked
    above it, selecting other than the top-k inside the kept groups, or regrouping a token whose
    groups tie only under the other group score is rejected."""
    case, workload, inputs = _grouped_tied_case(biased)
    ids = torch.tensor(ids)
    got = _routed(case, ids=ids, weights=_grouped_routing(inputs, ids))
    assert not _accepts(workload.verification(*inputs), got, workload.ref_program(*inputs))


@pytest.mark.parametrize("gap, accepted", [(0.75, True), (1.25, False)], ids=["inside", "outside"])
def test_fused_moe_group_tie_band_is_relative_plus_absolute(gap, accepted):
    """A group three quarters of a band below the last kept one, inside only by the band's
    absolute part, may take its place; one and a quarter bands below may not."""
    case, workload, inputs = _grouped_tied_case(False, gap)
    ids = torch.tensor([[1, 4, 0], [6, 2, 7]])
    got = _routed(case, ids=ids, weights=_grouped_routing(inputs, ids))
    assert _accepts(workload.verification(*inputs), got, workload.ref_program(*inputs)) == accepted


@pytest.mark.parametrize("fault", ["weights-not-renormalized", "value-off"])
def test_fused_moe_rejects_an_inexact_step_after_a_tied_group_set(fault):
    """Keeping the other tied group and then leaving the weights unrenormalized, or missing one
    element by more than its budget, is rejected; the untied token stays exact."""
    case, workload, inputs = _grouped_tied_case(False)
    ids = torch.tensor([[1, 4, 0], [6, 2, 7]])
    weights = _grouped_routing(inputs, ids)
    if fault == "weights-not-renormalized":
        weights[0] = _grouped_routing(inputs, ids, renormalize=False)[0]
    got = _routed(case, ids=ids, weights=weights)
    if fault == "value-off":
        got[0, 4] = (got[0, 4].float() + 0.5).to(_BF16)
    assert not _accepts(workload.verification(*inputs), got, workload.ref_program(*inputs))


# Decision: which experts and groups fill a tie, for FusedMoESharedExpertFwdOp's routed half;
# its shared half has none.


def _shared_accepts(routing, ids, shared, fault=None) -> bool:
    """Whether the verification of a CPU ``FusedMoESharedExpertFwdOp`` call accepts the shared
    reference output with the routed output of the experts *ids*, weighted by their
    renormalized scores, with at most one *fault*.

    The call is routed as *routing*: ``"expert-tie"`` is :func:`_tied_case`'s gating,
    ``"group-tie"`` and ``"biased-group-tie"`` :func:`_grouped_tied_case`'s, a
    ``"group-tie-<gap>"`` that one with token 0's tied group *gap* bands lower, and
    ``"expert-chain"`` and ``"group-chain"`` :func:`_band_edge_case`'s. The experts apply
    ``silu_and_mul`` with no biases; the shared expert is passed when *shared*.
    """
    if routing == "expert-tie":
        _, tied, tied_inputs = _tied_case()
    elif routing.endswith("-chain"):
        _, tied, tied_inputs = _band_edge_case(routing)
    else:
        gap = float(routing.rpartition("-")[2]) if routing[-1].isdigit() else 0.0
        _, tied, tied_inputs = _grouped_tied_case(routing.startswith("biased"), gap)
    gating, correction_bias = tied_inputs[1], tied_inputs[4]
    p = tied.call.params
    tokens, num_experts = gating.shape
    case = _routed_case("silu_and_mul", tokens=tokens, experts=num_experts, top_k=p["top_k"])
    case["b_gate_up"] = torch.zeros_like(case["b_gate_up"])
    case["b_down"] = torch.zeros_like(case["b_down"])
    some = ["correction_bias"] if correction_bias is not None else []
    call = manifest_call(
        "FusedMoESharedExpertFwdOp",
        {"D": "bfloat16"},
        T=tokens,
        E=num_experts,
        H=8,
        F=4,
        top_k=p["top_k"],
        num_expert_group=p["num_expert_group"],
        topk_group=p["topk_group"],
        scoring_func=p["scoring_func"],
        renormalize=True,
        some=some + (["shared_w_gate_up"] if shared else []),
        **({"S": 4} if shared else {}),
    )
    workload = FusedMoESharedExpertWorkload(call, device="cpu")
    g = torch.Generator().manual_seed(1)
    shared_weights = (
        (torch.randn(8, 8, generator=g) / 8**0.5).to(_BF16),
        (torch.randn(8, 4, generator=g) / 2).to(_BF16),
    )
    inputs = (
        case["hidden"],
        gating,
        case["w_gate_up"],
        case["w_down"],
        correction_bias,
        *(shared_weights if shared else (None, None)),
    )
    ids = torch.tensor(ids)
    weights = _grouped_routing(inputs, ids)
    if fault == "weights-not-renormalized":
        weights[0] = _grouped_routing(inputs, ids, renormalize=False)[0]
    shared_output, _ = workload.ref_program(*inputs)
    routed_output = _routed(case, ids=ids, weights=weights)
    if fault == "shared-output-off":
        shared_output = (shared_output.float() * 1.05).to(_BF16)
    elif fault == "shared-output-zero":
        shared_output = torch.zeros_like(shared_output)
    elif fault == "routed-output-zero":
        routed_output = torch.zeros_like(routed_output)
    got = (shared_output, routed_output)
    return _accepts(workload.verification(*inputs), got, workload.ref_program(*inputs))


@pytest.mark.parametrize(
    "routing, ids, shared",
    [
        pytest.param("expert-tie", [[0, 1], [1, 3]], True, id="first-tied-expert"),
        pytest.param("expert-tie", [[0, 2], [1, 3]], False, id="second-tied-expert-no-shared"),
        pytest.param("expert-tie", [[2, 0], [3, 1]], True, id="selection-order-permuted"),
        pytest.param("group-tie", [[0, 2, 1], [6, 2, 7]], False, id="lower-tied-group-no-shared"),
        pytest.param("group-tie", [[1, 4, 0], [6, 2, 7]], True, id="higher-tied-group"),
        pytest.param("biased-group-tie", [[0, 2, 1], [0, 2, 3]], True, id="two-best-sum-lower"),
        pytest.param(
            "biased-group-tie", [[1, 4, 0], [0, 2, 3]], False, id="two-best-sum-higher-no-shared"
        ),
        pytest.param("group-tie-0.75", [[1, 4, 0], [6, 2, 7]], True, id="inside-the-group-band"),
        pytest.param("expert-chain", [[0, 2, 3]], True, id="expert-chain-fourth-for-the-second"),
        pytest.param("group-chain", [[0, 4, 6]], True, id="group-chain-fourth-for-the-second"),
    ],
)
def test_fused_moe_shared_expert_accepts_every_routing_a_tie_admits(routing, ids, shared):
    """Either tied expert, or either tied group under either group score, may fill the tied
    place of the routed half, in any order, with or without the shared expert; inside a chain
    of scores each within the band of the next, so may the one past the last kept."""
    assert _shared_accepts(routing, ids, shared)


@pytest.mark.parametrize(
    "routing, ids, shared, fault",
    [
        pytest.param("expert-tie", [[0, 3], [1, 3]], True, None, id="expert-outside-the-tie"),
        pytest.param(
            "expert-tie", [[1, 2], [1, 3]], False, None, id="expert-above-the-tie-dropped"
        ),
        pytest.param("expert-tie", [[0, 1], [3, 0]], True, None, id="untied-token-reselected"),
        pytest.param(
            "expert-tie",
            [[0, 2], [1, 3]],
            True,
            "weights-not-renormalized",
            id="tied-expert-weights-not-renormalized",
        ),
        pytest.param(
            "expert-tie", [[0, 2], [1, 3]], True, "shared-output-off", id="shared-output-off"
        ),
        pytest.param(
            "expert-tie", [[0, 1], [1, 3]], True, "shared-output-zero", id="all-zero-shared-output"
        ),
        pytest.param(
            "expert-tie", [[0, 1], [1, 3]], True, "routed-output-zero", id="all-zero-routed-output"
        ),
        pytest.param("group-tie", [[0, 2, 4], [6, 2, 7]], False, None, id="both-tied-groups"),
        pytest.param("group-tie", [[0, 6, 1], [6, 2, 7]], True, None, id="group-outside-tie"),
        pytest.param(
            "group-tie", [[2, 4, 3], [6, 2, 7]], False, None, id="group-above-the-tie-dropped"
        ),
        pytest.param(
            "group-tie", [[0, 4, 5], [6, 2, 7]], True, None, id="top-k-not-inside-the-kept-groups"
        ),
        pytest.param(
            "group-tie", [[0, 4, 1], [6, 4, 7]], False, None, id="tie-by-two-best-sum-not-by-best"
        ),
        pytest.param(
            "biased-group-tie", [[0, 2, 1], [0, 4, 1]], True, None, id="tie-by-best-not-by-sum"
        ),
        pytest.param(
            "group-tie-1.25", [[1, 4, 0], [6, 2, 7]], False, None, id="outside-the-group-band"
        ),
        pytest.param(
            "expert-chain", [[1, 2, 3]], True, None, id="expert-chain-fourth-for-the-best"
        ),
        pytest.param("group-chain", [[2, 4, 6]], False, None, id="group-chain-fourth-for-the-best"),
    ],
)
def test_fused_moe_shared_expert_rejects_a_routing_no_tie_admits(routing, ids, shared, fault):
    """A routed half that takes an expert or group no tie admits, the best of a chain dropped
    among them, or that takes an admitted one and then leaves its weights unrenormalized, is
    rejected, with or without the shared expert; so is a shared output off its reference, and
    an all-zero shared or routed output."""
    assert not _shared_accepts(routing, ids, shared, fault)


# The verifier against its definition: every candidate output of a small domain.


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _definition(case, ids, weights, scale, held=False):
    """Each output element's exact value and rounding budget, by the definition the routed
    verification states, element by element in float64."""
    unit, unit32 = 2.0**-8, 2.0**-24
    tiny = torch.finfo(_BF16).smallest_normal
    factor_error = 2.0**-11

    def held(v, b):
        return b + unit * (abs(v) + tiny)

    h, w1, w2 = (case[k].double() for k in ("hidden", "w_gate_up", "w_down"))
    b1, b2 = case["b_gate_up"].double(), case["b_down"].double()
    tokens, top_k = ids.shape
    hidden, ffn = h.shape[1], w2.shape[2]
    value = [[0.0] * hidden for _ in range(tokens)]
    budget = [[0.0] * hidden for _ in range(tokens)]
    for t, k in itertools.product(range(tokens), range(top_k)):
        e, w = int(ids[t, k]), float(weights[t, k])
        gu, gu_b = [], []
        for i in range(2 * ffn):
            products = [float(h[t, j] * w1[e, i, j]) for j in range(hidden)]
            v = sum(products) + float(b1[e, i])
            b = (hidden + 1) * unit32 * (sum(abs(p) for p in products) + abs(float(b1[e, i])))
            gu.append(v)
            gu_b.append(held(v, b))
        a, a_b = [], []
        for c in range(ffn):
            g, u, gate_budget, up_budget = gu[2 * c], gu[2 * c + 1], gu_b[2 * c], gu_b[2 * c + 1]
            gc, uc = min(g, _LIMIT), max(min(u, _LIMIT), -_LIMIT)
            s = _sigmoid(_ALPHA * gc)
            v = gc * s * (uc + 1)
            d_gate = (
                s * (1 + _ALPHA * gc * (1 - s)) * (uc + 1) if g < _LIMIT + 2 * gate_budget else 0.0
            )
            d_up = gc * s if abs(u) < _LIMIT + 2 * up_budget else 0.0
            b = (
                abs(d_gate) * gate_budget
                + abs(d_up) * up_budget
                + factor_error * abs(gc * (uc + 1))
            )
            a.append(v)
            a_b.append(held(v, b + unit * abs(v)))
        for o in range(hidden):
            products = [
                (float(torch.tensor(a[c], dtype=_BF16)) if held else a[c]) * float(w2[e, o, c])
                for c in range(ffn)
            ]
            d = sum(products) + float(b2[e, o])
            spread = math.sqrt(sum((a_b[c] * float(w2[e, o, c])) ** 2 for c in range(ffn)))
            d_b = spread + (ffn + 1) * unit32 * (
                sum(abs(p) for p in products) + abs(float(b2[e, o]))
            )
            d_b = held(d, d_b)
            value[t][o] += w * d
            budget[t][o] += abs(w) * (d_b + unit * abs(d))
    for t, o in itertools.product(range(tokens), range(hidden)):
        v, b = value[t][o] * scale, budget[t][o] * abs(scale)
        if scale != 1.0:
            b = held(v, b)
        value[t][o], budget[t][o] = v, held(v, b)
    return value, budget


def _assert_verdicts_equal_the_definition(verification, case, gating, admitted, routing):
    """Over every 2-expert selection of a one-token, 2-element case and every per-element
    offset in {0, +-1.9, +-2.1} budgets, *verification* accepts exactly the outputs within two
    budgets of an *admitted* selection's exact result, by :func:`_definition`."""
    expected = torch.zeros(1, 2, dtype=_BF16)
    definitions = {
        (s, held): _definition(
            case, torch.tensor([list(s)]), routing(torch.tensor([list(s)])), 1.0, held
        )
        for s in itertools.combinations(range(gating.shape[-1]), 2)
        for held in (False, True)
    }
    offsets = (0.0, 1.9, -1.9, 2.1, -2.1)
    checked = 0
    for selection, (value, budget) in definitions.items():
        for steps in itertools.product(offsets, repeat=2):
            got = [value[0][o] + steps[o] * budget[0][o] for o in range(2)]
            got32 = torch.tensor([got], dtype=torch.float32)
            accepted_by_definition = any(
                all(abs(got[o] - v[0][o]) <= 2 * b[0][o] for o in range(2))
                for (s, _), (v, b) in definitions.items()
                if s in admitted
            )
            assert _accepts(verification, got32, expected.float()) == accepted_by_definition, (
                selection,
                steps,
            )
            checked += 1
    assert checked == len(definitions) * 25


def test_routed_verification_equals_its_definition_on_a_small_domain():
    """Over every selection of a tied token and every per-element offset in {0, +-1.9, +-2.1}
    budgets, the verification accepts exactly the outputs the definition accepts."""
    from workloads.moe import _tied_routings

    case = _routed_case(tokens=1, experts=3, top_k=2, hidden=2, ffn=2)
    gating = torch.tensor([[2.0, 0.0, 0.0]])
    canonical = torch.tensor([[0, 1]])
    routings = _topk_routings_by_definition(gating.softmax(-1)[0], 1, 1, 2, False)
    admitted = {ids for _, ids in routings}
    verification = routed_experts_verification(
        case["hidden"],
        case["w_gate_up"],
        case["w_down"],
        _routing(gating, canonical),
        canonical,
        "swigluoai",
        b_gate_up=case["b_gate_up"],
        b_down=case["b_down"],
        ties=_tied_routings(gating, None, 2, "softmax", True),
    )
    _assert_verdicts_equal_the_definition(
        verification, case, gating, admitted, lambda ids: _routing(gating, ids)
    )


def test_grouped_routing_verification_equals_its_definition_on_a_small_domain():
    """E = 4 in 2 groups of 2, one group kept, top_k = 2, the two groups tied by their best
    expert: over every selection and every per-element offset in {0, +-1.9, +-2.1} budgets,
    the fused MoE verification accepts exactly the outputs within two budgets of either whole
    group."""
    case = _routed_case(tokens=1, experts=4, top_k=2, hidden=2, ffn=2)
    gating = torch.tensor([[1.0, 0.0, 1.0, -1.0]])
    routings = _topk_routings_by_definition(gating.softmax(-1)[0], 2, 1, 2, False)
    call = manifest_call(
        "FusedMoEFwdOp",
        {"D": "bfloat16"},
        T=1,
        E=4,
        H=2,
        F=2,
        top_k=2,
        num_expert_group=2,
        topk_group=1,
        scoring_func="softmax",
        renormalize=True,
        activation="swigluoai",
        some=["b_gate_up", "b_down"],
    )
    workload = FusedMoEWorkload(call, device="cpu")
    inputs = (
        case["hidden"],
        gating,
        case["w_gate_up"],
        case["w_down"],
        None,
        case["b_gate_up"],
        case["b_down"],
    )
    _assert_verdicts_equal_the_definition(
        workload.verification(*inputs),
        case,
        gating,
        {ids for _, ids in routings},
        lambda ids: _routing(gating, ids),
    )


# The grouped GEMM, the expert MLP and the post-permute reduction.


def _grouped_case(activation=None, layout=None, out_dtype=None, seed=0):
    layout = layout or {
        "contiguous": {"packing": "tight", "metadata_kind": "physical_psum", "alignment": 1}
    }
    row = dict(P=12, K=16, E=3, N=8, layout=layout, some=["bias"])
    if activation is not None:
        row["activation"] = activation
    if out_dtype is not None:
        row["out_dtype"] = out_dtype
    if "masked" in layout:
        row.pop("P")
    call = manifest_call("MoEGroupedGemmFwdOp", {"D": "bfloat16"}, **row)
    torch.manual_seed(seed)
    workload = MoEGroupedGemmWorkload(call, device="cpu")
    return workload, workload.gen_inputs()


def _grouped(inputs, layout_rows, activation=None, hold_gate_up=False, fault=None, out=_BF16):
    """The grouped GEMM by definition over (group, rows) pairs, with at most one *fault*."""
    a, b, _, bias = inputs
    width = b.shape[1] // 2 if activation else b.shape[1]
    result = torch.full((*a.shape[:-1], width), 7.0)
    for g, rows in layout_rows:
        weight = b[(g + 1) % b.shape[0]] if fault == "wrong-expert" else b[g]
        product = a[rows].float() @ weight.float().T
        if fault == "product-held-before-bias":
            product = _hold(product, _BF16)
        gb = bias[(g + 1) % bias.shape[0]] if fault == "bias-of-another-expert" else bias[g]
        gu = product if fault == "drop-bias" else product + gb.float()
        if fault == "k-sum-reversed":
            gu = a[rows].float().flip(-1) @ weight.float().flip(-1).T + gb.float()
        if hold_gate_up:
            gu = _hold(gu, _BF16)
        if activation is not None:
            gu = _activate(
                gu, activation, "stacked-read" if fault == "stacked-read" else None, False, _BF16
            )
        result[rows] = gu
    if fault == "defined-row-zeroed":
        result[layout_rows[0][1]] = 0
    return result.to(out)


def _tight_rows(metadata):
    start, rows = 0, []
    for g, end in enumerate(metadata.tolist()):
        rows.append((g, slice(start, end)))
        start = end
    return rows


@pytest.mark.parametrize(
    ("activation", "variant", "out_dtype"),
    [
        pytest.param("swigluoai", None, None, id="swigluoai-exact"),
        pytest.param("swigluoai", "gate-up-held", None, id="swigluoai-gate-up-held"),
        pytest.param(None, "k-sum-reversed", None, id="k-sum-reversed"),
        pytest.param(None, None, "float32", id="float32-output"),
    ],
)
def test_grouped_gemm_accepts_every_admissible_rounding(activation, variant, out_dtype):
    """The activation may read the gate/up product held in the operand dtype, and the K sum may
    run in any order."""
    workload, inputs = _grouped_case(activation, out_dtype=out_dtype)
    rows = _tight_rows(inputs[2])
    out = torch.float32 if out_dtype else _BF16
    got = _grouped(inputs, rows, activation, variant == "gate-up-held", variant, out)
    assert _accepts(workload.verification(*inputs), got, workload.ref_program(*inputs))


def test_grouped_gemm_accepts_anything_in_the_rows_a_layout_pads():
    """A masked slab's rows past its count are not outputs: any value there is accepted."""
    workload, inputs = _grouped_case(layout={"masked": {"max_m": 8}})
    counts = inputs[2]
    rows = [(g, (g, slice(0, int(c)))) for g, c in enumerate(counts.tolist())]
    got = _grouped(inputs, rows)
    assert int(counts.min()) < 8, "a slab with padded rows"
    assert _accepts(workload.verification(*inputs), got, workload.ref_program(*inputs))


@pytest.mark.parametrize(
    ("activation", "fault"),
    [
        pytest.param(None, "product-held-before-bias", id="product-held-before-bias"),
        pytest.param(None, "drop-bias", id="drop-bias"),
        pytest.param(None, "bias-of-another-expert", id="bias-of-another-expert"),
        pytest.param(None, "wrong-expert", id="wrong-expert"),
        pytest.param(None, "defined-row-zeroed", id="defined-row-zeroed"),
        pytest.param("swigluoai", "stacked-read", id="swigluoai-stacked-read"),
    ],
)
def test_grouped_gemm_rejects_each_violation(activation, fault):
    """Holding the unbiased product, dropping or mismatching the bias or the expert weight,
    zeroing a defined row, or reading interleaved gate/up rows as stacked is rejected."""
    workload, inputs = _grouped_case(activation)
    rows = _tight_rows(inputs[2])
    if fault == "product-held-before-bias":
        # One element whose bias cancels its product to a fraction of the product's rounding.
        a, b, _, bias = inputs
        g = next(g for g, r in rows if r.stop > r.start)
        product = a[rows[g][1].start].float() @ b[g].float().T
        bias[g, 3] = (-product[3] + 1e-4 * product[3].abs()).to(_BF16)
    got = _grouped(inputs, rows, activation, fault=fault)
    assert not _accepts(workload.verification(*inputs), got, workload.ref_program(*inputs))


def _mlp_case(seed=0):
    layout = {"contiguous": {"packing": "tight", "metadata_kind": "physical_psum", "alignment": 1}}
    call = manifest_call(
        "MoEExpertMLPFwdOp",
        {"D": "bfloat16"},
        layout=layout,
        activation="swigluoai",
        some=["b_gate_up", "b_down"],
        P=12,
        H=8,
        E=3,
        F=4096,
    )
    torch.manual_seed(seed)
    workload = MoEExpertMLPWorkload(call, device="cpu")
    return workload, workload.gen_inputs()


def _mlp(inputs, hold_activated=True, fault=None):
    x, w1, w2, metadata, b1, b2 = inputs
    out = torch.zeros(x.shape)
    for g, rows in _tight_rows(metadata):
        gu = x[rows].float() @ w1[g].float().T + b1[g].float()
        a = _activate(
            gu, "swigluoai", "stacked-read" if fault == "stacked-read" else None, False, _BF16
        )
        if hold_activated:
            a = _hold(a, _BF16)
        product = a @ w2[g].float().T
        if fault == "product-held-before-bias":
            product = _hold(product, _BF16)
        out[rows] = product if fault == "drop-b-down" else product + b2[g].float()
    return out.to(_BF16)


@pytest.mark.parametrize("hold_activated", [True, False], ids=["activated-held", "activated-fp32"])
def test_expert_mlp_accepts_the_activated_rows_held_or_not(hold_activated):
    """The activated rows between the GEMMs may be held in the operand dtype or kept in fp32."""
    workload, inputs = _mlp_case()
    got = _mlp(inputs, hold_activated)
    assert _accepts(workload.verification(*inputs), got, workload.ref_program(*inputs))


@pytest.mark.parametrize("fault", ["product-held-before-bias", "drop-b-down", "stacked-read"])
def test_expert_mlp_rejects_each_violation(fault):
    """Holding the down product before its bias, dropping the down bias, or reading interleaved
    gate/up rows as stacked is rejected."""
    workload, inputs = _mlp_case()
    if fault == "product-held-before-bias":
        # One element whose down bias cancels its product to a fraction of the product's rounding.
        x, w1, w2, metadata, b1, b2 = inputs
        g, r = next((g, r) for g, r in _tight_rows(metadata) if r.stop > r.start)
        w1[g].zero_()
        b1[g, ::2] = _LIMIT + 1
        b1[g, 1::2] = 0
        gu = x[r.start].float() @ w1[g].float().T + b1[g].float()
        a = _hold(_activate(gu, "swigluoai", None, False, _BF16), _BF16)
        # The down product lies between dtype values; its rounded value is cancelled exactly.
        w2[g, 5] = 0.123046875
        product = a @ w2[g, 5].float()
        b2[g, 5] = -_hold(product, _BF16)
    got = _mlp(inputs, fault=fault)
    assert not _accepts(workload.verification(*inputs), got, workload.ref_program(*inputs))


def _post_case(seed=0, tokens=6, top_k=3, hidden=8):
    g = torch.Generator().manual_seed(seed)
    rows = torch.randn(tokens * top_k, hidden, generator=g).to(_BF16)
    inverse = torch.randperm(tokens * top_k, generator=g).int()
    weights = torch.rand(tokens, top_k, generator=g)
    return rows, weights, inverse


def _post(rows, weights, inverse, scale, hold=frozenset(), fault=None):
    tokens, top_k = weights.shape
    gathered = rows[inverse.long()].float().view(tokens, top_k, -1)
    if fault == "inverse-shifted":
        gathered = rows[inverse.roll(1).long()].float().view(tokens, top_k, -1)
    w = torch.ones_like(weights) if fault == "weights-missing" else weights
    total = torch.zeros(tokens, rows.shape[-1])
    for k in range(top_k - 1, -1, -1) if "reverse" in hold else range(top_k):
        term = gathered[:, k] * w[:, k].unsqueeze(-1)
        if "weighted" in hold:
            term = _hold(term, rows.dtype)
        total = total + term
        if "partial" in hold:
            total = _hold(total, rows.dtype, 3 if fault == "partial-held-coarser" else None)
    if "sum" in hold:
        total = _hold(total, rows.dtype)
    return (total * (1.0 if fault == "scale-missing" else scale)).to(rows.dtype)


@pytest.mark.parametrize(
    "hold",
    [frozenset(), frozenset({"weighted", "partial"}), frozenset({"partial", "sum", "reverse"})],
    ids=["nothing-held", "weighted-and-partials-held", "partials-and-sum-held-reversed"],
)
def test_post_permute_accepts_every_admissible_rounding(hold):
    """Each weighted row, each partial sum and the sum before the scale may be held in the
    operand dtype, and the rows may be summed in any order."""
    rows, weights, inverse = _post_case()
    got = _post(rows, weights, inverse, 2.5, hold)
    verification = post_permute_verification(rows, weights, inverse, 2.5)
    assert _accepts(verification, got, _post(rows, weights, inverse, 2.5))


@pytest.mark.parametrize(
    "fault", ["partial-held-coarser", "scale-missing", "weights-missing", "inverse-shifted"]
)
def test_post_permute_rejects_each_violation(fault):
    """Holding a partial sum coarser than the operand dtype, dropping the scale or the weights,
    or gathering the wrong rows is rejected."""
    rows, weights, inverse = _post_case()
    hold = frozenset({"partial"}) if fault == "partial-held-coarser" else frozenset()
    got = _post(rows, weights, inverse, 2.5, hold, fault)
    verification = post_permute_verification(rows, weights, inverse, 2.5)
    assert not _accepts(verification, got, _post(rows, weights, inverse, 2.5))


# The tie enumeration against its definition: band edges, chains, its cap and DeepSeek-V3 rows.


def _topk_routings_by_definition(selection, num_groups, topk_group, top_k, biased) -> set:
    """Every ``(groups, expert ids)`` routing ``FusedTopKFwdOp``'s docstring admits for one
    token's selection scores, decided in float64.

    The groups are ``topk_group`` of them holding every group scored above the
    ``topk_group``-th best group score by more than 1e-5 of it plus 1e-6 and none scored below
    it by more; a group scores the float32 sum of its two best selection scores when *biased*,
    else its best. The experts are every expert of those groups above their ``top_k``-th best
    selection score by more than 1e-5 of it plus float32 epsilon, and enough within that band to make
    ``top_k``.
    """
    values = [float(v) for v in selection.tolist()]
    size = len(values) // num_groups
    members = [sorted(values[g * size : (g + 1) * size], reverse=True) for g in range(num_groups)]
    group_scores = [
        torch.tensor(m[0] + m[1], dtype=torch.float32).item() if biased else m[0] for m in members
    ]
    last = sorted(group_scores, reverse=True)[topk_group - 1]
    group_band = 1e-5 * abs(last) + 1e-6
    routings = set()
    for groups in itertools.combinations(range(num_groups), topk_group):
        others = [g for g in range(num_groups) if g not in groups]
        if any(group_scores[g] - last > group_band for g in others):
            continue
        if any(last - group_scores[g] > group_band for g in groups):
            continue
        experts = [e for g in groups for e in range(g * size, (g + 1) * size)]
        kth = sorted((values[e] for e in experts), reverse=True)[top_k - 1]
        band = 1e-5 * abs(kth) + torch.finfo(torch.float32).eps
        above = [e for e in experts if values[e] - kth > band]
        tied = [e for e in experts if abs(values[e] - kth) <= band]
        for fill in itertools.combinations(tied, top_k - len(above)):
            routings.add((groups, tuple(sorted(above + list(fill)))))
    return routings


def _exact_selection_case(target, top_k, num_groups=1, topk_group=1):
    """Fused MoE inputs of one token whose selection scores are exactly *target*, each at least
    0.25: zero logits scored by sigmoid, plus a correction bias of *target* - 0.5."""
    experts = target.numel()
    case = _routed_case(experts=experts, tokens=1, top_k=top_k)
    call = manifest_call(
        "FusedMoEFwdOp",
        {"D": "bfloat16"},
        T=1,
        E=experts,
        H=8,
        F=4,
        top_k=top_k,
        num_expert_group=num_groups,
        topk_group=topk_group,
        scoring_func="sigmoid",
        renormalize=True,
        activation="swigluoai",
        some=["b_gate_up", "b_down", "correction_bias"],
    )
    gating = torch.zeros(1, experts)
    inputs = (
        case["hidden"],
        gating,
        case["w_gate_up"],
        case["w_down"],
        target - 0.5,
        case["b_gate_up"],
        case["b_down"],
    )
    assert torch.equal(gating.sigmoid()[0] + inputs[4], target)
    return case, FusedMoEWorkload(call, device="cpu"), inputs


def _pairs(scores: torch.Tensor) -> torch.Tensor:
    """Selection scores of groups of two scoring *scores*: each group's best expert 0.5."""
    return torch.stack([torch.full_like(scores, 0.5), scores - 0.5], -1).flatten()


def _band_edge_case(name):
    """One token on the edge of a tie band.

    ``expert-past-band``: ``[0.5, 0.5, 0.5 + band, 0.5, 0.4, 0.3]``, top_k = 2; expert 2 is
    above the k-th score by more than the band, and the band added to the k-th score rounds up
    to it. ``group-past-band``: the same at the group level, six groups of two scoring
    ``[0.9, 0.9, 0.9 + band, 0.9, 0.8, 0.6]``, two kept, top_k = 2. ``expert-chain``:
    ``[0.8, 0.8 - 7e-6, 0.8 - 14e-6, 0.8 - 21e-6, 0.4, 0.3]``, top_k = 3, each expert within
    the band of the next and the best past the k-th's. ``group-chain``: six groups of two
    scoring ``[0.9, 0.9 - 7e-6, 0.9 - 14e-6, 0.9 - 21e-6, 0.8, 0.6]``, three kept, top_k = 3.
    """
    if name == "expert-past-band":
        target = torch.tensor([0.5, 0.5, 0.5 + 5e-6, 0.5, 0.4, 0.3])
        last, best = target[0], target[2]
        band = 1e-5 * last + torch.finfo(torch.float32).eps
        target[2] = last + band
        best = target[2]
        assert best - last > band and last + band >= best
        return _exact_selection_case(target, 2)
    if name == "group-past-band":
        last = torch.tensor(0.9)
        band = 1e-5 * last + 1e-6
        best = last + band
        assert best - last > band
        scores = torch.stack([last, last, best, last, torch.tensor(0.8), torch.tensor(0.6)])
        return _exact_selection_case(_pairs(scores), 2, 6, 2)
    if name == "expert-chain":
        target = torch.tensor([0.8, 0.8 - 7e-6, 0.8 - 14e-6, 0.8 - 21e-6, 0.4, 0.3])
        return _exact_selection_case(target, 3)
    scores = torch.tensor([0.9, 0.9 - 7e-6, 0.9 - 14e-6, 0.9 - 21e-6, 0.8, 0.6])
    return _exact_selection_case(_pairs(scores), 3, 6, 3)


def _accepts_routing(case, workload, inputs, ids) -> bool:
    """Whether the verification accepts the routed output of the experts *ids*, weighted by
    their renormalized scores, or the reference output when *ids* is None."""
    if ids is None:
        got = workload.ref_program(*inputs)
    else:
        ids = torch.tensor(ids)
        got = _routed(case, ids=ids, weights=_grouped_routing(inputs, ids))
    return _accepts(workload.verification(*inputs), got, workload.ref_program(*inputs))


@pytest.mark.parametrize(
    "name, ids, accepted",
    [
        pytest.param("expert-past-band", None, True, id="expert-past-band-reference"),
        pytest.param("expert-past-band", [[2, 3]], True, id="expert-past-band-kept-with-a-tie"),
        pytest.param("expert-past-band", [[0, 1]], False, id="expert-past-band-dropped"),
        pytest.param("group-past-band", None, True, id="group-past-band-reference"),
        pytest.param("group-past-band", [[4, 2]], True, id="group-past-band-kept-with-a-tie"),
        pytest.param("group-past-band", [[0, 2]], False, id="group-past-band-dropped"),
        pytest.param("expert-chain", [[0, 2, 3]], True, id="expert-chain-fourth-for-the-second"),
        pytest.param("expert-chain", [[1, 2, 3]], False, id="expert-chain-fourth-for-the-best"),
        pytest.param("group-chain", [[0, 4, 6]], True, id="group-chain-fourth-for-the-second"),
        pytest.param("group-chain", [[2, 4, 6]], False, id="group-chain-fourth-for-the-best"),
    ],
)
def test_fused_moe_always_keeps_a_score_above_the_band_of_the_last_kept(name, ids, accepted):
    """An expert or group scored above the band of the last one kept, by any margin, is in
    every admitted routing; one inside the band may take another's place. A chain of scores
    each within the band of the next does not stretch the band."""
    assert _accepts_routing(*_band_edge_case(name), ids) == accepted


def _option_sets(gating, bias, top_k, num_groups, topk_group):
    """The expert sets the verification's tie enumeration admits for one sigmoid-scored,
    biased, renormalized token, and those the definition admits."""
    from workloads.moe import _tied_routings, ref_fused_topk

    ties = _tied_routings(gating, bias, top_k, "sigmoid", True, num_groups, topk_group)
    if 0 in ties:
        enumerated = {tuple(sorted(ids.tolist())) for _, ids in ties[0]}
    else:
        ids = ref_fused_topk(gating, bias, top_k, "sigmoid", True, num_groups, topk_group)[1]
        enumerated = {tuple(sorted(ids[0].tolist()))}
    selection = gating.sigmoid()[0] + bias
    routings = _topk_routings_by_definition(selection, num_groups, topk_group, top_k, True)
    return enumerated, {ids for _, ids in routings}


@pytest.mark.parametrize(
    "name", ["expert-past-band", "group-past-band", "expert-chain", "group-chain"]
)
def test_tie_enumeration_equals_the_definition_on_band_edges(name):
    """On a score past the band by less than its rounding, and on chains, the verification
    enumerates exactly the routings the definition admits."""
    _, workload, inputs = _band_edge_case(name)
    p = workload.call.params
    enumerated, defined = _option_sets(
        inputs[1], inputs[4], p["top_k"], p["num_expert_group"], p["topk_group"]
    )
    assert len(defined) == 3 and enumerated == defined


@pytest.mark.parametrize(
    "experts, equal, top_k, num_groups, topk_group, count",
    [
        pytest.param(66, 64, 1, 1, 1, None, id="64-routings-enumerated"),
        pytest.param(66, 65, 1, 1, 1, 65, id="65-routings-refused"),
        pytest.param(64, 64, 32, 1, 1, math.comb(64, 32), id="all-equal-experts-refused"),
        pytest.param(128, 128, 1, 64, 32, math.comb(64, 32), id="all-equal-groups-refused"),
    ],
)
def test_fused_moe_verification_enumerates_at_most_64_routings_a_tie_admits(
    experts, equal, top_k, num_groups, topk_group, count
):
    """A token whose *equal* best experts are exactly tied admits C(*equal*, top_k) routings,
    or, with groups, at least one per set of kept groups: up to 64 are enumerated, and a
    non-reference one passes; past 64 the verification refuses to build, naming the token and
    the count, without listing them."""
    target = torch.full((experts,), 0.5)
    target[equal:] = 0.25
    case, workload, inputs = _exact_selection_case(target, top_k, num_groups, topk_group)
    if count is None:
        assert _accepts_routing(case, workload, inputs, [[equal - 1]])
    else:
        with pytest.raises(ValueError, match=f"token 0: its tie admits at least {count} "):
            workload.verification(*inputs)


def _place(gating, bias, expert, value):
    """Set *expert*'s selection score to *value* exactly: a zero logit, a bias of *value* - 0.5."""
    gating[0, expert] = 0.0
    bias[expert] = value - 0.5


def _ds_v3_row(kind, generator):
    """``(gating, bias)`` of one DeepSeek-V3 token (E = 256 in 8 groups of 32), random, then by
    *kind*: four groups 0.7 of a group band apart straddling the last kept (``group-chain``);
    four experts of the kept groups 0.7 of a band apart straddling the top_k-th
    (``expert-chain``); a kept group and expert a few float32 steps from the upper edge of the
    band, and a dropped group from the lower edge (``band-edge``); a dropped group and expert
    exactly tied with the last kept (``exact-tie``)."""
    num_groups, group_size, topk_group, top_k = 8, 32, 4, 8
    gating = torch.randn(1, 256, generator=generator)
    bias = torch.randn(256, generator=generator) * 0.1

    def ranked():
        selection = gating.sigmoid()[0] + bias
        top2 = selection.view(num_groups, group_size).topk(2)
        pairs = top2.values.sum(-1)
        order = pairs.argsort(descending=True, stable=True).tolist()
        return selection, top2, pairs, order

    def place_group(g, score):
        best = g * group_size + int(top2.indices[g, 0])
        _place(gating, bias, best, score - top2.values[g, 1])

    def kept_experts():
        experts = torch.cat(
            [torch.arange(g * group_size, (g + 1) * group_size) for g in order[:topk_group]]
        )
        return experts[selection[experts].argsort(descending=True, stable=True)].tolist()

    steps = torch.randint(-2, 3, (3,), generator=generator).tolist()

    def edge(value, band, sign, step):
        x = value + sign * band
        towards = torch.tensor(float("inf") if step > 0 else float("-inf"))
        for _ in range(abs(step)):
            x = torch.nextafter(x, towards)
        return x

    selection, top2, pairs, order = ranked()
    last = pairs[order[topk_group - 1]]
    group_band = 1e-5 * last.abs() + 1e-6
    if kind == "group-chain":
        for g, step in zip(
            order[topk_group - 3 : topk_group + 1], (1.4, 0.7, 0.0, -0.7), strict=True
        ):
            place_group(g, last + step * group_band)
    elif kind == "band-edge":
        place_group(order[topk_group - 2], edge(last, group_band, 1, steps[0]))
        place_group(order[topk_group], edge(last, group_band, -1, steps[1]))
    elif kind == "exact-tie":
        g, h = order[topk_group - 1], order[topk_group]
        for i in range(2):
            _place(gating, bias, h * group_size + int(top2.indices[h, i]), top2.values[g, i])
    selection, top2, pairs, order = ranked()
    experts = kept_experts()
    kth = selection[experts[top_k - 1]]
    band = 1e-5 * kth.abs() + torch.finfo(torch.float32).eps
    if kind == "expert-chain":
        for e, step in zip(experts[top_k - 3 : top_k + 1], (1.4, 0.7, 0.0, -0.7), strict=True):
            _place(gating, bias, e, kth * (1 + step * 1e-5))
    elif kind == "band-edge":
        _place(gating, bias, experts[top_k - 2], edge(kth, band, 1, steps[2]))
    elif kind == "exact-tie":
        _place(gating, bias, experts[top_k], kth)
    return gating, bias


def test_tie_enumeration_equals_the_definition_at_the_deepseek_v3_shape():
    """At E = 256 in 8 groups, 4 kept, top_k = 8, on random tokens and tokens with group
    chains, expert chains, scores on the edges of both bands and exact ties, the verification
    enumerates exactly the routings the definition admits."""
    generator = torch.Generator().manual_seed(0)
    kinds = ("random", "group-chain", "expert-chain", "band-edge", "exact-tie")
    disagree = []
    for row in range(3000):
        gating, bias = _ds_v3_row(kinds[row % len(kinds)], generator)
        enumerated, defined = _option_sets(gating, bias, 8, 8, 4)
        if enumerated != defined:
            disagree.append((row, sorted(enumerated ^ defined)))
    assert disagree == []


# FusedTopK: which experts a tie admits.


def _topk_workload(num_experts, num_groups, topk_group, top_k, biased, **params):
    """The ``FusedTopKWorkload`` of a CPU call: one token, sigmoid scores, no renormalization,
    unless *params* say otherwise."""
    row = dict(T=1, scoring_func="sigmoid", renormalize=False) | params
    call = manifest_call(
        "FusedTopKFwdOp",
        {"G": "float32"},
        E=num_experts,
        top_k=top_k,
        num_expert_group=num_groups,
        topk_group=topk_group,
        some=["correction_bias"] if biased else [],
        **row,
    )
    return FusedTopKWorkload(call, device="cpu")


def _topk_accepts(workload, inputs, ids, weights=None) -> bool:
    """Whether the verification accepts the experts *ids* of each token, weighted by their own
    scores (renormalized and scaled as the call asks) unless *weights* are given."""
    p = workload.call.params
    returned = torch.tensor(ids, dtype=torch.int32)
    if weights is None:
        scores = inputs[0].softmax(-1) if p["scoring_func"] == "softmax" else inputs[0].sigmoid()
        weights = scores.gather(1, returned.long())
        if p["renormalize"]:
            weights = weights / weights.sum(-1, keepdim=True)
        weights = weights * p["routed_scaling_factor"]
    got = (weights, returned)
    return _accepts(workload.verification(*inputs), got, workload.ref_program(*inputs))


@pytest.mark.parametrize(
    "selection, topk_group, ids, accepted",
    [
        pytest.param([[0.9, 0.1], [0.5, 0.5]], 1, [2], True, id="tied-group-swapped-in"),
        pytest.param([[0.9, 0.1], [0.5, 0.5]], 1, [1], False, id="not-the-kept-group-best"),
        pytest.param(
            [[0.6, 0.6], [0.9, 0.1], [0.5, 0.5]], 2, [2, 3], False, id="skips-a-group-above-tie"
        ),
        pytest.param(
            [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]], 1, [0, 2], False, id="spans-more-groups-than-kept"
        ),
        pytest.param(
            [[0.8, 0.2], [0.6, 0.4], [0.7, 0.3]], 2, [4], True, id="kept-group-with-no-expert"
        ),
        pytest.param(
            [[0.8, 0.2], [0.6, 0.4], [0.7, 0.3]], 2, [2], False, id="every-second-group-beats-it"
        ),
        pytest.param(
            [[0.7, 0.7], [0.6, 0.4], [0.5, 0.5]], 2, [2], False, id="fills-past-a-group-above-tie"
        ),
        # Four experts 7e-6 apart: the best is 1.4e-5 above the third, past the band.
        pytest.param(
            [[0.8, 0.8 - 7e-6], [0.8 - 14e-6, 0.8 - 21e-6], [0.3, 0.2]],
            2,
            [1, 2, 3],
            False,
            id="chain-drops-an-expert-above-the-band",
        ),
        pytest.param(
            [[0.8, 0.8 - 7e-6], [0.8 - 14e-6, 0.8 - 21e-6], [0.3, 0.2]],
            2,
            [0, 2, 3],
            True,
            id="chain-keeps-a-tied-expert-below-the-kth",
        ),
        pytest.param(
            [[0.8, 0.8 - 7e-6], [0.8 - 14e-6, 0.8 - 21e-6], [0.3, 0.2]],
            2,
            [3, 0, 2],
            True,
            id="chain-three-experts-out-of-order",
        ),
        # Groups 1 and 2 tie, and either kept set leaves the same top-k in group 0.
        pytest.param(
            [[0.9, 0.85], [0.5, 0.4], [0.6, 0.3 + 4e-6]],
            2,
            [0, 1],
            True,
            id="group-tie-with-one-expert-set",
        ),
        pytest.param(
            [[0.9, 0.85], [0.5, 0.4], [0.6, 0.3 + 4e-6]],
            2,
            [0, 4],
            False,
            id="group-tie-with-one-expert-set-other-experts",
        ),
    ],
)
def test_fused_topk_on_a_near_tied_boundary(selection, topk_group, ids, accepted):
    """A token passes only when its experts are those of a routing a tie admits: ``topk_group``
    groups holding every group clearly above the last kept group score and the rest within its
    band, and inside them every expert clearly above the ``top_k``-th best selection score and
    the rest within its band."""
    num_experts = 2 * len(selection)
    workload = _topk_workload(num_experts, len(selection), topk_group, len(ids), True)
    inputs = (torch.logit(torch.tensor([sum(selection, [])])), torch.zeros(num_experts))
    assert _topk_accepts(workload, inputs, [ids]) == accepted


@pytest.mark.parametrize(
    "ids, fault, accepted",
    [
        pytest.param([[0, 2], [1, 3]], None, True, id="tied-token-takes-the-other-tied-expert"),
        pytest.param([[2, 0], [3, 1]], None, True, id="selection-order-permuted"),
        pytest.param([[0, 3], [1, 3]], None, False, id="expert-outside-the-tie"),
        pytest.param([[0, 2], [1, 0]], None, False, id="untied-token-reselected"),
        pytest.param([[0, 0], [1, 3]], None, False, id="duplicate-expert"),
        pytest.param([[0, 2], [1, 3]], "not-renormalized", False, id="weights-not-renormalized"),
        pytest.param([[0, 2], [1, 3]], "scaled-twice", False, id="weights-scaled-twice"),
        pytest.param([[0, 0], [0, 0]], "zero", False, id="all-zero-output"),
    ],
)
def test_fused_topk_holds_each_token_to_the_routings_its_tie_admits(ids, fault, accepted):
    """Token 0 ties experts 1 and 2 for its second place, token 1 has no tie: token 0 may take
    either, token 1 only the reference's experts, no token an expert twice, and the weights
    stay each expert's own score, renormalized and scaled once; an all-zero output is
    rejected."""
    workload = _topk_workload(
        4, 1, 1, 2, False, T=2, scoring_func="softmax", renormalize=True, routed_scaling_factor=2.5
    )
    gating = torch.tensor([[3.0, 1.0, 1.0, -2.0], [0.5, 2.0, -1.0, 1.0]])
    weights = None
    if fault == "zero":
        weights = torch.zeros(2, 2)
    elif fault is not None:
        weights = gating.softmax(-1).gather(1, torch.tensor(ids))
        if fault == "scaled-twice":
            weights = weights / weights.sum(-1, keepdim=True) * 2.5
        weights = weights * 2.5
    assert _topk_accepts(workload, (gating,), ids, weights) == accepted


def _topk_verdicts_disagree(workload, inputs, routings, candidates) -> list:
    """The candidate expert sets on which the verification's verdict differs from the
    definition's, which accepts exactly the expert sets of *routings*."""
    admitted = {ids for _, ids in routings}
    return [
        (ids, accepted)
        for ids in candidates
        if (accepted := _topk_accepts(workload, inputs, [list(ids)])) != (ids in admitted)
    ]


def _ds_v3_near_tied_candidates(selection, routings, group_size) -> list:
    """Each admitted routing's experts and its near misses: one of its three weakest experts or
    its best replaced by one of the two best experts its groups leave out or by the best expert
    of the first group no routing keeps, and the top-k of each group set that trades one of its
    groups for another group some routing keeps."""
    num_groups = selection.numel() // group_size
    grouped = selection.view(num_groups, group_size)
    kept_by_some = {g for groups, _ in routings for g in groups}
    dropped = [g for g in range(num_groups) if g not in kept_by_some]
    candidates = set()
    for groups, ids in routings:
        candidates.add(ids)
        experts = [e for g in groups for e in range(g * group_size, (g + 1) * group_size)]
        left_out = sorted((e for e in experts if e not in ids), key=lambda e: -selection[e])
        outside = [g * group_size + int(grouped[g].argmax()) for g in dropped[:1]]
        ranked = sorted(ids, key=lambda e: selection[e])
        for out in ranked[:3] + ranked[-1:]:
            for into in left_out[:2] + outside:
                candidates.add(tuple(sorted(e if e != out else into for e in ids)))
        for skipped in groups:
            for other in sorted(kept_by_some - set(groups)):
                pool = [
                    e
                    for g in (set(groups) - {skipped}) | {other}
                    for e in range(g * group_size, (g + 1) * group_size)
                ]
                pool.sort(key=lambda e: -selection[e])
                candidates.add(tuple(sorted(pool[: len(ids)])))
    return sorted(candidates)


def test_fused_topk_verification_equals_its_definition():
    """The verification accepts a token's experts exactly when the docstring's definition
    admits them: every candidate of small routings, grouped and not, and at the DeepSeek-V3
    shape each admitted routing's experts and their near misses on random tokens and tokens
    with group chains, expert chains, scores on the edges of both bands and exact ties."""
    near_tied6 = [
        [[0.9, 0.1], [0.6, 0.4], [0.3, 0.2]],
        [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]],
        [[0.8, 0.2], [0.6, 0.4], [0.7, 0.3]],
        [[0.7, 0.7], [0.6, 0.4], [0.5, 0.5]],
        [[0.6, 0.6], [0.9, 0.1], [0.5, 0.5]],
        [[0.5 + 2.2e-5, 0.5], [0.5, 0.5], [0.5 - 5e-6, 0.5]],
        [[0.5, 0.5], [0.5 + 5e-6, 0.5], [0.4, 0.3]],
        [[0.9, 0.1], [0.2, 0.9], [0.9, 0.5]],
        [[0.9, 0.1], [0.9 - 4e-6, 0.8], [0.5, 0.3]],
        # Chains: groups whose first experts are 7e-6 apart, experts 5e-6 apart straddling the
        # top_k-th.
        [[0.5 + 7e-6, 0.5], [0.5, 0.5], [0.5 - 7e-6, 0.5]],
        [[0.8, 0.8 - 5e-6], [0.8 - 10e-6, 0.8 - 15e-6], [0.3, 0.2]],
    ]
    near_tied8 = [
        [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]],
        [[0.95, 0.05], [0.6, 0.4], [0.5, 0.5], [0.7, 0.3]],
        [[0.9, 0.9], [0.9, 0.1], [0.5, 0.5], [0.5, 0.5 + 5e-6]],
        # Chains: experts 7e-6 apart, groups whose first experts are 4e-6 and 6e-6 apart.
        [[0.8, 0.8 - 7e-6], [0.8 - 14e-6, 0.8 - 21e-6], [0.3, 0.2], [0.2, 0.1]],
        [[0.5 + 8e-6, 0.5], [0.5 + 4e-6, 0.5], [0.5, 0.5], [0.5 - 4e-6, 0.5]],
        [[0.5 + 12e-6, 0.3], [0.5 + 6e-6, 0.3], [0.5, 0.3], [0.5 - 6e-6, 0.3]],
    ]
    rng = torch.Generator().manual_seed(0)
    random = [torch.rand(3, 2, generator=rng).tolist() for _ in range(4)]
    random += [(torch.randint(1, 10, (3, 2), generator=rng) / 10).tolist() for _ in range(2)]
    random8 = [torch.rand(4, 2, generator=rng).tolist() for _ in range(3)]
    configs = [
        (num_groups, top_k, topk_group, biased, matrices)
        for biased in (True, False)
        for num_groups, top_k, topk_group, matrices in (
            (None, 2, 1, near_tied6 + random),
            (None, 2, 2, near_tied6 + random),
            (None, 3, 2, near_tied8 + random8),
            (1, 2, 1, near_tied6 + random),
            (1, 3, 1, near_tied8 + random8),
        )
    ]
    disagree = []
    for num_groups, top_k, topk_group, biased, matrices in configs:
        for values in matrices:
            groups, num_experts = num_groups or len(values), 2 * len(values)
            workload = _topk_workload(num_experts, groups, topk_group, top_k, biased)
            target = torch.tensor([sum(values, [])])
            if biased:
                inputs = (torch.zeros(1, num_experts), target[0] - 0.5)
                selection = inputs[0].sigmoid() + inputs[1]
            else:
                inputs = (torch.logit(target),)
                selection = inputs[0].sigmoid()
            routings = _topk_routings_by_definition(selection[0], groups, topk_group, top_k, biased)
            candidates = itertools.combinations(range(num_experts), top_k)
            disagree += _topk_verdicts_disagree(workload, inputs, routings, candidates)

    workload = _topk_workload(256, 8, 4, 8, True)
    kinds = ("random", "group-chain", "expert-chain", "band-edge", "exact-tie")
    for row in range(400):
        gating, bias = _ds_v3_row(kinds[row % len(kinds)], rng)
        selection = gating.sigmoid()[0] + bias
        routings = _topk_routings_by_definition(selection, 8, 4, 8, True)
        candidates = _ds_v3_near_tied_candidates(selection, routings, 32)
        disagree += _topk_verdicts_disagree(workload, (gating, bias), routings, candidates)
    assert disagree == []


def test_fused_topk_verification_refuses_a_tie_past_64_routings():
    """Eight exactly equal experts for four places admit C(8, 4) = 70 routings: the
    verification refuses to build, naming the token and the count."""
    workload = _topk_workload(8, 1, 1, 4, False, scoring_func="softmax")
    with pytest.raises(ValueError, match="token 0: its tie admits at least 70 "):
        workload.verification(torch.zeros(1, 8))


def _coherent_precision_case(activation, gate):
    """Identical activated columns whose low-precision product is cancelled by the bias."""
    layout = {"contiguous": {"packing": "tight", "metadata_kind": "physical_psum", "alignment": 1}}
    ffn, hidden = 128, 8
    call = manifest_call(
        "MoEExpertMLPFwdOp",
        {"D": "bfloat16"},
        layout=layout,
        activation=activation,
        some=["b_gate_up", "b_down"],
        P=1,
        H=hidden,
        E=1,
        F=ffn,
    )
    workload = MoEExpertMLPWorkload(call, device="cpu")
    x = torch.zeros(1, hidden, dtype=_BF16)
    w1 = torch.zeros(1, 2 * ffn, hidden, dtype=_BF16)
    w2 = torch.ones(1, hidden, ffn, dtype=_BF16)
    bias = torch.ones(1, 2 * ffn, dtype=_BF16)
    if activation == "swigluoai":
        bias[:, ::2] = gate
    else:
        bias[:, :ffn] = gate
    activated = _activate(bias.float(), activation, None, False, _BF16)
    down_bias = (-(_hold(activated, _BF16) @ w2[0].float().T)).to(_BF16)
    inputs = (x, w1, w2, torch.tensor([1], dtype=torch.int32), bias, down_bias)
    got = (activated @ w2[0].float().T + down_bias.float()).to(_BF16)
    held = (_hold(activated, _BF16) @ w2[0].float().T + down_bias.float()).to(_BF16)
    return workload, inputs, got, held


@pytest.mark.parametrize(
    "activation, gate",
    [
        ("silu_and_mul", -1.6),
        ("gelu_and_mul", -1.6),
        ("swigluoai", -1.05),
    ],
)
def test_expert_mlp_accepts_fp32_activated_rows_under_coherent_cancellation(activation, gate):
    """The fp32 precision choice remains admissible under coherent cancellation."""
    workload, inputs, got, _ = _coherent_precision_case(activation, gate)
    assert _accepts(workload.verification(*inputs), got, workload.ref_program(*inputs))


def test_expert_mlp_rejects_different_precision_choices_per_output_column():
    """Every output column of one row must fit the same precision choice."""
    workload, inputs, fp32, held = _coherent_precision_case("silu_and_mul", -1.6)
    mixed = held.clone()
    mixed[:, :1] = fp32[:, :1]
    with pytest.raises(AssertionError, match="no complete row-wise activation precision"):
        compare_outputs(mixed, workload.ref_program(*inputs), workload.verification(*inputs))


def test_routed_accepts_held_activated_rows_under_coherent_cancellation():
    """The operand-dtype precision choice remains admissible under coherent cancellation."""
    _, inputs, fp32, held = _coherent_precision_case("silu_and_mul", -1.6)
    x, w1, w2, _, b1, b2 = inputs
    weights = torch.ones(1, 1)
    ids = torch.zeros(1, 1, dtype=torch.int32)
    verification = routed_experts_verification(
        x,
        w1,
        w2,
        weights,
        ids,
        "silu_and_mul",
        b_gate_up=b1,
        b_down=b2,
    )
    assert _accepts(verification, held, fp32)


@pytest.mark.parametrize("rows", [0, 1])
def test_scalar_and_empty_workload_conditioning_remains_finite(rows):
    """A one-element or empty conditioning sample produces finite generated operands."""
    layout = {"contiguous": {"packing": "tight", "metadata_kind": "physical_psum", "alignment": 1}}
    call = manifest_call(
        "MoEGroupedGemmFwdOp",
        {"D": "bfloat16"},
        layout=layout,
        some=["bias"],
        P=rows,
        E=1,
        K=1,
        N=1,
    )
    inputs = MoEGroupedGemmWorkload(call, device="cpu").gen_inputs()
    assert all(torch.isfinite(tensor).all() for tensor in inputs if tensor is not None)


@pytest.mark.parametrize(
    "ids, accepted",
    [([0, 1], True), ([0, 2], True), ([0, 3], True), ([1, 2], False), ([0, 4], False)],
    ids=["canonical", "middle-tie", "floor-edge", "mandatory-dropped", "below-floor"],
)
def test_zero_selection_band_keeps_mandatory_experts_and_admits_only_its_floor(ids, accepted):
    """Zero-score ties admit a three-expert chain through epsilon, excluding the next bias step."""
    epsilon = torch.finfo(torch.float32).eps
    selection = torch.tensor([2 * epsilon, 0, -epsilon / 2, -epsilon, -1.5 * epsilon])
    gating = torch.zeros(1, selection.numel())
    bias = selection - 0.5
    assert torch.equal(gating.sigmoid()[0] + bias, selection)
    workload = _topk_workload(selection.numel(), 1, 1, 2, True)
    assert _topk_accepts(workload, (gating, bias), [ids]) == accepted
    admitted = _topk_routings_by_definition(selection, 1, 1, 2, True)
    for candidate in itertools.combinations(range(selection.numel()), 2):
        assert _topk_accepts(workload, (gating, bias), [candidate]) == (
            ((0,), candidate) in admitted
        )


@pytest.mark.parametrize("direction", [-1, 1], ids=["lower", "upper"])
@pytest.mark.parametrize("past", [False, True], ids=["equal", "illegal-neighbor"])
def test_post_permute_budget_includes_equality_and_rejects_the_next_output(direction, past):
    """Each exact two-budget edge passes; its adjacent outward BF16 value fails."""
    rows = torch.ones(1, 1, dtype=_BF16)
    weights = torch.ones(1, 1)
    inverse = torch.zeros(1, 1, dtype=torch.int32)
    unit = torch.finfo(_BF16).eps / 2
    budget = unit + unit
    got = torch.full_like(rows, 1 + direction * 2 * budget)
    if past:
        got = torch.nextafter(got, torch.full_like(got, direction * math.inf))
    verification = post_permute_verification(rows, weights, inverse, 1.0)
    assert _accepts(verification, got, rows) == (not past)
