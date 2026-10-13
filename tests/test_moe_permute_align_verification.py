"""``MoEPermuteAlignWorkload``'s verifier, held to the result ``MoEPermuteAlignFwdOp`` states.

The op leaves one decision open, the order of each expert's routes within its first
``count[e]`` slots, and one value free, that of a block past the padded length. A positive
control makes a non-reference choice of both and must pass. A negative control starts from
such a choice, breaks one constraint, keeps the rest consistent and must be rejected: either
the choice lies outside what the op allows, or a step after it is not exact. An exhaustive and
a differential test compare the verifier with an oracle written from the op's docstring.

Tensors stay on the host: these tests run the verifier alone, on constructed outputs.
"""

import itertools
import math
import random

import pytest
import torch

from tileops.manifest import load_manifest
from workloads.moe import MoEPermuteAlignWorkload
from workloads.numerics import compare_outputs
from workloads.workload_base import manifest_call


def _workload(total_tokens: int, top_k: int, num_experts: int, block_size: int):
    return MoEPermuteAlignWorkload(
        manifest_call(
            "MoEPermuteAlignFwdOp",
            T=total_tokens,
            K=top_k,
            num_experts=num_experts,
            block_size=block_size,
        )
    )


def _accepts(workload, topk_ids, got) -> bool:
    try:
        compare_outputs(got, workload.ref_program(topk_ids), workload.verification(topk_ids))
    except AssertionError:
        return False
    return True


def _docstring_oracle(topk_ids, num_experts: int, block_size: int, got) -> bool:
    """Whether *got* is a result ``MoEPermuteAlignFwdOp``'s docstring allows, read literally."""
    tokens, experts, count = got
    flat = topk_ids.flatten().tolist()
    numel = len(flat)
    capacity = numel + (num_experts + 1) * (block_size - 1)
    # forward's Returns: the shapes, all int32.
    if tuple(tokens.shape) != (capacity,) or tuple(count.shape) != (1,):
        return False
    if tuple(experts.shape) != (math.ceil(capacity / block_size),):
        return False
    if any(t.dtype != torch.int32 for t in got):
        return False
    tokens, experts, count = tokens.tolist(), experts.tolist(), count.tolist()
    routes = [[] for _ in range(num_experts)]
    for index, expert in enumerate(flat):
        routes[expert].append(index)
    lengths = [math.ceil(len(r) / block_size) * block_size for r in routes]
    total = sum(lengths)
    # num_tokens_post_pad: the sum of the rounded-up counts.
    if count != [total]:
        return False
    start = 0
    for expert in range(num_experts):
        span = tokens[start : start + lengths[expert]]
        held = len(routes[expert])
        # The first count[e] slots hold the expert's routes, each once, in any order; the rest
        # of the range holds numel.
        if sorted(span[:held]) != routes[expert] or any(x != numel for x in span[held:]):
            return False
        # expert_ids names the owner of each block of the range.
        if any(
            experts[b] != expert
            for b in range(start // block_size, (start + lengths[expert]) // block_size)
        ):
            return False
        start += lengths[expert]
    # Past those lengths: numel in sorted_token_ids, no expert in expert_ids.
    if any(x != numel for x in tokens[total:]):
        return False
    return not any(0 <= x < num_experts for x in experts[total // block_size :])


def _another_choice(order: bool = True, marker: bool = True):
    """A reference result with the open choices made differently.

    The call has 12 routes over 4 experts in blocks of 4. Expert 0 holds routes 0, 2, 5, 7, 8 in
    slots 0-7, so its routes cross from block 0 into block 1 and three padding slots follow;
    expert 1 holds routes 1, 4, 9, 10 in slots 8-11; expert 2 routes 3, 6, 11 in slots 12-15
    and one padding slot; expert 3 has no route and no range. Slots 16-26 and blocks 4-6 lie
    past the padded length.

    *order* reverses each expert's routes within their slots, which moves routes of expert 0
    across its block boundary and leaves expert 3 as it is. *marker* puts -1 instead of the
    reference's ``num_experts`` in every block past the padded length.
    """
    topk_ids = torch.tensor([[0, 1], [0, 2], [1, 0], [2, 0], [0, 1], [1, 2]], dtype=torch.int32)
    workload = _workload(6, 2, 4, 4)
    tokens, experts, count = (t.clone() for t in workload.ref_program(topk_ids))
    if order:
        for start, held in ((0, 5), (8, 4), (12, 3)):
            tokens[start : start + held] = tokens[start : start + held].flip(0)
    if marker:
        experts[4:] = -1
    return workload, topk_ids, (tokens, experts, count)


@pytest.mark.parametrize(
    "order, marker",
    [
        pytest.param(True, False, marks=pytest.mark.smoke, id="route-order"),
        pytest.param(False, True, marks=pytest.mark.full, id="no-expert-marker"),
    ],
)
def test_verifier_accepts_another_choice_of_what_the_op_leaves_open(
    order: bool, marker: bool
) -> None:
    """Another route order within each expert's slots, or -1 past the padded length, passes."""
    workload, topk_ids, got = _another_choice(order, marker)
    assert _accepts(workload, topk_ids, got)


def _route_twice(tokens, experts, count):
    tokens[1] = tokens[0]
    return tokens, experts, count


def _route_in_another_experts_range(tokens, experts, count):
    tokens[[0, 8]] = tokens[[8, 0]]
    return tokens, experts, count


def _padding_before_routes(tokens, experts, count):
    tokens[0:6] = torch.cat([tokens[5:6], tokens[0:5]])
    return tokens, experts, count


def _block_past_the_end_names_the_last_expert(tokens, experts, count):
    experts[4] = 3
    return tokens, experts, count


@pytest.mark.parametrize(
    "change, reason",
    [
        # Two copies of one route and none of another, both on the same expert.
        pytest.param(
            _route_twice, "sorted_token_ids", marks=pytest.mark.smoke, id="each-route-once"
        ),
        # A route in another expert's slots.
        pytest.param(
            _route_in_another_experts_range,
            "sorted_token_ids",
            marks=pytest.mark.full,
            id="own-range",
        ),
        # A padding slot before an expert's routes.
        pytest.param(
            _padding_before_routes, "sorted_token_ids", marks=pytest.mark.full, id="routes-first"
        ),
        # A block past the padded length naming expert num_experts - 1.
        pytest.param(
            _block_past_the_end_names_the_last_expert,
            "names an expert",
            marks=pytest.mark.full,
            id="tail-names-no-expert",
        ),
    ],
)
def test_verifier_rejects_a_choice_the_op_does_not_allow(change, reason: str) -> None:
    """An output whose open choice lies outside what the op allows fails that check."""
    workload, topk_ids, got = _another_choice()
    with pytest.raises(AssertionError, match=reason):
        compare_outputs(
            change(*got), workload.ref_program(topk_ids), workload.verification(topk_ids)
        )


def _wrong_count(tokens, experts, count):
    return tokens, experts, count * 2


def _swap_expert_ranges(tokens, experts, count):
    tokens[8:16] = torch.cat([tokens[12:16], tokens[8:12]])
    experts[2:4] = experts[2:4].flip(0)
    return tokens, experts, count


def _wrong_block_owner(tokens, experts, count):
    experts[1] = 1
    return tokens, experts, count


def _padding_not_numel(tokens, experts, count):
    tokens[5] = tokens[-1] + 1
    return tokens, experts, count


def _slot_past_the_end_holds_a_route(tokens, experts, count):
    tokens[16] = 0
    return tokens, experts, count


def _longer_sorted_token_ids(tokens, experts, count):
    return torch.cat([tokens, tokens[-1:]]), experts, count


def _sorted_token_ids_widened(tokens, experts, count):
    return tokens.long(), experts, count


@pytest.mark.parametrize(
    "change, reason",
    [
        pytest.param(
            _wrong_count, "num_tokens_post_pad", marks=pytest.mark.smoke, id="num-tokens-post-pad"
        ),
        # Expert 1's and expert 2's ranges trade places, routes and owners alike.
        pytest.param(_swap_expert_ranges, "owner", marks=pytest.mark.full, id="expert-order"),
        pytest.param(_wrong_block_owner, "owner", marks=pytest.mark.full, id="block-owner"),
        pytest.param(
            _padding_not_numel, "sorted_token_ids", marks=pytest.mark.full, id="padding-is-numel"
        ),
        pytest.param(
            _slot_past_the_end_holds_a_route,
            "sorted_token_ids",
            marks=pytest.mark.full,
            id="sorted-ids-tail",
        ),
        pytest.param(_longer_sorted_token_ids, "shape", marks=pytest.mark.full, id="shape"),
        pytest.param(_sorted_token_ids_widened, "dtype", marks=pytest.mark.full, id="dtype"),
    ],
)
def test_verifier_rejects_an_inexact_step_after_the_choice(change, reason: str) -> None:
    """An output that makes an allowed choice and then breaks one exact step fails that check."""
    workload, topk_ids, got = _another_choice()
    with pytest.raises(AssertionError, match=reason):
        compare_outputs(
            change(*got), workload.ref_program(topk_ids), workload.verification(topk_ids)
        )


@pytest.mark.parametrize(
    "topk",
    [
        pytest.param([[0], [1]], marks=pytest.mark.smoke, id="both-experts-routed"),
        pytest.param([[0], [0]], marks=pytest.mark.full, id="one-expert-routed"),
        pytest.param([[1], [1]], marks=pytest.mark.full, id="first-expert-unrouted"),
    ],
)
def test_verifier_matches_the_docstring_oracle_on_every_small_output(topk) -> None:
    """Every output of a two-route call is accepted exactly when the docstring allows it.

    The call has two experts and blocks of 2. sorted_token_ids draws from the route ids, numel
    and one value past it; expert_ids from both experts, num_experts and -1;
    num_tokens_post_pad is right or one block too large.
    """
    num_experts, block_size = 2, 2
    topk_ids = torch.tensor(topk, dtype=torch.int32)
    workload = _workload(2, 1, num_experts, block_size)
    reference = workload.ref_program(topk_ids)
    evidence = workload.verification(topk_ids)
    numel = 2
    capacity = numel + (num_experts + 1) * (block_size - 1)
    blocks = math.ceil(capacity / block_size)
    accepted = 0
    for count in (reference[2], reference[2] + block_size):
        for token_values in itertools.product((0, 1, numel, numel + 1), repeat=capacity):
            tokens = torch.tensor(token_values, dtype=torch.int32)
            for expert_values in itertools.product((0, 1, num_experts, -1), repeat=blocks):
                got = (tokens, torch.tensor(expert_values, dtype=torch.int32), count)
                try:
                    compare_outputs(got, reference, evidence)
                    verdict = True
                except AssertionError:
                    verdict = False
                assert verdict == _docstring_oracle(topk_ids, num_experts, block_size, got), got
                accepted += verdict
    assert accepted > 0


def _manifest_rows():
    return [
        {k: row[k] for k in ("T", "K", "num_experts", "block_size")}
        for row in load_manifest()["MoEPermuteAlignFwdOp"]["workloads"]
    ]


@pytest.mark.parametrize(
    "num_experts",
    [
        pytest.param(128, marks=pytest.mark.smoke, id="128-experts"),
        pytest.param(256, marks=pytest.mark.full, id="256-experts"),
        pytest.param(384, marks=pytest.mark.full, id="384-experts"),
    ],
)
def test_verifier_matches_the_docstring_oracle_on_manifest_shapes(num_experts: int) -> None:
    """Perturbed outputs on the manifest shapes are accepted exactly when the docstring allows.

    On each row with *num_experts*, an output that makes other allowed choices, and six outputs
    with one more slot changed, are judged by the verifier and the oracle alike.
    """
    rng = random.Random(num_experts)
    rows = [row for row in _manifest_rows() if row["num_experts"] == num_experts]
    assert rows
    for row in rows:
        total_tokens, top_k, block_size = row["T"], row["K"], row["block_size"]
        workload = _workload(total_tokens, top_k, num_experts, block_size)
        topk_ids = torch.stack(
            [torch.tensor(rng.sample(range(num_experts), top_k)) for _ in range(total_tokens)]
        ).to(torch.int32)
        tokens, experts, count = (t.clone() for t in workload.ref_program(topk_ids))
        held = torch.bincount(topk_ids.flatten().long(), minlength=num_experts).tolist()
        start = 0
        for expert in range(num_experts):
            routed = tokens[start : start + held[expert]].tolist()
            rng.shuffle(routed)
            tokens[start : start + held[expert]] = torch.tensor(routed, dtype=torch.int32)
            start += math.ceil(held[expert] / block_size) * block_size
        tail = int(count.item()) // block_size
        experts[tail:] = torch.tensor(
            [
                rng.choice((-1, num_experts, num_experts + 7, -100))
                for _ in range(experts.numel() - tail)
            ],
            dtype=torch.int32,
        )
        allowed = (tokens, experts, count)
        assert _accepts(workload, topk_ids, allowed)
        assert _docstring_oracle(topk_ids, num_experts, block_size, allowed)
        numel = total_tokens * top_k
        for _ in range(6):
            got = [t.clone() for t in allowed]
            which = rng.choice((0, 0, 0, 1, 1, 2))
            slot = rng.randrange(got[which].numel())
            if which == 0:
                got[0][slot] = rng.randrange(-1, numel + 2)
            elif which == 1:
                got[1][slot] = rng.randrange(-1, num_experts + 1)
            else:
                got[2][slot] += rng.choice((-block_size, block_size))
            got = tuple(got)
            assert _accepts(workload, topk_ids, got) == _docstring_oracle(
                topk_ids, num_experts, block_size, got
            ), (row, which, slot)
