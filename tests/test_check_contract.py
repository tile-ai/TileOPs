"""The shared verifier's correctness protocol, and what ``check()`` accepts to run it."""

import pytest
import torch

from workloads.numerics import (
    Custom,
    Exact,
    NegativeControl,
    Partial,
    Request,
    Unestablished,
    assert_quantized,
    compare_outputs,
    verify,
    zeroed_input,
)

pytestmark = pytest.mark.smoke


def _verify_one(subject, inputs, *, reference, evidence, subject_inputs=None):
    """Check one call the way ``verify`` checks each of its requests."""
    args = inputs if subject_inputs is None else subject_inputs
    return verify(
        reference, inputs, evidence=evidence, requests={"subject": Request(subject, args)}
    )["subject"]


@pytest.mark.parametrize("fault", ["shape", "dtype", "arity", "value", "none", "nan", "inf"])
def test_incorrect_results_are_rejected(fault):
    expected = torch.ones(2)
    got = {
        "shape": torch.ones(1),
        "dtype": torch.ones(2, dtype=torch.float64),
        "arity": (expected, expected),
        "value": torch.zeros(2),
        "none": None,
        "nan": torch.full((2,), float("nan")),
        "inf": torch.full((2,), float("inf")),
    }[fault]
    with pytest.raises((AssertionError, ValueError)):
        compare_outputs(got, expected, Exact())


def test_empty_tensors_compare_and_matching_nonfinite_values_are_allowed():
    for values in (torch.empty(0), torch.tensor([float("nan"), float("inf"), -float("inf")])):
        result = compare_outputs(values, values.clone(), Exact())
        assert result.checked_outputs == 1
        assert result.max_abs_err == 0


def test_no_numeric_output_does_not_establish_verification():
    output = (None, {"state": None})
    result = compare_outputs(output, output, Exact())
    assert result.checked_outputs == 0
    assert result.max_abs_err is None
    assert result.unchecked_reason


def test_none_is_a_value_not_permission_to_skip_an_output():
    with pytest.raises(AssertionError):
        compare_outputs(torch.ones(1), None, Exact())


def test_partial_is_explicit_and_reports_its_coverage():
    value = torch.ones(2)
    result = compare_outputs((value, value * 9), value, Partial(1, "saved state not checked"))
    assert (result.checked_outputs, result.total_outputs) == (1, 2)
    assert "saved state" in result.unchecked_reason
    with pytest.raises(ValueError):
        Partial(0, "nothing")


def test_custom_comparison_cannot_bypass_structure():
    got = torch.ones(1)
    with pytest.raises(AssertionError):
        compare_outputs(got, torch.ones(2), Custom(lambda *_: None, "always accepts"))


def test_each_output_uses_its_own_dtype_bound():
    expected = (torch.ones(2, dtype=torch.bfloat16), torch.ones(2))
    with pytest.raises(AssertionError):
        compare_outputs((expected[0], expected[1] + 0.01), expected, Exact())


def test_explicit_absolute_bound_does_not_inherit_relative_slack():
    with pytest.raises(AssertionError):
        compare_outputs(torch.tensor([100001.0]), torch.tensor([100000.0]), Exact(atol=0.1))


@pytest.mark.parametrize("failure", ["reference", "subject", "comparison"])
def test_inputs_are_restored_on_every_failure(failure):
    x = torch.ones(2)

    def reference(value):
        if failure == "reference":
            value.add_(5)
            raise RuntimeError("reference failed")
        return value

    def subject(value):
        value.add_(7)
        if failure == "subject":
            raise RuntimeError("subject failed")
        return value

    with pytest.raises((RuntimeError, AssertionError)):
        _verify_one(subject, (x,), reference=reference, evidence=Exact())
    torch.testing.assert_close(x, torch.ones(2))


def test_in_place_output_is_copied_before_restoration():
    x = torch.ones(2)
    with pytest.raises(AssertionError):
        _verify_one(
            lambda v: {"out": v.add_(1)}, (x,), reference=lambda v: {"out": v}, evidence=Exact()
        )
    torch.testing.assert_close(x, torch.ones(2))


@pytest.mark.cuda_only
def test_an_unwritten_output_element_does_not_read_back_as_expected():
    """A subject that writes half its output is rejected.

    The reference's result stays allocated while the subject runs, so the subject's
    output cannot be carved from memory that already holds the expected values.
    """
    from workloads.device import run_device

    x = torch.randn(1 << 16, device=run_device())

    def half_written(v):
        out = torch.empty_like(v)
        out[: v.numel() // 2] = v[: v.numel() // 2] * 2
        return out

    with pytest.raises(AssertionError):
        _verify_one(half_written, (x,), reference=lambda v: v * 2, evidence=Exact())


def test_every_block_of_a_large_output_is_compared():
    """A mismatch past the first block of a multi-block output is rejected."""
    x = torch.randn(3 << 20, dtype=torch.float64)
    result = _verify_one(lambda v: v * 2, (x,), reference=lambda v: v * 2, evidence=Exact())
    assert result.checked_outputs == 1

    def last_element_wrong(v):
        out = v * 2
        out[-1] += 1
        return out

    with pytest.raises(AssertionError, match="first failing block"):
        _verify_one(last_element_wrong, (x,), reference=lambda v: v * 2, evidence=Exact())


def test_a_reference_returning_a_cached_buffer_keeps_its_expected_values():
    """The subject overwriting a buffer the reference returned does not change what it is held to."""
    cache = torch.zeros(1)

    def reference(v):
        cache.copy_(v * 3)
        return cache

    with pytest.raises(AssertionError):
        _verify_one(
            lambda v: cache.zero_(), (torch.full((1,), 2.0),), reference=reference, evidence=Exact()
        )


def test_argument_aliases_are_preserved_and_subject_arguments_are_isolated():
    x, alternate = torch.ones(2), torch.ones(2)

    def subject(a, b):
        assert a is b
        return a.add_(1) - 1

    result = _verify_one(
        subject,
        (x,),
        reference=lambda v: v,
        evidence=Exact(),
        subject_inputs=(alternate, alternate),
    )
    assert result.checked_outputs == 1
    torch.testing.assert_close(alternate, x)


def test_reference_oom_is_distinct_from_subject_failure():
    def oom(*_):
        raise torch.OutOfMemoryError("out of memory")

    result = _verify_one(lambda x: x, (torch.ones(1),), reference=oom, evidence=Exact())
    assert result.checked_outputs == 0 and "reference" in result.unchecked_reason
    with pytest.raises(torch.OutOfMemoryError):
        _verify_one(oom, (torch.ones(1),), reference=lambda x: x, evidence=Exact())


def test_missing_oracle_requires_an_explicit_unestablished_declaration():
    with pytest.raises(ValueError, match="ref_program"):
        _verify_one(lambda: torch.ones(1), (), reference=None, evidence=Exact())
    result = _verify_one(
        lambda: pytest.fail("executed"), (), reference=None, evidence=Unestablished()
    )
    assert result.checked_outputs == 0 and result.unchecked_reason


@pytest.mark.parametrize("atol", [0.0, 2.0])
def test_negative_control_exposes_a_weak_contract(atol):
    evidence = Exact(atol=atol, rtol=0, controls=(zeroed_input(0, "dropped-input"),))
    if atol:
        with pytest.raises(ValueError, match="was accepted"):
            _verify_one(lambda x: x, (torch.ones(1),), reference=lambda x: x, evidence=evidence)
    else:
        assert (
            _verify_one(
                lambda x: x, (torch.ones(1),), reference=lambda x: x, evidence=evidence
            ).checked_outputs
            == 1
        )


def test_broken_negative_control_is_not_a_successful_rejection():
    def broken(*_):
        raise RuntimeError("invalid control")

    with pytest.raises(RuntimeError, match="invalid control"):
        _verify_one(
            lambda x: x,
            (torch.ones(1),),
            reference=lambda x: x,
            evidence=Exact(controls=(NegativeControl("broken", broken),)),
        )


def test_partial_control_must_change_the_checked_output():
    mark = Partial(
        1,
        "state omitted",
        controls=(NegativeControl("state only", lambda ref, args: (args[0], args[0] + 9)),),
    )
    with pytest.raises(ValueError, match="was accepted"):
        _verify_one(lambda x: (x, x), (torch.ones(1),), reference=lambda x: (x, x), evidence=mark)


def test_custom_probe_failure_is_a_verification_failure():
    def reject(subject, inputs):
        raise AssertionError("distribution wrong")

    with pytest.raises(AssertionError, match="distribution"):
        _verify_one(
            lambda x: x,
            (torch.ones(1),),
            reference=lambda x: x,
            evidence=Custom(lambda *_: None, "statistical check", probe=reject),
        )


def test_custom_probe_cannot_overwrite_shared_inputs():
    inputs = (torch.ones(1),)

    def probe(subject, args):
        subject(*args)

    calls = 0

    def subject(x):
        nonlocal calls
        calls += 1
        if calls == 2:
            x.mul_(2)
        return x.clone()

    with pytest.raises(AssertionError, match="overwrote the shared inputs"):
        verify(
            lambda x: x.clone(),
            inputs,
            evidence=Custom(lambda *_: None, "statistical check", probe=probe),
            requests={"impl": Request(subject, inputs, preserve_inputs=True)},
        )
    torch.testing.assert_close(inputs[0], torch.ones(1))


@pytest.mark.parametrize("fault", ["scale", "codes"])
def test_quantization_rejects_wrong_scales_and_codes(fault):
    codes = torch.tensor([12, 20], dtype=torch.int8)
    scale = torch.ones(2)
    got = (codes, scale * 2) if fault == "scale" else (torch.zeros_like(codes), scale)
    with pytest.raises(AssertionError):
        compare_outputs(got, (codes, scale), Custom(assert_quantized, "quantization"))


def test_broadcast_input_views_restore_shared_storage():
    base = torch.ones(3)
    expanded = base.expand(4, 3)

    def subject(value):
        value[0].add_(1)
        return value - 1

    result = _verify_one(subject, (expanded,), reference=lambda value: value, evidence=Exact())
    assert result.checked_outputs == 1
    assert expanded.stride(0) == 0
    torch.testing.assert_close(base, torch.ones(3))


@pytest.mark.parametrize("family", ["gla", "deltanet"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("wrong_output", [0, 1])
def test_inference_decode_keeps_its_strict_output_and_state_bound(family, dtype, wrong_output):
    from types import SimpleNamespace

    from workloads.linear_attention.deltanet import DeltaNetFwdCall, DeltaNetFwdWorkload
    from workloads.linear_attention.gla import GLAFwdCall, GLAFwdWorkload

    workload_type, call_type = {
        "gla": (GLAFwdWorkload, GLAFwdCall),
        "deltanet": (DeltaNetFwdWorkload, DeltaNetFwdCall),
    }[family]
    dimensions = {"dim_k": 64, "dim_v": 64} if family == "gla" else {"dim": 64}
    workload = workload_type(batch=1, seq_len=1, heads=1, dtype=dtype, **dimensions)
    q = torch.zeros(1, 1, 1, 64, dtype=dtype)
    expected = (torch.zeros(1, dtype=dtype), torch.zeros(1))
    got = list(expected)
    got[wrong_output] = got[wrong_output] + 1e-4
    # Both unit and manifest consumers must reject a drift accepted by prefill's bound.
    manifest_consumer = SimpleNamespace(
        call=SimpleNamespace(indices={"use_qk_l2norm_in_kernel": False})
    )
    for evidence in (workload.verification(q), call_type.verification(manifest_consumer, q)):
        with pytest.raises(AssertionError):
            compare_outputs(tuple(got), expected, evidence)


@pytest.mark.parametrize("family", ["gla", "deltanet"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_decode_rounding_allowance_stops_at_one_adjacent_value(family, dtype):
    from workloads.linear_attention.deltanet import inference_verification as delta_policy
    from workloads.linear_attention.gla import inference_verification as gla_policy

    evidence = {"gla": gla_policy, "deltanet": delta_policy}[family](dtype, decode=True)
    output, state = torch.ones(1, dtype=dtype), torch.zeros(1)
    adjacent = torch.nextafter(output, torch.full_like(output, float("inf")))
    compare_outputs((adjacent, state), (output, state), evidence)
    second = torch.nextafter(adjacent, torch.full_like(adjacent, float("inf")))
    with pytest.raises(AssertionError):
        compare_outputs((second, state), (output, state), evidence)
    with pytest.raises(AssertionError):
        compare_outputs((output, state + 1e-6), (output, state), evidence)


def test_batch_norm_rejects_correct_output_without_running_stat_updates():
    from workloads.norm import BatchNormFwdWorkload, batch_norm_forward_result

    workload = BatchNormFwdWorkload(2, 2, (2,), torch.float32, True)
    inputs = (torch.ones(2, 2, 2), torch.zeros(2), torch.ones(2), torch.ones(2), torch.zeros(2))

    def missing_update(*args):
        return workload.ref_program(*args)[0]

    with pytest.raises(AssertionError):
        _verify_one(
            lambda *args: batch_norm_forward_result(missing_update, *args),
            inputs,
            reference=workload.ref_program,
            evidence=workload.verification(*inputs),
        )


def test_check_refuses_a_kernel_in_place_of_the_op_path():
    """A result is reported under an Op and runs what a caller reaches, before any reference."""
    import types

    from tests.workload_test_base import TestBase
    from tileops.kernels.elementwise import ReluFwdKernel
    from tileops.ops.elementwise import ReluFwdOp

    def reference(x):
        raise AssertionError("the reference ran")

    workload = types.SimpleNamespace(ref_program=reference, verification=lambda *inputs: Exact())
    kernel = ReluFwdKernel.__new__(ReluFwdKernel)
    for op, runs in ((kernel, None), (ReluFwdOp(), kernel), (ReluFwdOp(), kernel.forward)):
        with pytest.raises(AssertionError, match="runs="):
            TestBase.check(workload, op, torch.ones(1), runs=runs)


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
    ],
)
def test_grouped_routing_on_a_near_tied_boundary(selection, topk_group, ids, accepted):
    """A row whose group boundary is within rounding passes only when some valid set of
    ``topk_group`` groups (every group clearly above the boundary, each group an id lands in,
    the rest from the groups within rounding) has the returned experts as its top-k."""
    num_experts = 2 * len(selection)
    workload = _grouped_topk_workload(num_experts, len(selection), topk_group, len(ids), True)
    gating, bias = torch.logit(torch.tensor([sum(selection, [])])), torch.zeros(num_experts)
    returned = torch.tensor([ids], dtype=torch.int32)
    got = (gating.sigmoid().gather(1, returned.long()), returned)
    expected = workload.ref_program(gating, bias)
    evidence = workload.verification(gating, bias)
    if accepted:
        compare_outputs(got, expected, evidence)
    else:
        with pytest.raises(AssertionError):
            compare_outputs(got, expected, evidence)


def _grouped_topk_workload(num_experts, num_groups, topk_group, top_k, biased):
    """The ``FusedTopKWorkload`` of one token's sigmoid grouped routing, on the CPU."""
    from workloads.moe import FusedTopKWorkload
    from workloads.workload_base import manifest_call

    call = manifest_call(
        "FusedTopKFwdOp",
        {"G": "float32"},
        T=1,
        E=num_experts,
        top_k=top_k,
        scoring_func="sigmoid",
        renormalize=False,
        num_expert_group=num_groups,
        topk_group=topk_group,
        some=["correction_bias"] if biased else [],
    )
    return FusedTopKWorkload(call, device="cpu")


def _valid_group_set_top_k(selection, num_groups, topk_group, top_k, biased):
    """``(groups, top-k ids, sorted top-k selection)`` of every valid set of ``topk_group``
    groups: each holds the groups above the last kept score by more than the slack and none
    below it by more, the slack being 1e-5 of that score plus 1e-6."""
    import itertools

    group_size = selection.numel() // num_groups
    grouped = selection.view(num_groups, group_size)
    if biased:
        pairs = grouped.sort(descending=True).values
        group_scores = pairs[:, 0] + pairs[:, 1]
    else:
        group_scores = grouped.amax(-1)
    last = group_scores.sort(descending=True).values[topk_group - 1]
    slack = 1e-5 * last.abs() + 1e-6
    always = {g for g in range(num_groups) if group_scores[g] > last + slack}
    allowed = {g for g in range(num_groups) if group_scores[g] >= last - slack}
    for groups in itertools.combinations(sorted(allowed), topk_group):
        if always <= set(groups):
            experts = torch.cat(
                [torch.arange(g * group_size, (g + 1) * group_size) for g in groups]
            )
            top = experts[selection[experts].topk(top_k).indices]
            yield set(groups), top.tolist(), selection[top].sort().values


def _verdicts_disagree(workload, inputs, selection, valid, group_size, candidates):
    """The candidates on which the verification's verdict differs from the valid group sets'."""
    expected = workload.ref_program(*inputs)
    evidence = workload.verification(*inputs)
    disagree = []
    for ids in candidates:
        chosen = selection[list(ids)].sort().values
        in_spec = any(
            {i // group_size for i in ids} <= groups
            and torch.isclose(chosen, top, rtol=1e-5, atol=1e-6).all()
            for groups, _, top in valid
        )
        returned = torch.tensor([ids], dtype=torch.int32)
        got = (inputs[0].sigmoid().gather(1, returned.long()), returned)
        try:
            compare_outputs(got, expected, evidence)
            accepted = True
        except AssertionError:
            accepted = False
        if accepted != in_spec:
            disagree.append((ids, accepted))
    return disagree


def _ds_v3_near_tied_candidates(selection, valid, group_size, topk_group):
    """Candidate outputs around each valid DeepSeek-V3 top-k selection."""
    num_groups = selection.numel() // group_size
    grouped = selection.view(num_groups, group_size)
    kept_by_some = set().union(*(groups for groups, _, _ in valid))
    kept_by_all = set.intersection(*(groups for groups, _, _ in valid))
    dropped = [g for g in range(num_groups) if g not in kept_by_some]
    candidates = []
    for groups, top, _ in valid:
        candidates.append(tuple(top))
        weakest = min(top, key=lambda i: selection[i])
        group = weakest // group_size
        below = [i for i in range(group * group_size, (group + 1) * group_size) if i not in top]
        swap = max(below, key=lambda i: selection[i])
        candidates.append(tuple(i if i != weakest else swap for i in top))
        if dropped:
            outside = dropped[0] * group_size + int(grouped[dropped[0]].argmax())
            candidates.append(tuple(i if i != weakest else outside for i in top))
        for skipped in kept_by_all:
            for other in sorted(kept_by_some - groups):
                experts = [
                    i
                    for g in (groups - {skipped}) | {other}
                    for i in range(g * group_size, (g + 1) * group_size)
                ]
                best = sorted(experts, key=lambda i: selection[i], reverse=True)
                candidates.append(tuple(best[: len(top)]))
    spread = selection.view(num_groups, group_size).amax(-1).argsort(descending=True)
    firsts = [int(g) * group_size + int(grouped[g].argmax()) for g in spread[: topk_group + 1]]
    rest = sorted(
        (i for i in range(selection.numel()) if i not in firsts),
        key=lambda i: selection[i],
        reverse=True,
    )
    candidates.append(tuple(firsts + rest[: len(valid[0][1]) - len(firsts)]))
    return candidates


def test_grouped_routing_accepts_exactly_the_selections_of_a_valid_group_set():
    """The verification accepts an output exactly when some valid group set's top-k has its
    selection scores, with every id inside that set: every candidate of small routings, and
    at the DeepSeek-V3 shape the valid sets' top-k and their near misses on near-tied rows."""
    import itertools

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
    ]
    near_tied8 = [
        [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]],
        [[0.95, 0.05], [0.6, 0.4], [0.5, 0.5], [0.7, 0.3]],
        [[0.9, 0.9], [0.9, 0.1], [0.5, 0.5], [0.5, 0.5 + 5e-6]],
    ]
    rng = torch.Generator().manual_seed(0)
    random = [torch.rand(3, 2, generator=rng).tolist() for _ in range(4)]
    random += [(torch.randint(1, 10, (3, 2), generator=rng) / 10).tolist() for _ in range(2)]
    random8 = [torch.rand(4, 2, generator=rng).tolist() for _ in range(3)]
    configs = [
        (top_k, topk_group, biased, matrices)
        for biased in (True, False)
        for top_k, topk_group, matrices in (
            (2, 1, near_tied6 + random),
            (2, 2, near_tied6 + random),
            (3, 2, near_tied8 + random8),
        )
    ]
    disagree = []
    for top_k, topk_group, biased, matrices in configs:
        for values in matrices:
            num_groups, num_experts = len(values), 2 * len(values)
            workload = _grouped_topk_workload(num_experts, num_groups, topk_group, top_k, biased)
            target = torch.tensor([sum(values, [])])
            if biased:
                inputs = (torch.zeros(1, num_experts), target[0] - 0.5)
                selection = inputs[0].sigmoid() + inputs[1]
            else:
                inputs = (torch.logit(target),)
                selection = inputs[0].sigmoid()
            valid = list(
                _valid_group_set_top_k(selection[0], num_groups, topk_group, top_k, biased)
            )
            candidates = itertools.combinations(range(num_experts), top_k)
            disagree += _verdicts_disagree(workload, inputs, selection[0], valid, 2, candidates)

    # Near-tied DeepSeek-V3 groups, including one clearly above the boundary.
    num_groups, group_size, topk_group, top_k = 8, 32, 4, 8
    workload = _grouped_topk_workload(256, num_groups, topk_group, top_k, True)
    for row in range(200):
        gating = torch.randn(1, 256, generator=rng)
        bias = torch.randn(256, generator=rng) * 0.1
        selection = gating[0].sigmoid() + bias
        pairs = selection.view(num_groups, group_size).topk(2).values.sum(-1)
        order = pairs.argsort(descending=True)
        last = pairs[order[topk_group - 1]]
        slack = float(1e-5 * last.abs() + 1e-6)
        near = (
            order[topk_group - 2 : topk_group + 1]
            if row % 2
            else order[topk_group - 1 : topk_group + 1]
        )
        offsets = (torch.rand(len(near), generator=rng) - 0.5) * 1.8 * slack
        if row % 3 == 0:
            near, offsets = order[topk_group - 1 : topk_group + 1], offsets[:2]
            above = order[topk_group - 2]
            bias.view(num_groups, group_size)[above] += (last + 2 * slack - pairs[above]) / 2
        for g, offset in zip(near.tolist(), offsets.tolist(), strict=True):
            bias.view(num_groups, group_size)[g] += (last + offset - pairs[g]) / 2
        selection = gating[0].sigmoid() + bias
        valid = list(_valid_group_set_top_k(selection, num_groups, topk_group, top_k, True))
        candidates = _ds_v3_near_tied_candidates(selection, valid, group_size, topk_group)
        disagree += _verdicts_disagree(
            workload, (gating, bias), selection, valid, group_size, candidates
        )
    assert disagree == []
