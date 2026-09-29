"""The sampling ops against the references in ``workloads/sampling.py``.

Each correctness case checks the reference first and skips when no implementation serves
the op. The random ops are compared by support, structure and distribution: a kernel's
Philox stream need not match the reference's.
"""

import pytest
import torch

from tileops.backend import OpNotAvailableError
from tileops.sampling import (
    ChainSpeculativeSamplingFwdOp,
    MinPMaskFwdOp,
    SamplingFromProbsFwdOp,
    TopKMaskFwdOp,
    TopKTopPMaskFwdOp,
    TopPMaskFwdOp,
)
from workloads.device import run_device
from workloads.sampling import (
    ChainSpeculativeSamplingWorkload,
    MinPMaskWorkload,
    SamplingFromProbsWorkload,
    TopKMaskWorkload,
    TopKTopPMaskWorkload,
    TopPMaskWorkload,
    probability_above,
    sampling_call,
    top_k_mask,
)

pytestmark = pytest.mark.smoke

_DTYPES = ["float16", "bfloat16", "float32"]
_V = 32000
# k = 1, a k that bf16 ties overrun, k == V and k > V.
_K = [1, 50, _V, _V + 7]
# Reference and kernel round the threshold differently; a token this close to it may differ.
_MARGIN = 1e-4
_INF = float("inf")


def _run(op, *inputs):
    try:
        return op(*inputs)
    except OpNotAvailableError as e:
        pytest.skip(str(e))


def _assert_top_set(masked, logits):
    """Per row: the largest logit is kept, and no masked logit exceeds a kept one."""
    kept = masked != -_INF
    values = logits.float()
    assert kept.gather(1, values.argmax(-1, keepdim=True)).all()
    lowest_kept = values.masked_fill(~kept, _INF).amin(-1)
    highest_masked = values.masked_fill(kept, -_INF).amax(-1)
    assert (highest_masked <= lowest_kept).all()


def _assert_nucleus(masked, logits, p):
    """The kept tokens hold at least ``p``, less than ``p`` without their least probable value,
    and every token of that value."""
    _assert_top_set(masked, logits)
    kept = masked != -_INF
    probs = logits.float().softmax(-1)
    lowest = probs.masked_fill(~kept, _INF).amin(-1, keepdim=True)
    assert ((probs * kept).sum(-1) >= p - _MARGIN).all()
    assert ((probs * (kept & (probs > lowest))).sum(-1) < p + _MARGIN).all()
    # Tokens tied with the least probable kept one are kept with it.
    assert not (~kept & (probs == lowest)).any()


def _assert_same_mask(out, ref, logits, near):
    """Equal kept sets away from the threshold, and kept logits passed through unchanged."""
    assert out.dtype == logits.dtype
    kept = out != -_INF
    assert not ((kept ^ (ref != -_INF)) & ~near).any()
    assert torch.equal(out[kept], logits[kept])


def _assert_follows(samples, probs):
    """Each index's count within ``6 sigma + 5`` of its expectation; a zero-probability index
    never drawn. The ``+ 5`` covers indices expected fewer than a few times."""
    n = samples.numel()
    count = torch.bincount(samples.long(), minlength=probs.numel()).double()
    p = probs.double()
    bound = 6 * (n * p * (1 - p)).sqrt() + 5 * (p > 0)
    assert ((count - n * p).abs() <= bound).all(), (count, n * p)


def _tie_at_the_boundary(logits, p):
    """Row 1 becomes three equal top tokens and ``p[1] = 0.5``: all three survive top-p."""
    logits[1] = -100.0
    logits[1, :3] = 0.0
    p[1] = 0.5


@pytest.mark.parametrize("dtype", _DTYPES)
def test_top_k_mask(dtype):
    workload = TopKMaskWorkload(sampling_call("TopKMaskFwdOp", {"T": dtype}, V=_V, k_list=_K))
    logits, k = workload.gen_inputs()
    ref = workload.ref_program(logits, k)
    _assert_top_set(ref, logits)
    kept = (ref != -_INF).sum(-1)
    assert (kept >= k.clamp(max=_V)).all()
    assert torch.equal(kept[k >= _V], torch.full_like(kept[k >= _V], _V))
    out = _run(TopKMaskFwdOp(), logits, k)
    assert out.dtype == logits.dtype and torch.equal(out, ref)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_min_p_mask(dtype):
    workload = MinPMaskWorkload(sampling_call("MinPMaskFwdOp", {"T": dtype}, B=4, V=_V))
    logits, min_p = workload.gen_inputs()
    ref = workload.ref_program(logits, min_p)
    _assert_top_set(ref, logits)
    probs = logits.float().softmax(-1)
    relative = probs / probs.amax(-1, keepdim=True) - min_p[:, None]
    assert (relative[ref == -_INF] < _MARGIN).all() and (relative[ref != -_INF] > -_MARGIN).all()
    out = _run(MinPMaskFwdOp(), logits, min_p)
    values = logits.float()
    threshold = values.amax(-1, keepdim=True) + min_p[:, None].log()
    _assert_same_mask(out, ref, logits, (values - threshold).abs() <= _MARGIN)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_top_p_mask(dtype):
    workload = TopPMaskWorkload(sampling_call("TopPMaskFwdOp", {"T": dtype}, B=4, V=_V))
    logits, p = workload.gen_inputs()
    _tie_at_the_boundary(logits, p)
    ref = workload.ref_program(logits, p)
    _assert_nucleus(ref, logits, p)
    out = _run(TopPMaskFwdOp(), logits, p)
    above = probability_above(logits.float().softmax(-1))
    _assert_same_mask(out, ref, logits, (above - p[:, None]).abs() <= _MARGIN)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_top_k_top_p_mask(dtype):
    call = sampling_call("TopKTopPMaskFwdOp", {"T": dtype}, V=_V, k_list=_K)
    workload = TopKTopPMaskWorkload(call)
    logits, k, p = workload.gen_inputs()
    _tie_at_the_boundary(logits, p)
    ref = workload.ref_program(logits, k, p)
    top_k = top_k_mask(logits, k)
    assert ((ref != -_INF) <= (top_k != -_INF)).all()
    _assert_nucleus(ref, top_k, p)
    out = _run(TopKTopPMaskFwdOp(), logits, k, p)
    above = probability_above(top_k.float().softmax(-1))
    _assert_same_mask(out, ref, logits, (above - p[:, None]).abs() <= _MARGIN)


def test_sampling_from_probs():
    """Unnormalized rows with every fourth weight zero, drawn in 65536 identical rows."""
    n, vocab = 65536, 64
    device = run_device()
    weights = torch.rand(vocab, device=device)
    weights[::4] = 0
    probs = weights.expand(n, vocab).contiguous()
    seed = torch.tensor([1234], dtype=torch.int64, device=device)
    offset = torch.tensor([7], dtype=torch.int64, device=device)
    workload = SamplingFromProbsWorkload(sampling_call("SamplingFromProbsFwdOp", B=n, V=vocab))
    ref = workload.ref_program(probs, seed, offset)
    _assert_follows(ref, weights / weights.sum())
    assert torch.equal(ref, workload.ref_program(probs, seed, offset))
    op = SamplingFromProbsFwdOp()
    out = _run(op, probs, seed, offset)
    assert out.dtype == torch.int32
    _assert_follows(out, weights / weights.sum())
    assert torch.equal(out, op(probs, seed, offset))


def _assert_verifies_chains(tokens, num, draft_ids, draft, target, accepted):
    """The accepted prefix is the drafts and ``-1`` follows the drawn token; the drawn token
    follows the residual of its position, or target row ``N`` after a whole chain; the first
    token follows target row 0; ``num`` follows the acceptance probabilities ``accepted``."""
    assert tokens.dtype == num.dtype == torch.int32
    num_draft = draft.shape[0]
    position = torch.arange(num_draft + 1, device=tokens.device)[None]
    prefix = position[:, :-1] < num[:, None]
    assert torch.equal(tokens[:, :-1][prefix], draft_ids[prefix])
    assert (tokens[position > num[:, None]] == -1).all()
    for stop in range(num_draft + 1):
        weights = target[stop] if stop == num_draft else (target[stop] - draft[stop]).clamp_min(0)
        _assert_follows(tokens[num == stop, stop], weights / weights.sum())
    _assert_follows(tokens[:, 0], target[0])
    a0, a1 = accepted
    _assert_follows(num, torch.stack([1 - a0, a0 * (1 - a1), a0 * a1]))


def test_chain_speculative_sampling():
    """Two drafts per row over 16 tokens, each draft drawn from its draft row, in 65536 rows."""
    n, num_draft, vocab = 65536, 2, 16
    device = run_device()
    draft = torch.rand(num_draft, vocab, device=device) ** 3
    draft /= draft.sum(-1, keepdim=True)
    target = torch.rand(num_draft + 1, vocab, device=device) ** 3
    target /= target.sum(-1, keepdim=True)
    draft_ids = torch.multinomial(draft, n, replacement=True).T.to(torch.int32).contiguous()
    # A draft drawn from its row is accepted with probability sum(min(draft, target)).
    accepted = torch.minimum(draft, target[:num_draft]).sum(-1)
    inputs = (
        draft.expand(n, num_draft, vocab).contiguous(),
        draft_ids,
        target.expand(n, num_draft + 1, vocab).contiguous(),
        torch.tensor([1234], dtype=torch.int64, device=device),
        torch.tensor([7], dtype=torch.int64, device=device),
    )
    call = sampling_call("ChainSpeculativeSamplingFwdOp", B=n, N=num_draft, V=vocab)
    workload = ChainSpeculativeSamplingWorkload(call)
    ref = workload.ref_program(*inputs)
    _assert_verifies_chains(*ref, draft_ids, draft, target, accepted)
    assert all(map(torch.equal, ref, workload.ref_program(*inputs)))
    op = ChainSpeculativeSamplingFwdOp()
    out = _run(op, *inputs)
    _assert_verifies_chains(*out, draft_ids, draft, target, accepted)
    assert all(map(torch.equal, out, op(*inputs)))


_SMALL_CALLS = {
    TopKMaskFwdOp: (TopKMaskWorkload, {"T": "float32"}, {"V": 64, "k_list": [1, 5]}),
    MinPMaskFwdOp: (MinPMaskWorkload, {"T": "float32"}, {"B": 2, "V": 64}),
    TopPMaskFwdOp: (TopPMaskWorkload, {"T": "float32"}, {"B": 2, "V": 64}),
    TopKTopPMaskFwdOp: (TopKTopPMaskWorkload, {"T": "float32"}, {"V": 64, "k_list": [1, 5]}),
    SamplingFromProbsFwdOp: (SamplingFromProbsWorkload, None, {"B": 2, "V": 64}),
    ChainSpeculativeSamplingFwdOp: (
        ChainSpeculativeSamplingWorkload,
        None,
        {"B": 2, "N": 2, "V": 64},
    ),
}


@pytest.mark.parametrize("cls", list(_SMALL_CALLS), ids=lambda cls: cls.__name__)
def test_generated_checks_reject_an_integer_first_input(cls):
    workload_cls, dtype_case, row = _SMALL_CALLS[cls]
    inputs = list(workload_cls(sampling_call(cls.__name__, dtype_case, **row)).gen_inputs())
    inputs[0] = inputs[0].to(torch.int32)
    with pytest.raises(ValueError):
        cls()(*inputs)
