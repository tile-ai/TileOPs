"""The sampling ops against the references in ``workloads/sampling.py``.

Each correctness case checks the reference first and skips when no implementation serves
the op. The random ops are compared by support, structure and distribution: a kernel's
Philox stream need not match the reference's.
"""

import pytest
import torch

from tileops.backend import OpNotAvailableError
from tileops.kernels.sampling import SamplingCall
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
    chain_speculative_sampling,
    min_p_mask,
    probability_above,
    sampling_call,
    sampling_from_probs,
    top_k_mask,
    top_p_mask,
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


def _special_rows(vocab: int, dtype: torch.dtype) -> torch.Tensor:
    """Six rows: noise, both zeros, both infinities, NaN, one repeated value, and two."""
    device = run_device()
    rows = torch.randn(6, vocab, device=device).to(dtype)
    rows[1, ::2], rows[1, 1::2] = 0.0, -0.0
    rows[2, :7], rows[2, 7:14] = _INF, -_INF
    rows[3, :5] = float("nan")
    rows[4] = 1.5
    rows[5] = torch.where(torch.arange(vocab, device=device) % 3 == 0, 2.0, -2.0).to(dtype)
    return rows


@pytest.mark.parametrize(
    "dtype, vocab",
    [
        # 16-bit keys with the row on the vector, 16-bit and 32-bit keys with it off.
        (torch.bfloat16, 4096),
        (torch.float16, 999),
        (torch.float32, 999),
    ],
)
def test_top_k_mask_matches_the_reference_bit_for_bit(dtype: torch.dtype, vocab: int):
    """Each row is cut where its own special values sit: at a zero, at an infinity, at a
    NaN, among equal values, and at a k the row leaves whole."""
    logits = _special_rows(vocab, dtype)
    k = torch.tensor(
        [1, vocab // 2, 3, 2, vocab // 4, vocab], dtype=torch.int32, device=logits.device
    )
    ref = top_k_mask(logits, k)
    out = _run(TopKMaskFwdOp(), logits, k)
    bits = torch.int16 if logits.element_size() == 2 else torch.int32
    # Bit patterns, so that -0.0 kept where the reference keeps 0.0 is a failure.
    assert torch.equal(out.view(bits), ref.view(bits))


@pytest.mark.packaging(family="sampling")
def test_top_k_mask_cuts_a_row_its_samples_misplace():
    """Every value a sample can land on is the row's smallest, so the bracket the samples
    give holds no rank the k-th value can take."""
    vocab = 8192
    device = run_device()
    columns = torch.arange(vocab, device=device)
    logits = torch.where(columns % 4 == 0, 0.0, 1.0).to(torch.float32)[None]
    k = torch.tensor([vocab // 2], dtype=torch.int32, device=device)
    out = _run(TopKMaskFwdOp(), logits, k)
    assert torch.equal(out, top_k_mask(logits, k))


@pytest.mark.in_tree_kernels
def test_top_k_mask_selects_its_one_implementation():
    call = SamplingCall(arch=90, sm_count=132, batch=64, vocab=128256, dtype=torch.bfloat16)
    assert TopKMaskFwdOp().select_implementation("top_k_mask_fwd", call) == "top_k_mask_fwd"


@pytest.mark.in_tree_kernels
def test_top_k_mask_refuses_a_call_int32_cannot_index():
    call = SamplingCall(arch=90, sm_count=132, batch=2**16, vocab=2**16, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="B \\* V"):
        TopKMaskFwdOp().select_implementation("top_k_mask_fwd", call)


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


def test_min_p_mask_rows_without_a_finite_threshold():
    """A row whose maximum is not a number passes through, at each shape the kernel plans for.

    ``V`` picks the plan: a row split across blocks, which reduces across a grid barrier; a
    row one block holds, which takes no barrier; and a row whose bytes are not a whole number
    of 16-byte vectors, which is read and written element by element.
    """
    device = run_device()
    for vocab in (_V, 512, 4099):
        logits = torch.randn(5, vocab, device=device)
        logits[1] = -_INF
        logits[2, vocab // 2] = float("nan")
        logits[3, 0] = _INF
        min_p = torch.tensor([0.0, 0.5, 0.5, 0.5, 1.0], device=device)
        out = _run(MinPMaskFwdOp(), logits, min_p)
        ref = min_p_mask(logits, min_p)
        assert ((out == ref) | (out.isnan() & ref.isnan())).all(), vocab


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


@pytest.mark.in_tree_kernels
def test_top_p_mask_selects_its_one_implementation():
    call = SamplingCall(arch=90, sm_count=132, batch=64, vocab=128256, dtype=torch.bfloat16)
    assert TopPMaskFwdOp().select_implementation("top_p_mask_fwd", call) == "top_p_mask_fwd"


def test_top_p_mask_rows_at_the_contract_endpoints():
    """``p = 0`` masks a row whole, ``p = 1`` keeps every token the softmax gives mass to,
    and a row with no finite maximum passes through, at each shape the kernel plans for.

    ``V`` picks the plan: a row split across blocks, which reduces across a grid barrier; a
    row one block holds, which takes no barrier; and a row whose bytes are not a whole number
    of 16-byte vectors, which is read and written element by element. The endpoints are held
    to the contract rather than to the reference, which rounds its own cumulative sum past 1
    at ``p = 1``; the rows with no finite maximum are held to the reference, which leaves
    them whole because a NaN never reaches ``p``.
    """
    device = run_device()
    p = torch.tensor([0.0, 1.0, 0.5], device=device)
    for vocab in (_V, 512, 4099):
        logits = torch.randn(3, vocab, device=device)
        out = _run(TopPMaskFwdOp(), logits, p)
        assert (out[0] == -_INF).all(), vocab
        assert torch.equal(out[1] != -_INF, logits.float().softmax(-1)[1] > 0), vocab
        logits[0, vocab // 2] = float("nan")
        logits[1, 0] = _INF
        logits[2] = -_INF
        out = _run(TopPMaskFwdOp(), logits, p)
        ref = top_p_mask(logits, p)
        assert ((out == ref) | (out.isnan() & ref.isnan())).all(), vocab


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


@pytest.mark.parametrize(
    "dtype, vocab",
    [
        # 16-bit keys with the row on the vector, 16-bit and 32-bit keys with it off.
        (torch.bfloat16, 4096),
        (torch.float16, 999),
        (torch.float32, 999),
    ],
)
def test_top_k_top_p_mask_cuts_special_rows_where_the_reference_does(dtype, vocab):
    """Row 1's top-k bound lands on a zero and the other zero is kept with it; rows 2 and 3
    have no finite largest logit once ``k`` leaves them whole, so the reference's
    probabilities are NaN, top-p removes nothing and the kept sets have to agree exactly.
    The NaN row is cut both ways: at a ``k`` that drops its NaNs and at one that keeps them."""
    logits = _special_rows(vocab, dtype)
    device = logits.device
    p = torch.tensor([0.9, 0.95, 0.8, 0.95, 0.6, 0.5], dtype=torch.float32, device=device)
    bits = torch.int16 if logits.element_size() == 2 else torch.int32
    for nan_k in (2, vocab):
        k = torch.tensor(
            [1, vocab // 2, vocab, nan_k, vocab // 4, vocab // 3], dtype=torch.int32, device=device
        )
        ref = top_p_mask(top_k_mask(logits, k), p)
        out = _run(TopKTopPMaskFwdOp(), logits, k, p)
        above = probability_above(top_k_mask(logits, k).float().softmax(-1))
        kept = out != -_INF
        # A row of NaN probabilities is near no boundary, so its kept set agrees exactly.
        assert not (((ref != -_INF) ^ kept) & ~((above - p[:, None]).abs() <= _MARGIN)).any()
        # Bit patterns, so that -0.0 kept where the reference keeps 0.0 is a failure. A NaN
        # a row keeps comes back as a quiet NaN, not its own payload.
        passed = kept & ~logits.isnan()
        assert torch.equal(out[passed].view(bits), logits[passed].view(bits))
        assert out[kept & logits.isnan()].isnan().all()


@pytest.mark.in_tree_kernels
def test_top_k_top_p_mask_selects_its_one_implementation():
    call = SamplingCall(arch=90, sm_count=132, batch=64, vocab=128256, dtype=torch.bfloat16)
    op = TopKTopPMaskFwdOp()
    assert op.select_implementation("top_k_top_p_mask_fwd", call) == "top_k_top_p_mask_fwd"
    wide = SamplingCall(arch=90, sm_count=132, batch=2**16, vocab=2**16, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="B \\* V"):
        op.select_implementation("top_k_top_p_mask_fwd", wide)


@pytest.mark.in_tree_kernels
def test_sampling_from_probs_refuses_a_call_int32_cannot_index():
    call = SamplingCall(arch=90, sm_count=132, batch=2**16, vocab=2**16, dtype=torch.float32)
    with pytest.raises(ValueError, match="B \\* V"):
        SamplingFromProbsFwdOp().select_implementation("sampling_from_probs", call)


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


def test_sampling_from_probs_draws_the_same_token_at_every_launch_shape():
    """A row drawn inside batches that take different launches gives one token, and weight.

    The batch settles how many blocks share a row, and with it how the prefix sums the
    search descends are grouped. ``B = 1`` splits the row across the device and takes the
    grid barrier; a batch past the block count leaves the row to one block and takes none.
    Every seventh weight is zero, so the same case says the split row draws no zero weight.
    """
    device = run_device()
    torch.manual_seed(3)
    weights = torch.rand(151936, device=device)
    weights[::7] = 0
    row = weights / weights.sum()
    seed = torch.tensor([1234], dtype=torch.int64, device=device)
    offset = torch.tensor([7], dtype=torch.int64, device=device)
    drawn = {
        batch: _run(SamplingFromProbsFwdOp(), row.expand(batch, -1).contiguous(), seed, offset)[0]
        for batch in (1, 17, 300)
    }
    assert len(set(int(token) for token in drawn.values())) == 1, drawn
    assert (row[torch.stack(list(drawn.values())).long()] > 0).all()
    assert int(drawn[1]) == int(sampling_from_probs(row[None], seed, offset)[0])


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


@pytest.mark.in_tree_kernels
def test_chain_speculative_sampling_selects_its_one_implementation():
    call = SamplingCall(arch=90, sm_count=132, batch=64, vocab=128256, num_draft=4)
    op = ChainSpeculativeSamplingFwdOp()
    assert (
        op.select_implementation("chain_speculative_sampling", call) == "chain_speculative_sampling"
    )


@pytest.mark.in_tree_kernels
def test_chain_speculative_sampling_refuses_a_call_int32_cannot_index():
    call = SamplingCall(arch=90, sm_count=132, batch=2**16, vocab=2**16, num_draft=1)
    with pytest.raises(ValueError, match=r"B \* \(N \+ 1\) \* V"):
        ChainSpeculativeSamplingFwdOp().select_implementation("chain_speculative_sampling", call)


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "batch, vocab, num_draft",
    [
        # A row split across blocks, a row one block holds, a row whose bytes are not a whole
        # number of 16-byte vectors, which is folded weight by weight, and a chain longer than
        # the block, whose uniforms one thread each takes over several rounds.
        (2, 151936, 3),
        (300, 32000, 3),
        (8, 4099, 3),
        (4, 64, 600),
    ],
)
def test_chain_speculative_sampling_accepts_the_reference_prefix(batch, vocab, num_draft):
    """``num_accepted`` and the accepted prefix are the reference's, at every launch shape.

    The uniforms and the float32 ``u * draft < target`` test are the reference's, and neither
    names a launch fact, so the verification agrees exactly however the row is split. The
    token after the prefix follows the same distribution but is not required to be the
    reference's index.
    """
    device = run_device()
    torch.manual_seed(11)
    draft = torch.randn(batch, num_draft, vocab, device=device).softmax(-1)
    target = torch.randn(batch, num_draft + 1, vocab, device=device).softmax(-1)
    ids = torch.multinomial(draft.reshape(-1, vocab), 1).view(batch, num_draft)
    inputs = (
        draft,
        ids.to(torch.int32).contiguous(),
        target,
        torch.tensor([1234], dtype=torch.int64, device=device),
        torch.tensor([7], dtype=torch.int64, device=device),
    )
    ref_tokens, ref_num = chain_speculative_sampling(*inputs)
    tokens, num = _run(ChainSpeculativeSamplingFwdOp(), *inputs)
    assert torch.equal(num, ref_num)
    position = torch.arange(num_draft + 1, device=device)[None]
    prefix = position < num[:, None]
    assert torch.equal(tokens[prefix], ref_tokens[prefix])
    assert (tokens[position > num[:, None]] == -1).all()
    # The drawn token carries residual weight at the position the chain stopped on.
    rows = torch.arange(batch, device=device)
    padded = torch.cat([draft, torch.zeros_like(draft[:, :1])], 1)
    weights = (target[rows, num.long()] - padded[rows, num.long()]).clamp_min(0)
    assert (weights[rows, tokens[rows, num.long()].long()] > 0).all()


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
