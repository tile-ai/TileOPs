"""The sampling ops against the references in ``workloads/sampling.py``.

Each correctness case checks the reference first and skips when no implementation serves
the op. The random ops are compared by support, structure and distribution: a kernel's
Philox stream need not match the reference's.
"""

import pytest
import torch

from tests.workload_test_base import TestBase
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
from workloads.numerics import compare_outputs
from workloads.sampling import (
    ChainSpeculativeSamplingWorkload,
    MinPMaskWorkload,
    SamplingFromProbsWorkload,
    TopKMaskWorkload,
    TopKTopPMaskWorkload,
    TopPMaskWorkload,
    min_p_mask,
    min_p_mask_verification,
    sampling_from_probs,
    top_k_mask,
    top_k_mask_verification,
    top_p_mask,
    top_p_mask_verification,
)
from workloads.workload_base import manifest_call

_DTYPES = ["float16", "bfloat16", "float32"]
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


def _assert_nucleus(masked, logits, p, *, margin):
    """The kept tokens hold at least ``p``, less than ``p`` without their least probable value,
    and every token of that value."""
    _assert_top_set(masked, logits)
    kept = masked != -_INF
    probs = logits.float().softmax(-1)
    lowest = probs.masked_fill(~kept, _INF).amin(-1, keepdim=True)
    assert ((probs * kept).sum(-1) >= p - margin).all()
    assert ((probs * (kept & (probs > lowest))).sum(-1) < p + margin).all()
    # Tokens tied with the least probable kept one are kept with it.
    assert not (~kept & (probs == lowest)).any()


def _tie_at_the_boundary(logits, p):
    """Row 1 becomes three equal top tokens and ``p[1] = 0.5``: all three survive top-p."""
    logits[1] = -100.0
    logits[1, :3] = 0.0
    p[1] = 0.5


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", _DTYPES)
def test_top_k_mask(dtype):
    vocab = 32000
    # k = 1, a k that bf16 ties overrun, k == vocab and k > vocab.
    workload = TopKMaskWorkload(
        manifest_call("TopKMaskFwdOp", {"T": dtype}, V=vocab, k_list=[1, 50, vocab, vocab + 7])
    )
    logits, k = workload.gen_inputs()
    ref = workload.ref_program(logits, k)
    _assert_top_set(ref, logits)
    kept = (ref != -_INF).sum(-1)
    assert (kept >= k.clamp(max=vocab)).all()
    assert torch.equal(kept[k >= vocab], torch.full_like(kept[k >= vocab], vocab))
    out = _run(TopKMaskFwdOp(), logits, k)
    compare_outputs(out, ref, workload.verification(logits, k))


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


@pytest.mark.smoke
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
    # Bit patterns, so that -0.0 kept where the reference keeps 0.0 is a failure.
    compare_outputs(out, ref, top_k_mask_verification(logits))


@pytest.mark.smoke
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
    compare_outputs(out, top_k_mask(logits, k), top_k_mask_verification(logits))


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", _DTYPES)
def test_min_p_mask(dtype):
    # Allow threshold-rounding differences only for tokens this close to the cutoff.
    margin = 1e-4
    vocab = 32000
    workload = MinPMaskWorkload(manifest_call("MinPMaskFwdOp", {"T": dtype}, B=4, V=vocab))
    logits, min_p = workload.gen_inputs()
    ref = workload.ref_program(logits, min_p)
    _assert_top_set(ref, logits)
    probs = logits.float().softmax(-1)
    relative = probs / probs.amax(-1, keepdim=True) - min_p[:, None]
    assert (relative[ref == -_INF] < margin).all() and (relative[ref != -_INF] > -margin).all()
    out = _run(MinPMaskFwdOp(), logits, min_p)
    compare_outputs(out, ref, workload.verification(logits, min_p))


@pytest.mark.parametrize(
    "vocab",
    [
        pytest.param(32000, marks=pytest.mark.smoke),
        pytest.param(512, marks=pytest.mark.full),
        pytest.param(4099, marks=pytest.mark.full),
    ],
)
def test_min_p_mask_rows_without_a_finite_threshold(vocab: int):
    """A row whose maximum is not a number passes through. ``V`` picks the plan: split across
    blocks, one block, element by element."""
    device = run_device()
    logits = torch.randn(5, vocab, device=device)
    logits[1] = -_INF
    logits[2, vocab // 2] = float("nan")
    logits[3, 0] = _INF
    min_p = torch.tensor([0.0, 0.5, 0.5, 0.5, 1.0], device=device)
    out = _run(MinPMaskFwdOp(), logits, min_p)
    ref = min_p_mask(logits, min_p)
    compare_outputs(out, ref, min_p_mask_verification(logits, min_p))


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", _DTYPES)
def test_top_p_mask(dtype):
    # Allow threshold-rounding differences only for tokens this close to the cutoff.
    margin = 1e-4
    vocab = 32000
    workload = TopPMaskWorkload(manifest_call("TopPMaskFwdOp", {"T": dtype}, B=4, V=vocab))
    logits, p = workload.gen_inputs()
    _tie_at_the_boundary(logits, p)
    ref = workload.ref_program(logits, p)
    _assert_nucleus(ref, logits, p, margin=margin)
    out = _run(TopPMaskFwdOp(), logits, p)
    compare_outputs(out, ref, workload.verification(logits, p))


@pytest.mark.parametrize(
    "vocab",
    [
        pytest.param(32000, marks=pytest.mark.smoke),
        pytest.param(512, marks=pytest.mark.full),
        pytest.param(4099, marks=pytest.mark.full),
    ],
)
def test_top_p_mask_rows_at_the_contract_endpoints(vocab: int):
    """``p = 0`` masks a row whole, ``p = 1`` passes it through whole (tokens 20 and 200 below
    the maximum included), and a row with no finite maximum passes through. ``V`` picks the
    plan: split across blocks, one block, element by element. The endpoints are held to the
    contract, since the reference rounds its cumulative sum past 1 at ``p = 1``.
    """
    device = run_device()
    p = torch.tensor([0.0, 1.0, 0.5], device=device)
    logits = torch.randn(3, vocab, device=device)
    logits[1, 1] = logits[1].max() - 20
    logits[1, 2] = logits[1].max() - 200
    out = _run(TopPMaskFwdOp(), logits, p)
    assert (out[0] == -_INF).all()
    assert torch.equal(out[1], logits[1])
    logits[0, vocab // 2] = float("nan")
    logits[1, 0] = _INF
    logits[2] = -_INF
    out = _run(TopPMaskFwdOp(), logits, p)
    ref = top_p_mask(logits, p)
    compare_outputs(out, ref, top_p_mask_verification(logits, p))


@pytest.mark.smoke
def test_top_p_mask_below_one_keeps_a_tail_of_tiny_weights():
    """``p`` just below 1 keeps a split row's tail of 31999 tokens of weight ``exp(-19)`` each,
    which together hold about 1.8e-4 of the total."""
    device = run_device()
    logits = torch.full((1, 32000), -19.0, device=device)
    logits[0, 0] = 0.0
    p = torch.full((1,), 1 - 2**-24, device=device)
    ref = top_p_mask(logits, p)
    assert (ref != -_INF).all()
    out = _run(TopPMaskFwdOp(), logits, p)
    assert torch.equal(out, ref)


@pytest.mark.parametrize(
    ("batch", "vocab"),
    [
        pytest.param(64, 201088, marks=pytest.mark.smoke),
        pytest.param(2, 1 << 20, marks=pytest.mark.full),
        pytest.param(17, 50257, marks=pytest.mark.full),
    ],
)
def test_top_p_mask_split_rows_cut_where_the_reference_does(batch: int, vocab: int):
    """Split rows cut where the reference does: three fixed-point pieces (``V = 201088``),
    float32 bins past a 64-bit fixed-point total (``V = 2^20``), and element-wise reads
    (``V = 50257``)."""
    device = run_device()
    logits = torch.randn(batch, vocab, device=device)
    p = torch.rand(batch, device=device) * 0.45 + 0.5
    p[0] = 1.0
    out = _run(TopPMaskFwdOp(), logits, p)
    compare_outputs(out, top_p_mask(logits, p), top_p_mask_verification(logits, p))


@pytest.mark.parametrize(
    "batch",
    [
        # A split row: fixed-point bins and a row barrier counter, both integer inputs.
        pytest.param(4, marks=pytest.mark.smoke),
        # One block holds each row; its plan still takes an integer counter input.
        pytest.param(256, marks=pytest.mark.full),
    ],
)
def test_top_p_mask_tuned(batch: int):
    """Tuned mode cuts where the reference does, on plans whose integer inputs the tuner
    must supply rather than draw at random."""
    device = run_device()
    logits = torch.randn(batch, 32000, device=device)
    p = torch.rand(batch, device=device) * 0.45 + 0.5
    p[0] = 1.0
    op = TopPMaskFwdOp()
    op.request_tune()
    out = _run(op, logits, p)
    compare_outputs(out, top_p_mask(logits, p), top_p_mask_verification(logits, p))


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, marks=pytest.mark.smoke),
        # Three passes, so each call leaves the row's barrier counter in the other phase.
        pytest.param(torch.float16, marks=pytest.mark.smoke),
    ],
)
def test_top_p_mask_split_row_repeats_its_mask(dtype):
    """A row split across blocks gets the same mask on every call, at a ``p`` where another
    rounding of the bins' totals would move the cut, whichever phase the previous call left
    the row's barrier counter in."""
    device = run_device()
    generator = torch.Generator(device=device).manual_seed(0)
    logits = (-30 * torch.rand(1, 128256, device=device, generator=generator)).to(dtype)
    p = torch.full((1,), 1 - 2**-24, device=device)
    op = TopPMaskFwdOp()
    first = _run(op, logits, p)
    compare_outputs(first, top_p_mask(logits, p), top_p_mask_verification(logits, p))
    for _ in range(8):
        assert torch.equal(op(logits, p), first)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", _DTYPES)
def test_top_k_top_p_mask(dtype):
    # Allow threshold-rounding differences only for tokens this close to the cutoff.
    margin = 1e-4
    vocab = 32000
    call = manifest_call(
        "TopKTopPMaskFwdOp", {"T": dtype}, V=vocab, k_list=[1, 50, vocab, vocab + 7]
    )
    workload = TopKTopPMaskWorkload(call)
    logits, k, p = workload.gen_inputs()
    _tie_at_the_boundary(logits, p)
    ref = workload.ref_program(logits, k, p)
    top_k = top_k_mask(logits, k)
    assert ((ref != -_INF) <= (top_k != -_INF)).all()
    _assert_nucleus(ref, top_k, p, margin=margin)
    out = _run(TopKTopPMaskFwdOp(), logits, k, p)
    compare_outputs(out, ref, workload.verification(logits, k, p))


@pytest.mark.smoke
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
    for nan_k in (2, vocab):
        k = torch.tensor(
            [1, vocab // 2, vocab, nan_k, vocab // 4, vocab // 3], dtype=torch.int32, device=device
        )
        ref = top_p_mask(top_k_mask(logits, k), p)
        out = _run(TopKTopPMaskFwdOp(), logits, k, p)
        compare_outputs(out, ref, top_p_mask_verification(logits, p, k=k))


@pytest.mark.smoke
@pytest.mark.sm89
def test_top_k_masks_hold_a_row_in_one_cta():
    """Thread block clusters start at SM90, so on SM89 one CTA holds qwen3's bfloat16 row."""
    vocab = 151936
    k_list = [1, 50, vocab]
    top_k = TopKMaskWorkload(
        manifest_call("TopKMaskFwdOp", {"T": "bfloat16"}, V=vocab, k_list=k_list)
    )
    logits, k = top_k.gen_inputs()
    out = TopKMaskFwdOp()(logits, k)
    compare_outputs(out, top_k.ref_program(logits, k), top_k.verification(logits, k))

    nucleus = TopKTopPMaskWorkload(
        manifest_call("TopKTopPMaskFwdOp", {"T": "bfloat16"}, V=vocab, k_list=k_list)
    )
    logits, k, p = nucleus.gen_inputs()
    out = TopKTopPMaskFwdOp()(logits, k, p)
    compare_outputs(out, nucleus.ref_program(logits, k, p), nucleus.verification(logits, k, p))


def _wide_row(vocab: int):
    device = run_device()
    logits = torch.randn(1, vocab, device=device, dtype=torch.bfloat16)
    return logits, torch.ones(1, dtype=torch.int32, device=device)


@pytest.mark.smoke
@pytest.mark.sm89
@pytest.mark.in_tree_kernels
def test_top_k_mask_refuses_a_row_one_cta_cannot_hold():
    """Thread block clusters start at SM90, so on SM89 a row wider than one CTA holds is
    refused before anything is built."""
    op = TopKMaskFwdOp()
    with pytest.raises(ValueError, match="supports rows of at most"):
        op(*_wide_row(1 << 19))
    for interface in TopKMaskFwdOp.interfaces:
        assert not op._entries_built(interface)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_top_k_top_p_mask_refuses_a_row_it_cannot_hold():
    """A row of 2**19 values is wider than the kernel holds on any architecture, so it is
    refused before anything is built."""
    op = TopKTopPMaskFwdOp()
    p = torch.full((1,), 0.9, device=run_device())
    with pytest.raises(ValueError, match="supports rows of at most"):
        op(*_wide_row(1 << 19), p)
    for interface in TopKTopPMaskFwdOp.interfaces:
        assert not op._entries_built(interface)


@pytest.mark.smoke
def test_sampling_from_probs():
    """Unnormalized rows with every fourth weight zero, drawn in 65536 identical rows."""
    n, vocab = 65536, 64
    device = run_device()
    weights = torch.rand(vocab, device=device)
    weights[::4] = 0
    probs = weights.expand(n, vocab).contiguous()
    seed = torch.tensor([1234], dtype=torch.int64, device=device)
    offset = torch.tensor([7], dtype=torch.int64, device=device)
    workload = SamplingFromProbsWorkload(manifest_call("SamplingFromProbsFwdOp", B=n, V=vocab))
    ref = workload.ref_program(probs, seed, offset)
    assert torch.equal(ref, workload.ref_program(probs, seed, offset))
    op = SamplingFromProbsFwdOp()
    out = _run(op, probs, seed, offset)
    assert out.dtype == torch.int32
    TestBase.check(workload, op, probs, seed, offset, runs=lambda *args: _run(op, *args))
    assert torch.equal(out, op(probs, seed, offset))


@pytest.mark.smoke
def test_sampling_from_probs_draws_the_same_token_at_every_launch_shape():
    """A row drawn inside batches that split it differently (``B = 1`` across the device,
    ``B = 300`` one block) gives one token, never one of zero weight."""
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


@pytest.mark.smoke
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
    inputs = (
        draft.expand(n, num_draft, vocab).contiguous(),
        draft_ids,
        target.expand(n, num_draft + 1, vocab).contiguous(),
        torch.tensor([1234], dtype=torch.int64, device=device),
        torch.tensor([7], dtype=torch.int64, device=device),
    )
    call = manifest_call("ChainSpeculativeSamplingFwdOp", B=n, N=num_draft, V=vocab)
    workload = ChainSpeculativeSamplingWorkload(call)
    ref = workload.ref_program(*inputs)
    assert all(map(torch.equal, ref, workload.ref_program(*inputs)))
    op = ChainSpeculativeSamplingFwdOp()
    out = _run(op, *inputs)
    TestBase.check(workload, op, *inputs, runs=lambda *args: _run(op, *args))
    assert all(map(torch.equal, out, op(*inputs)))


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "batch, vocab, num_draft",
    [
        # A row split across blocks, a row one block holds, a row whose bytes are not a whole
        # number of 16-byte vectors, which is folded weight by weight, and a chain longer than
        # the block, whose uniforms one thread each takes over several rounds.
        pytest.param(2, 151936, 3, marks=pytest.mark.smoke),
        pytest.param(300, 32000, 3, marks=pytest.mark.full),
        pytest.param(8, 4099, 3, marks=pytest.mark.full),
        pytest.param(4, 64, 600, marks=pytest.mark.full),
    ],
)
def test_chain_speculative_sampling_accepts_the_reference_prefix(batch, vocab, num_draft):
    """``num_accepted`` and the accepted prefix are the reference's at every launch shape; the
    token after the prefix need only follow the same distribution."""
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
    workload = ChainSpeculativeSamplingWorkload(
        manifest_call("ChainSpeculativeSamplingFwdOp", B=batch, N=num_draft, V=vocab)
    )
    op = ChainSpeculativeSamplingFwdOp()
    out, ref = _run(op, *inputs), workload.ref_program(*inputs)
    compare_outputs(out, ref, workload.verification(*inputs))


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


@pytest.mark.smoke
@pytest.mark.parametrize("cls", list(_SMALL_CALLS), ids=lambda cls: cls.__name__)
def test_generated_checks_reject_an_integer_first_input(cls):
    workload_cls, dtype_case, row = _SMALL_CALLS[cls]
    inputs = list(workload_cls(manifest_call(cls.__name__, dtype_case, **row)).gen_inputs())
    inputs[0] = inputs[0].to(torch.int32)
    with pytest.raises(ValueError):
        cls()(*inputs)


@pytest.mark.smoke
@pytest.mark.parametrize("cls", list(_SMALL_CALLS), ids=lambda cls: cls.__name__)
def test_probabilities_off_a_vector_boundary_give_the_aligned_result(cls):
    """Logits or probabilities starting one element past a 16-byte boundary sample as their
    aligned copy does; the draws are seeded."""
    workload_cls, dtype_case, row = _SMALL_CALLS[cls]
    inputs = list(workload_cls(manifest_call(cls.__name__, dtype_case, **row)).gen_inputs())
    want = _run(cls(), *inputs)
    # The float inputs a row is read from: chain speculative sampling's draft and target.
    for i in (0, 2) if cls is ChainSpeculativeSamplingFwdOp else (0,):
        moved = torch.empty(inputs[i].numel() + 1, dtype=inputs[i].dtype, device=run_device())
        inputs[i] = moved[1:].copy_(inputs[i].reshape(-1)).view(inputs[i].shape)
    got = cls()(*inputs)
    if isinstance(want, tuple):
        assert all(map(torch.equal, got, want))
    else:
        assert torch.equal(got, want)
