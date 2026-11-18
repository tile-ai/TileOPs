"""GatedPoolCompressFwdOp against the torch reference in
``workloads/attention/gated_pool_compress.py``."""

import pytest
import torch

from tests.workload_test_base import TestBase
from tileops.attention import GatedPoolCompressFwdOp
from workloads.attention.gated_pool_compress import (
    GatedPoolCompressFwdCall,
    gated_pool_compress_result,
)
from workloads.device import run_device
from workloads.workload_base import manifest_call


class GatedPoolCompressFwdTest(GatedPoolCompressFwdCall, TestBase):
    pass


def _test(dtype, *, dim, ratio, overlap, seq_lens, positions, slots=8):
    call = manifest_call(
        "GatedPoolCompressFwdOp",
        {"T": str(dtype).removeprefix("torch.")},
        T_q=sum(seq_lens),
        NS=slots,
        D=dim,
        ratio=ratio,
        overlap=overlap,
        seq_lens=seq_lens,
        positions=positions,
    )
    return GatedPoolCompressFwdTest(call)


def _check(test):
    op = GatedPoolCompressFwdOp(**test.arguments())
    test.check(op, *test.gen_inputs(), runs=lambda *args: gated_pool_compress_result(op, *args))


@pytest.mark.parametrize(
    "dtype, dim, ratio, overlap, seq_lens, positions",
    [
        # A request pooled from position 0, overlapping windows.
        pytest.param(
            torch.float32, 512, 4, True, [67], [0], marks=pytest.mark.smoke, id="fp32-overlap"
        ),
        # Ragged requests resuming mid-window from their rings, an empty one among them.
        pytest.param(
            torch.bfloat16,
            128,
            4,
            False,
            [9, 0, 21, 3],
            [6, 40, 101, 2],
            marks=pytest.mark.smoke,
            id="bf16-ragged-resume",
        ),
        # A window longer than the register path: the online softmax.
        pytest.param(
            torch.float32, 512, 128, False, [300], [0], marks=pytest.mark.full, id="ratio128"
        ),
        # Decode: one token a request, each closing a window from its ring.
        pytest.param(
            torch.float32,
            128,
            4,
            True,
            [1, 1, 1, 1],
            [7, 11, 31, 3],
            marks=pytest.mark.full,
            id="decode",
        ),
        # A window longer than the registers hold, pooled across the block online, with
        # bfloat16 kv; the second request resumes from its ring.
        pytest.param(
            torch.bfloat16,
            64,
            10,
            True,
            [57, 3],
            [0, 15],
            marks=pytest.mark.full,
            id="bf16-long-window",
        ),
        # Channels off the 4-wide vector, and a ratio off a power of two.
        pytest.param(
            torch.bfloat16, 10, 3, True, [20, 5], [0, 4], marks=pytest.mark.full, id="dim10-ratio3"
        ),
    ],
)
def test_gated_pool_compress_matches_reference(
    dtype: torch.dtype,
    dim: int,
    ratio: int,
    overlap: bool,
    seq_lens: list,
    positions: list,
) -> None:
    _check(
        _test(dtype, dim=dim, ratio=ratio, overlap=overlap, seq_lens=seq_lens, positions=positions)
    )


@pytest.mark.parametrize(
    "ratio, overlap",
    [
        pytest.param(4, True, marks=pytest.mark.smoke, id="short"),
        pytest.param(32, False, marks=pytest.mark.full, id="long"),
    ],
)
def test_gated_pool_compress_finds_each_window_among_many_requests(
    ratio: int, overlap: bool
) -> None:
    """Forty ragged requests, some empty, resuming at scattered positions: a window's request
    is found among more candidates than a warp tests at once."""
    n = 40
    _check(
        _test(
            torch.float32,
            dim=64,
            ratio=ratio,
            overlap=overlap,
            seq_lens=[(3 * i) % (ratio + 3) for i in range(n)],
            positions=[(17 * i) % (5 * ratio) for i in range(n)],
            slots=n,
        )
    )


@pytest.mark.smoke
def test_gated_pool_compress_reads_a_ring_row_its_later_token_writes() -> None:
    """Two tokens a request, the first closing a window that reads the ring row the second
    token's position takes: the window reads the row before the call writes it."""
    _check(
        _test(
            torch.float32,
            dim=128,
            ratio=4,
            overlap=True,
            seq_lens=[2, 2, 2],
            positions=[11, 15, 3],
        )
    )


@pytest.mark.smoke
def test_gated_pool_compress_calls_without_tokens() -> None:
    """No request, and requests without tokens: no entry, and the rings as they were."""
    dim, ratio = 64, 4
    op = GatedPoolCompressFwdOp(ratio, True)
    kv = torch.randn(0, 2 * dim, device=run_device())
    ape = torch.randn(ratio, 2 * dim, device=run_device())
    kv_state, score_state = _ring(dim, ratio, True)
    rings = kv_state.clone(), score_state.clone()
    for n in (0, 2):
        cu = torch.zeros(n + 1, dtype=torch.int32, device=run_device())
        position = torch.full((n,), 5, dtype=torch.int32, device=run_device())
        slots = torch.arange(n, dtype=torch.int32, device=run_device())
        entries, valid = op(kv, kv, ape, kv_state, score_state, cu, position, slots)
        assert entries.shape == (0, dim) and valid.shape == (0,)
    assert torch.equal(kv_state, rings[0]) and torch.equal(score_state, rings[1])


@pytest.mark.smoke
def test_gated_pool_compress_rejects_float16() -> None:
    dim, ratio = 64, 4
    op = GatedPoolCompressFwdOp(ratio)
    kv = torch.randn(4, dim, device=run_device(), dtype=torch.float16)
    kv_state, score_state = _ring(dim, ratio, False)
    cu = torch.tensor([0, 4], dtype=torch.int32, device=run_device())
    zero = torch.zeros(1, dtype=torch.int32, device=run_device())
    with pytest.raises(ValueError, match="dtype is outside"):
        op(kv, kv, kv[:ratio], kv_state, score_state, cu, zero, zero)


def _ring(dim, ratio, overlap):
    c = 2 if overlap else 1
    shape = (2, c * ratio, c * dim)
    return torch.randn(shape, device=run_device()), torch.randn(shape, device=run_device())


def _call(op, kv, score, ape, kv_state, score_state, start, slot=1):
    cu = torch.tensor([0, kv.shape[0]], dtype=torch.int32, device=run_device())
    position = torch.tensor([start], dtype=torch.int32, device=run_device())
    slots = torch.tensor([slot], dtype=torch.int32, device=run_device())
    return op(kv, score, ape, kv_state, score_state, cu, position, slots)


@pytest.mark.parametrize(
    "ratio, overlap, split",
    [
        pytest.param(4, True, 7, marks=pytest.mark.smoke, id="overlap-mid-ring"),
        pytest.param(4, False, 4, marks=pytest.mark.full, id="window-boundary"),
    ],
)
def test_gated_pool_compress_prefill_then_decode_equals_one_shot(
    ratio: int, overlap: bool, split: int
) -> None:
    """A prefill of the first tokens, then one call a token, pools the windows and leaves the
    ring a single prefill of every token does."""
    torch.manual_seed(0)
    dim, length = 128, 41
    c = 2 if overlap else 1
    kv = torch.randn(length, c * dim, device=run_device())
    score = torch.randn(length, c * dim, device=run_device())
    ape = torch.randn(ratio, c * dim, device=run_device())
    op = GatedPoolCompressFwdOp(ratio, overlap)
    kv_state, score_state = _ring(dim, ratio, overlap)
    once = _call(op, kv, score, ape, kv_state, score_state, 0)
    steps_kv, steps_score = _ring(dim, ratio, overlap)
    entries, valid = _call(op, kv[:split], score[:split], ape, steps_kv, steps_score, 0)
    pooled = {p: entries[p] for p in range(split) if valid[p]}
    for p in range(split, length):
        entries, valid = _call(op, kv[p : p + 1], score[p : p + 1], ape, steps_kv, steps_score, p)
        if valid[0]:
            pooled[p] = entries[0]
    closing = once[1].nonzero().squeeze(1).tolist()
    assert sorted(pooled) == closing
    assert torch.equal(torch.stack([pooled[p] for p in closing]), once[0][closing])
    assert torch.equal(steps_kv[1], kv_state[1]) and torch.equal(steps_score[1], score_state[1])


@pytest.mark.smoke
def test_gated_pool_compress_rollback_equals_one_token_decode() -> None:
    """Calls of a token and a draft, each next call starting at the draft and replacing it,
    pool the windows and fill the ring one call a token does."""
    torch.manual_seed(0)
    dim, ratio, start, length = 128, 4, 9, 30
    kv = torch.randn(length, 2 * dim, device=run_device())
    score = torch.randn(length, 2 * dim, device=run_device())
    ape = torch.randn(ratio, 2 * dim, device=run_device())
    op = GatedPoolCompressFwdOp(ratio, True)
    one_kv, one_score = _ring(dim, ratio, True)
    _call(op, kv[:start], score[:start], ape, one_kv, one_score, 0)
    two_kv, two_score = one_kv.clone(), one_score.clone()
    for p in range(start, length):
        draft = torch.randn(1, 2 * dim, device=run_device())
        two = _call(
            op,
            torch.cat([kv[p : p + 1], draft]),
            torch.cat([score[p : p + 1], -draft]),
            ape,
            two_kv,
            two_score,
            p,
        )
        one = _call(op, kv[p : p + 1], score[p : p + 1], ape, one_kv, one_score, p)
        assert torch.equal(two[1][:1], one[1])
        if one[1][0]:
            assert torch.equal(two[0][0], one[0][0])
    # The last call left its draft at position length; every other row agrees.
    kept = [row for row in range(2 * ratio) if row != length % (2 * ratio)]
    assert torch.equal(two_kv[1][kept], one_kv[1][kept])
    assert torch.equal(two_score[1][kept], one_score[1][kept])
