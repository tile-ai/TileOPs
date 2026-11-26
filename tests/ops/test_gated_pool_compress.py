"""GatedPoolCompressFwdOp against the torch reference in
``workloads/attention/gated_pool_compress.py``."""

import pytest
import torch

from tests.cuda_graph_contract import check_replay
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
        NS=max(slots, len(seq_lens)),
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
        # One token a request over channels off the 4-wide vector: a lane's channels past the
        # entry's are masked; the first window of a request at position ratio - 1.
        pytest.param(
            torch.bfloat16,
            10,
            3,
            True,
            [1, 1, 1, 1],
            [2, 8, 13, 5],
            marks=pytest.mark.full,
            id="decode-dim10",
        ),
        # At most a token a request, a window over a block: forty requests of eight channel
        # chunks fit one wave of the one-token program's 8-warp blocks on an SM90 part, not of
        # 16-warp blocks. The three-token request pools the window its token at 127 closes on
        # the general request path, from its ring and its tokens, and writes its rows; the
        # others are empty or one token, closing a window and not.
        pytest.param(
            torch.float32,
            512,
            128,
            False,
            [3, 0, 0] + [1] * 37,
            [126, 40, 300] + [128 * i + 127 if i % 4 else 128 * i + 5 for i in range(3, 40)],
            marks=pytest.mark.full,
            id="decode-long",
        ),
        # One token a request, a block's window over two windows' positions: the ring rows
        # offset by half the span, and the first window's previous positions masked.
        pytest.param(
            torch.float32,
            64,
            64,
            True,
            [1, 1, 1, 1],
            [63, 127, 191, 100],
            marks=pytest.mark.full,
            id="decode-long-overlap",
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
        # A window of two batches a warp, which rescales a warp's sums across its batches, and
        # channels off the block's lane width, which masks the last chunk's lanes.
        pytest.param(
            torch.float32,
            96,
            128,
            True,
            [300],
            [0],
            marks=pytest.mark.full,
            id="ratio128-overlap-dim96",
        ),
        # A short window over two windows' positions whose request's ring rows take its row
        # warps two batches; the second request resumes mid-window from its ring.
        pytest.param(
            torch.float32,
            64,
            8,
            True,
            [37, 5],
            [0, 13],
            marks=pytest.mark.full,
            id="ratio8-overlap-two-row-batches",
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


class _StridedRowsTest(GatedPoolCompressFwdTest):
    """``kv`` and ``score`` laid out by ``layout``: ``halves``, the column halves of one
    ``[T_q, 2 * C * D]`` projection, as DeepSeek-V4-Flash's compressor passes them;
    ``padded``, the leading columns of two tensors whose rows are two elements wider, off a
    lane's vector; ``broadcast``, one row each repeated over the tokens by a zero stride;
    ``broadcast-kv``, ``kv`` so repeated and ``score`` the second column half of a
    ``[T_q, 2 * C * D]`` projection, rows of two strides; ``transposed``, the transposes of
    ``[C * D, T_q]`` tensors, whose channels lie ``T_q`` elements apart."""

    def __init__(self, call, layout: str) -> None:
        super().__init__(call)
        self.layout = layout

    def gen_inputs(self) -> tuple:
        kv, score, *rest = super().gen_inputs()
        width = kv.shape[1]
        if self.layout == "halves":
            both = torch.cat([kv, score], dim=1)
            return both[:, :width], both[:, width:], *rest
        if self.layout == "padded":
            rows = []
            for t in (kv, score):
                wide = torch.zeros(t.shape[0], width + 2, dtype=t.dtype, device=t.device)
                wide[:, :width] = t
                rows.append(wide[:, :width])
            return *rows, *rest
        if self.layout == "broadcast":
            return kv[:1].expand_as(kv), score[:1].expand_as(score), *rest
        if self.layout == "transposed":
            return kv.t().contiguous().t(), score.t().contiguous().t(), *rest
        both = torch.cat([kv, score], dim=1)
        return kv[:1].expand_as(kv), both[:, width:], *rest


@pytest.mark.parametrize(
    "dtype, dim, ratio, overlap, seq_lens, positions, layout",
    [
        pytest.param(
            torch.float32,
            512,
            4,
            True,
            [67],
            [0],
            "halves",
            marks=pytest.mark.smoke,
            id="deepseek-ratio-4",
        ),
        pytest.param(
            torch.bfloat16,
            128,
            4,
            False,
            [9, 0, 21, 3],
            [6, 40, 101, 2],
            "halves",
            marks=pytest.mark.smoke,
            id="bf16-ragged-resume",
        ),
        pytest.param(
            torch.float32,
            512,
            128,
            False,
            [300],
            [0],
            "halves",
            marks=pytest.mark.full,
            id="deepseek-ratio-128",
        ),
        pytest.param(
            torch.float32,
            128,
            4,
            True,
            [1, 1, 1, 1],
            [7, 11, 31, 3],
            "halves",
            marks=pytest.mark.full,
            id="decode",
        ),
        pytest.param(
            torch.float32,
            128,
            4,
            True,
            [21],
            [0],
            "padded",
            marks=pytest.mark.full,
            id="rows-off-the-vector",
        ),
        pytest.param(
            torch.float32,
            128,
            4,
            True,
            [67],
            [0],
            "broadcast",
            marks=pytest.mark.full,
            id="broadcast-rows",
        ),
        pytest.param(
            torch.float32,
            128,
            4,
            True,
            [67],
            [0],
            "broadcast-kv",
            marks=pytest.mark.full,
            id="two-row-strides",
        ),
        pytest.param(
            torch.float32,
            128,
            4,
            True,
            [21],
            [0],
            "transposed",
            marks=pytest.mark.full,
            id="transposed-channels",
        ),
    ],
)
def test_gated_pool_compress_reads_strided_rows(
    dtype: torch.dtype,
    dim: int,
    ratio: int,
    overlap: bool,
    seq_lens: list,
    positions: list,
    layout: str,
) -> None:
    """``kv`` and ``score`` whose rows lie apart from their channels' width, read where they
    lie: wider apart, off a lane's vector, or all at one address; and whose channels lie apart,
    read through a contiguous copy."""
    call = _test(
        dtype, dim=dim, ratio=ratio, overlap=overlap, seq_lens=seq_lens, positions=positions
    ).call
    _check(_StridedRowsTest(call, layout))


@pytest.mark.smoke
def test_gated_pool_compress_builds_a_program_for_each_row_stride() -> None:
    """One op called on contiguous rows, then on the column halves of one projection, then on
    contiguous rows again: each call runs the program of its own row strides."""
    call = _test(torch.float32, dim=512, ratio=4, overlap=True, seq_lens=[67], positions=[0]).call
    contiguous = GatedPoolCompressFwdTest(call)
    op = GatedPoolCompressFwdOp(**contiguous.arguments())
    for test in (contiguous, _StridedRowsTest(call, "halves"), contiguous):
        test.check(op, *test.gen_inputs(), runs=lambda *args: gated_pool_compress_result(op, *args))


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
    """The signature check refuses the dtype before any kernel is chosen: CPU tensors suffice."""
    dim, ratio = 64, 4
    op = GatedPoolCompressFwdOp(ratio)
    kv = torch.randn(4, dim, dtype=torch.float16)
    kv_state, score_state = torch.randn(2, 2, ratio, dim)
    cu = torch.tensor([0, 4], dtype=torch.int32)
    zero = torch.zeros(1, dtype=torch.int32)
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


@pytest.mark.parametrize(
    "n",
    [
        pytest.param(4, marks=pytest.mark.smoke, id="16-warp"),
        pytest.param(40, marks=pytest.mark.full, id="8-warp"),
    ],
)
def test_gated_pool_compress_long_window_decode_equals_prefill(n: int) -> None:
    """A window over a block, closed by one token a request after a prefill of its other
    positions, pools exactly what one prefill of them all pools, and leaves the same rings:
    forty requests of eight channel chunks take the 8-warp program on an SM90 part."""
    torch.manual_seed(0)
    dim, ratio = 512, 128
    kv = torch.randn(n, ratio, dim, device=run_device())
    score = torch.randn(n, ratio, dim, device=run_device())
    ape = torch.randn(ratio, dim, device=run_device())
    op = GatedPoolCompressFwdOp(ratio)
    slots = torch.arange(n, dtype=torch.int32, device=run_device())

    def call(start, tokens, kv_state, score_state):
        cu = torch.arange(0, n * tokens + 1, tokens, dtype=torch.int32, device=run_device())
        position = torch.full((n,), start, dtype=torch.int32, device=run_device())
        rows = slice(start, start + tokens)
        return op(
            kv[:, rows].reshape(-1, dim),
            score[:, rows].reshape(-1, dim),
            ape,
            kv_state,
            score_state,
            cu,
            position,
            slots,
        )

    once_kv = torch.randn(n, ratio, dim, device=run_device())
    once_score = torch.randn(n, ratio, dim, device=run_device())
    steps_kv, steps_score = once_kv.clone(), once_score.clone()
    entries, valid = call(0, ratio, once_kv, once_score)
    call(0, ratio - 1, steps_kv, steps_score)
    last = call(ratio - 1, 1, steps_kv, steps_score)
    assert bool(valid[ratio - 1 :: ratio].all()) and bool(last[1].all())
    assert torch.equal(last[0], entries[ratio - 1 :: ratio])
    assert torch.equal(steps_kv, once_kv) and torch.equal(steps_score, once_score)


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "tokens",
    [
        pytest.param(1, marks=pytest.mark.smoke, id="one-token"),
        pytest.param(3, marks=pytest.mark.full, id="request"),
    ],
)
def test_gated_pool_compress_flushes_subnormal_score_rows(tokens: int) -> None:
    """A float32 score below the normal range with zero ape leaves its ring rows' score + ape
    at zero, as the in-tree kernel's fast math flushes subnormals: a request's only token and
    a longer request's tokens alike. The reference keeps them, so another target need not."""
    dim, ratio, start = 128, 4, 11
    kv = torch.randn(tokens, 2 * dim, device=run_device())
    score = torch.full((tokens, 2 * dim), 1e-39, device=run_device())
    ape = torch.zeros(ratio, 2 * dim, device=run_device())
    op = GatedPoolCompressFwdOp(ratio, True)
    kv_state, score_state = _ring(dim, ratio, True)
    _call(op, kv, score, ape, kv_state, score_state, start)
    rows = [p % (2 * ratio) for p in range(start, start + tokens)]
    assert torch.equal(score_state[1, rows], torch.zeros(tokens, 2 * dim, device=run_device()))


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


@pytest.mark.sm90
@pytest.mark.cuda_graph(op="GatedPoolCompressFwdOp")
@pytest.mark.parametrize(
    "dim, ratio, overlap",
    [
        pytest.param(512, 4, True, marks=pytest.mark.smoke, id="deepseek-v4-flash-ratio-4"),
        pytest.param(128, 4, True, marks=pytest.mark.full, id="deepseek-v4-flash-indexer-ratio-4"),
        pytest.param(512, 128, False, marks=pytest.mark.full, id="deepseek-v4-flash-ratio-128"),
    ],
)
def test_gated_pool_compress_graph_replay(dim: int, ratio: int, overlap: bool) -> None:
    """DeepSeek-V4-Flash's decode pooling of a token's float32 kv and score into its ring, at a
    position that closes a window. The replay holds the packing and the ring slot, which a
    request of one token cannot change."""
    test = _test(
        torch.float32, dim=dim, ratio=ratio, overlap=overlap, seq_lens=[1], positions=[1], slots=1
    )
    inputs = test.gen_inputs()
    inputs[6].fill_(4096 - 1)
    op = GatedPoolCompressFwdOp(**test.arguments())
    check_replay(test, op, inputs, held=(5, 7), result=gated_pool_compress_result)
