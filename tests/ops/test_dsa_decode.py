import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops._kernels.attention import DSADecodeBasicKernel, DSADecodeKernel, DSADecodeWSKernel
from tileops._utils import get_sm_version
from tileops.attention import DSADenseFwdOp
from tileops.backend import OpNotAvailableError
from tileops.backend import _registry as registry
from tileops.backend._registry import register_kernel_type
from workloads.attention.dsa import DSADecodeWorkload
from workloads.device import run_device


class DSADecodeTest(DSADecodeWorkload, TestBase):
    pass


class DSADecodeFixture(FixtureBase):
    PARAMS = [
        (
            "batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv, "
            "q_start_index_s, sm_scale, dtype, tune",
            [
                pytest.param(
                    1,
                    128,
                    1024,
                    2048,
                    512,
                    64,
                    2048,
                    1,
                    1,
                    1024,
                    None,
                    torch.float16,
                    False,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    1,
                    128,
                    1024,
                    2048,
                    512,
                    64,
                    2048,
                    1,
                    1,
                    1024,
                    None,
                    torch.float16,
                    True,
                    marks=pytest.mark.full,
                    id="full-fp16-tuned",
                ),
                # Every causal query sees topk keys, so no row is scanned for dead slots. More
                # rows than SMs: each CTA runs one row after another.
                pytest.param(
                    1,
                    64,
                    300,
                    1024,
                    512,
                    0,
                    192,
                    1,
                    1,
                    1024,
                    None,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="rows-past-sms-see-topk",
                ),
                # Rows that fill fewer than half the SMs: a CTA pair shares each row's keys.
                pytest.param(
                    2,
                    128,
                    3,
                    1024,
                    512,
                    0,
                    512,
                    1,
                    1,
                    1024,
                    None,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="few-rows-see-topk",
                ),
            ],
        ),
    ]


@DSADecodeFixture
def test_dsa_decode_decode(
    batch: int,
    heads: int,
    seq_len_q: int,
    seq_len_kv: int,
    dim: int,
    dim_tail: int,
    topk: int,
    stride_kv: int,
    heads_kv: int,
    q_start_index_s: int,
    sm_scale: float,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = DSADecodeTest(
        batch,
        heads,
        seq_len_q,
        seq_len_kv,
        dim,
        dim_tail,
        topk,
        stride_kv,
        heads_kv,
        q_start_index_s,
        sm_scale=sm_scale,
        dtype=dtype,
    )
    op = DSADenseFwdOp(dim_tail, stride_kv, q_start_index_s, sm_scale=sm_scale)
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())


@pytest.mark.parametrize(
    ("dim_tail", "dtype"),
    [
        pytest.param(0, torch.bfloat16, marks=pytest.mark.smoke, id="bf16-no-tail"),
        pytest.param(0, torch.float16, marks=pytest.mark.smoke, id="fp16-no-tail"),
        pytest.param(64, torch.bfloat16, marks=pytest.mark.full, id="bf16-tail"),
    ],
)
def test_dsa_decode_decode_tail_and_dtype(dim_tail, dtype) -> None:
    """BF16 preserves the output dtype; a zero tail omits the extra QK contraction."""
    test = DSADecodeTest(1, 64, 7, 256, 512, dim_tail, 128, 1, 1, 256, dtype=dtype)
    op = DSADenseFwdOp(dim_tail, 1, 256)
    test.check(op, *test.gen_inputs())


@pytest.mark.sm89
@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_dsa_decode_refuses_what_99_kb_cannot_hold() -> None:
    """SM89 has no WGMMA and gives a block 99 KB of opt-in shared memory, too little for a
    head dim 2048 query tile, so the op refuses it before anything is built."""
    test = DSADecodeTest(1, 16, 1, 2048, 2048, 64, 128, 1, 1, 2048)
    op = DSADenseFwdOp(64, 1, 2048)
    with pytest.raises(ValueError, match="bytes of shared memory per block"):
        op(*test.gen_inputs())
    for interface in DSADenseFwdOp.interfaces:
        assert not op._entries_built(interface)


def _padded_topk_indices(
    batch: int,
    seq_len: int,
    heads_kv: int,
    topk: int,
    seq_len_kv: int,
    pad: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """A top-k list that fills half its slots and pads the rest with *pad*."""
    indices = torch.full(
        (batch, seq_len, heads_kv, topk), pad, dtype=torch.int32, device=run_device()
    )
    for b in range(batch):
        for t in range(seq_len):
            for h in range(heads_kv):
                selected = torch.randperm(seq_len_kv, generator=generator)[: topk // 2]
                indices[b, t, h, : topk // 2] = selected.to(torch.int32).to(run_device())
    return indices


@pytest.mark.smoke
def test_dsa_decode_decode_ignores_padded_topk_slots() -> None:
    """A slot no row fills must not reach kv, whatever value pads it.

    The cache here is shorter than the causal window, so the causal limit alone
    accepts a padding index; the kernel has to reject it on the kv extent.
    """
    batch, heads, seq_len, seq_len_kv = 1, 128, 64, 1024
    dim, dim_tail, topk, stride_kv, heads_kv, q_start = 512, 64, 128, 1, 1, 1024
    generator = torch.Generator().manual_seed(0)

    test = DSADecodeTest(
        batch, heads, seq_len, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv, q_start
    )
    op = DSADenseFwdOp(dim_tail, stride_kv, q_start)
    q, kv, _ = test.gen_inputs()

    # seq_len_kv is the padding the workloads write and the reference reads.
    in_range_pad = _padded_topk_indices(
        batch, seq_len, heads_kv, topk, seq_len_kv, seq_len_kv, generator
    )
    # The reference sums in float32 and rounds once; one fp16 ulp here is 1e-3.
    test.check(op, q, kv, in_range_pad)

    expected = op(q, kv, in_range_pad)
    assert torch.isfinite(expected).all(), "an in-range padding slot produced a non-finite output"
    for pad in (-1, 2**30):
        padded = in_range_pad.clone()
        padded[padded == seq_len_kv] = pad
        assert torch.equal(op(q, kv, padded), expected), f"padding with {pad} changed the output"

    # Masked rows carry no state from one launch into the next: a launch that stages
    # NaN rows in every slot must not change the output.
    op(q, torch.full_like(kv, float("nan")), torch.zeros_like(in_range_pad))
    assert torch.equal(op(q, kv, in_range_pad), expected)


@pytest.mark.sm90
@pytest.mark.parametrize(
    ("dim", "dim_tail", "heads", "heads_kv", "stride_kv", "dtype"),
    [
        pytest.param(512, 0, 128, 2, 1, torch.float16, marks=pytest.mark.smoke, id="512-kv-groups"),
        pytest.param(
            512, 64, 64, 1, 2, torch.bfloat16, marks=pytest.mark.smoke, id="512-kv-stride"
        ),
        pytest.param(256, 64, 64, 1, 1, torch.bfloat16, marks=pytest.mark.full, id="256"),
    ],
)
def test_dsa_decode_ignores_keys_past_the_causal_limit(
    dim, dim_tail, heads, heads_kv, stride_kv, dtype
) -> None:
    """A selected key past the causal limit, which stride_kv scales, carries no weight.

    Each row selects distinct keys from the whole cache, so most of them lie past the limit.
    """
    seq_len, seq_len_kv, topk, q_start = 33, 1024, 256, 256
    test = DSADecodeTest(
        1, heads, seq_len, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv, q_start,
        dtype=dtype,
    )  # fmt: skip
    op = DSADenseFwdOp(dim_tail, stride_kv, q_start)
    q, kv, _ = test.gen_inputs()
    generator = torch.Generator().manual_seed(0)
    rows = [
        torch.randperm(seq_len_kv, generator=generator)[:topk] for _ in range(seq_len * heads_kv)
    ]
    indices = torch.stack(rows).view(1, seq_len, heads_kv, topk).to(torch.int32).to(q.device)
    test.check(op, q, kv, indices)


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "kernel", [DSADecodeWSKernel, DSADecodeKernel, DSADecodeBasicKernel], ids=lambda k: k.__name__
)
@pytest.mark.parametrize(
    "row",
    [
        pytest.param("no-valid-index", marks=pytest.mark.smoke),
        pytest.param("very-negative-score", marks=pytest.mark.full),
    ],
)
def test_dsa_decode_row_normalization(kernel, row) -> None:
    """Every kernel zeroes a row with no valid key and normalizes a very negative score."""
    if get_sm_version() not in kernel.supported_archs:
        pytest.skip(f"{kernel.__name__} does not run on this device")

    class Pinned(kernel):
        general = False
        preferred_over = frozenset(DSADenseFwdOp.kernel_types)

    seq_len, seq_len_kv, q_start = 8, 512, 512
    # The one valid score of each row is 576 * 4096 * -4096, about -9.7e9.
    sm_scale = 1e-10 if row == "very-negative-score" else None
    state = registry.snapshot()
    try:
        register_kernel_type("DSADenseFwdOp", "dsa_decode_pinned", Pinned)
        op = DSADenseFwdOp(64, 1, q_start, sm_scale=sm_scale)
    finally:
        registry.restore(state)
    test = DSADecodeTest(1, 64, seq_len, seq_len_kv, 512, 64, 128, 1, 1, q_start, sm_scale=sm_scale)
    q, kv, indices = test.gen_inputs()
    if row == "no-valid-index":
        indices[:, ::2] = seq_len_kv
    else:
        q.fill_(4096)
        kv[:, 0] = -4096
        indices.fill_(seq_len_kv)
        indices[..., 0] = 0
    test.check(op, q, kv, indices)
    assert isinstance(next(op.iter_kernels()), Pinned)
    if row == "no-valid-index":
        assert op(q, kv, indices)[:, ::2].eq(0).all()


@pytest.mark.sm90
@pytest.mark.parametrize(
    ("batch", "seq_len", "seq_len_kv", "topk", "with_sink"),
    [
        # The bf16-no-tail shape, index-only with a sink over whole pairs.
        pytest.param(1, 7, 256, 128, True, marks=pytest.mark.smoke, id="sink-whole-pairs"),
        pytest.param(1, 9, 1000, 160, True, marks=pytest.mark.full, id="odd-block-past-topk"),
        pytest.param(1, 9, 1000, 200, True, marks=pytest.mark.full, id="both-blocks-ragged"),
        pytest.param(4, 2, 1000, 640, False, marks=pytest.mark.full, id="mtp-rows"),
    ],
)
def test_dsa_decode_index_only(batch, seq_len, seq_len_kv, topk, with_sink) -> None:
    """With is_causal false only the indices select keys: -1 skips a slot, an index past
    the query's position still counts, and a sink of +inf zeroes its head while -inf leaves
    it as without a sink."""
    heads = 64
    test = DSADecodeTest(
        batch, heads, seq_len, seq_len_kv, 512, 0, topk, 1, 1, 0, is_causal=False,
        dtype=torch.bfloat16,
    )  # fmt: skip
    q, kv, _ = test.gen_inputs()
    generator = torch.Generator().manual_seed(0)
    rows = [torch.randperm(seq_len_kv, generator=generator)[:topk] for _ in range(batch * seq_len)]
    indices = torch.stack(rows).view(batch, seq_len, 1, topk).to(torch.int32)
    indices[torch.rand(indices.shape, generator=generator) < 0.3] = -1
    inputs = [q, kv, indices.to(q.device)]
    if with_sink:
        sink = torch.randn(heads, generator=generator)
        sink[0], sink[1] = float("inf"), float("-inf")
        inputs.append(sink.to(q.device))
    test.check(DSADenseFwdOp(0, 1, 0, is_causal=False), *inputs)


@pytest.mark.sm90
@pytest.mark.parametrize(
    ("batch", "seq_len", "seq_len_kv", "heads", "topk", "is_causal"),
    [
        # The bf16-no-tail shape: a CTA a row.
        pytest.param(1, 7, 256, 64, 128, False, marks=pytest.mark.smoke, id="bf16-no-tail-shape"),
        # More rows than SMs: each CTA runs one row after another.
        pytest.param(1, 300, 1024, 64, 200, False, marks=pytest.mark.full, id="rows-past-sms"),
        # Rows that fill fewer than half the SMs: a CTA pair shares each row's keys.
        pytest.param(4, 2, 1024, 64, 640, False, marks=pytest.mark.full, id="few-rows"),
        pytest.param(2, 3, 1024, 128, 512, True, marks=pytest.mark.full, id="few-rows-causal"),
        # More rows than SMs with an odd block count: a row of one block leaves consumer 1
        # no odd block to score. A topk that is not a multiple of 4 scans one slot a thread.
        pytest.param(1, 300, 1024, 64, 190, False, marks=pytest.mark.full, id="rows-past-sms-odd"),
        pytest.param(
            1, 300, 1024, 64, 192, True, marks=pytest.mark.full, id="rows-past-sms-causal"
        ),
        # Between half and all of the SMs: a CTA a row.
        pytest.param(1, 100, 1024, 64, 640, False, marks=pytest.mark.full, id="row-per-cta"),
    ],
)
def test_dsa_decode_skips_dead_tails(batch, seq_len, seq_len_kv, heads, topk, is_causal) -> None:
    """Rows end in dead slots of every length, a row with none live included: the blocks
    past a row's last live slot weigh nothing, however the rows are spread over the SMs."""
    # Every causal row sees fewer than topk keys.
    q_start = topk // 2 if is_causal else 0
    test = DSADecodeTest(
        batch, heads, seq_len, seq_len_kv, 512, 0, topk, 1, 1, q_start, is_causal=is_causal,
        dtype=torch.bfloat16,
    )  # fmt: skip
    q, kv, _ = test.gen_inputs()
    generator = torch.Generator().manual_seed(0)
    rows = batch * seq_len
    # A causal row pads with seq_len_kv, an index-only one with -1.
    indices = torch.full((rows, topk), seq_len_kv if is_causal else -1, dtype=torch.int32)
    for r, live in enumerate(torch.randint(0, topk + 1, (rows,), generator=generator)):
        live = 0 if r == 0 else int(live)
        indices[r, :live] = torch.randperm(seq_len_kv, generator=generator)[:live]
    inputs = [q, kv, indices.view(batch, seq_len, 1, topk).to(q.device)]
    if not is_causal:
        sink = torch.randn(heads, generator=generator)
        sink[0], sink[1] = float("inf"), float("-inf")
        inputs.append(sink.to(q.device))
    test.check(DSADenseFwdOp(0, 1, q_start, is_causal=is_causal), *inputs)


@pytest.mark.sm90
@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_dsa_decode_refuses_index_only_selection_off_dim_512() -> None:
    """Only the SM90 kernel for value dim 512 serves is_causal false; the others need the
    causal mask, so the op refuses a dim-256 call before anything is built."""
    test = DSADecodeTest(1, 64, 1, 256, 256, 64, 128, 1, 1, 0, is_causal=False)
    op = DSADenseFwdOp(64, 1, 0, is_causal=False)
    with pytest.raises(OpNotAvailableError, match="requires dim 512.*requires the causal mask"):
        op(*test.gen_inputs())
    for interface in DSADenseFwdOp.interfaces:
        assert not op._entries_built(interface)
