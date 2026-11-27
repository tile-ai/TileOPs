"""Tests for the Native Sparse Attention (NSA) ops: the sparse forward, compression and top-k selection."""

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.attention import NSACompressedVarlenFwdOp, NSATopKVarlenFwdOp, NSAVarlenFwdOp
from tileops.backend import OpNotAvailableError
from tileops.manifest import load_adts, load_manifest, load_workloads
from tileops.manifest._plan import entry_plan
from tileops.manifest._workload import instantiate
from workloads.attention.nsa import (
    NSACompressedFwdWorkload,
    NSAFwdCall,
    NSAFwdWorkload,
    NSATopKWorkload,
)


class NSAFwdTest(NSAFwdWorkload, TestBase):
    pass


class NSAFwdFixture(FixtureBase):
    PARAMS = [
        (
            "batch, heads, c_seq_len, dim, is_causal, scale, block_size, "
            "groups, selected_blocks, dtype, seq_lens",
            [
                # Non-causal, with lengths no block size divides.
                pytest.param(
                    2,
                    16,
                    256,
                    64,
                    False,
                    0.1,
                    32,
                    16,
                    4,
                    torch.float16,
                    [100, 156],
                    marks=pytest.mark.smoke,
                ),
                # bfloat16 over two KV heads, eight slots through a two-stage K/V ring.
                pytest.param(
                    2,
                    32,
                    256,
                    128,
                    True,
                    128**-0.5,
                    32,
                    16,
                    8,
                    torch.bfloat16,
                    [100, 156],
                    marks=pytest.mark.smoke,
                ),
                # A head dim the MMA K step does not tile selects the general kernel.
                pytest.param(
                    2,
                    16,
                    256,
                    72,
                    True,
                    0.1,
                    32,
                    16,
                    4,
                    torch.float16,
                    [100, 156],
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    4,
                    16,
                    8192,
                    64,
                    True,
                    0.1,
                    32,
                    16,
                    1,
                    torch.float16,
                    None,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    2,
                    16,
                    8192,
                    64,
                    True,
                    0.1,
                    32,
                    16,
                    4,
                    torch.float16,
                    None,
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


@NSAFwdFixture
def test_nsa_varlen_op(
    batch: int,
    heads: int,
    c_seq_len: int,
    dim: int,
    is_causal: bool,
    scale: float,
    block_size: int,
    groups: int,
    selected_blocks: int,
    dtype: torch.dtype,
    seq_lens: "list[int] | None",
) -> None:
    assert groups % 16 == 0, "Group size must be a multiple of 16 in NSA"

    test = NSAFwdTest(
        batch,
        heads,
        c_seq_len,
        dim,
        is_causal,
        scale,
        block_size,
        groups,
        selected_blocks,
        dtype,
        seq_lens,
    )
    op = NSAVarlenFwdOp(is_causal=is_causal, sm_scale=scale, block_size=block_size)
    test.check(op, *test.gen_inputs())


# Qwen3.8-Flash-Next full attention: 24 query heads over 2 KV heads, head dim 256, the
# top-512 4-key blocks plus the token's own, scale 1/16. 2051 keys is the most a token's
# selection holds, so a context past it makes the selection a strict subset of the keys.
_WIDE = {"heads": 24, "dim": 256, "scale": 0.0625, "block_size": 4, "groups": 12}


def _wide_test(
    seq_lens: "list[int]",
    kv_lens: "list[int] | None",
    scale: float,
    selected_blocks: int,
    **options,
) -> NSAFwdTest:
    """A causal bfloat16 call of the ``_WIDE`` head shape at *scale*, its q, k and v drawn
    along the head dimension too."""
    return NSAFwdTest(
        len(seq_lens),
        _WIDE["heads"],
        sum(seq_lens),
        _WIDE["dim"],
        True,
        scale,
        _WIDE["block_size"],
        _WIDE["groups"],
        selected_blocks,
        torch.bfloat16,
        seq_lens,
        kv_lens=kv_lens,
        varied_dims=True,
        **options,
    )


class NSAWideGroupFixture(FixtureBase):
    PARAMS = [
        (
            "seq_lens, kv_lens, is_causal, selected_blocks, dtype, shape",
            [
                # Prefill: requests below and above 2051 tokens, one KV split per token.
                pytest.param(
                    [300, 2300], None, True, 513, torch.bfloat16, _WIDE, marks=pytest.mark.smoke
                ),
                # float16, and a non-causal token that sees its whole request.
                pytest.param(
                    [100, 156], None, False, 64, torch.float16, _WIDE, marks=pytest.mark.smoke
                ),
                # Decode and MTP over cached keys below and above 2051: queries are the last
                # keys, and few queries split each selection across CTAs.
                pytest.param(
                    [1, 1, 2],
                    [1500, 2101, 3000],
                    True,
                    513,
                    torch.bfloat16,
                    _WIDE,
                    marks=pytest.mark.full,
                ),
                # Blocks wider than a 64-key step: each step gathers one whole block.
                pytest.param(
                    [300],
                    None,
                    True,
                    4,
                    torch.bfloat16,
                    {**_WIDE, "dim": 128, "block_size": 128, "scale": 128**-0.5},
                    marks=pytest.mark.full,
                ),
                # 4097 requests of 16 tokens: more tokens than a grid's 65535-block y and z
                # axes hold.
                pytest.param(
                    [16] * 4097,
                    None,
                    True,
                    4,
                    torch.bfloat16,
                    {"heads": 1, "dim": 16, "scale": 0.25, "block_size": 4, "groups": 1},
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


@NSAWideGroupFixture
def test_nsa_varlen_wide_group(
    seq_lens: "list[int]",
    kv_lens: "list[int] | None",
    is_causal: bool,
    selected_blocks: int,
    dtype: torch.dtype,
    shape: dict,
) -> None:
    """The gather kernel's group, head dim and block shapes, over the queries' own tokens
    and over separate keys."""
    test = NSAFwdTest(
        len(seq_lens),
        shape["heads"],
        sum(seq_lens),
        shape["dim"],
        is_causal,
        shape["scale"],
        shape["block_size"],
        shape["groups"],
        selected_blocks,
        dtype,
        seq_lens,
        kv_lens=kv_lens,
        varied_dims=True,
    )
    op = NSAVarlenFwdOp(
        is_causal=is_causal, sm_scale=shape["scale"], block_size=shape["block_size"]
    )
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_nsa_varlen_topk_select_selection() -> None:
    """The selection TopKSelectFwdOp makes: each token's top-512 blocks before its own
    plus its own block, one list for every KV head, in no order, padded with the block
    count where fewer than 512 blocks precede the token."""
    test = _wide_test([2100], None, _WIDE["scale"], 513, selection="topk")
    op = NSAVarlenFwdOp(sm_scale=_WIDE["scale"], block_size=_WIDE["block_size"])
    test.check(op, *test.gen_inputs())


@pytest.mark.parametrize(
    "seq_lens, kv_lens, heads, groups, dim, block_size, selected_blocks",
    [
        # One CTA per token: a token whose every slot is padding outputs zeros.
        pytest.param([300], None, 24, 12, 256, 4, 513, marks=pytest.mark.smoke),
        # Split selections over 2**24 cached keys: the padding id times the 256-key block
        # passes the int32 range and must not wrap onto the first block.
        pytest.param([2], [2**24], 1, 1, 16, 256, 2, marks=pytest.mark.full),
    ],
)
def test_nsa_varlen_padding_only_selection(
    seq_lens: "list[int]",
    kv_lens: "list[int] | None",
    heads: int,
    groups: int,
    dim: int,
    block_size: int,
    selected_blocks: int,
) -> None:
    """A selection of padding ids alone gives the token no key: its output is zeros."""
    test = NSAFwdTest(
        len(seq_lens),
        heads,
        sum(seq_lens),
        dim,
        True,
        dim**-0.5,
        block_size,
        groups,
        selected_blocks,
        torch.bfloat16,
        seq_lens,
        kv_lens=kv_lens,
        varied_dims=True,
        selection="padding",
    )
    op = NSAVarlenFwdOp(sm_scale=dim**-0.5, block_size=block_size)
    test.check(op, *test.gen_inputs())


@pytest.mark.parametrize(
    "seq_lens, kv_lens, scale, selected_blocks, q_value, k_value",
    [
        # Every key scores 2**28 / 16 * 256: the splits' log-sum-exps are equal and so
        # large that a sum of their weights added back to one is lost to rounding.
        pytest.param(
            [1, 1, 2],
            [1500, 2101, 3000],
            _WIDE["scale"],
            513,
            4096.0,
            4096.0,
            marks=pytest.mark.smoke,
            id="huge-scores",
        ),
        # Every key scores -3 * 2**126: finite, but past float32 once scaled by log2(e).
        pytest.param(
            [1, 1, 2],
            [1500, 2101, 3000],
            1.0,
            513,
            2.0**59,
            -3 * 2.0**59,
            marks=pytest.mark.full,
            id="negative-scores",
        ),
        # Every score is nearly 0, and a key the selection does not reach still weighs
        # nothing: early tokens see a few keys of a step full of slots.
        pytest.param([100], None, 1e-38, 513, None, None, marks=pytest.mark.full, id="tiny-scale"),
        # A finite scale near float32's largest value: equal (zero) scores weigh alike.
        pytest.param([64], None, 1.2e38, 16, 0.0, 0.0, marks=pytest.mark.full, id="huge-scale"),
    ],
)
def test_nsa_varlen_scores_and_scales_at_the_float32_range(
    seq_lens: "list[int]",
    kv_lens: "list[int] | None",
    scale: float,
    selected_blocks: int,
    q_value: "float | None",
    k_value: "float | None",
) -> None:
    """Splits whose scores are all equal, huge or hugely negative merge to the mean of the
    selected values, and a scale near 0 or float32's largest value weighs equal scores
    alike. ``q_value`` and ``k_value`` fill q and k, or ``None`` keeps the random inputs."""
    test = _wide_test(seq_lens, kv_lens, scale, selected_blocks)
    q, k, *rest = test.gen_inputs()
    if q_value is not None:
        q, k = torch.full_like(q, q_value), torch.full_like(k, k_value)
    op = NSAVarlenFwdOp(sm_scale=scale, block_size=_WIDE["block_size"])
    test.check(op, q, k, *rest)


@pytest.mark.smoke
def test_nsa_varlen_split_merge_opposite_extreme_scores() -> None:
    """Finite scores of opposite sign whose difference passes float32, at a scale that
    brings them back near 1: one split scores -3 * 2**126, the other +3 * 2**126."""
    test = _wide_test([1], [64], 1e-38, 16)
    q, k, v, block_indices, block_counts, *rest = test.gen_inputs()
    q = torch.full_like(q, 2.0**59)
    k = torch.cat([torch.full_like(k[:32], -3 * 2.0**59), torch.full_like(k[32:], 3 * 2.0**59)])
    v = torch.cat([torch.full_like(v[:32], 10.0), torch.zeros_like(v[32:])])
    block_indices = torch.arange(16, dtype=torch.int32, device=q.device).expand_as(block_indices)
    block_counts = torch.full_like(block_counts, 16)
    op = NSAVarlenFwdOp(sm_scale=1e-38, block_size=_WIDE["block_size"])
    test.check(op, q, k, v, block_indices.contiguous(), block_counts, *rest)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_nsa_varlen_scale_past_float32_is_refused() -> None:
    """A finite scale whose float32 root of 2 * scale * log2(e) does not exist reaches no
    kernel: the call is refused, not answered wrong."""
    test = _wide_test([16], None, 1e308, 4)
    op = NSAVarlenFwdOp(sm_scale=1e308, block_size=_WIDE["block_size"])
    with pytest.raises(OpNotAvailableError, match="no normal float32 root"):
        op(*test.gen_inputs())


@pytest.mark.smoke
def test_nsa_varlen_strided_inputs() -> None:
    """Strided views, a key tensor sliced from a wider one and one selection expanded over
    the KV heads, give the result their contiguous copies give: the op copies them before
    any kernel."""
    dim, block_size = 128, 32
    test = NSAFwdTest(
        2, 32, 256, dim, True, dim**-0.5, block_size, 16, 8, torch.bfloat16, [100, 156],
        varied_dims=True,
    )  # fmt: skip
    q, k, v, block_indices, block_counts, cu_seqlens, token_indices = test.gen_inputs()
    block_indices = block_indices[:, :1].expand_as(block_indices)
    k = torch.cat([k, k], -1)[..., :dim]
    assert not (k.is_contiguous() or block_indices.is_contiguous())
    op = NSAVarlenFwdOp(sm_scale=dim**-0.5, block_size=block_size)
    test.check(op, q, k, v, block_indices, block_counts, cu_seqlens, token_indices)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_nsa_varlen_heads_past_grid_y_limit_are_refused() -> None:
    """More heads than a grid's y axis holds reach no kernel: the call is refused."""
    heads = 65536
    test = NSAFwdTest(1, heads, 4, 16, True, 0.25, 4, 1, 1, torch.bfloat16, [4])
    op = NSAVarlenFwdOp(sm_scale=0.25, block_size=4)
    with pytest.raises(OpNotAvailableError, match="65535 blocks"):
        op(*test.gen_inputs())


class NSACompressedFwdTest(NSACompressedFwdWorkload, TestBase):
    pass


class NSACompressedFwdFixture(FixtureBase):
    PARAMS = [
        (
            "seq_num, c_seq_len, heads, dim_k, dim_v, group, scale, bs, dtype",
            [
                pytest.param(
                    9,
                    8192,
                    32,
                    128,
                    128,
                    16,
                    128**-0.5,
                    32,
                    torch.float16,
                    marks=pytest.mark.smoke,
                ),
                # BF16 MMA requires FP32 accumulation.
                pytest.param(
                    1, 65, 16, 64, 64, 16, 0.125, 32, torch.bfloat16, marks=pytest.mark.smoke
                ),
            ],
        ),
    ]


@NSACompressedFwdFixture
def test_nsa_compressed_fwd_varlen_op(
    seq_num: int,
    c_seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    group: int,
    scale: float,
    bs: int,
    dtype: torch.dtype,
) -> None:
    assert group % 16 == 0, "Group size must be a multiple of 16 in NSA"
    test = NSACompressedFwdTest(seq_num, c_seq_len, heads, dim_k, dim_v, group, scale, bs, dtype)
    op = NSACompressedVarlenFwdOp(sm_scale=scale, block_size=bs)
    test.check(op, *test.gen_inputs())


class NSATopKTest(NSATopKWorkload, TestBase):
    pass


class NSATopKFixture(FixtureBase):
    PARAMS = [
        (
            "seq_num, c_seq_len, heads, dim, group, scale, selected_block_num, bs, dtype",
            [
                pytest.param(5, 1024, 32, 128, 16, 1.0, 16, 32, dtype, marks=pytest.mark.smoke)
                for dtype in (torch.float16, torch.bfloat16)
            ]
            + [
                # A 32-head group fills a 64-row tile with two tokens, so half the warps sort.
                pytest.param(
                    4,
                    2048,
                    64,
                    64,
                    32,
                    0.125,
                    8,
                    32,
                    torch.bfloat16,
                    marks=pytest.mark.full,
                ),
                # Two sort pairs per lane must not put barriers under a partial-lane mask.
                pytest.param(
                    1, 65, 16, 64, 16, 0.125, 8, 64, torch.float16, marks=pytest.mark.full
                ),
                # A narrow sort band pads all 64 requested output slots.
                pytest.param(
                    1, 65, 32, 64, 32, 0.125, 64, 64, torch.float16, marks=pytest.mark.full
                ),
                # A 48-head group uses the bounded SM89 layout or the per-token kernel.
                pytest.param(
                    2, 817, 48, 64, 48, 0.125, 16, 32, torch.float16, marks=pytest.mark.full
                ),
            ],
        ),
    ]


@NSATopKFixture
def test_nsa_topk_varlen_op(
    seq_num: int,
    c_seq_len: int,
    heads: int,
    dim: int,
    group: int,
    scale: float,
    selected_block_num: int,
    bs: int,
    dtype: torch.dtype,
) -> None:
    assert group % 16 == 0, "Group size must be a multiple of 16 in NSA"

    test = NSATopKTest(seq_num, c_seq_len, heads, dim, group, scale, selected_block_num, bs, dtype)
    op = NSATopKVarlenFwdOp(sm_scale=scale, selected_block_num=selected_block_num, block_size=bs)
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_nsa_topk_reference_keeps_fp32_dot_products() -> None:
    """Promoting a rounded half matmul is too late to rank close candidates."""
    torch.manual_seed(1235)
    workload = NSATopKWorkload(1, 512, 32, 64, 16, 1.0, 16, 32, torch.float16)
    q, k, *metadata = workload.gen_inputs()
    assert torch.equal(
        workload.ref_program(q, k, *metadata),
        workload.ref_program(q.float(), k.float(), *metadata),
    )


@pytest.mark.smoke
def test_nsa_topk_ranks_unquantized_scores() -> None:
    """Priority blocks survive; the remaining slot takes the highest raw score."""
    workload = NSATopKTest(1, 256, 16, 16, 16, 1.0, 4, 32, torch.float16)
    q, k, *metadata = workload.gen_inputs()
    q = torch.zeros_like(q)
    q[-1].fill_(1)
    k = torch.empty_like(k)
    k.copy_(torch.arange(k.shape[0] - 1, -1, -1, device=k.device)[:, None, None] * 2**-24)
    op = NSATopKVarlenFwdOp(sm_scale=1.0, selected_block_num=4, block_size=32)
    workload.check(op, q, k, *metadata)
    result = op(q, k, *metadata)
    # Sub-1e-5 gaps must still select block 1, alongside priority blocks 0, 6, 7.
    assert torch.equal(
        result[-1, 0].sort().values,
        torch.tensor([0, 1, 6, 7], dtype=torch.int32, device=q.device),
    )


@pytest.mark.smoke
def test_nsa_varlen_reference_returns_the_declared_output() -> None:
    """Benchmarks check every implementation against this reference, so it returns ``o`` as declared."""
    name = "NSAVarlenFwdOp"
    plan = entry_plan(name, load_manifest()[name], load_adts(), resolve=False)
    calls = [
        instantiate(plan, row, case)
        for row in load_workloads(name)
        for case in row.get("dtype_cases") or [{}]
    ]
    call = min(calls, key=lambda c: c.indices["T_q"] * c.indices["H"] * c.indices["D"])
    workload = NSAFwdCall(call)
    output = workload.ref_program(*workload.gen_inputs())
    spec = call.specs["o"]
    assert tuple(output.shape) == tuple(spec.shape) and output.dtype == getattr(torch, spec.dtype)


@pytest.mark.sm89
@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nsa_topk_group48_wide_head(dtype: torch.dtype) -> None:
    """Cover both dtypes of the group48 layout, two KV heads and nontrivial ranking."""
    test = NSATopKTest(2, 258, 96, 256, 48, 0.0625, 4, 64, dtype, seq_lens=[1, 257])
    inputs = test.gen_inputs()
    op = NSATopKVarlenFwdOp(sm_scale=0.0625, selected_block_num=4, block_size=64)
    test.check(op, *inputs)


@pytest.mark.sm89
@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nsa_topk_group48_many_requests(dtype: torch.dtype) -> None:
    """Packed short requests must not allocate sorting space for the whole batch."""
    count = 2049
    test = NSATopKTest(count, count, 48, 256, 48, 0.0625, 1, 64, dtype, seq_lens=[1] * count)
    inputs = test.gen_inputs()
    result = NSATopKVarlenFwdOp(sm_scale=0.0625, selected_block_num=1, block_size=64)(*inputs)
    assert torch.equal(result, torch.zeros_like(result))


@pytest.mark.smoke
def test_nsa_compressed_lse_is_head_major_float32_and_minus_inf_without_a_chunk() -> None:
    """A token before its request's first closed chunk sees no key: zero output and lse
    -inf, the convention every attention op and MergeAttentionStatesFwdOp share."""
    test = NSACompressedFwdTest(1, 65, 16, 64, 64, 16, 0.125, 32, torch.bfloat16)
    inputs = test.gen_inputs()
    o, lse = NSACompressedVarlenFwdOp(sm_scale=0.125, block_size=32)(*inputs)
    assert lse.dtype == torch.float32
    assert lse.shape == (16, 65)
    assert torch.isneginf(lse[:, :31]).all()
    assert torch.isfinite(lse[:, 31:]).all()
    assert (o[:31] == 0).all()


@pytest.mark.smoke
def test_nsa_single_selected_block_respects_changed_request_lengths() -> None:
    """One op serves two calls of different request lengths, one selected block a token and
    every fifth token selecting none."""
    scale = 0.2
    op = NSAVarlenFwdOp(sm_scale=scale, block_size=_WIDE["block_size"])
    for q_lengths, kv_lengths in (([11, 21], [17, 44]), ([3, 29], [24, 37])):
        test = _wide_test(q_lengths, kv_lengths, scale, 1)
        q, k, v, block_indices, block_counts, *rest = test.gen_inputs()
        block_counts[::5] = 0
        test.check(op, q, k, v, block_indices, block_counts, *rest)
