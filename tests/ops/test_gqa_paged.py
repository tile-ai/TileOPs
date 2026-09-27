"""GroupedQueryAttentionPagedFwdOp tests against the workload's reference."""

import pytest
import torch

from tests.test_base import FixtureBase, standard_tolerance
from tileops.ops import GroupedQueryAttentionPagedFwdOp
from workloads.device import run_device
from workloads.gqa import GroupedQueryAttentionPagedFwdWorkload


def _decode(
    batch: int,
    heads: int,
    heads_kv: int,
    cache_lens: list[int],
    dim: int,
    page_size: int,
    dtype: torch.dtype = torch.float16,
    *,
    pool_pages: int | None = None,
    **semantics,
) -> GroupedQueryAttentionPagedFwdWorkload:
    """A decode call: one query token per request, tables as wide as the longest cache."""
    width = -(-max(cache_lens) // page_size)
    return GroupedQueryAttentionPagedFwdWorkload(
        heads,
        heads_kv,
        dim,
        [1] * batch,
        cache_lens,
        page_size,
        width,
        pool_pages or batch * width,
        dtype,
        **semantics,
    )


def _check(op, workload, inputs) -> None:
    torch.testing.assert_close(
        op(*inputs), workload.ref_program(*inputs), **standard_tolerance(workload.dtype)
    )


def _built_kernel(op, inputs):
    """The kernel *op* selects and builds for *inputs*."""
    q, k_pages, _, page_table, _, cu_seqlens_q = inputs[:6]
    call = op.paged_call(q, k_pages, page_table, cu_seqlens_q)
    return op.kernel_for("gqa_paged", inputs, call)


class GroupedQueryAttentionPagedDecodeFixture(FixtureBase):
    PARAMS = [
        (
            "batch, heads, heads_kv, cache_lens, dim, page_size, dtype",
            [
                pytest.param(
                    2, 16, 4, [513, 1000], 128, 128, torch.float16, marks=pytest.mark.smoke
                ),
                pytest.param(
                    2, 16, 4, [513, 1000], 128, 128, torch.bfloat16, marks=pytest.mark.smoke
                ),
                pytest.param(1, 16, 8, [512], 128, 128, torch.float16, marks=pytest.mark.full),
                pytest.param(2, 8, 4, [1024, 700], 64, 256, torch.float16, marks=pytest.mark.full),
                pytest.param(1, 32, 8, [200], 128, 64, torch.float16, marks=pytest.mark.full),
                pytest.param(1, 16, 4, [2048], 128, 512, torch.float16, marks=pytest.mark.full),
                pytest.param(1, 32, 16, [512], 64, 128, torch.float16, marks=pytest.mark.full),
                pytest.param(
                    4, 32, 8, [100, 16, 77, 300], 128, 16, torch.float16, marks=pytest.mark.full
                ),
            ],
        ),
    ]


@GroupedQueryAttentionPagedDecodeFixture
def test_gqa_paged_decode_op(
    batch: int,
    heads: int,
    heads_kv: int,
    cache_lens: list[int],
    dim: int,
    page_size: int,
    dtype: torch.dtype,
) -> None:
    workload = _decode(batch, heads, heads_kv, cache_lens, dim, page_size, dtype)
    _check(GroupedQueryAttentionPagedFwdOp(), workload, workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize(
    "sm_scale, softcap, cache_lens",
    [
        pytest.param(0.25, None, [512, 300], id="custom-sm-scale"),
        pytest.param(None, 2.0, [512, 300], id="softcap"),
        # Unequal lengths on the split path leave the short request with empty splits.
        pytest.param(0.0, None, [4096, 128], id="zero-scale-split"),
    ],
)
def test_gqa_paged_decode_score_controls(
    sm_scale: float | None, softcap: float | None, cache_lens: list[int]
) -> None:
    workload = _decode(2, 16, 8, cache_lens, 128, 128, sm_scale=sm_scale, softcap=softcap)
    op = GroupedQueryAttentionPagedFwdOp(sm_scale=sm_scale, softcap=softcap)
    _check(op, workload, workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("batch", [pytest.param(2, id="generic"), pytest.param(1, id="bs1")])
def test_gqa_paged_decode_pool_is_independent_of_table_width(batch: int) -> None:
    """A pool holding more pages than the tables name reads only the named ones."""
    workload = _decode(batch, 32, 4, [2048] * batch, 128, 256, pool_pages=4 * batch * 8)
    _check(GroupedQueryAttentionPagedFwdOp(), workload, workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_gqa_paged_decode_non_divisible_128_page_split() -> None:
    """page_size=192 uses 64-token tiles without skipping page tails."""
    workload = _decode(1, 16, 4, [3072], 128, 192)
    inputs = list(workload.gen_inputs())
    inputs[3] = inputs[3].flip(-1).contiguous()
    op = GroupedQueryAttentionPagedFwdOp()
    kernel = _built_kernel(op, inputs)
    assert 192 % kernel.config["block_N"] == 0
    assert {config["block_N"] for config in kernel.autotune_configs} == {64}
    _check(op, workload, inputs)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("page_size", [96, 160, 224])
def test_gqa_paged_decode_rejects_unsupported_page_tile(page_size: int) -> None:
    """A page layout no supported key tile covers exactly is refused at selection."""
    workload = _decode(1, 16, 4, [page_size * 16], 128, page_size)
    with pytest.raises(ValueError, match="matches no supported block_N"):
        _built_kernel(GroupedQueryAttentionPagedFwdOp(), workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    ("cache_len", "reverse_pages"),
    [
        pytest.param(512, False, id="no-split"),
        pytest.param(4096, True, id="ctx-reversed-pages"),
    ],
)
def test_gqa_paged_decode_bs1_tiers(cache_len: int, reverse_pages: bool) -> None:
    """Both batch-1 tiers, including a page translation that changes the output."""
    torch.manual_seed(0)
    workload = _decode(1, 32, 4, [cache_len], 128, 256)
    inputs = list(workload.gen_inputs())
    if reverse_pages:
        inputs[3] = inputs[3].flip(-1).contiguous()
    op = GroupedQueryAttentionPagedFwdOp()
    kernel = _built_kernel(op, inputs)
    assert kernel.__class__.__name__ == "GQADecodePagedBs1Kernel"
    assert kernel._select_tier(cache_len) == ("ctx" if cache_len >= 1024 else "no_split")
    _check(op, workload, inputs)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_gqa_paged_decode_bs1_dispatch() -> None:
    """An eligible batch-1 call selects the batch-1 kernel and its tiers."""
    workload = _decode(1, 32, 4, [8192], 128, 256)
    kernel = _built_kernel(GroupedQueryAttentionPagedFwdOp(), workload.gen_inputs())
    assert kernel.__class__.__name__ == "GQADecodePagedBs1Kernel"
    assert kernel._select_tier(1024) == "ctx"
    assert kernel._select_tier(512) == "no_split"
    assert kernel._ctx_splits_for(8192) == 32
    assert kernel._ctx_splits_for(2048) == 16
    assert kernel._ctx_splits_for(3072) == 8


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    ("cu_seqlens_q", "op_kwargs", "reason"),
    [
        pytest.param([0, 0, 2], {}, "one query token per request", id="uneven-requests"),
        pytest.param([0, 2, 4], {}, "one query token per request", id="prefill"),
        pytest.param([0, 1, 2], {"window_size_left": 128}, "sliding windows", id="window"),
    ],
)
def test_gqa_paged_refuses_calls_outside_the_decode_region(
    cu_seqlens_q: list[int], op_kwargs: dict, reason: str
) -> None:
    """Decode is read from every step of cu_seqlens_q, not from the packed total."""
    workload = _decode(2, 8, 2, [512, 512], 64, 64)
    inputs = list(workload.gen_inputs())
    total_q = cu_seqlens_q[-1]
    inputs[0] = torch.randn(total_q, 8, 64, dtype=torch.float16, device=run_device())
    inputs[5] = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=run_device())
    with pytest.raises(ValueError, match=reason):
        GroupedQueryAttentionPagedFwdOp(**op_kwargs)(*inputs)
