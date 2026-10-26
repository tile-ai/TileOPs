"""GQAPagedFwdOp tests against the workload's reference."""

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.attention import GQAPagedFwdOp, PagedKVCacheWriteFwdOp
from tileops.backend import _registry as registry
from tileops.backend._registry import register_kernel_type
from workloads.attention.gqa.paged import GQAPagedFwdWorkload
from workloads.attention.paged_cache_write import paged_cache_write_result
from workloads.device import run_device


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
    q_len: int = 1,
    **semantics,
) -> GQAPagedFwdWorkload:
    """A decode call: ``q_len`` query tokens per request, tables as wide as the longest cache."""
    width = -(-max(cache_lens) // page_size)
    return GQAPagedFwdWorkload(
        heads,
        heads_kv,
        dim,
        [q_len] * batch,
        cache_lens,
        page_size,
        width,
        pool_pages or batch * width,
        dtype,
        **semantics,
    )


def _check(op, workload, inputs) -> None:
    TestBase.check(workload, op, *inputs)


def _kernel(op: GQAPagedFwdOp, inputs: tuple) -> object:
    """The in-tree kernel *op* selects for a call on *inputs*."""
    q, k_cache, _, block_table, _, _, _, k_scale, _, rope_cos, _, sinks = inputs
    call = op._paged_call(q, k_cache, block_table, k_scale, rope_cos, sinks)
    return op.kernel_for("gqa_paged", lambda: call)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("num_split", [1, 4], ids=["unsplit", "split"])
@pytest.mark.parametrize("pos_encoding_mode", ["none", "rope"])
def test_gqa_paged_fixed_split(num_split: int, pos_encoding_mode: str) -> None:
    """One split or several preserve masks and the split LSE merge."""
    from tileops._kernels.attention.gqa.paged import GQAPagedFwdKernel

    class FixedSplitKernel(GQAPagedFwdKernel):
        @property
        def default_config(self):
            return {**super().default_config, "num_split": num_split}

    semantics = dict(pos_encoding_mode=pos_encoding_mode)
    workload = GQAPagedFwdWorkload(
        16, 4, 128, [3, 1], [257, 33], 64, 5, 10, torch.float16, **semantics
    )

    class PreferredFixedSplitKernel(FixedSplitKernel):
        preferred_over = frozenset(GQAPagedFwdOp.kernel_types)

    state = registry.snapshot()
    try:
        register_kernel_type("GQAPagedFwdOp", "gqa_paged_fixed_split", PreferredFixedSplitKernel)
        op = GQAPagedFwdOp(**{"is_causal": True, **semantics})
    finally:
        registry.restore(state)
    _check(op, workload, workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "q_len,cache_len", [(1, 1025), (4, 4097)], ids=["single-token", "causal-split"]
)
def test_gqa_paged_decode_kernel(q_len: int, cache_len: int) -> None:
    """The decode kernel keeps masked keys out of a single-token and a causal call."""
    from tileops._kernels.attention.gqa.paged_decode import GQADecodePagedKernel

    workload = _decode(1, 16, 4, [cache_len], 128, 128, q_len=q_len)
    # Keep single-token dispatch natural to guard the batch-1 specialization's refusal; a
    # longer query is held to the decode kernel, registered above every in-tree key.
    state = registry.snapshot()
    try:
        if q_len > 1:

            class PreferredDecodePagedKernel(GQADecodePagedKernel):
                general = False
                preferred_over = frozenset(GQAPagedFwdOp.kernel_types)

            register_kernel_type(
                "GQAPagedFwdOp", "gqa_decode_paged_any_query", PreferredDecodePagedKernel
            )
        op = GQAPagedFwdOp(is_causal=True)
    finally:
        registry.restore(state)
    _check(op, workload, workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dtype,q_lens,cache_lens,page_size,dim,semantics",
    [
        pytest.param(torch.float16, [1, 3], [129, 63], 64, 128, {}, id="fp16-split"),
        pytest.param(torch.bfloat16, [1, 3], [129, 63], 64, 128, {}, id="bf16-split"),
        pytest.param(
            torch.float16,
            [160, 129],
            [256, 193],
            256,
            64,
            {"rope_layout": "interleaved"},
            id="interleaved-unsplit",
        ),
        pytest.param(
            torch.float16,
            [7, 0, 130],
            [99, 0, 145],
            48,
            128,
            {"rotary_dim": 48, "sm_scale": 0.25},
            id="partial-odd-pages",
        ),
        pytest.param(
            torch.float16,
            [2, 1],
            [65, 97],
            16,
            256,
            {"rotary_dim": 64, "rope_layout": "interleaved", "softcap": 2.0},
            id="partial-interleaved-softcap",
        ),
        pytest.param(
            torch.float16,
            [65, 3],
            [129, 33],
            64,
            128,
            {"is_causal": False, "window_size_left": 17, "window_size_right": 3},
            id="two-sided-window",
        ),
        pytest.param(
            torch.float16,
            [3, 1],
            [65, 33],
            64,
            128,
            {"is_causal": False},
            id="noncausal",
        ),
    ],
)
def test_gqa_paged_rope_reads_logical_positions(
    dtype: torch.dtype,
    q_lens: list[int],
    cache_lens: list[int],
    page_size: int,
    dim: int,
    semantics: dict,
) -> None:
    """Rotate logical positions through fragmented pages without modifying the cache.

    Poisoned cache tails must not enter either the rotary table lookup or the value sum.
    """
    semantics = {"pos_encoding_mode": "rope", **semantics}
    width = -(-max(cache_lens) // page_size)
    workload = GQAPagedFwdWorkload(
        16,
        4,
        dim,
        q_lens,
        cache_lens,
        page_size,
        width,
        len(q_lens) * width,
        dtype,
        **semantics,
    )
    inputs = workload.gen_inputs()
    for request, length in enumerate(cache_lens):
        slots = torch.arange(length, width * page_size, device=inputs[0].device)
        stale = (inputs[3][request, slots // page_size], slots % page_size)
        inputs[1][stale] = float("nan")
        inputs[2][stale] = float("inf")
    _check(GQAPagedFwdOp(**{"is_causal": True, **semantics}), workload, inputs)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "cache_dtype, width, served",
    [(torch.float16, 528, True), (torch.float8_e4m3fn, 528, False), (torch.float16, 768, False)],
    ids=["16-bit-cache", "fp8-cache", "16-bit-at-budget"],
)
def test_gqa_paged_rope_scratch_counts_written_channels(
    cache_dtype: torch.dtype, width: int, served: bool
) -> None:
    """RoPE preparation over a 16-bit cache writes only the rotated key channels, so eight
    33792-key requests at dimension 256 with 64 rotated channels fit the warp-specialized
    scratch budget; an FP8 cache, whose keys and values are written whole, does not. At 768
    pages a request, the query and key channels fill the budget exactly and the values'
    one-row placeholder exceeds it."""
    from tileops._kernels.attention.gqa.paged_ws import GQAPagedFwdWSKernel

    meta = torch.device("meta")
    q = torch.empty(8192, 32, 256, dtype=torch.float16, device=meta)
    k_pages = torch.empty(8 * width, 64, 8, 256, dtype=cache_dtype, device=meta)
    table = torch.empty(8, width, dtype=torch.int32, device=meta)
    rope_cos = torch.empty(width * 64, 32, dtype=torch.float16, device=meta)
    k_scale = torch.empty(1, device=meta) if cache_dtype == torch.float8_e4m3fn else None
    op = GQAPagedFwdOp(is_causal=True, pos_encoding_mode="rope", rotary_dim=64)
    refusal = GQAPagedFwdWSKernel.refusal(op._paged_call(q, k_pages, table, k_scale, rope_cos))
    assert (refusal is None) == served, refusal


@pytest.mark.smoke
@pytest.mark.cuda_only
def test_gqa_paged_rope_replays_positions_from_device_lengths() -> None:
    """Replay must rotate at the current cache positions and current packed Q boundaries."""
    workload = GQAPagedFwdWorkload(
        16,
        4,
        128,
        [1, 3],
        [129, 65],
        64,
        3,
        6,
        torch.float16,
        pos_encoding_mode="rope",
        rotary_dim=64,
    )
    inputs = workload.gen_inputs()
    op = GQAPagedFwdOp(is_causal=True, pos_encoding_mode="rope", rotary_dim=64)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = op(*inputs)
    inputs[4].copy_(torch.tensor([33, 63], device=inputs[0].device, dtype=torch.int32))
    inputs[5][1] = 2
    workload.q_lens, workload.cache_lens = [2, 2], [33, 63]
    graph.replay()
    TestBase.check(workload, op, *inputs, runs=lambda *args: output)


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.in_tree_kernels
def test_gqa_paged_rope_prefill_replays_rotated_key_channels() -> None:
    """A replayed warp-specialized RoPE prefill rotates its 64 leading key channels at the
    current device lengths and query offsets, reads the rest of each key and the values from
    the poisoned cache, and drops the stale rows past each cache end."""
    from tileops._kernels.attention.gqa.paged_ws import GQAPagedFwdWSKernel

    semantics = {"pos_encoding_mode": "rope", "rotary_dim": 64}
    workload = _paged([64, 40], [300, 200], 64, torch.float16, heads=16, heads_kv=4, **semantics)
    inputs = _poisoned_inputs(workload)
    op = GQAPagedFwdOp(**{"is_causal": True, **semantics})
    assert isinstance(_kernel(op, inputs), GQAPagedFwdWSKernel)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = op(*inputs)
    inputs[4].copy_(torch.tensor([250, 180], device=inputs[0].device, dtype=torch.int32))
    inputs[5][1] = 50
    workload.q_lens, workload.cache_lens = [50, 54], [250, 180]
    # The rows past the shorter caches turn stale: poison them too.
    for request, length in enumerate(workload.cache_lens):
        slots = torch.arange(length, 300, device=inputs[0].device)
        stale = (inputs[3][request, slots // 64], slots % 64)
        inputs[1][stale] = float("nan")
        inputs[2][stale] = float("nan")
    graph.replay()
    TestBase.check(workload, op, *inputs, runs=lambda *args: output)


class GQAPagedDecodeFixture(FixtureBase):
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
                # The default tile at head dim 512 overflows every GPU's shared memory.
                pytest.param(
                    2, 16, 4, [513, 1000], 512, 128, torch.float16, marks=pytest.mark.smoke
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


@GQAPagedDecodeFixture
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
    _check(GQAPagedFwdOp(is_causal=True), workload, workload.gen_inputs())


@pytest.mark.parametrize(
    "sm_scale, softcap, cache_lens, q_len",
    [
        pytest.param(0.25, None, [512, 300], 1, marks=pytest.mark.smoke, id="custom-sm-scale"),
        pytest.param(None, 2.0, [512, 300], 1, marks=pytest.mark.smoke, id="softcap"),
        # Unequal lengths on the split path leave the short request with empty splits.
        pytest.param(None, None, [4096, 128], 1, marks=pytest.mark.smoke, id="unequal-split"),
        pytest.param(None, None, [700, 300], 4, marks=pytest.mark.smoke, id="causal"),
    ],
)
def test_gqa_paged_decode_score_controls(
    sm_scale: float | None, softcap: float | None, cache_lens: list[int], q_len: int
) -> None:
    """Score controls preserve masking."""
    workload = _decode(
        len(cache_lens),
        16,
        8,
        cache_lens,
        128,
        128,
        q_len=q_len,
        sm_scale=sm_scale,
        softcap=softcap,
    )
    op = GQAPagedFwdOp(is_causal=True, sm_scale=sm_scale, softcap=softcap)
    _check(op, workload, workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("batch", [pytest.param(2, id="generic"), pytest.param(1, id="bs1")])
def test_gqa_paged_decode_pool_is_independent_of_table_width(batch: int) -> None:
    """A pool holding more pages than the tables name reads only the named ones."""
    workload = _decode(batch, 32, 4, [2048] * batch, 128, 256, pool_pages=4 * batch * 8)
    _check(GQAPagedFwdOp(is_causal=True), workload, workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("page_size", [192, 96, 48])
def test_gqa_paged_reversed_page_table(page_size: int) -> None:
    """A page table that permutes the pool is read through, for pages of any length."""
    workload = _decode(1, 16, 4, [page_size * 16], 128, page_size)
    inputs = list(workload.gen_inputs())
    inputs[3] = inputs[3].flip(-1).contiguous()
    _check(GQAPagedFwdOp(is_causal=True), workload, inputs)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "cache_lens",
    [pytest.param([700, 130], id="no-split"), pytest.param([2048, 1500], id="split")],
)
def test_gqa_paged_multi_token_causal(cache_lens: list[int]) -> None:
    """Several query tokens per request, with a head group's rows spanning two row blocks."""
    workload = _decode(2, 32, 2, cache_lens, 64, 64, q_len=5)
    _check(GQAPagedFwdOp(is_causal=True), workload, workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("cache_lens", "page_size", "q_len"),
    [
        # Pages of 64 hold copied key tiles of 64 rows, and pages of 256 tiles of 128. Five
        # query tokens a request leave the device short of row tiles, so the key range is
        # split; 160 fill it, so it is not, and on SM90 they fill warp-specialized query
        # tiles whose key tile is two 64-row pages or part of a 256-row one.
        pytest.param([63, 130], 64, 5, id="tile-64-split"),
        pytest.param([2047, 1501], 256, 5, id="tile-128-split"),
        pytest.param([700, 450], 64, 160, id="tile-64"),
        pytest.param([700, 450], 256, 160, id="tile-128"),
    ],
)
def test_gqa_paged_drops_stale_rows_past_the_cache(
    cache_lens: list[int], page_size: int, q_len: int
) -> None:
    """A page keeps stale rows past a request's cache; a NaN among them must not reach the
    output through the key tile the cache end cuts."""
    workload = _decode(2, 32, 8, cache_lens, 128, page_size, q_len=q_len)
    inputs = workload.gen_inputs()
    k_cache, v_cache, block_table = inputs[1], inputs[2], inputs[3]
    for request, length in enumerate(cache_lens):
        slots = torch.arange(length, block_table.shape[1] * page_size, device=v_cache.device)
        stale = (block_table[request, slots // page_size], slots % page_size)
        k_cache[stale] = float("nan")
        v_cache[stale] = float("nan")
    _check(GQAPagedFwdOp(is_causal=True), workload, inputs)


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    (
        "heads",
        "heads_kv",
        "dim",
        "q_lens",
        "cache_lens",
        "page_size",
        "dtype",
        "op_kwargs",
        "split_small",
    ),
    [
        # One query head a KV head: a head's queries span both consumer warpgroups.
        pytest.param(
            8, 8, 128, [300, 0, 130], [900, 64, 130], 128, torch.bfloat16, {}, True, id="group-1"
        ),
        # Sixteen query heads a KV head, eight query positions each, over 16-row pages.
        pytest.param(
            32, 2, 64, [70, 1, 9], [301, 40, 9], 16, torch.float16, {}, False, id="group-16"
        ),
        # A key tile of two 48-row pages, narrower than the 128 rows a page of 64 gives.
        pytest.param(
            32, 8, 128, [200, 1], [700, 333], 48, torch.bfloat16, {}, False, id="pages-48"
        ),
        pytest.param(
            32,
            8,
            128,
            [200, 1],
            [700, 333],
            64,
            torch.float16,
            {"is_causal": False},
            False,
            id="bidirectional",
        ),  # fmt: skip
        pytest.param(
            32,
            8,
            128,
            [200, 1],
            [700, 333],
            64,
            torch.float16,
            {"softcap": 30.0},
            False,
            id="softcap",
        ),  # fmt: skip
        # Requests of one and three tokens over 17 and 12 key tiles: their rows fit one
        # warpgroup, so the two warpgroups split the tiles and merge.
        pytest.param(
            32,
            8,
            128,
            [200, 1, 3],
            [700, 2049, 1500],
            64,
            torch.float16,
            {},
            True,
            id="split-small-requests",
        ),  # fmt: skip
        # Dimension 256: one query slot beside the key and value rings.
        pytest.param(32, 8, 256, [300, 1], [700, 333], 64, torch.float16, {}, True, id="dim-256"),
        # 48 query heads over one KV head: flat rows, 128 of the (T * H, D) query a tile.
        pytest.param(
            48, 1, 128, [200, 70], [700, 333], 64, torch.bfloat16, {}, False, id="flat-group-48"
        ),
        # 24 query heads over one KV head at dimension 64: three consumer warpgroups, 192
        # flat rows a tile.
        pytest.param(
            24, 1, 64, [200, 70], [700, 333], 64, torch.bfloat16, {}, False, id="flat-dim-64"
        ),
        # A serving step: one long request and thirty single-token requests, ranked, whose
        # tiles split between the warpgroups.
        pytest.param(
            32,
            8,
            128,
            [1024] + [1] * 30 + [0],
            [4096] * 32,
            64,
            torch.float16,
            {},
            True,
            id="ranked-serving-step",
        ),  # fmt: skip
        # Over a table of split width, 128 requests are ranked by cache length before they are
        # claimed and 129 are claimed in packing order; both with tied lengths and empty
        # requests over empty caches.
        pytest.param(
            32,
            8,
            128,
            [64, 0, 66] * 42 + [64, 64],
            [64, 0, 70] * 42 + [64, 1100],
            64,
            torch.float16,
            {},
            True,
            id="ranked-128",
        ),  # fmt: skip
        pytest.param(
            32,
            8,
            128,
            [64, 0, 66] * 43,
            [64, 0, 70] * 42 + [64, 0, 1100],
            64,
            torch.float16,
            {},
            True,
            id="claimed-in-order-129",
        ),  # fmt: skip
        # More query tiles than multiprocessors: each persistent CTA claims several.
        pytest.param(
            32,
            8,
            128,
            [700, 1, 333],
            [1500, 777, 333],
            64,
            torch.bfloat16,
            {},
            True,
            id="tiles-past-multiprocessors",
        ),  # fmt: skip
    ],
)
def test_gqa_paged_warp_specialized_query_tiles(
    heads: int,
    heads_kv: int,
    dim: int,
    q_lens: list[int],
    cache_lens: list[int],
    page_size: int,
    dtype: torch.dtype,
    op_kwargs: dict,
    split_small: bool,
) -> None:
    """SM90 copies pages by TMA into tiles that warp-specialized WGMMA reads; requests that
    fill query tiles are cut by their own query and cache lengths. A table of at least eight
    key tiles builds the split of a small request's tiles between the warpgroups."""
    from tileops._kernels.attention.gqa.paged_ws import GQAPagedFwdWSKernel

    width = -(-max(cache_lens) // page_size)
    workload = GQAPagedFwdWorkload(
        heads,
        heads_kv,
        dim,
        q_lens,
        cache_lens,
        page_size,
        width,
        len(q_lens) * width,
        dtype,
        **op_kwargs,
    )
    op, inputs = GQAPagedFwdOp(**{"is_causal": True, **op_kwargs}), _poisoned_inputs(workload)
    _check(op, workload, inputs)
    kernel = _kernel(op, inputs)
    assert isinstance(kernel, GQAPagedFwdWSKernel), type(kernel).__name__
    assert kernel.split_small == split_small


@pytest.mark.smoke
@pytest.mark.sm90
def test_gqa_paged_repeated_calls_claim_every_query_tile() -> None:
    """SM90 persistent CTAs claim query tiles from a counter one op keeps across calls on a
    stream; every call claims them all again."""
    workload = _decode(2, 32, 8, [700, 450], 128, 64, q_len=160)
    inputs = workload.gen_inputs()
    op = GQAPagedFwdOp(is_causal=True)
    for length in (700, 513):
        inputs[4][0] = length
        workload.cache_lens[0] = length
        _check(op, workload, inputs)


@pytest.mark.smoke
@pytest.mark.sm90
def test_gqa_paged_calls_overlapping_on_two_streams_each_claim_every_tile() -> None:
    """A warm op launched on two streams at once claims each stream's tiles from that
    stream's counter, so neither call takes the other's."""
    # Long enough that the two streams' launches run at the same time.
    workload = _decode(16, 32, 8, [8192] * 16, 128, 64, q_len=256)
    inputs = workload.gen_inputs()
    op = GQAPagedFwdOp(is_causal=True)
    _check(op, workload, inputs)
    expected = op(*inputs)
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    outputs = []
    for _ in range(4):
        for stream in streams:
            with torch.cuda.stream(stream):
                outputs.append(op(*inputs))
    torch.cuda.synchronize()
    for output in outputs:
        torch.testing.assert_close(output, expected)


@pytest.mark.smoke
@pytest.mark.sm90
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
    op = GQAPagedFwdOp(is_causal=True)
    _check(op, workload, inputs)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("q_lens", "cache_lens", "page_size", "dim", "dtype", "op_kwargs"),
    [
        pytest.param(
            [0, 1, 5, 1], [300, 512, 600, 7], 64, 128, torch.float16, {}, id="ragged-with-empty"
        ),
        # Head dimension 64, where the key tile is gathered at a width the page does not hold.
        pytest.param(
            [1, 1], [1000, 513], 64, 64, torch.float16, {"window_size_left": 128}, id="window"
        ),
        pytest.param(
            [7, 3],
            [300, 200],
            128,
            128,
            torch.float16,
            {"is_causal": False, "window_size_left": 64, "window_size_right": 32},
            id="both-windows-noncausal",
        ),
        # A page no key tile fits inside: the key tile is gathered row by row.
        pytest.param([1, 1], [400, 131], 65, 128, torch.float16, {}, id="page-no-tile-fits"),
        # The only shape where the causal bound and the window bound both bind: a request
        # with one query token sees its whole cache, so its causal bound masks nothing.
        pytest.param(
            [6, 1, 0],
            [300, 500, 128],
            64,
            128,
            torch.float16,
            {"window_size_left": 64},
            id="causal-window-multi",
        ),
        pytest.param(
            [2, 0, 9], [700, 64, 33], 16, 128, torch.float16, {"softcap": 30.0}, id="softcap"
        ),
        pytest.param([3, 1], [129, 48], 48, 128, torch.bfloat16, {}, id="bf16"),
        # The three shapes that isolate one term of the key-tile classification each. A tile
        # is masked per element only where a bound cuts it, so a term that never fires on any
        # case is a term no test pays for.
        # Causal only: a cache length the key tile divides, so the cache end never cuts and
        # the causal bound is the only term that can.
        pytest.param([5, 2], [512, 256], 64, 128, torch.float16, {}, id="causal-aligned-cache"),
        # Neither causal nor windowed: the cache end is then the only term that can cut.
        pytest.param(
            [3, 2], [250, 130], 64, 128, torch.float16, {"is_causal": False}, id="cache-end-only"
        ),
        # A right window narrower than the request's own query span, over a cache the key
        # tile divides: the right bound is then the only term that cuts.
        pytest.param(
            [64, 8],
            [256, 128],
            64,
            128,
            torch.float16,
            {"is_causal": False, "window_size_right": 8},
            id="right-window-only",
        ),
    ],
)
def test_gqa_paged_packed_query_lengths(
    q_lens: list[int],
    cache_lens: list[int],
    page_size: int,
    dim: int,
    dtype: torch.dtype,
    op_kwargs: dict,
) -> None:
    """Calls the decode region does not serve: ragged lengths, windows, and odd pages."""
    width = -(-max(cache_lens) // page_size)
    workload = GQAPagedFwdWorkload(
        32, 8, dim, q_lens, cache_lens, page_size, width, len(q_lens) * width, dtype, **op_kwargs
    )
    _check(GQAPagedFwdOp(**{"is_causal": True, **op_kwargs}), workload, workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.cuda_graph(op="GQAPagedFwdOp")
@pytest.mark.parametrize("batch", [2, 1], ids=["dynamic", "single-request"])
def test_gqa_paged_graph_replays_device_metadata(batch: int) -> None:
    """Dispatch is static; packed lengths and cache lengths remain replay inputs."""
    workload = _decode(batch, 32, 4, [4096, 64][:batch], 128, 64)
    inputs = workload.gen_inputs()
    inputs[4][0] = 128  # A mostly empty table must also be safe on the split path.
    workload.cache_lens[0] = 128
    op = GQAPagedFwdOp(is_causal=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = op(*inputs)
    # Same tensor addresses and shapes, different sequence boundaries. The dynamic
    # path must not keep the uniform packing it saw during warmup or capture.
    if batch == 2:
        inputs[5][1] = 0
        workload.q_lens = [0, 2]
        inputs[4][-1] = 63
        workload.cache_lens[-1] = 63
    v_cache, block_table, page_size = inputs[2], inputs[3][0], inputs[2].shape[1]
    live_values = v_cache.clone()
    for length in (128, 2048, 63, 0):
        inputs[4][0] = length
        workload.cache_lens[0] = length
        # Slots past the cache keep stale contents; a non-finite one must not reach the output.
        v_cache.copy_(live_values)
        slots = torch.arange(length, block_table.numel() * page_size, device=v_cache.device)
        v_cache[block_table[slots // page_size], slots % page_size] = float("nan")
        graph.replay()
        TestBase.check(workload, op, *inputs, runs=lambda *args: output)


@pytest.mark.smoke
def test_gqa_paged_sink_normalization() -> None:
    """A small paged prefill includes one denominator-only logit per query head."""
    workload = GQAPagedFwdWorkload(8, 2, 64, [17], [65], 32, 3, 3, torch.bfloat16, has_sinks=True)
    inputs = workload.gen_inputs()
    TestBase.check(workload, GQAPagedFwdOp(is_causal=True), *inputs)


@pytest.mark.smoke
@pytest.mark.skipif(
    torch.device(run_device()).type == "cpu",
    reason="mixes a run-device tensor with a cpu one, so the run device is not cpu",
)
def test_gqa_paged_rejects_invalid_tensor_contract() -> None:
    """Unsupported Q dtypes and malformed sinks fail at the public signature check."""
    from tileops._ops._signature_codegen import CheckError

    workload = GQAPagedFwdWorkload(8, 2, 64, [1], [1], 16, 1, 1, torch.float16, has_sinks=True)
    inputs = workload.gen_inputs()
    op = GQAPagedFwdOp(is_causal=True)
    with pytest.raises(CheckError, match="dtype"):
        op(inputs[0].float(), *inputs[1:])
    for bad_sink, message in (
        (inputs[-1].half(), "dtype"),
        (inputs[-1][:-1], "H|shape|size"),
        (inputs[-1].cpu(), "device"),
    ):
        with pytest.raises(CheckError, match=message):
            op(*inputs[:-1], bad_sink)
    # A cache scale is per tensor, [1], or per request and KV head, [B, H_kv].
    fp8 = _paged([1], [1], 16, torch.float16, cache_dtype=FP8, heads=8, heads_kv=2, dim=64)
    inputs = list(fp8.gen_inputs())
    inputs[7] = inputs[8] = torch.ones(2, dtype=torch.float32, device=inputs[0].device)
    with pytest.raises(CheckError, match="SK|shape|rule"):
        op(*inputs)


FP8 = torch.float8_e4m3fn


def _paged(
    q_lens: list[int],
    cache_lens: list[int],
    page_size: int,
    dtype: torch.dtype,
    *,
    heads: int = 32,
    heads_kv: int = 8,
    dim: int = 128,
    **semantics,
) -> GQAPagedFwdWorkload:
    """A packed call over tables as wide as the longest cache, one table's pages a request."""
    width = -(-max(cache_lens) // page_size)
    return GQAPagedFwdWorkload(
        heads,
        heads_kv,
        dim,
        q_lens,
        cache_lens,
        page_size,
        width,
        len(q_lens) * width,
        dtype,
        **semantics,
    )


def _poisoned_inputs(workload: GQAPagedFwdWorkload) -> tuple:
    """The workload's inputs with NaN in every page row past each request's cache."""
    inputs = workload.gen_inputs()
    k_cache, v_cache, block_table = inputs[1], inputs[2], inputs[3]
    page_size = workload.page_size
    for request, length in enumerate(workload.cache_lens):
        slots = torch.arange(length, block_table.shape[1] * page_size, device=v_cache.device)
        stale = (block_table[request, slots // page_size], slots % page_size)
        k_cache[stale] = float("nan")
        v_cache[stale] = float("nan")
    return inputs


def _causal_canary(workload: GQAPagedFwdWorkload, inputs: tuple) -> tuple:
    """*inputs* with the largest e4m3 value in the values of every key past each request's
    first query position, so a row that sees a later key stands out past the FP8 tolerance."""
    v_cache, block_table, page_size = inputs[2], inputs[3], workload.page_size
    for request, (q_len, kv_len) in enumerate(
        zip(workload.q_lens, workload.cache_lens, strict=True)
    ):
        if q_len < 2:
            continue
        keys = torch.arange(kv_len - q_len + 1, kv_len, device=v_cache.device)
        v_cache[block_table[request, keys // page_size], keys % page_size] = 448.0
    return inputs


def _fp8_op(workload: GQAPagedFwdWorkload, **semantics) -> GQAPagedFwdOp:
    return GQAPagedFwdOp(out_dtype=workload.out_dtype, **{"is_causal": True, **semantics})


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    ("q_lens", "cache_lens", "page_size", "dtype", "semantics"),
    [
        # One copy a key tile; the cache end cuts the last tile of the first request.
        pytest.param(
            [1, 1], [513, 1000], 64, torch.float16, {}, id="f16-tensor-scale-cut"
        ),  # fmt: skip
        # Four 16-row copies a key tile; causal rows of three query tokens a request.
        pytest.param(
            [3, 3],
            [700, 333],
            16,
            torch.bfloat16,
            {"per_request_scales": True},
            id="bf16-request-scales-pages-16",
        ),
        # An FP8 query, its score descale under a softcap and a left window.
        pytest.param(
            [2, 2],
            [1000, 300],
            128,
            FP8,
            {"per_request_scales": True, "softcap": 30.0, "window_size_left": 200},
            id="fp8-softcap-window",
        ),
    ],
)
def test_gqa_paged_fp8_decode(
    q_lens: list[int],
    cache_lens: list[int],
    page_size: int,
    dtype: torch.dtype,
    semantics: dict,
) -> None:
    """SM90 FP8 decode converts each stored tile in registers and dequantizes by the scales;
    a NaN row past a cache must not reach the output."""
    from tileops._kernels.attention.gqa.paged_fp8_decode import GQAPagedFP8DecodeKernel

    out_dtype = torch.bfloat16 if dtype == FP8 else None
    workload = _paged(
        q_lens, cache_lens, page_size, dtype, cache_dtype=FP8, out_dtype=out_dtype, **semantics
    )
    op_semantics = {k: v for k, v in semantics.items() if k != "per_request_scales"}
    op, inputs = _fp8_op(workload, **op_semantics), _poisoned_inputs(workload)
    _check(op, workload, inputs)
    assert isinstance(_kernel(op, inputs), GQAPagedFP8DecodeKernel)


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    ("q_lens", "cache_lens", "page_size", "dtype", "out_dtype", "semantics", "split_small"),
    [
        # A 16-bit query: the producer warpgroup converts both tiles, 64-row key tiles; the
        # one-token request over 16 of them splits its tiles between the warpgroups.
        pytest.param(
            [200, 1], [700, 1000], 64, torch.float16, None, {}, True, id="f16-convert-k-v"
        ),
        # An FP8 query with one- and three-token requests over 17 and 12 key tiles: split,
        # causal within the three-token request.
        pytest.param(
            [200, 1, 3],
            [700, 2049, 1500],
            64,
            FP8,
            torch.bfloat16,
            {"per_request_scales": True},
            True,
            id="fp8-split-small-requests",
        ),
        # Seven one-token requests split ahead of a 300-token one over 43 key tiles each:
        # the split handover must not share shared memory with the stored value ring.
        pytest.param(
            [1, 1, 1, 1, 1, 1, 1, 300, 0],
            [4099] * 9,
            48,
            FP8,
            torch.bfloat16,
            {},
            True,
            id="fp8-split-before-prefill-pages-48",
        ),
        # 360 requests at 64-row key tiles and three stages: the largest shared-memory
        # footprint the tiling picks, at the opt-in limit.
        pytest.param(
            [32] * 360,
            [64] * 360,
            64,
            torch.bfloat16,
            None,
            {},
            False,
            id="bf16-shared-memory-limit",
        ),
        # Two 48-row pages a 96-row key tile, scales per request.
        pytest.param(
            [200, 1],
            [700, 333],
            48,
            torch.bfloat16,
            None,
            {"per_request_scales": True},
            False,
            id="bf16-pages-48",
        ),
        # An FP8 query reads the stored keys; float16 output, no causal mask.
        pytest.param(
            [200, 1],
            [700, 333],
            64,
            FP8,
            torch.float16,
            {"per_request_scales": True, "is_causal": False},
            False,
            id="fp8-out-f16-bidirectional",
        ),
        # Part of one 256-row page a key tile; the softcap reads the dequantized score.
        pytest.param(
            [300, 0, 130],
            [900, 64, 130],
            256,
            FP8,
            torch.bfloat16,
            {"softcap": 30.0},
            True,
            id="fp8-softcap-pages-256",
        ),
    ],
)
def test_gqa_paged_fp8_warp_specialized(
    q_lens: list[int],
    cache_lens: list[int],
    page_size: int,
    dtype: torch.dtype,
    out_dtype: torch.dtype | None,
    semantics: dict,
    split_small: bool,
) -> None:
    """SM90 FP8 prefill: the producer warpgroup converts stored tiles, zeroing value rows past
    a cache, while the consumers dequantize the scores and the output. A table of at least
    eight key tiles builds the split of a small request's tiles between the warpgroups."""
    from tileops._kernels.attention.gqa.paged_fp8_ws import GQAPagedFP8WSKernel

    workload = _paged(
        q_lens, cache_lens, page_size, dtype, cache_dtype=FP8, out_dtype=out_dtype, **semantics
    )
    op_semantics = {k: v for k, v in semantics.items() if k != "per_request_scales"}
    op = _fp8_op(workload, **op_semantics)
    inputs = _causal_canary(workload, _poisoned_inputs(workload))
    _check(op, workload, inputs)
    kernel = _kernel(op, inputs)
    assert isinstance(kernel, GQAPagedFP8WSKernel), type(kernel).__name__
    assert kernel.split_small == split_small


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    ("mode", "semantics"),
    [
        pytest.param("capped", {}, id="one-run-over-capped-workspace"),
        pytest.param("runs", {"softcap": 30.0}, id="softcap-runs"),
        pytest.param("graph", {}, id="graph"),
    ],
)
def test_gqa_paged_fp8_split_runs(
    mode: str, semantics: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One-token requests over 43 key tiles split them into runs that the last one combines.

    With the partials' workspace capped below one run each request keeps a single run. The
    softcap applies inside each run before the combine. Under graph replay, the arrival
    counts return to zero after each call, so a second replay over new values combines again.
    """
    from tileops._kernels.attention.gqa.paged_fp8_ws import GQAPagedFP8WSKernel

    if mode == "capped":
        monkeypatch.setattr(GQAPagedFP8WSKernel, "_SPLIT_WORKSPACE", 0)
    workload = _paged(
        [1, 1, 1, 1, 1, 1, 1, 300, 0], [4099] * 9, 48, FP8, cache_dtype=FP8,
        out_dtype=torch.bfloat16, **semantics,
    )  # fmt: skip
    op = _fp8_op(workload, **semantics)
    inputs = _poisoned_inputs(workload)
    kernel = _kernel(op, inputs)
    assert isinstance(kernel, GQAPagedFP8WSKernel), type(kernel).__name__
    assert kernel.split_chunks == (1 if mode == "capped" else 2)
    if mode != "graph":
        _check(op, workload, inputs)
        return
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    cuda_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cuda_graph):
        output = op(*inputs)
    v_cache = inputs[2]
    for scale in (1.0, -1.0):
        v_cache.copy_((v_cache.float() * scale).to(FP8))
        cuda_graph.replay()
        TestBase.check(workload, op, *inputs, runs=lambda *args: output)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    ("q_lens", "cache_lens", "page_size", "dtype", "semantics"),
    [
        # RoPE over a decode packing, which the warp-specialized kernels do not serve.
        pytest.param(
            [1, 1],
            [129, 65],
            64,
            torch.bfloat16,
            {
                "per_request_scales": True,
                "pos_encoding_mode": "rope",
                "rotary_dim": 64,
                "rope_layout": "interleaved",
            },
            id="rope-decode",
        ),
        # Sinks with an FP8 query; one request and one KV head make the per-request scales
        # [1, 1], one element each.
        pytest.param(
            [3],
            [65],
            32,
            FP8,
            {"has_sinks": True, "per_request_scales": True},
            id="fp8-sinks-request-scales-1x1",
        ),
        # Both window bounds, over enough row tiles that the key range is not split.
        pytest.param(
            [300, 260],
            [700, 333],
            64,
            torch.float16,
            {"is_causal": False, "window_size_left": 17, "window_size_right": 3},
            id="two-sided-window-unsplit",
        ),
        # Dimension 256 over rows averaging a 128-row tile a request: two warpgroups share
        # each converted key tile.
        pytest.param([300, 1], [700, 333], 64, torch.float16, {"dim": 256}, id="dim-256-filled"),
    ],
)
def test_gqa_paged_fp8_tile_scan(
    q_lens: list[int],
    cache_lens: list[int],
    page_size: int,
    dtype: torch.dtype,
    semantics: dict,
) -> None:
    """Calls the FP8 kernels refuse gather each FP8 tile into the tile scan's 16-bit buffers."""
    from tileops._kernels.attention.gqa.paged import GQAPagedFwdKernel

    out_dtype = torch.bfloat16 if dtype == FP8 else None
    extra = {"heads": 8, "heads_kv": 1, "dim": 64} if semantics.get("has_sinks") else {}
    semantics = dict(semantics)
    if "dim" in semantics:
        extra["dim"] = semantics.pop("dim")
    workload = _paged(
        q_lens,
        cache_lens,
        page_size,
        dtype,
        cache_dtype=FP8,
        out_dtype=out_dtype,
        **extra,
        **semantics,
    )
    op_semantics = {
        k: v for k, v in semantics.items() if k not in ("per_request_scales", "has_sinks")
    }
    op, inputs = _fp8_op(workload, **op_semantics), _poisoned_inputs(workload)
    _check(op, workload, inputs)
    kernel = _kernel(op, inputs)
    assert isinstance(kernel, GQAPagedFwdKernel)
    assert (kernel.config["threads"] == 256) == (workload.dim > 128), kernel.config


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    ("dtype", "per_request_scales"),
    [pytest.param(FP8, False, id="fp8-query"), pytest.param(torch.bfloat16, True, id="bf16")],
)
def test_gqa_paged_fp8_rope_prefill(dtype: torch.dtype, per_request_scales: bool) -> None:
    """RoPE preparation dequantizes an FP8 query and cache into the 16-bit pages the
    warp-specialized kernel reads."""
    from tileops._kernels.attention.gqa.paged_ws import GQAPagedFwdWSKernel

    out_dtype = torch.bfloat16 if dtype == FP8 else None
    semantics = {"pos_encoding_mode": "rope", "rotary_dim": 64}
    workload = _paged(
        [200, 1],
        [700, 333],
        64,
        dtype,
        cache_dtype=FP8,
        out_dtype=out_dtype,
        per_request_scales=per_request_scales,
        **semantics,
    )
    op, inputs = _fp8_op(workload, **semantics), _poisoned_inputs(workload)
    _check(op, workload, inputs)
    assert isinstance(_kernel(op, inputs), GQAPagedFwdWSKernel)


@pytest.mark.parametrize(
    "cache_dtype",
    [
        pytest.param(torch.float16, id="fp16", marks=pytest.mark.smoke),
        pytest.param(torch.float8_e4m3fn, id="fp8-cache", marks=pytest.mark.full),
    ],
)
def test_gqa_paged_write_then_read(cache_dtype: torch.dtype) -> None:
    """Append across a fragmented page boundary, then attend the stored new tokens."""
    case = GQAPagedFwdWorkload(
        8, 2, 64, [3, 1], [66, 128], 64, 3, 6, torch.float16, cache_dtype=cache_dtype
    )
    inputs = case.gen_inputs()
    write_workload, write_inputs = case.append_workload(inputs)
    write_op = PagedKVCacheWriteFwdOp()
    TestBase.check(
        write_workload,
        write_op,
        *write_inputs,
        runs=lambda *args: paged_cache_write_result(write_op, *args),
    )
    write_op(*write_inputs)
    _check(GQAPagedFwdOp(is_causal=True), case, inputs)
