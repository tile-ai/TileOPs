"""GQAPagedFwdOp tests against the workload's reference."""

import dataclasses

import pytest
import torch

from tests.cuda_graph_contract import CapturedGraph
from tests.workload_test_base import FixtureBase, TestBase
from tileops.attention import GQAPagedFwdOp, PagedKVCacheWriteFwdOp
from tileops.backend import OpNotAvailableError
from workloads.attention.gqa.paged import GQAPagedFwdWorkload
from workloads.attention.paged_cache_write import paged_cache_write_result
from workloads.attention.paged_kv_cache import make_fragmented_block_table
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
@pytest.mark.parametrize(
    ("q_lens", "cache_lens", "pool_pages", "split"),
    [([3] + [1] * 23, [257] + [33] * 23, 120, False), ([3, 1], [257, 33], 10, True)],
    ids=["unsplit", "split"],
)
@pytest.mark.parametrize("pos_encoding_mode", ["none", "rope"])
def test_gqa_paged_split(
    q_lens: list[int], cache_lens: list[int], pool_pages: int, split: bool, pos_encoding_mode: str
) -> None:
    """One split or several preserve masks and the split LSE merge.

    Enough request tiles fill the multiprocessors without a split; two requests take several.
    """
    from tileops._kernels.attention.gqa.paged import GQAPagedFwdKernel

    semantics = dict(pos_encoding_mode=pos_encoding_mode)
    workload = GQAPagedFwdWorkload(
        16, 4, 128, q_lens, cache_lens, 64, 5, pool_pages, torch.float16, **semantics
    )
    op = GQAPagedFwdOp(**{"is_causal": True, **semantics})
    inputs = workload.gen_inputs()
    kernel = _kernel(op, inputs)
    assert isinstance(kernel, GQAPagedFwdKernel)
    config = kernel.config
    tiles = kernel._tile_estimate(inputs[0].shape[0], config["block_M"])
    assert (kernel._splits_for(tiles, config) > 1) == split
    _check(op, workload, inputs)


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    ("heads", "heads_kv", "dim", "q_len", "semantics"),
    [
        pytest.param(16, 2, 64, 4, {"has_sinks": True}, marks=pytest.mark.smoke, id="dim-64-sinks"),
        pytest.param(16, 4, 256, 2, {}, marks=pytest.mark.full, id="dim-256"),
        pytest.param(8, 2, 144, 4, {}, marks=pytest.mark.full, id="dim-144"),
    ],
)
def test_gqa_paged_split_combine_row_blocks(
    heads: int, heads_kv: int, dim: int, q_len: int, semantics: dict
) -> None:
    """The combine merges a query tile in blocks of rows (16 at dimension 64, 4 at 256, and 8
    at 144, where four rows of 16-byte vectors leave a thread a part of a row but eight rows
    split evenly over the threads): each request's rows span two blocks or more, a block past
    them stores nothing, and the weight table of sixteen chunks or more is stored split-major."""
    from tileops._kernels.attention.gqa.paged import GQAPagedFwdKernel

    workload = _decode(2, heads, heads_kv, [4096] * 2, dim, 64, q_len=q_len, **semantics)
    op = GQAPagedFwdOp(is_causal=True)
    inputs = workload.gen_inputs()
    kernel = _kernel(op, inputs)
    assert isinstance(kernel, GQAPagedFwdKernel)
    config = kernel.config
    tiles = kernel._tile_estimate(inputs[0].shape[0], config["block_M"])
    assert kernel._splits_for(tiles, config) >= 16
    _check(op, workload, inputs)


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    ("semantics", "kernel_name"),
    [
        ({"is_causal": True, "window_size_left": 254}, "GQAPagedFwdKernel"),
        (
            {"is_causal": False, "window_size_left": 255, "window_size_right": 255},
            "GQAPagedFwdWSKernel",
        ),
        # INT32_MAX hides no key; added to a query position it overflows int32.
        ({"is_causal": True, "window_size_right": 2**31 - 1}, "GQAPagedFwdWSKernel"),
    ],
    ids=["left-inside-table", "both-at-last-key", "right-int32-max"],
)
def test_gqa_paged_window_reaching_the_last_key_is_no_window(
    semantics: dict, kernel_name: str
) -> None:
    """Without RoPE, the WS kernel serves a window reaching key 255, the last of four 64-row
    pages, and refuses one a key narrower. A window of INT32_MAX keys is no window."""
    workload = GQAPagedFwdWorkload(16, 4, 128, [256], [256], 64, 4, 4, torch.float16, **semantics)
    op = GQAPagedFwdOp(**semantics)
    inputs = workload.gen_inputs()
    assert type(_kernel(op, inputs)).__name__ == kernel_name
    _check(op, workload, inputs)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("left", "right", "windows"),
    [
        pytest.param(254, 254, (254, 254), id="inside-table"),
        pytest.param(255, -1, (-1, -1), id="left-at-last-key"),
        pytest.param(-1, 255, (-1, -1), id="right-at-last-key"),
        pytest.param(-1, 2**31 - 1, (-1, -1), id="right-int32-max"),
    ],
)
def test_gqa_paged_call_names_no_window_that_hides_no_key(
    left: int, right: int, windows: tuple[int, int]
) -> None:
    """A table of four 64-row pages holds 256 positions: a window reaching key 255 hides no
    key, and the call the op builds names no window there."""
    meta = torch.device("meta")
    q = torch.empty(16, 4, 128, dtype=torch.float16, device=meta)
    k_cache = torch.empty(4, 64, 4, 128, dtype=torch.float16, device=meta)
    table = torch.empty(1, 4, dtype=torch.int32, device=meta)
    op = GQAPagedFwdOp(is_causal=False, window_size_left=left, window_size_right=right)
    call = op._paged_call(q, k_cache, table)
    assert (call.window_size_left, call.window_size_right) == windows


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_gqa_paged_decode_keeps_the_narrow_query_tile() -> None:
    """A decode over pages that hold only a key tile narrower than one WGMMA keeps the 64-row
    query tile: its four rows a request would pad the wide one."""
    from tileops._kernels.attention.gqa.paged import GQAPagedFwdKernel

    semantics = {"window_size_left": 512}
    workload = _decode(4, 32, 8, [700] * 4, 128, 48, torch.bfloat16, **semantics)
    op = GQAPagedFwdOp(is_causal=True, **semantics)
    inputs = workload.gen_inputs()
    kernel = _kernel(op, inputs)
    assert isinstance(kernel, GQAPagedFwdKernel)
    assert (kernel.config["block_M"], kernel.config["block_N"]) == (64, 48)
    _check(op, workload, inputs)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    ("batch", "page_size", "pages", "semantics", "expected"),
    [
        (2, 64, 256, {}, 8),
        (2, 16, 8192, {"dim": 64}, 41),
        (2, 16, 4096, {"dim": 64}, 32),
        (2, 64, 256, {"fuse_rope": True}, 16),
        (8, 16, 160, {"dim": 64}, 5),
        (2, 16, 128, {"dim": 64}, 9),
        (8, 16, 64, {}, 4),
        (1, 16, 1024, {}, 17),
        (1, 16, 1024, {"fuse_rope": True}, 33),
        (2, 16, 1024, {"dim": 64, "window_size_left": 1024}, 9),
        (2, 64, 256, {"window_size_left": 64}, 2),
        (40, 16, 1024, {}, 1),
    ],
    ids=[
        "copied",
        "gathered",
        "combine-bound",
        "rotated",
        "chunk-limit",
        "short-range",
        "stacked-start",
        "stacked-combine-bound",
        "stacked-rotated",
        "left-window",
        "window-key-tiles",
        "unsplit",
    ],
)
def test_gqa_paged_split_count(
    batch: int, page_size: int, pages: int, semantics: dict, expected: int
) -> None:
    """Fixed H200 geometry makes each split-policy branch testable on any device."""
    from tileops._kernels.attention.gqa.paged import GQAPagedFwdKernel

    kernel = object.__new__(GQAPagedFwdKernel)
    kernel.batch, kernel.dim, kernel.heads_kv = batch, 128, 8
    kernel.page_size, kernel.max_pages_per_req, kernel.dtype = page_size, pages, torch.float16
    kernel.is_causal, kernel.window_size_left, kernel.window_size_right = True, -1, -1
    kernel.fuse_rope = False
    kernel._processors, kernel._sm_shared_bytes = 132, 228 * 1024
    kernel._cta_reserved_bytes, kernel._sm_threads = 1024, 2048
    for name, value in semantics.items():
        setattr(kernel, name, value)
    config = {"block_M": 64, "block_N": 64, "num_stages": 2, "threads": 128}
    assert kernel._splits_for(batch, config) == expected


@pytest.mark.smoke
def test_gqa_paged_single_token_cut_key_tile() -> None:
    """A single-token call keeps the keys past a cache end that cuts a key tile out."""
    workload = _decode(1, 16, 4, [1025], 128, 128)
    _check(GQAPagedFwdOp(is_causal=True), workload, workload.gen_inputs())


@pytest.mark.parametrize(
    ("q_len", "cache_len", "pool_pages"),
    [
        pytest.param(3, 13, 1, marks=pytest.mark.smoke, id="chunk-one-page"),
        pytest.param(2100, 2100, 36, marks=pytest.mark.full, id="prefill-36-pages"),
    ],
)
def test_gqa_paged_one_request_over_the_whole_pool(
    q_len: int, cache_len: int, pool_pages: int
) -> None:
    """Qwen3.8-27B's attention calls: 24 query heads over 4 KV heads of 256, causal, bfloat16,
    one request whose table spans the whole pool of 64-row pages. Guards their compile: the
    pipelined wait's index reached codegen unfolded and failed with ``Downcast from tirx.Sub
    to ir.IntImm``."""
    workload = GQAPagedFwdWorkload(
        24,
        4,
        256,
        [q_len],
        [cache_len],
        64,
        max_pages_per_req=pool_pages,
        num_pages=pool_pages,
        dtype=torch.bfloat16,
    )
    _check(GQAPagedFwdOp(is_causal=True), workload, workload.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_gqa_paged_refuses_a_head_dim_off_mma_k_slices() -> None:
    """A head dim of 72 is refused before anything is built.

    Every paged kernel contracts the head dim in 16-wide matrix-multiply-accumulate (MMA)
    k-slices.
    """
    workload = _decode(1, 8, 2, [300], 72, 64)
    op = GQAPagedFwdOp(is_causal=True)
    with pytest.raises(OpNotAvailableError, match="multiple of 16"):
        op(*workload.gen_inputs())
    for interface in GQAPagedFwdOp.interfaces:
        assert not op._entries_built(interface)


@pytest.mark.parametrize(
    "dtype,q_lens,cache_lens,page_size,dim,semantics",
    [
        pytest.param(
            torch.float16, [1, 3], [129, 63], 64, 128, {}, marks=pytest.mark.smoke, id="fp16-split"
        ),
        pytest.param(
            torch.bfloat16, [1, 3], [129, 63], 64, 128, {}, marks=pytest.mark.smoke, id="bf16-split"
        ),
        pytest.param(
            torch.float16,
            [160, 129],
            [256, 193],
            256,
            64,
            {"rope_layout": "interleaved"},
            marks=pytest.mark.full,
            id="interleaved-unsplit",
        ),
        pytest.param(
            torch.float16,
            [7, 0, 130],
            [99, 0, 145],
            48,
            128,
            {"rotary_dim": 48, "sm_scale": 0.25},
            marks=pytest.mark.full,
            id="partial-odd-pages",
        ),
        pytest.param(
            torch.float16,
            [2, 1],
            [65, 97],
            16,
            256,
            {"rotary_dim": 64, "rope_layout": "interleaved", "softcap": 2.0},
            marks=pytest.mark.full,
            id="partial-interleaved-softcap",
        ),
        pytest.param(
            torch.float16,
            [65, 3],
            [129, 33],
            64,
            128,
            {"is_causal": False, "window_size_left": 17, "window_size_right": 3},
            marks=pytest.mark.full,
            id="two-sided-window",
        ),
        pytest.param(
            torch.float16,
            [3, 1],
            [65, 33],
            64,
            128,
            {"is_causal": False},
            marks=pytest.mark.full,
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


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "cache_dtype",
    [
        pytest.param(torch.float16, marks=pytest.mark.smoke, id="rotated-channels"),
        pytest.param(torch.float8_e4m3fn, marks=pytest.mark.full, id="whole-pages"),
    ],
)
def test_gqa_paged_rope_window_prepares_from_the_first_window(cache_dtype: torch.dtype) -> None:
    """Under a left window the RoPE preparation writes keys from the key tile holding the
    request's first query's window start, which lies tiles before its last query's."""
    from tileops._kernels.attention.gqa.paged_ws import GQAPagedFwdWSKernel

    semantics = {"pos_encoding_mode": "rope", "window_size_left": 100}
    workload = _paged(
        [300, 200], [1500, 1300], 64, torch.float16, cache_dtype=cache_dtype, **semantics
    )
    op = GQAPagedFwdOp(is_causal=True, **semantics)
    inputs = workload.gen_inputs()
    assert isinstance(_kernel(op, inputs), GQAPagedFwdWSKernel)
    _check(op, workload, inputs)


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "rotary_dim, rope_layout, cache_dtype, dim",
    [
        pytest.param(40, "neox", torch.float16, 128, marks=pytest.mark.smoke, id="rotary-40-neox"),
        pytest.param(
            36,
            "interleaved",
            torch.float16,
            128,
            marks=pytest.mark.full,
            id="rotary-36-interleaved",
        ),
        pytest.param(
            36, "neox", torch.float8_e4m3fn, 128, marks=pytest.mark.full, id="rotary-36-fp8"
        ),
        pytest.param(
            34, "interleaved", torch.float16, 256, marks=pytest.mark.full, id="rotary-34-dim256"
        ),
    ],
)
def test_gqa_paged_rope_prepares_whole_vectors(
    rotary_dim: int, rope_layout: str, cache_dtype: torch.dtype, dim: int
) -> None:
    """On SM90, where dispatch selects the warp-specialized kernel for packings of whole
    query tiles, RoPE serves rotated widths whose half and whose unrotated tails are not a
    multiple of the widest vector access, in both layouts, over a 16-bit and an FP8 cache, and
    at head dimension 256."""
    from tileops._kernels.attention.gqa.paged_ws import GQAPagedFwdWSKernel

    semantics = {"pos_encoding_mode": "rope", "rotary_dim": rotary_dim, "rope_layout": rope_layout}
    workload = _paged(
        [300, 200], [1500, 1300], 64, torch.float16, dim=dim, cache_dtype=cache_dtype, **semantics
    )
    op = GQAPagedFwdOp(is_causal=True, **semantics)
    inputs = workload.gen_inputs()
    assert isinstance(_kernel(op, inputs), GQAPagedFwdWSKernel)
    _check(op, workload, inputs)


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
@pytest.mark.cuda_only("cuda-graph")
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
        pytest.param(None, 2.0, [512, 300], 1, marks=pytest.mark.full, id="softcap"),
        # Unequal lengths on the split path leave the short request with empty splits.
        pytest.param(None, None, [4096, 128], 1, marks=pytest.mark.full, id="unequal-split"),
        pytest.param(None, None, [700, 300], 4, marks=pytest.mark.full, id="causal"),
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


@pytest.mark.parametrize(
    ("cache_lens", "page_size", "q_len"),
    [
        # Pages of 64 hold copied key tiles of 64 rows, and pages of 256 tiles of 128. Five
        # query tokens a request leave the device short of row tiles, so the key range is
        # split; 160 fill it, so it is not, and on SM90 they fill warp-specialized query
        # tiles whose key tile is two 64-row pages or part of a 256-row one.
        pytest.param([63, 130], 64, 5, marks=pytest.mark.smoke, id="tile-64-split"),
        pytest.param([2047, 1501], 256, 5, marks=pytest.mark.full, id="tile-128-split"),
        pytest.param([700, 450], 64, 160, marks=pytest.mark.full, id="tile-64"),
        pytest.param([700, 450], 256, 160, marks=pytest.mark.full, id="tile-128"),
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
            8,
            8,
            128,
            [300, 0, 130],
            [900, 64, 130],
            128,
            torch.bfloat16,
            {},
            True,
            marks=pytest.mark.smoke,
            id="group-1",
        ),
        # Sixteen query heads a KV head, eight query positions each, over 16-row pages.
        pytest.param(
            32,
            2,
            64,
            [70, 1, 9],
            [301, 40, 9],
            16,
            torch.float16,
            {},
            False,
            marks=pytest.mark.smoke,
            id="group-16",
        ),
        # A key tile of two 48-row pages, narrower than the 128 rows a page of 64 gives.
        pytest.param(
            32,
            8,
            128,
            [200, 1],
            [700, 333],
            48,
            torch.bfloat16,
            {},
            False,
            marks=pytest.mark.full,
            id="pages-48",
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
            marks=pytest.mark.full,
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
            marks=pytest.mark.full,
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
            marks=pytest.mark.full,
            id="split-small-requests",
        ),  # fmt: skip
        # Dimension 256: one query slot beside the key and value rings.
        pytest.param(
            32,
            8,
            256,
            [300, 1],
            [700, 333],
            64,
            torch.float16,
            {},
            True,
            marks=pytest.mark.full,
            id="dim-256",
        ),
        # 48 query heads over one KV head: flat rows, 128 of the (T * H, D) query a tile.
        pytest.param(
            48,
            1,
            128,
            [200, 70],
            [700, 333],
            64,
            torch.bfloat16,
            {},
            False,
            marks=pytest.mark.full,
            id="flat-group-48",
        ),
        # 24 query heads over one KV head at dimension 64: three consumer warpgroups, 192
        # flat rows a tile.
        pytest.param(
            24,
            1,
            64,
            [200, 70],
            [700, 333],
            64,
            torch.bfloat16,
            {},
            False,
            marks=pytest.mark.full,
            id="flat-dim-64",
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
            marks=pytest.mark.full,
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
            marks=pytest.mark.full,
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
            marks=pytest.mark.full,
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
            marks=pytest.mark.full,
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


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    ("heads", "heads_kv", "dim", "q_lens", "cache_lens", "page_size", "dtype", "semantics"),
    [
        pytest.param(
            24, 4, 256, [300, 3, 0, 130], [900, 1500, 64, 131], 64, torch.bfloat16, {},
            marks=pytest.mark.smoke, id="group-6",
        ),
        pytest.param(
            24, 4, 128, [200, 9, 450], [700, 333, 900], 48, torch.bfloat16,
            {"pos_encoding_mode": "rope", "rotary_dim": 64, "window_size_left": 256},
            marks=pytest.mark.full, id="group-6-rope-window",
        ),
        pytest.param(
            24, 4, 128, [200, 1, 450], [700, 333, 900], 64, torch.bfloat16,
            {"pos_encoding_mode": "rope", "rotary_dim": 64, "cache_dtype": torch.float8_e4m3fn},
            marks=pytest.mark.full, id="group-6-fp8-cache-rope",
        ),
        pytest.param(
            24, 4, 128, [250, 309], [250, 309], 64, torch.bfloat16, {"has_sinks": True},
            marks=pytest.mark.full, id="group-6-sinks",
        ),
        pytest.param(
            12, 4, 128, [800, 1, 300], [901, 40, 301], 16, torch.bfloat16, {},
            marks=pytest.mark.full, id="group-3",
        ),
        pytest.param(
            24, 2, 128, [600, 2, 33], [1100, 1200, 40], 64, torch.bfloat16, {},
            marks=pytest.mark.full, id="group-12",
        ),
        pytest.param(
            48, 2, 128, [280, 70], [700, 333], 64, torch.bfloat16, {},
            marks=pytest.mark.full, id="group-24-over-two-kv-heads",
        ),
        pytest.param(
            24, 4, 128, [44] * 129 + [1], [1100] * 129 + [1500], 64, torch.bfloat16, {},
            marks=pytest.mark.full, id="group-6-claimed-in-order-130",
        ),
    ],
)  # fmt: skip
def test_gqa_paged_warp_specialized_head_groups(
    heads: int,
    heads_kv: int,
    dim: int,
    q_lens: list[int],
    cache_lens: list[int],
    page_size: int,
    dtype: torch.dtype,
    semantics: dict,
) -> None:
    """A query group whose heads cannot each take whole swizzle atoms of a query tile is cut
    into head groups, each its own work item over its KV head's key tiles. Ragged requests end
    inside a page whose later rows are NaN; a request of a few tokens over a long cache splits
    its key tiles between the warpgroups; keys past each request's first query position carry
    large values, so a row that sees a later key, or takes another head's sink, stands out.
    The op selects this kernel for each call on the 132 multiprocessors of an H200, and the
    kernel runs on the test's device whatever that device would select."""
    from tileops._kernels.attention.gqa.paged_ws import GQAPagedFwdWSKernel

    workload = _paged(
        q_lens, cache_lens, page_size, dtype, heads=heads, heads_kv=heads_kv, dim=dim, **semantics
    )
    op_kwargs = {k: v for k, v in semantics.items() if k not in ("has_sinks", "cache_dtype")}
    op = GQAPagedFwdOp(is_causal=True, **op_kwargs)
    inputs = _causal_canary(workload, _poisoned_inputs(workload))
    q, k_cache, v_cache, block_table, cache_seqlens, cu_seqlens_q, *scales, cos, sin, sinks = inputs
    call = op._paged_call(q, k_cache, block_table, scales[1], cos, sinks)
    assert op.key_for("gqa_paged", dataclasses.replace(call, sm_count=132)) == "gqa_paged_ws_kernel"
    _, build = GQAPagedFwdWSKernel.entry_for(call)
    pools = k_cache.flatten(0, 1), v_cache.flatten(0, 1)
    output = build()(q, *pools, cache_seqlens, block_table, cu_seqlens_q, *scales, cos, sin, sinks)
    TestBase.check(workload, op, *inputs, runs=lambda *args: output)


def _laid_out_queries(q: torch.Tensor, layout: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    """*q*, ``(tokens, heads, dim)``, in *layout*, and the storage the view lies in:
    ``("columns", row)``, the leading columns of rows *row* elements wide;
    ``("padded-heads", pad)``, each head followed by *pad* elements of another tensor;
    ``("expanded",)``, its first token repeated over every token."""
    tokens, heads, dim = q.shape
    if layout[0] == "expanded":
        storage = q[:1].clone()
        return storage.expand(q.shape), storage
    if layout[0] == "columns":
        storage = torch.randn(tokens, layout[1], dtype=q.dtype, device=q.device)
        view = storage[:, : heads * dim].view(q.shape)
    else:
        storage = torch.randn(tokens, heads, dim + layout[1], dtype=q.dtype, device=q.device)
        view = storage[:, :, :dim]
    view.copy_(q)
    return view, storage


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    ("heads", "heads_kv", "dim", "dtype", "q_lens", "cache_lens", "semantics", "layout", "key"),
    [
        pytest.param(
            24, 4, 256, torch.bfloat16, [300, 3, 0, 130], [900, 1500, 64, 131], {},
            ("columns", 8192), "gqa_paged_ws_kernel", marks=pytest.mark.smoke, id="qwen3.8-27b",
        ),
        pytest.param(
            32, 4, 128, torch.float16, [1], [4096], {}, ("padded-heads", 8),
            "gqa_decode_paged_bs1_kernel", marks=pytest.mark.smoke, id="batch-1-decode",
        ),
        pytest.param(
            24, 4, 256, torch.bfloat16, [600, 5], [700, 40], {}, ("columns", 6148),
            "gqa_paged_ws_kernel", marks=pytest.mark.full, id="unaligned-tokens",
        ),
        pytest.param(
            24, 4, 256, torch.bfloat16, [600, 5], [700, 40], {}, ("padded-heads", 8),
            "gqa_paged_ws_kernel", marks=pytest.mark.full, id="padded-heads",
        ),
        pytest.param(
            24, 4, 256, torch.bfloat16, [600, 5], [700, 40], {}, ("expanded",),
            "gqa_paged_ws_kernel", marks=pytest.mark.full, id="expanded",
        ),
        pytest.param(
            24, 4, 256, torch.bfloat16, [600, 5], [700, 40],
            {"pos_encoding_mode": "rope", "rotary_dim": 64}, ("columns", 8192),
            "gqa_paged_ws_kernel", marks=pytest.mark.full, id="rope",
        ),
        pytest.param(
            32, 1, 256, torch.bfloat16, [400, 3], [500, 30], {}, ("columns", 8448),
            "gqa_paged_ws_kernel", marks=pytest.mark.full, id="flat-rows",
        ),
        pytest.param(
            24, 4, 256, torch.bfloat16, [1] * 8, [300] * 8, {}, ("columns", 8192),
            "gqa_paged_varlen_kernel", marks=pytest.mark.full, id="split-scan",
        ),
        pytest.param(
            32, 8, 128, torch.bfloat16, [1] * 4, [300] * 4, {"cache_dtype": torch.float8_e4m3fn},
            ("columns", 8192), "gqa_paged_fp8_decode_kernel", marks=pytest.mark.full,
            id="fp8-decode",
        ),
        pytest.param(
            32, 8, 128, torch.bfloat16, [600, 5], [700, 40], {"cache_dtype": torch.float8_e4m3fn},
            ("columns", 8192), "gqa_paged_fp8_ws_kernel", marks=pytest.mark.full,
            id="fp8-prefill",
        ),
    ],
)  # fmt: skip
def test_gqa_paged_reads_projection_columns(
    heads, heads_kv, dim, dtype, q_lens, cache_lens, semantics, layout, key
) -> None:
    """q whose tokens' heads are packed and lie a whole number of 16-byte vectors apart is
    read in place by the warp-specialized kernel; every other layout, the RoPE preparation,
    flat rows and the other kernels read a packed copy: the output matches the reference,
    and q's storage stays as it was.

    The cells: four ragged requests whose q is the first 6144 columns of Qwen3.8-27B's fused
    8192-wide q/k/v projection, among them short requests over long caches, whose key tiles
    split between the warpgroups, which load their queries directly rather than by TMA;
    tokens 6148 elements apart, off 16-byte alignment; heads 264
    elements apart, so a token's heads are not packed; an expanded q, its tokens zero
    elements apart; RoPE preparation, which rotates a packed copy; flat rows over one KV
    head; decode tokens of many requests, through the split scan; one request's decode
    token with its heads 136 elements apart, through the batch-1 decode kernel; and an FP8
    cache's decode and prefill, through the FP8 kernels.
    """
    workload = _paged(
        q_lens, cache_lens, 64, dtype, heads=heads, heads_kv=heads_kv, dim=dim, **semantics
    )
    op = GQAPagedFwdOp(is_causal=True, **{k: v for k, v in semantics.items() if k != "cache_dtype"})
    q, *rest = _poisoned_inputs(workload)
    q, storage = _laid_out_queries(q, layout)
    before = storage.clone()
    k_cache, block_table, cos, sinks = rest[0], rest[2], rest[8], rest[10]
    call = op._paged_call(q, k_cache, block_table, rest[6], cos, sinks)
    assert op.key_for("gqa_paged", dataclasses.replace(call, sm_count=132)) == key

    def run(*args):
        out = op(*args)
        assert torch.equal(storage, before)
        return out

    TestBase.check(workload, op, q, *rest, runs=run)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("heads", "heads_kv", "dim", "batch", "q_len", "rope", "served"),
    [
        pytest.param(24, 4, 128, 9, 22, False, True, id="one-wave-a-third-full"),
        pytest.param(28, 4, 128, 4, 40, False, False, id="under-a-third-full"),
        pytest.param(24, 4, 128, 8, 22, False, False, id="partly-idle-wave"),
        pytest.param(24, 4, 128, 22, 22, False, False, id="thin-tiles-two-waves"),
        pytest.param(24, 4, 128, 22, 44, False, True, id="fuller-tiles-two-waves"),
        pytest.param(24, 4, 256, 8, 22, False, True, id="dim-256-one-wave"),
        pytest.param(24, 4, 256, 7, 24, False, False, id="dim-256-partial-wave"),
        pytest.param(24, 4, 128, 6, 22, True, True, id="rope-half-a-wave"),
        pytest.param(24, 4, 128, 5, 22, True, False, id="rope-under-half-a-wave"),
        pytest.param(24, 4, 64, 11, 22, False, False, id="dimension-64"),
    ],
)
def test_gqa_paged_split_group_dispatch(
    heads: int, heads_kv: int, dim: int, batch: int, q_len: int, rope: bool, served: bool
) -> None:
    """The warp-specialized kernel serves a split query group only where its head-group tiles
    are full enough and fill the waves of multiprocessors they take, and not at dimension 64,
    leaving the rest to the key splits of ``GQAPagedFwdKernel``. The packings sit on either
    side of those bounds for the 132 multiprocessors of an H200, which the call states."""
    from tileops._kernels.attention.gqa.paged_ws import GQAPagedFwdWSKernel

    semantics = {"pos_encoding_mode": "rope", "rotary_dim": 64} if rope else {}
    meta = torch.device("meta")
    q = torch.empty(batch * q_len, heads, dim, dtype=torch.bfloat16, device=meta)
    k_pages = torch.empty(batch * 8, 64, heads_kv, dim, dtype=torch.bfloat16, device=meta)
    table = torch.empty(batch, 8, dtype=torch.int32, device=meta)
    rope_cos = torch.empty(512, 32, device=meta) if rope else None
    call = GQAPagedFwdOp(is_causal=True, **semantics)._paged_call(q, k_pages, table, None, rope_cos)
    call = dataclasses.replace(call, sm_count=132)
    refusal = GQAPagedFwdWSKernel.refusal(call)
    assert (refusal is None) == served, refusal


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "batch",
    [
        pytest.param(64, marks=pytest.mark.smoke, id="batch-64"),
        # A 508-byte tile prefix: the planner rounds it to the handover's 512 bytes.
        pytest.param(42, marks=pytest.mark.full, id="batch-42"),
    ],
)
def test_gqa_paged_warp_specialized_split_handover_beside_tile_prefix(batch: int) -> None:
    """Split decodes beside a long request, the requests ranked: a CTA claims several items
    after its consumers have written a split handover, so the producer still reads the tile
    prefix while they write it, and each keeps its own shared bytes."""
    from tileops._kernels.attention.gqa.paged_ws import GQAPagedFwdWSKernel

    q_lens = [32 * batch - (batch - 1)] + [1] * (batch - 1)
    workload = _paged(q_lens, [4096] * batch, 64, torch.float16)
    op, inputs = GQAPagedFwdOp(is_causal=True), _poisoned_inputs(workload)
    kernel = _kernel(op, inputs)
    assert isinstance(kernel, GQAPagedFwdWSKernel) and kernel.split_small
    _check(op, workload, inputs)


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "batch",
    [
        pytest.param(300, marks=pytest.mark.smoke, id="batch-300"),
        # A 1004-byte tile prefix, which the planner merges with free space to fit the anchor.
        pytest.param(250, marks=pytest.mark.full, id="batch-250"),
    ],
)
def test_gqa_paged_warp_specialized_value_anchor_beside_tile_prefix(batch: int) -> None:
    """At dimension 256 the consumers store to a value-wait anchor on every key tile while the
    producer reads the tile prefix of decodes beside one long request: each keeps its own
    shared bytes. Sinks build the shared scan, so the anchor is the only consumer-written
    buffer."""
    from tileops._kernels.attention.gqa.paged_ws import GQAPagedFwdWSKernel

    q_lens = [32 * batch - (batch - 1)] + [1] * (batch - 1)
    cache_lens = [q_lens[0] + 64] + [1024] * (batch - 1)
    width = -(-max(cache_lens) // 64)
    # Tables are permutations of one small pool, so every request stays a few pages.
    workload = GQAPagedFwdWorkload(
        8, 2, 256, q_lens, cache_lens, 64, width, width + 16, torch.float16, has_sinks=True
    )
    op, inputs = GQAPagedFwdOp(is_causal=True), workload.gen_inputs()
    kernel = _kernel(op, inputs)
    assert isinstance(kernel, GQAPagedFwdWSKernel) and not kernel.split_small
    _check(op, workload, inputs)


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
        pytest.param(512, False, id="four-splits"),
        pytest.param(4096, True, id="ctx-reversed-pages"),
    ],
)
def test_gqa_paged_decode_bs1_tiers(cache_len: int, reverse_pages: bool) -> None:
    """A 4-split and a 32-split table, the second with reversed pages, a page translation
    that changes the output."""
    torch.manual_seed(0)
    workload = _decode(1, 32, 4, [cache_len], 128, 256)
    inputs = list(workload.gen_inputs())
    if reverse_pages:
        inputs[3] = inputs[3].flip(-1).contiguous()
    op = GQAPagedFwdOp(is_causal=True)
    _check(op, workload, inputs)


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    ("heads", "heads_kv", "dim", "page_size", "pages", "cache_len", "dtype"),
    [
        pytest.param(
            24, 4, 256, 64, 65, 4097, torch.bfloat16, marks=pytest.mark.smoke, id="bf16-full-table"
        ),
        pytest.param(
            24, 4, 256, 64, 65, 4097, torch.float16, marks=pytest.mark.smoke, id="fp16-full-table"
        ),
        pytest.param(
            24,
            4,
            256,
            64,
            65,
            513,
            torch.bfloat16,
            marks=pytest.mark.full,
            id="bf16-short-of-table",
        ),
        pytest.param(
            32, 8, 128, 64, 64, 1000, torch.bfloat16, marks=pytest.mark.full, id="bf16-head-dim-128"
        ),
        pytest.param(
            24, 4, 256, 128, 16, 1000, torch.bfloat16, marks=pytest.mark.full, id="128-row-pages"
        ),
        pytest.param(
            24, 4, 256, 64, 15, 900, torch.bfloat16, marks=pytest.mark.full, id="short-table"
        ),
        pytest.param(
            32, 8, 128, 64, 2, 127, torch.bfloat16, marks=pytest.mark.full, id="two-page-table"
        ),
        pytest.param(
            32, 8, 128, 64, 8, 300, torch.float16, marks=pytest.mark.full, id="fp16-short-table"
        ),
        pytest.param(
            32, 8, 128, 64, 1, 1, torch.float16, marks=pytest.mark.full, id="one-page-table"
        ),
    ],
)
def test_gqa_paged_decode_bs1_wider_region(
    heads: int,
    heads_kv: int,
    dim: int,
    page_size: int,
    pages: int,
    cache_len: int,
    dtype: torch.dtype,
) -> None:
    """The batch-1 kernel, whose warp-specialized producer feeds SM90 WGMMA through TMA,
    serves bfloat16 and head dimension 256: Qwen3.8-27B's decode attention over a full and
    an eighth-full table, bfloat16 at head dimension 128, 128-row pages at head dimension
    256, and short tables, down to one page. NaN in every row past the cache keeps a split
    that reads one key too many from passing."""
    torch.manual_seed(0)
    workload = GQAPagedFwdWorkload(
        heads, heads_kv, dim, [1], [cache_len], page_size, pages, pages, dtype
    )
    inputs = _poisoned_inputs(workload)
    op = GQAPagedFwdOp(is_causal=True)
    q, k_cache, _, block_table, _, _, _, k_scale, _, rope_cos, _, sinks = inputs
    call = op._paged_call(q, k_cache, block_table, k_scale, rope_cos, sinks)
    assert op.key_for("gqa_paged", call) == "gqa_decode_paged_bs1_kernel"
    _check(op, workload, inputs)


@pytest.mark.parametrize(
    ("q_lens", "cache_lens", "page_size", "dim", "dtype", "op_kwargs"),
    [
        pytest.param(
            [0, 1, 5, 1],
            [300, 512, 600, 7],
            64,
            128,
            torch.float16,
            {},
            marks=pytest.mark.smoke,
            id="ragged-with-empty",
        ),
        pytest.param(
            [3, 1], [129, 48], 48, 128, torch.bfloat16, {}, marks=pytest.mark.smoke, id="bf16"
        ),
        # Head dimension 64, where the key tile is gathered at a width the page does not hold.
        pytest.param(
            [1, 1],
            [1000, 513],
            64,
            64,
            torch.float16,
            {"window_size_left": 128},
            marks=pytest.mark.full,
            id="window",
        ),
        pytest.param(
            [7, 3],
            [300, 200],
            128,
            128,
            torch.float16,
            {"is_causal": False, "window_size_left": 64, "window_size_right": 32},
            marks=pytest.mark.full,
            id="both-windows-noncausal",
        ),
        # A page no key tile fits inside: the key tile is gathered row by row.
        pytest.param(
            [1, 1],
            [400, 131],
            65,
            128,
            torch.float16,
            {},
            marks=pytest.mark.full,
            id="page-no-tile-fits",
        ),
        pytest.param(
            [1],
            [300],
            32,
            256,
            torch.bfloat16,
            {},
            marks=pytest.mark.full,
            id="batch-1-head-dim-256-on-32-row-pages",
        ),
        # The only shape where the causal bound and the window bound both bind: a request
        # with one query token sees its whole cache, so its causal bound masks nothing.
        pytest.param(
            [6, 1, 0],
            [300, 500, 128],
            64,
            128,
            torch.float16,
            {"window_size_left": 64},
            marks=pytest.mark.full,
            id="causal-window-multi",
        ),
        pytest.param(
            [2, 0, 9],
            [700, 64, 33],
            16,
            128,
            torch.float16,
            {"softcap": 30.0},
            marks=pytest.mark.full,
            id="softcap",
        ),
        # The three shapes that isolate one term of the key-tile classification each. A tile
        # is masked per element only where a bound cuts it, so a term that never fires on any
        # case is a term no test pays for.
        # Causal only: a cache length the key tile divides, so the cache end never cuts and
        # the causal bound is the only term that can.
        pytest.param(
            [5, 2],
            [512, 256],
            64,
            128,
            torch.float16,
            {},
            marks=pytest.mark.full,
            id="causal-aligned-cache",
        ),
        # Neither causal nor windowed: the cache end is then the only term that can cut.
        pytest.param(
            [3, 2],
            [250, 130],
            64,
            128,
            torch.float16,
            {"is_causal": False},
            marks=pytest.mark.full,
            id="cache-end-only",
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
            marks=pytest.mark.full,
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
    """Calls the decode region does not serve: ragged lengths, windows, and odd pages, among
    them a batch-1 head-dimension-256 decode on pages that hold no whole 64-row key tile."""
    width = -(-max(cache_lens) // page_size)
    workload = GQAPagedFwdWorkload(
        32, 8, dim, q_lens, cache_lens, page_size, width, len(q_lens) * width, dtype, **op_kwargs
    )
    _check(GQAPagedFwdOp(**{"is_causal": True, **op_kwargs}), workload, workload.gen_inputs())


@pytest.mark.sm90
@pytest.mark.cuda_graph(op="GQAPagedFwdOp")
@pytest.mark.parametrize(
    "batch, longest, call",
    [
        pytest.param(2, 4096, {"heads": 32, "dim": 128}, marks=pytest.mark.smoke, id="dynamic"),
        pytest.param(
            1, 4096, {"heads": 32, "dim": 128}, marks=pytest.mark.full, id="single-request"
        ),
        pytest.param(
            1, 4096, {"heads": 24, "dim": 256, "dtype": torch.bfloat16}, marks=pytest.mark.full,
            id="qwen3.8-27b",
        ),
        pytest.param(
            1, 4128, {"heads": 24, "dim": 256, "dtype": torch.bfloat16}, marks=pytest.mark.full,
            id="qwen3.8-27b-4128",
        ),
    ],
)  # fmt: skip
def test_gqa_paged_graph_replay(batch: int, longest: int, call: dict) -> None:
    """Dispatch is static; packed lengths and cache lengths remain replay inputs.

    A single request holds the one packing of its one query token, and skips the captured
    cache length, at which its cache lengths would hold their captured values.
    """
    workload = _decode(batch, heads_kv=4, cache_lens=[longest, 64][:batch], page_size=64, **call)
    inputs = workload.gen_inputs()
    inputs[4][0] = 128  # A mostly empty table must also be safe on the split path.
    workload.cache_lens[0] = 128
    op = GQAPagedFwdOp(is_causal=True)
    graph = CapturedGraph((op, inputs))
    graph.renew(workload.gen_inputs())
    pages = torch.randperm(inputs[1].shape[0], device=inputs[3].device)
    inputs[3].copy_(pages[inputs[3]])
    # Same tensor addresses and shapes, different sequence boundaries. The dynamic
    # path must not keep the uniform packing it saw during warmup or capture.
    if batch == 2:
        inputs[5][1] = 0
        workload.q_lens = [0, 2]
        inputs[4][-1] = 63
        workload.cache_lens[-1] = 63
    v_cache, block_table, page_size = inputs[2], inputs[3][0], inputs[2].shape[1]
    live_values = v_cache.clone()
    for length in (128, 2048, longest, 63, 0)[2 - batch :]:
        inputs[4][0] = length
        workload.cache_lens[0] = length
        # Slots past the cache keep stale contents; a non-finite one must not reach the output.
        v_cache.copy_(live_values)
        slots = torch.arange(length, block_table.numel() * page_size, device=v_cache.device)
        v_cache[block_table[slots // page_size], slots % page_size] = float("nan")
        graph.replay(held=[inputs[5]] if batch == 1 else [])
        TestBase.check(workload, op, *inputs, runs=lambda *args: graph.outputs[0])


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


@pytest.mark.smoke
@pytest.mark.sm89
@pytest.mark.in_tree_kernels
def test_gqa_paged_autotune_supplies_packed_inputs(monkeypatch) -> None:
    """Guard the supplier's three missing scale inputs through real Op autotuning.

    SM89 selects the ordinary paged kernel here; SM90 can select another implementation.
    """
    from tileops._kernels.attention.gqa.paged import GQAPagedFwdKernel

    production_configs = GQAPagedFwdKernel.autotune_configs.fget

    def one_config(entry):
        candidates = production_configs(entry)
        return [
            next(
                config
                for config in candidates
                if config["block_M"] == 64
                and config["block_N"] == 64
                and config["num_stages"] == 1
                and config["threads"] == 128
            )
        ]

    # Keep the initial seed in the same single-stage compilation path as the sweep.
    monkeypatch.setattr(GQAPagedFwdKernel, "default_config", property(lambda e: one_config(e)[0]))
    monkeypatch.setattr(GQAPagedFwdKernel, "autotune_configs", property(one_config))
    workload = GQAPagedFwdWorkload(32, 8, 128, [160, 160], [700, 450], 64, 11, 22, torch.float16)
    op = GQAPagedFwdOp(is_causal=True)
    op.request_tune()
    TestBase.check(workload, op, *workload.gen_inputs())
    entries = [
        entry for interface in op.interfaces for entry in op._entries_built(interface).values()
    ]
    assert entries and all(isinstance(entry, GQAPagedFwdKernel) for entry in entries)


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


# A chunk count a case does not bound.
ANY = (1, None)


def _fp8_op(workload: GQAPagedFwdWorkload, **semantics) -> GQAPagedFwdOp:
    return GQAPagedFwdOp(out_dtype=workload.out_dtype, **{"is_causal": True, **semantics})


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    ("q_lens", "cache_lens", "page_size", "dtype", "semantics", "tile_rows", "splits"),
    [
        # One copy a key tile; the cache end cuts the last tile of the first request.
        pytest.param(
            [1, 1],
            [513, 1000],
            64,
            torch.float16,
            {},
            8,
            ANY,
            marks=pytest.mark.smoke,
            id="f16-tensor-scale-cut",
        ),  # fmt: skip
        # Four 16-row copies a key tile; causal rows of three query tokens a request.
        pytest.param(
            [3, 3],
            [700, 333],
            16,
            torch.bfloat16,
            {"per_request_scales": True},
            16,
            ANY,
            marks=pytest.mark.smoke,
            id="bf16-request-scales-pages-16",
        ),
        # An FP8 query, its score descale under a softcap and a left window.
        pytest.param(
            [2, 2],
            [1000, 300],
            128,
            FP8,
            {"per_request_scales": True, "softcap": 30.0, "window_size_left": 200},
            8,
            ANY,
            marks=pytest.mark.smoke,
            id="fp8-softcap-window",
        ),
        # Bidirectional rows: each query sees every key of its request.
        pytest.param(
            [2, 2],
            [700, 300],
            64,
            torch.float16,
            {"is_causal": False},
            8,
            ANY,
            marks=pytest.mark.full,
            id="f16-bidirectional",
        ),
        # Rows on M with an FP8 query, request scales, a softcap and a left window, bidirectional.
        pytest.param(
            [16, 16],
            [900, 400],
            128,
            FP8,
            {
                "per_request_scales": True,
                "softcap": 30.0,
                "window_size_left": 200,
                "is_causal": False,
            },
            64,
            ANY,
            marks=pytest.mark.full,
            id="fp8-rows-on-m-softcap-window-bidirectional",
        ),
        # Head dimension 64 with the keys on M, a one-key cache among them.
        pytest.param(
            [1, 1, 1],
            [700, 1, 1500],
            64,
            torch.float16,
            {"heads": 64, "dim": 64},
            8,
            ANY,
            marks=pytest.mark.full,
            id="f16-d64-keys-on-m",
        ),
        # Head dimension 64 with the rows on M, two 32-row pages a key tile.
        pytest.param(
            [8, 8],
            [900, 300],
            32,
            torch.bfloat16,
            {"heads": 64, "dim": 64},
            64,
            ANY,
            marks=pytest.mark.full,
            id="bf16-d64-rows-on-m",
        ),
        # One-row pages at head dimension 128 and two-row pages at 64: each page is a TMA
        # box of fewer rows than the 8-row swizzle atom of the tile it lands in.
        pytest.param(
            [1, 1],
            [300, 517],
            1,
            torch.float16,
            {},
            8,
            ANY,
            marks=pytest.mark.full,
            id="f16-pages-1",
        ),  # fmt: skip
        pytest.param(
            [8, 8],
            [301, 500],
            2,
            torch.bfloat16,
            {"heads": 64, "dim": 64},
            64,
            ANY,
            marks=pytest.mark.full,
            id="bf16-d64-pages-2",
        ),
        # A right window and a bidirectional mask, with a request of no query token.
        pytest.param(
            [3, 0, 2],
            [600, 64, 300],
            64,
            torch.float16,
            {"window_size_left": 100, "window_size_right": 0, "is_causal": False},
            8,
            ANY,
            marks=pytest.mark.full,
            id="f16-right-window-empty-request",
        ),
        # 16-row tiles: the first request's 36 rows span three of them.
        pytest.param(
            [9, 1, 1, 1],
            [600, 100, 1000, 70],
            64,
            torch.float16,
            {},
            16,
            ANY,
            marks=pytest.mark.full,
            id="f16-request-spans-tiles",
        ),  # fmt: skip
        # The left window hides the first key tile of a chunk from the later rows, whose maximum
        # stays at its -1e38 floor; a score scale of 8 overflows float32 if it scales that floor.
        pytest.param(
            [4, 4],
            [1000, 1000],
            64,
            torch.float16,
            {"sm_scale": 8.0, "window_size_left": 102},
            16,
            ANY,
            marks=pytest.mark.full,
            id="f16-keys-on-m-large-scale-hidden-tile",
        ),
        pytest.param(
            [16, 16],
            [1000, 1000],
            64,
            torch.float16,
            {"sm_scale": 8.0, "window_size_left": 92},
            64,
            ANY,
            marks=pytest.mark.full,
            id="f16-rows-on-m-large-scale-hidden-tile",
        ),
        # A KV head stored as zeros quantizes by amax / 448 to a zero scale: every score of
        # request 0's first head is zero, so its rows average the values they see.
        pytest.param(
            [1, 1],
            [1000, 1000],
            64,
            torch.float16,
            {"per_request_scales": True, "zero_key_scale": True},
            8,
            ANY,
            marks=pytest.mark.full,
            id="f16-keys-on-m-zero-key-scale",
        ),
        # A table of 64 key tiles takes several chunks: a 70-key request beside a 4096-key one
        # leaves most of its chunks without a key tile, and they must weigh nothing in the
        # combine.
        pytest.param(
            [16, 16],
            [70, 4096],
            64,
            torch.float16,
            {},
            64,
            (2, None),
            marks=pytest.mark.full,
            id="f16-rows-on-m-empty-chunks",
        ),  # fmt: skip
        # A table too short for a chunk to outlast filling its ring stays whole, so the tile
        # writes its rows without the combine.
        pytest.param(
            [1, 1],
            [200, 256],
            64,
            torch.float16,
            {},
            8,
            (1, 1),
            marks=pytest.mark.full,
            id="f16-keys-on-m-unsplit",
        ),  # fmt: skip
        pytest.param(
            [16, 16],
            [200, 256],
            64,
            torch.float16,
            {},
            64,
            (1, 1),
            marks=pytest.mark.full,
            id="f16-rows-on-m-unsplit",
        ),  # fmt: skip
    ],
)
def test_gqa_paged_fp8_decode(
    q_lens: list[int],
    cache_lens: list[int],
    page_size: int,
    dtype: torch.dtype,
    semantics: dict,
    tile_rows: int,
    splits: tuple[int, int | None],
) -> None:
    """SM90 FP8 decode widens each stored tile and dequantizes by the scales, holding the rows
    on M at a full 64-row tile and the keys on M below it, its chunk count within *splits*; a
    NaN row past a cache, or a value a causal row must not see, must not reach the output."""
    from tileops._kernels.attention.gqa.paged_fp8_decode import GQAPagedFP8DecodeKernel

    out_dtype = torch.bfloat16 if dtype == FP8 else None
    workload_semantics = {k: v for k, v in semantics.items() if k != "zero_key_scale"}
    workload = _paged(
        q_lens, cache_lens, page_size, dtype, cache_dtype=FP8, out_dtype=out_dtype,
        **workload_semantics,
    )  # fmt: skip
    op_semantics = {
        k: v
        for k, v in workload_semantics.items()
        if k not in ("per_request_scales", "heads", "dim")
    }
    op = _fp8_op(workload, **op_semantics)
    inputs = _causal_canary(workload, _poisoned_inputs(workload))
    if semantics.get("zero_key_scale"):
        inputs[7][0, 0] = 0.0
    _check(op, workload, inputs)
    kernel = _kernel(op, inputs)
    assert isinstance(kernel, GQAPagedFP8DecodeKernel)
    assert kernel.block_m == tile_rows
    least, most = splits
    chunks = kernel.chunks(inputs[0].shape[0])
    assert least <= chunks <= (most or chunks)


# H200's shared memory a block may take, in bytes.
_H200_BLOCK_SHARED = 232448


@pytest.mark.smoke
@pytest.mark.parametrize(("batch", "served"), [(44287, True), (44288, False)])
def test_gqa_paged_fp8_decode_refuses_a_tile_prefix_past_shared_memory(
    batch: int, served: bool
) -> None:
    """The FP8 decode kernel holds a per-request tile prefix in shared memory beside its tiles.
    An 8-row tile at head dimension 128 takes 54 KB beside the prefix, so a block of 227 KB
    serves at most 44287 requests and leaves larger batches to another kernel."""
    from tileops._kernels.attention.gqa.paged_fp8_decode import GQAPagedFP8DecodeKernel

    meta = torch.device("meta")
    q = torch.empty(batch, 8, 128, dtype=torch.float16, device=meta)
    k_pages = torch.empty(batch, 64, 1, 128, dtype=FP8, device=meta)
    table = torch.empty(batch, 1, dtype=torch.int32, device=meta)
    k_scale = torch.empty(1, device=meta)
    call = GQAPagedFwdOp(is_causal=True)._paged_call(q, k_pages, table, k_scale)
    call = dataclasses.replace(call, smem_budget=_H200_BLOCK_SHARED)
    refusal = GQAPagedFP8DecodeKernel.refusal(call)
    assert (refusal is None) == served, refusal


@pytest.mark.parametrize(
    ("page_size", "served"),
    [
        pytest.param(1, False, marks=pytest.mark.smoke, id="one-row-refused"),
        pytest.param(2, True, marks=pytest.mark.full, id="two-row-served"),
    ],
)
def test_gqa_paged_fp8_decode_refuses_pages_under_128_bytes(page_size: int, served: bool) -> None:
    """Each page's TMA copy lands at a multiple of its bytes in shared memory, which TMA needs
    128-byte aligned, so at head dimension 64 the FP8 decode kernel serves two-row pages and
    leaves one-row pages to another kernel."""
    from tileops._kernels.attention.gqa.paged_fp8_decode import GQAPagedFP8DecodeKernel

    meta = torch.device("meta")
    q = torch.empty(2, 8, 64, dtype=torch.float16, device=meta)
    k_pages = torch.empty(2, page_size, 1, 64, dtype=FP8, device=meta)
    table = torch.empty(2, 1, dtype=torch.int32, device=meta)
    k_scale = torch.empty(1, device=meta)
    call = GQAPagedFwdOp(is_causal=True)._paged_call(q, k_pages, table, k_scale)
    call = dataclasses.replace(call, smem_budget=_H200_BLOCK_SHARED)
    refusal = GQAPagedFP8DecodeKernel.refusal(call)
    assert (refusal is None) == served, refusal


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    ("q_lens", "cache_lens", "page_size", "dtype", "out_dtype", "semantics", "split_small"),
    [
        # A 16-bit query: the producer warpgroup converts both tiles, 64-row key tiles; the
        # one-token request over 16 of them splits its tiles between the warpgroups.
        pytest.param(
            [200, 1],
            [700, 1000],
            64,
            torch.float16,
            None,
            {},
            True,
            marks=pytest.mark.smoke,
            id="f16-convert-k-v",
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
            marks=pytest.mark.smoke,
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
            marks=pytest.mark.smoke,
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
            marks=pytest.mark.smoke,
            id="bf16-shared-memory-limit",
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
            marks=pytest.mark.smoke,
            id="fp8-out-f16-bidirectional",
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
            marks=pytest.mark.full,
            id="bf16-pages-48",
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
            marks=pytest.mark.full,
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


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    ("mode", "semantics"),
    [
        pytest.param("capped", {}, marks=pytest.mark.smoke, id="one-run-over-capped-workspace"),
        pytest.param("runs", {"softcap": 30.0}, marks=pytest.mark.full, id="softcap-runs"),
        pytest.param("graph", {}, marks=pytest.mark.full, id="graph"),
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
    assert kernel.row_sums
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
            marks=pytest.mark.smoke,
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
            marks=pytest.mark.smoke,
            id="fp8-sinks-request-scales-1x1",
        ),
        # Both window bounds, over enough row tiles that the key range is not split.
        pytest.param(
            [300, 260],
            [700, 333],
            64,
            torch.float16,
            {"is_causal": False, "window_size_left": 17, "window_size_right": 3},
            marks=pytest.mark.smoke,
            id="two-sided-window-unsplit",
        ),
        # Dimension 256 over rows averaging a 128-row tile a request: two warpgroups share
        # each converted key tile.
        pytest.param(
            [300, 1],
            [700, 333],
            64,
            torch.float16,
            {"dim": 256},
            marks=pytest.mark.full,
            id="dim-256-filled",
        ),
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


@pytest.mark.smoke
def test_fragmented_block_table() -> None:
    batch, pages_per_req = 4, 8
    pool_pages = batch * pages_per_req

    disjoint = make_fragmented_block_table(batch, pages_per_req, pool_pages)
    assert disjoint.shape == (batch, pages_per_req)
    assert sorted(disjoint.flatten().tolist()) == list(range(pool_pages))
    assert not torch.equal(
        disjoint,
        torch.arange(pool_pages, dtype=torch.int32, device=run_device()).reshape(batch, -1),
    )

    shared = make_fragmented_block_table(batch, pages_per_req, pages_per_req).tolist()
    for row in shared:
        assert sorted(row) == list(range(pages_per_req))
    assert not all(row == sorted(row) for row in shared)

    assert torch.equal(disjoint, make_fragmented_block_table(batch, pages_per_req, pool_pages))

    case = GQAPagedFwdWorkload(16, 4, 128, [64, 64], [192, 192], 64, 3, 6, torch.float16)
    table = case.gen_inputs()[3]
    assert sorted(table.flatten().tolist()) == list(range(6))
