import pytest
import torch

from tests.cuda_graph_contract import check_replay
from tests.workload_test_base import TestBase
from tileops._kernels.constants import SM_RESIDENT_BLOCKS
from tileops._kernels.indexing import EmbeddingCall
from tileops.backend import OpNotAvailableError
from tileops.indexing import EmbeddingFwdOp
from workloads.device import run_device
from workloads.indexing import EmbeddingWorkload
from workloads.workload_base import manifest_call


class EmbeddingTest(EmbeddingWorkload, TestBase):
    pass


def _test(S, V, D, index_dtype, dtype, **params) -> EmbeddingTest:
    return EmbeddingTest(
        manifest_call("EmbeddingFwdOp", {"I": index_dtype, "T": dtype}, S=S, V=V, D=D, **params)
    )


_SMOKE = pytest.mark.smoke
_FULL = pytest.mark.full
# One case per dtype and per tiling path. Every case reads the table's last row.
_CASES = (
    # An odd row of 2-byte elements: 2-byte vectors, which cp.async does not move.
    pytest.param([7], 50, 1001, "int64", "bfloat16", {}, marks=_SMOKE, id="two-byte-vectors"),
    pytest.param([129], 300, 1152, "int64", "float32", {}, marks=_SMOKE, id="fp32"),
    # Fewer ids than a tile of narrow rows holds: a partial row tile. The gradient-only
    # params ride on it.
    pytest.param(
        [3, 5],
        700,
        64,
        "int32",
        "float16",
        {"padding_idx": -1, "sparse": True},
        marks=_SMOKE,
        id="fp16-row-tail",
    ),  # fmt: skip
    pytest.param([500], 129, 6, "int64", "int64", {}, marks=_SMOKE, id="int64-table"),
    # 12-byte rows: 4-byte vectors.
    pytest.param([2, 9], 40, 3, "int32", "int32", {}, marks=_SMOKE, id="int32-table"),
    # 24-byte rows: 8-byte vectors.
    pytest.param([5, 7], 90, 3, "int32", "int64", {}, marks=_FULL, id="eight-byte-vectors"),
    # 1021 vectors, a prime: no thread count divides them, so the last pass is guarded.
    pytest.param([2], 1, 8168, "int64", "bfloat16", {}, marks=_FULL, id="guarded-pass"),
    # 1025-vector rows: two column tiles, the second holding one vector; more than 2**20
    # tiles, so the grid's y axis holds the rest.
    pytest.param([2**19 + 1], 4, 1025, "int64", "float16", {}, marks=_FULL, id="column-tiles"),
    pytest.param([], 10, 128, "int64", "bfloat16", {}, marks=_FULL, id="rank-0"),
    pytest.param([0, 4], 10, 128, "int64", "bfloat16", {}, marks=_FULL, id="empty"),
    pytest.param([3], 10, 0, "int64", "bfloat16", {}, marks=_FULL, id="zero-width"),
    # A table of more rows than an int32 extent holds.
    pytest.param([3], 2**31 + 1, 1, "int64", "bfloat16", {}, marks=_FULL, id="rows-past-int32"),
    # int32 ids whose last row starts at element 2**31: the id stays int32 until the table
    # offset is formed, which must widen it before multiplying by the row width.
    pytest.param(
        [3], 2**20 + 1, 2048, "int32", "bfloat16", {}, marks=_FULL, id="int32-ids-past-int32-offset"
    ),
    # Column offsets past 2**31 elements in one row, on 16-byte vectors; the last column
    # tile holds one vector.
    pytest.param([2], 1, 2**31 + 8, "int64", "float16", {}, marks=_FULL, id="widest-row"),
)


def _column_tiles(cfg: dict, S: list, D: int) -> bool:
    """Two column tiles, the second one vector, and more tiles than grid x's 2**20."""
    vectors = D // cfg["vec"]
    col_tiles = -(-vectors // cfg["block_vectors"])
    tiles = -(-S[0] // cfg["block_rows"]) * col_tiles
    return col_tiles == 2 and vectors % cfg["block_vectors"] == 1 and tiles > 2**20


# The kernel config each path-named case relies on, so a retune fails the case instead of
# moving it off its path.
_ON_PATH = {
    "guarded-pass": lambda cfg, S, D: cfg["block_rows"] * cfg["block_vectors"] % cfg["threads"]
    != 0,
    "column-tiles": _column_tiles,
}


@pytest.mark.parametrize("S, V, D, index_dtype, dtype, params", _CASES)
def test_embedding(S, V, D, index_dtype, dtype, params, request) -> None:
    test = _test(S, V, D, index_dtype, dtype, **params)
    op = EmbeddingFwdOp(**test.arguments())
    on_path = _ON_PATH.get(request.node.callspec.id)
    if on_path is not None:
        call = EmbeddingCall(
            dim=D,
            dtype=getattr(torch, dtype),
            index_dtype=getattr(torch, index_dtype),
            device=torch.device(run_device()),
        )
        cfg = op.kernel_for("embedding", lambda: call).config
        assert on_path(cfg, S, D), cfg
    test.check(op, *test.gen_inputs())


@pytest.mark.sm90
@pytest.mark.cuda_graph(op="EmbeddingFwdOp")
@pytest.mark.parametrize(
    "S, V, D, T",
    [
        pytest.param([1], 129280, 6, "int32", marks=_SMOKE, id="deepseek-v4-flash-hash-routing"),
        pytest.param([1, 1], 129280, 4096, "bfloat16", marks=_FULL, id="deepseek-v4-flash"),
        pytest.param([1], 248320, 5120, "bfloat16", marks=_FULL, id="qwen3.8-27b"),
    ],
)
def test_embedding_graph_replay(S, V, D, T) -> None:
    """One decode token's int64 id looked up in a table. Every draw names the table's last row,
    so the replay reads the row before it."""
    test = _test(S, V, D, "int64", T)
    ids, table = test.gen_inputs()
    check_replay(test, EmbeddingFwdOp(**test.arguments()), (ids - 1, table))


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_embedding_wave_limit_counts_whole_warps() -> None:
    """A 1152-wide float32 row takes 144 threads, five warps, of which an SM holds 12 CTAs by
    its warp limit, not the 14 its thread limit alone would give."""
    call = EmbeddingCall(
        dim=1152, dtype=torch.float32, index_dtype=torch.int64, device=torch.device(run_device())
    )
    kernel = EmbeddingFwdOp().kernel_for("embedding", lambda: call)
    cfg = kernel.config
    assert cfg["threads"] == 144, cfg
    assert kernel._resident(cfg, cfg["threads"]) == call.sm_count * (call.threads_per_sm // 32 // 5)


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "D",
    [
        # 256 vectors: the spread program takes two warps; no one-wave program.
        pytest.param(2048, marks=_SMOKE, id="spread-row"),
        # 320 vectors: a one-wave program of two warps, five passes a thread against the
        # spread program's two.
        pytest.param(2560, marks=_FULL, id="five-pass-row"),
    ],
)
def test_embedding_launches_by_id_count(D, monkeypatch) -> None:
    """A one-row tile's call launches the default program within one wave of its CTAs and the
    spread program past it; where the row has a one-wave program, the call launches that,
    asking for the CTAs an SM holds of it, while those fit the call in one wave and the spread
    program's waves outnumber the ratio of passes a thread (five for two)."""
    call = EmbeddingCall(
        dim=D, dtype=torch.bfloat16, index_dtype=torch.int64, device=torch.device(run_device())
    )
    op = EmbeddingFwdOp()
    kernel = op.kernel_for("embedding", lambda: call)
    cfg = kernel.config
    threads, spread, wave = cfg["threads"], cfg["spread_threads"], cfg["wave_threads"]
    one_wave = kernel._resident(cfg, threads)
    launches = [(one_wave, (threads, 0)), (one_wave + 1, (spread, 0))]
    if wave != threads:
        assert (cfg["block_rows"], spread, wave) == (1, 160, 64), cfg
        passes_over = kernel._resident(cfg, spread) * 5 // 2
        two_warp_wave = kernel._resident(cfg, wave)
        held = two_warp_wave // call.sm_count
        if passes_over < two_warp_wave:
            launches += [
                (passes_over, (spread, 0)),
                (passes_over + 1, (wave, held)),
                (two_warp_wave, (wave, held)),
                (two_warp_wave + 1, (spread, 0)),
            ]
        else:
            launches.append((passes_over + 1, (spread, 0)))
    launched = _record_launches(kernel, monkeypatch)
    for tokens, facts in launches:
        launched.clear()
        test = _test([tokens], 50, D, "int64", "bfloat16")
        test.check(op, *test.gen_inputs())
        assert launched[-1] == facts, (tokens, launched)
    for launch_threads, min_blocks in {facts for _, facts in launches if facts[1]}:
        source = kernel.program(kernel._compile, launch_threads, min_blocks).get_kernel_source()
        assert f"__launch_bounds__({launch_threads}, {min_blocks})" in source, source[:2000]


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_embedding_one_wave_program_counts_shared_memory() -> None:
    """Where an SM holds its CTA limit at two-warp CTAs, a 320-vector row has a two-warp
    one-wave program, of which an SM holds as many CTAs as its CTA limit and its shared
    memory, the 5 KiB stage and the per-block reservation each, allow."""
    call = EmbeddingCall(
        dim=2560, dtype=torch.bfloat16, index_dtype=torch.int64, device=torch.device(run_device())
    )
    if not _two_warps_reach_cta_limit(call):
        pytest.skip("an SM of this device does not hold its CTA limit at two-warp CTAs")
    kernel = EmbeddingFwdOp().kernel_for("embedding", lambda: call)
    cfg = kernel.config
    assert (cfg["block_rows"], cfg["threads"], cfg["spread_threads"], cfg["wave_threads"]) == (
        1,
        160,
        160,
        64,
    ), cfg
    stage = 320 * 16
    by_smem = call.smem_per_sm // (stage + call.smem_per_sm - call.smem_budget)
    held = min(SM_RESIDENT_BLOCKS[call.arch], by_smem)
    assert kernel._resident(cfg, 64) == call.sm_count * held


def _record_launches(kernel, monkeypatch) -> list:
    """The facts of every program *kernel* launches from now on, newest last."""
    launched = []
    program = kernel.program

    def recording(build, *facts):
        launched.append(facts)
        return program(build, *facts)

    monkeypatch.setattr(kernel, "program", recording)
    return launched


def _two_warps_reach_cta_limit(call: EmbeddingCall) -> bool:
    """Whether an SM of *call*'s device holds its CTA limit at two-warp CTAs."""
    limit = SM_RESIDENT_BLOCKS.get(call.arch)
    return limit is not None and call.threads_per_sm // limit == 2 * 32


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "D",
    [
        # 512 vectors: eight a thread on two warps, more than a thread's registers hold at the
        # CTA limit.
        pytest.param(4096, id="past-register-bound"),
        # 288 vectors: two warps do not divide them.
        pytest.param(2304, id="undivided-row"),
    ],
)
def test_embedding_has_no_one_wave_program(D) -> None:
    """Where an SM holds its CTA limit at two-warp CTAs, a row that two warps cannot hold
    within a thread's registers at that limit, or do not divide, has no one-wave program."""
    call = EmbeddingCall(
        dim=D, dtype=torch.bfloat16, index_dtype=torch.int64, device=torch.device(run_device())
    )
    if not _two_warps_reach_cta_limit(call):
        pytest.skip("an SM of this device does not hold its CTA limit at two-warp CTAs")
    cfg = EmbeddingFwdOp().kernel_for("embedding", lambda: call).config
    assert cfg["block_rows"] == 1 and cfg["wave_threads"] == cfg["threads"], cfg


@pytest.mark.smoke
def test_embedding_strided_inputs() -> None:
    """Every second element of a wider id tensor and of a wider table."""
    test = _test([4, 33], 1000, 2048, "int64", "bfloat16")
    input, weight = test.gen_inputs()
    wide_input = torch.zeros(4, 66, dtype=input.dtype, device=run_device())
    wide_input[:, ::2] = input
    wide_weight = torch.zeros(1000, 4096, dtype=weight.dtype, device=run_device())
    wide_weight[:, ::2] = weight
    test.check(EmbeddingFwdOp(), wide_input[:, ::2], wide_weight[:, ::2])


@pytest.mark.smoke
@pytest.mark.parametrize("bad", [-1, 1000], ids=["negative", "past-end"])
def test_embedding_validate_inputs_rejects_out_of_range_ids(bad: int) -> None:
    input, weight = _test([4, 33], 1000, 2048, "int64", "bfloat16").gen_inputs()
    input[1, 3] = bad
    with pytest.raises(ValueError, match="input"):
        EmbeddingFwdOp(validate_inputs=True)(input, weight)


@pytest.mark.smoke
def test_embedding_rejects_float_ids() -> None:
    input, weight = _test([4, 33], 1000, 2048, "int64", "bfloat16").gen_inputs()
    with pytest.raises(ValueError, match="dtype is outside"):
        EmbeddingFwdOp()(input.float(), weight)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_embedding_refuses_max_norm() -> None:
    """No in-tree kernel renormalizes the table in place; the op refuses before building."""
    input, weight = _test([129], 300, 1152, "int64", "float32").gen_inputs()
    before = weight.clone()
    op = EmbeddingFwdOp(max_norm=1.0)
    with pytest.raises(OpNotAvailableError, match="max_norm"):
        op(input, weight)
    assert not op._entries_built("embedding")
    assert torch.equal(weight, before)
