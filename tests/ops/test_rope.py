"""Tests for the rotary position embedding ops.

Each op rotates ``query`` and ``key`` at the cache rows ``positions`` names, as vLLM's
``RotaryEmbedding.forward_native(positions, query, key)`` does. The ops are checked against
an independent reference of the vLLM formulas, and against vLLM itself at positions that do
not start at 0.
"""

import pytest
import torch

from tests.workload_test_base import TestBase
from tileops.rope import RoPECosSinFwdOp
from workloads.device import run_device
from workloads.numerics import compare_outputs
from workloads.rope import (
    RoPECosSinCall,
    RoPECosSinViewCall,
    rope_reference,
    rope_tables,
    rope_verification,
)
from workloads.workload_base import manifest_call

_SMOKE, _FULL = pytest.mark.smoke, pytest.mark.full

# Each op's parameters for a small cache, and its class name.
_PARAMS = {
    "RoPEFwdOp": {
        "head_size": 64,
        "rotary_dim": 64,
        "max_position_embeddings": 4096,
        "base": 10000.0,
        "is_neox_style": True,
    },
    "RoPELlama31FwdOp": {
        "head_size": 128,
        "rotary_dim": 128,
        "max_position_embeddings": 16384,
        "base": 500000.0,
        "is_neox_style": True,
        "scaling_factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "orig_max_position": 8192,
    },
    "YaRNFwdOp": {
        "head_size": 128,
        "rotary_dim": 128,
        "max_position_embeddings": 4096,
        "base": 10000.0,
        "is_neox_style": True,
        "scaling_factor": 4.0,
    },
    "LongRoPEFwdOp": {
        "head_size": 96,
        "rotary_dim": 96,
        "max_position_embeddings": 16384,
        "original_max_position_embeddings": 4096,
        "base": 10000.0,
        "is_neox_style": True,
        "rescale_factors": [1.0 + i / 16 for i in range(48)],
    },
}
_YARN_DEFAULTS = {
    "extrapolation_factor": 1,
    "attn_factor": 1,
    "beta_fast": 32,
    "beta_slow": 1,
    "apply_yarn_scaling": True,
    "truncate": True,
}


def _op(name: str, dtype: torch.dtype, **changes):
    """Op *name* built from :data:`_PARAMS` with *changes*, and its full parameters."""
    import tileops._ops.rope as rope

    params = {**_PARAMS[name], **changes, "dtype": dtype}
    full = dict(params)
    if name == "YaRNFwdOp":
        full = {**_YARN_DEFAULTS, **full}
    if name == "LongRoPEFwdOp":
        full.setdefault("mscale", None)
    return getattr(rope, name)(**params), full


def _inputs(name, params, tokens, q_heads, k_heads, dtype, first=0):
    """Positions from *first* on, and a query and key of *tokens* rows."""
    hs = params["head_size"]
    positions = torch.arange(first, first + tokens, device=run_device())
    query = torch.randn(tokens, q_heads * hs, device=run_device(), dtype=dtype)
    key = (
        None
        if k_heads is None
        else torch.randn(tokens, k_heads * hs, device=run_device(), dtype=dtype)
    )
    return positions, query, key


def _reference(name, params, positions, query, key, offsets=None):
    cos, sin = rope_tables(name, params, query)
    return rope_reference(params, cos, sin, positions, query, key, offsets)


@pytest.mark.smoke
@pytest.mark.packaging(family="rope")
@pytest.mark.parametrize("name", sorted(_PARAMS))
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_rope_matches_the_reference_at_positions_past_0(name, dtype) -> None:
    op, params = _op(name, dtype)
    positions, query, key = _inputs(name, params, 37, 4, 2, dtype, first=1000)
    got_q, got_k = op(positions, query, key)
    ref_q, ref_k = _reference(name, params, positions, query, key)
    compare_outputs(got_q, ref_q, rope_verification())
    compare_outputs(got_k, ref_k, rope_verification())


@pytest.mark.parametrize("neox", [True, False], ids=["neox", "gptj"])
@pytest.mark.parametrize(
    ("head_size", "rotary_dim"),
    [
        pytest.param(64, 64, marks=pytest.mark.smoke, id="full"),
        pytest.param(64, 32, marks=pytest.mark.full, id="partial"),
        pytest.param(65, 32, marks=pytest.mark.full, id="odd-head"),
    ],
)
def test_rope_rotates_the_leading_columns_in_either_convention(neox, head_size, rotary_dim) -> None:
    """An odd head size leaves no column offset on a vector boundary: scalar accesses."""
    dtype = torch.float16
    op, params = _op(
        "RoPEFwdOp", dtype, head_size=head_size, rotary_dim=rotary_dim, is_neox_style=neox
    )
    positions, query, key = _inputs("RoPEFwdOp", params, 64, 8, 2, dtype, first=4000)
    got_q, got_k = op(positions.flip(0), query, key)
    ref_q, ref_k = _reference("RoPEFwdOp", params, positions.flip(0), query, key)
    compare_outputs(got_q, ref_q, rope_verification())
    compare_outputs(got_k, ref_k, rope_verification())


@pytest.mark.parametrize(
    "first",
    [
        pytest.param(True, marks=pytest.mark.smoke, id="tune-first"),
        pytest.param(False, marks=pytest.mark.full, id="tune-after"),
    ],
)
def test_rope_runs_after_a_tuning_request(first) -> None:
    """The kernel tunes nothing: a request before or after the first call leaves the call
    running."""
    dtype = torch.bfloat16
    op, params = _op("RoPEFwdOp", dtype)
    positions = torch.randint(0, 4096, (16,), device=run_device())
    query = torch.randn(16, 3 * 64, device=run_device(), dtype=dtype)
    if first:
        op.request_tune()
    op(positions, query)
    if not first:
        op.request_tune()
    got_q, _ = op(positions, query)
    ref_q, _ = _reference("RoPEFwdOp", params, positions, query, None)
    compare_outputs(got_q, ref_q, rope_verification())


@pytest.mark.smoke
def test_rope_takes_batched_positions_and_a_query_alone() -> None:
    dtype = torch.bfloat16
    op, params = _op("RoPEFwdOp", dtype)
    positions = torch.randint(0, 4096, (2, 16), device=run_device())
    query = torch.randn(2, 16, 3 * 64, device=run_device(), dtype=dtype)
    got_q, got_k = op(positions, query)
    ref_q, _ = _reference("RoPEFwdOp", params, positions, query, None)
    assert got_k is None
    compare_outputs(got_q, ref_q, rope_verification())


@pytest.mark.smoke
def test_longrope_reads_the_row_positions_plus_offsets_name() -> None:
    dtype = torch.float16
    op, params = _op("LongRoPEFwdOp", dtype)
    positions, query, key = _inputs("LongRoPEFwdOp", params, 32, 4, 4, dtype, first=10)
    offsets = torch.full_like(positions, 4096)
    got_q, got_k = op(positions, query, key, offsets)
    ref_q, ref_k = _reference("LongRoPEFwdOp", params, positions, query, key, offsets)
    compare_outputs(got_q, ref_q, rope_verification())
    compare_outputs(got_k, ref_k, rope_verification())


@pytest.mark.smoke
@pytest.mark.parametrize("neox", [True, False], ids=["neox", "gptj"])
@pytest.mark.parametrize(
    ("dtype", "bad"),
    [(torch.int32, 8), (torch.int64, 2**32 + 1), (torch.int64, -(2**32) + 1)],
    ids=["int32", "int64-past-int32", "int64-negative"],
)
def test_rope_validate_inputs_rejects_a_position_outside_the_cache(neox, dtype, bad) -> None:
    op, params = _op(
        "RoPEFwdOp",
        torch.float16,
        max_position_embeddings=8,
        is_neox_style=neox,
        validate_inputs=True,
    )
    positions = torch.zeros(2, device=run_device(), dtype=dtype)
    positions[-1] = bad
    query = torch.randn(2, 64, device=run_device(), dtype=torch.float16)
    with pytest.raises(ValueError, match=r"^positions must be in \[0, 8\)"):
        op(positions, query)


@pytest.mark.smoke
@pytest.mark.sm90
def test_partial_neox_query_clamps_both_table_edges():
    op, params = _op(
        "RoPEFwdOp", torch.bfloat16, head_size=80, rotary_dim=32, max_position_embeddings=64
    )
    positions = torch.tensor([-1, 0, 63, 64], device=run_device(), dtype=torch.int32)
    query = torch.randn(4, 32 * 80, device=run_device(), dtype=torch.bfloat16)
    got, key = op(positions, query)
    expected, _ = _reference("RoPEFwdOp", params, positions.clamp(0, 63), query, None)
    compare_outputs(got, expected, rope_verification())
    assert key is None


@pytest.mark.smoke
@pytest.mark.parametrize("neox", [True, False], ids=["neox", "gptj"])
def test_rope_unchecked_position_outside_the_cache_reads_its_nearest_row(neox) -> None:
    dtype = torch.float16
    op, params = _op("RoPEFwdOp", dtype, max_position_embeddings=8, is_neox_style=neox)
    positions = torch.tensor([-1, 8, 3], device=run_device())
    query = torch.randn(3, 2 * 64, device=run_device(), dtype=dtype)
    key = torch.randn(3, 64, device=run_device(), dtype=dtype)
    got_q, got_k = op(positions, query, key)
    ref_q, ref_k = _reference("RoPEFwdOp", params, positions.clamp(0, 7), query, key)
    compare_outputs(got_q, ref_q, rope_verification())
    compare_outputs(got_k, ref_k, rope_verification())


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.cuda_graph(op="RoPEFwdOp")
@pytest.mark.parametrize("neox", [True, False], ids=["neox", "gptj"])
def test_rope_graph_replay(neox) -> None:
    """Replay rotates changed positions, query and key into the captured outputs."""
    dtype = torch.bfloat16
    op, params = _op("RoPEFwdOp", dtype, is_neox_style=neox)
    positions, query, key = _inputs("RoPEFwdOp", params, 64, 8, 2, dtype, first=100)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(positions, query, key)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got_q, got_k = op(positions, query, key)
    for _ in range(2):
        positions.add_(1000)
        query.neg_()
        key.mul_(0.5)
        got_q.fill_(float("nan"))
        got_k.fill_(float("nan"))
        graph.replay()
        ref_q, ref_k = _reference("RoPEFwdOp", params, positions, query, key)
        compare_outputs(got_q, ref_q, rope_verification())
        compare_outputs(got_k, ref_k, rope_verification())


@pytest.mark.parametrize(
    ("name", "neox", "dtype", "head_size", "rotary_dim", "heads", "tokens"),
    [
        pytest.param(
            "RoPEFwdOp", False, torch.float16, 64, 24, (32, 8), 16384, marks=_SMOKE, id="gptj"
        ),
        pytest.param(
            "LongRoPEFwdOp",
            True,
            torch.float32,
            64,
            24,
            (8, 8),
            16384,
            marks=_SMOKE,
            id="longrope-offsets-float32",
        ),
        pytest.param(
            "RoPEFwdOp", False, torch.float16, 65, 32, (3, 1), 16384, marks=_FULL, id="scalar-gptj"
        ),
        pytest.param(
            "RoPEFwdOp", True, torch.float16, 64, 24, (1100, 8), 512, marks=_FULL, id="block-limit"
        ),
    ],
)
def test_rope_stages_each_row_where_heads_copy_columns(
    name, neox, dtype, head_size, rotary_dim, heads, tokens
) -> None:
    """Heads with copied columns, each token's query row and key row staged in shared memory.

    Each call's rows outnumber the staged blocks any supported device holds at once, so the
    rows are staged: rotated in shared memory and written back whole. A 65-column head moves
    single columns, and its 3-head query row of 390 bytes is no whole number of 16-byte
    vectors; its key row is shorter than its query row. LongRoPE's offsets move each token's
    table row. A 140800-byte query row takes more threads than a block has, so its block holds
    1024.
    """
    changes = {"head_size": head_size, "rotary_dim": rotary_dim, "is_neox_style": neox}
    if name == "LongRoPEFwdOp":
        changes["rescale_factors"] = [1.0 + i / 16 for i in range(rotary_dim // 2)]
    op, params = _op(name, dtype, **changes)
    positions, query, key = _inputs(name, params, tokens, *heads, dtype)
    positions = positions.flip(0) % 4096
    (got_q, got_k), read = _call_with_offsets(name, op, positions, query, key)
    ref_q, ref_k = _reference(name, params, read, query, key)
    compare_outputs(got_q, ref_q, rope_verification())
    compare_outputs(got_k, ref_k, rope_verification())


@pytest.mark.smoke
def test_rope_stages_a_query_alone() -> None:
    """A NeoX query of 80-column heads rotating 32, alone, at int32 positions: one staged block
    a token."""
    dtype = torch.bfloat16
    op, params = _op("RoPEFwdOp", dtype, head_size=80, rotary_dim=32)
    positions, query, _ = _inputs("RoPEFwdOp", params, 16384, 32, None, dtype)
    positions = positions.int() % params["max_position_embeddings"]
    got_q, got_k = op(positions.flip(0), query)
    assert got_k is None
    compare_outputs(
        got_q,
        _reference("RoPEFwdOp", params, positions.flip(0), query, None)[0],
        rope_verification(),
    )


@pytest.mark.parametrize(
    ("neox", "head_size", "rotary_dim", "heads"),
    [
        pytest.param(False, 64, 24, 1824, marks=_SMOKE, id="vector-gptj"),
        pytest.param(True, 64, 24, 1824, marks=_FULL, id="vector-neox"),
        pytest.param(False, 65, 32, 1800, marks=_FULL, id="scalar-gptj"),
    ],
)
def test_rope_rotates_several_units_a_thread_where_a_row_passes_shared_memory(
    neox, head_size, rotary_dim, heads
) -> None:
    """A query row past a block's shared memory on any supported device (233472 and 234000
    bytes), so each thread loads several units of it before it stores any, though its 256
    tokens outnumber one wave of such rows. A NeoX head of 64 columns rotating 24 copies
    8-column runs past 4-column rotating units, so a copied run is wider than a rotating unit;
    a 65-column head moves single columns."""
    dtype = torch.float16
    op, params = _op(
        "RoPEFwdOp", dtype, head_size=head_size, rotary_dim=rotary_dim, is_neox_style=neox
    )
    positions, query, key = _inputs("RoPEFwdOp", params, 256, heads, 8, dtype)
    got_q, got_k = op(positions, query, key)
    ref_q, ref_k = _reference("RoPEFwdOp", params, positions, query, key)
    compare_outputs(got_q, ref_q, rope_verification())
    compare_outputs(got_k, ref_k, rope_verification())


def _call_with_offsets(name, op, positions, query, key):
    """*op*'s outputs, LongRoPE's at offsets of 4096, and the positions they read."""
    if name != "LongRoPEFwdOp":
        return op(positions, query, key), positions
    offsets = torch.full_like(positions, 4096)
    return op(positions, query, key, offsets), positions + offsets


@pytest.mark.parametrize(
    ("name", "neox", "dtype"),
    [
        pytest.param("RoPEFwdOp", False, torch.float32, marks=pytest.mark.smoke, id="gptj-float32"),
        pytest.param(
            "LongRoPEFwdOp", True, torch.float16, marks=pytest.mark.smoke, id="longrope-offsets"
        ),
    ],
)
def test_rope_rotates_one_unit_a_thread_where_heads_copy_columns(name, neox, dtype) -> None:
    """A small call whose heads copy columns: one unit a thread, its columns loaded before the
    token's position.

    5 tokens of 64 query heads and one key head, 64 columns a head and 24 rotating. In float32
    a token has more units than the 1024-thread block limit, so its second block is partly
    empty. Positions outside the cache read its nearest row; LongRoPE's offsets move each row.
    """
    changes = {"head_size": 64, "rotary_dim": 24, "is_neox_style": neox}
    if name == "LongRoPEFwdOp":
        changes["rescale_factors"] = [1.0 + i / 16 for i in range(12)]
    op, params = _op(name, dtype, **changes)
    _, query, key = _inputs(name, params, 5, 64, 1, dtype)
    rows = rope_tables(name, params, query)[0].shape[0]
    positions = torch.tensor([-1, rows, 3, rows - 1, 1000], device=run_device())
    (got_q, got_k), read = _call_with_offsets(name, op, positions, query, key)
    ref_q, ref_k = _reference(name, params, read.clamp(0, rows - 1), query, key)
    compare_outputs(got_q, ref_q, rope_verification())
    compare_outputs(got_k, ref_k, rope_verification())


@pytest.mark.parametrize(
    ("name", "changes", "tokens"),
    [
        pytest.param("RoPEFwdOp", {}, 5, marks=_SMOKE, id="rope"),
        pytest.param("LongRoPEFwdOp", {}, 5, marks=_FULL, id="longrope-offsets"),
        pytest.param(
            "LongRoPEFwdOp",
            {"head_size": 64, "rotary_dim": 24, "rescale_factors": [1.0] * 12},
            16384,
            marks=_FULL,
            id="longrope-offsets-staged",
        ),
    ],
)
def test_rope_takes_a_key_of_no_heads(name, changes, tokens) -> None:
    """A 0-column key failed to reshape; it comes back empty, shaped as the key, and the query
    rotates. 16384 tokens of heads with copied columns take the staged program."""
    dtype = torch.float16
    op, params = _op(name, dtype, **changes)
    positions, query, key = _inputs(name, params, tokens, 3, 0, dtype)
    positions = positions % 4096
    (got_q, got_k), read = _call_with_offsets(name, op, positions, query, key)
    assert got_k.shape == key.shape
    compare_outputs(got_q, _reference(name, params, read, query, None)[0], rope_verification())


@pytest.mark.smoke
@pytest.mark.parametrize("neox", [True, False], ids=["neox", "gptj"])
def test_rope_query_off_a_vector_boundary(neox) -> None:
    """A contiguous query starting one element past a 16-byte boundary matches the reference.

    8192 tokens of 8 heads fill more than half of any device's resident threads, so each
    thread rotates several units.
    """
    dtype = torch.float16
    op, params = _op("RoPEFwdOp", dtype, is_neox_style=neox)
    positions, aligned, _ = _inputs("RoPEFwdOp", params, 8192, 8, None, dtype)
    positions = positions % params["max_position_embeddings"]
    query = torch.empty(aligned.numel() + 1, device=run_device(), dtype=dtype)[1:]
    query = query.view(aligned.shape).copy_(aligned)
    got_q, _ = op(positions, query)
    compare_outputs(
        got_q, _reference("RoPEFwdOp", params, positions, aligned, None)[0], rope_verification()
    )


@pytest.mark.smoke
def test_rope_longrope_rejects_a_zero_rescale_factor() -> None:
    with pytest.raises(ValueError, match="rescale_factors"):
        _op("LongRoPEFwdOp", torch.float16, rescale_factors=[1.0, 0.0] * 24)


@pytest.mark.smoke
def test_rope_longrope_rescale_factors_edited_in_place_recompute_its_cache() -> None:
    """An edit in place to the factors list, and one back, is seen by the next call."""
    dtype = torch.float16
    rescale = [1.0] * 48
    op, params = _op("LongRoPEFwdOp", dtype, rescale_factors=rescale)
    positions, query, key = _inputs("LongRoPEFwdOp", params, 8, 1, 1, dtype, first=100)
    op(positions, query, key)
    for value in (2.0, 1.0):
        rescale[:] = [value] * 48
        got_q, _ = op(positions, query, key)
        ref_q, _ = _reference(
            "LongRoPEFwdOp", {**params, "rescale_factors": rescale}, positions, query, key
        )
        compare_outputs(got_q, ref_q, rope_verification())


# ---------------------------------------------------------------- regression against vLLM


def _vllm_rope(name: str, params: dict):
    """The vLLM rotary embedding op *name* follows, built from *params*."""
    vllm_rope = pytest.importorskip("vllm.model_executor.layers.rotary_embedding")
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        return _vllm_class(vllm_rope, name, params)


def _vllm_class(vllm_rope, name: str, params: dict):
    common = [params[k] for k in ("head_size", "rotary_dim", "max_position_embeddings")]
    if name == "RoPEFwdOp":
        return vllm_rope.base.RotaryEmbedding(
            *common, params["base"], params["is_neox_style"], params["dtype"]
        )
    if name == "RoPELlama31FwdOp":
        return vllm_rope.llama3_rope.Llama3RotaryEmbedding(
            *common,
            params["base"],
            params["is_neox_style"],
            params["dtype"],
            params["scaling_factor"],
            params["low_freq_factor"],
            params["high_freq_factor"],
            params["orig_max_position"],
        )
    if name == "YaRNFwdOp":
        return vllm_rope.yarn_scaling_rope.YaRNScalingRotaryEmbedding(
            *common,
            params["base"],
            params["is_neox_style"],
            params["scaling_factor"],
            params["dtype"],
        )
    raise AssertionError(name)


@pytest.mark.smoke
@pytest.mark.parametrize("name", ["RoPEFwdOp", "RoPELlama31FwdOp", "YaRNFwdOp"])
@pytest.mark.parametrize("neox", [True, False], ids=["neox", "gptj"])
def test_rope_matches_vllm_at_positions_past_0(name, neox) -> None:
    """Decode at position p > 0: the rows vLLM reads for each token, not rows 0..S-1."""
    dtype = torch.bfloat16
    op, params = _op(name, dtype, is_neox_style=neox)
    reference = _vllm_rope(name, params).to(run_device())
    positions, query, key = _inputs(name, params, 64, 8, 2, dtype, first=3000)
    got_q, got_k = op(positions, query, key)
    want_q, want_k = reference.forward_native(positions, query.clone(), key.clone())
    # vLLM rounds each product and sum in the input dtype, the op once from float32.
    torch.testing.assert_close(got_q, want_q, atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(got_k, want_k, atol=3e-2, rtol=2e-2)


@pytest.mark.smoke
@pytest.mark.parametrize("long_context", [True, False], ids=["long-factors", "short-factors"])
def test_longrope_matches_phi3_at_positions_past_0(long_context, monkeypatch) -> None:
    """The chosen Phi-3 factors and magnitude, passed in, give Phi-3's rotation."""
    phi3 = pytest.importorskip(
        "vllm.model_executor.layers.rotary_embedding.phi3_long_rope_scaled_rope"
    )
    from types import SimpleNamespace

    dtype = torch.bfloat16
    original, extended = 4096, 16384
    short = [1.0 + i / 64 for i in range(48)]
    long = [1.0 + i / 4 for i in range(48)]
    max_model_len = extended if long_context else original
    config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=max_model_len))
    monkeypatch.setattr(phi3, "get_current_vllm_config", lambda: config)
    reference = phi3.Phi3LongRoPEScaledRotaryEmbedding(
        96, 96, extended, original, 10000.0, True, dtype, short, long
    ).to(run_device())
    chosen, mscale = (
        (long, reference.long_mscale) if long_context else (short, reference.short_mscale)
    )
    op, params = _op("LongRoPEFwdOp", dtype, rescale_factors=chosen, mscale=mscale)
    # With the long factors Phi-3 skips its short cache: the contract's row is Phi-3's row
    # less original_max_position_embeddings.
    first = 9000 if long_context else 2000
    positions, query, key = _inputs("LongRoPEFwdOp", params, 64, 4, 4, dtype, first=first)
    got_q, got_k = op(positions, query, key)
    want_q, want_k = reference(positions, query.clone(), key.clone())
    torch.testing.assert_close(got_q, want_q, atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(got_k, want_k, atol=3e-2, rtol=2e-2)


# ---------------------------------------------------------------- caller-given cos and sin


class RoPECosSinTest(RoPECosSinCall, TestBase):
    pass


class RoPECosSinViewTest(RoPECosSinViewCall, TestBase):
    pass


_F32_TABLES = {"C": "float32", "Cs": "float32"}


def _dense(B, S, H, D, Rh, P=None, **row):
    return {"B": B, "S": S, "H": H, "D": D, "Rh": Rh, "P": P or S, **row}


def _packed(seq_lens, H, D, Rh, P, **row):
    some = row.pop("some", [])
    return {
        "Tt": sum(seq_lens),
        "seq_lens": seq_lens,
        "max_seqlen": max(seq_lens),
        "H": H,
        "D": D,
        "Rh": Rh,
        "P": P,
        "some": ["cu_seqlens", *some],
        **row,
    }


_COS_SIN_CASES = (
    # Qwen3.6 vision: a full 72-wide NeoX head, float32 tables.
    pytest.param(
        _dense(1, 37, 4, 72, 36), {"T": "bfloat16", **_F32_TABLES}, marks=_SMOKE, id="bf16"
    ),
    pytest.param(
        _dense(2, 19, 3, 128, 64, P=32),
        {"T": "float16", "C": "float16", "Cs": "float16"},
        marks=_SMOKE,
        id="fp16",
    ),
    pytest.param(
        _dense(2, 19, 3, 64, 32), {"T": "float32", **_F32_TABLES}, marks=_SMOKE, id="fp32"
    ),
    # Each table converts from its own dtype.
    pytest.param(
        _dense(1, 21, 2, 64, 32),
        {"T": "bfloat16", "C": "float16", "Cs": "float32"},
        marks=_FULL,
        id="mixed-tables",
    ),
    pytest.param(
        _dense(2, 17, 3, 128, 32, P=40, interleaved=True, seqlen_offsets=9),
        {"T": "bfloat16", **_F32_TABLES},
        marks=_FULL,
        id="interleaved-partial-offset",
    ),
    # Decode: one token per sequence, each at its own offset.
    pytest.param(
        _dense(6, 1, 4, 128, 64, P=64, some=["seqlen_offsets"]),
        {"T": "bfloat16", **_F32_TABLES, "O": "int32"},
        marks=_FULL,
        id="decode-offsets",
    ),
    # An empty sequence among packed ones, each sequence at its own offset.
    pytest.param(
        _packed([5, 0, 9], 3, 64, 32, P=24, some=["seqlen_offsets"]),
        {"T": "bfloat16", "C": "bfloat16", "Cs": "bfloat16", "I": "int32", "O": "int64"},
        marks=_FULL,
        id="packed-offsets",
    ),
    # An odd head leaves no column offset on a vector boundary: scalar accesses.
    pytest.param(
        _dense(1, 9, 3, 65, 16), {"T": "float16", **_F32_TABLES}, marks=_FULL, id="odd-head"
    ),
    pytest.param(
        _dense(1, 9, 3, 65, 16, interleaved=True),
        {"T": "float16", **_F32_TABLES},
        marks=_FULL,
        id="odd-head-interleaved",
    ),
    pytest.param(
        _packed([3, 8], 2, 72, 36, P=16, seqlen_offsets=4, interleaved=True),
        {"T": "float16", **_F32_TABLES, "I": "int64"},
        marks=_FULL,
        id="packed-int-offset",
    ),
    # In place with unrotated columns: the head stride exceeds the rotated width.
    pytest.param(
        _dense(1, 37, 4, 136, 16, inplace=True),
        {"T": "bfloat16", **_F32_TABLES},
        marks=_FULL,
        id="inplace-unrotated-columns",
    ),
    # Calls whose work fills more than half the device's resident threads give each thread
    # several heads, an odd head count leaving the last thread of a token one.
    pytest.param(
        _dense(1, 4096, 5, 128, 32),
        {"T": "bfloat16", **_F32_TABLES},
        marks=_FULL,
        id="many-units-partial",
    ),
    # Regression: more dense sequences than CUDA's grid has rows, each at its own offset.
    pytest.param(
        _dense(65537, 1, 1, 2, 1, P=4, some=["seqlen_offsets"]),
        {"T": "bfloat16", **_F32_TABLES, "O": "int64"},
        marks=_FULL,
        id="more-sequences-than-grid-rows",
    ),
)


@pytest.mark.parametrize("row, dtype_case", _COS_SIN_CASES)
def test_rope_cos_sin_matches_the_reference(row, dtype_case) -> None:
    test = RoPECosSinTest(manifest_call("RoPECosSinFwdOp", dtype_case, **row))
    test.check(RoPECosSinFwdOp(**test.arguments()), *test.gen_inputs())


def _strided_view_call(interleaved, inplace, columns, width):
    """The op and its inputs on a column view of heads *width* columns wide."""
    row = _dense(1, 37, 4, len(range(width)[columns]), 16, interleaved=interleaved, inplace=inplace)
    call = manifest_call("RoPECosSinFwdOp", {"T": "bfloat16", **_F32_TABLES}, **row)
    test = RoPECosSinViewTest(call, width, columns)
    return test, RoPECosSinFwdOp(**test.arguments()), test.gen_inputs()


# The trailing columns of NeoX heads are the smoke case; the other pairing and a
# column stride other than 1 are full.
_VIEWS = pytest.mark.parametrize(
    "interleaved, columns",
    [
        pytest.param(False, slice(64, 136), marks=_SMOKE, id="neox-tail"),
        pytest.param(True, slice(64, 136), marks=_FULL, id="interleaved-tail"),
        pytest.param(False, slice(8, 136, 2), marks=_FULL, id="neox-every-2nd"),
        pytest.param(True, slice(8, 136, 2), marks=_FULL, id="interleaved-every-2nd"),
    ],
)


@_VIEWS
def test_rope_cos_sin_rotates_a_strided_view_in_place(interleaved, columns) -> None:
    """A column view of wider heads rotated in place, as DeepSeek-V4-Flash rotates the
    trailing columns of each head: x is returned, and only the view's columns change."""
    test, op, inputs = _strided_view_call(interleaved, True, columns, 136)
    test.check(op, *inputs)
    x, wide = inputs[0], inputs[0]._base
    before = wide.clone()
    assert op(*inputs) is x
    kept = torch.ones_like(wide, dtype=torch.bool)
    kept[..., columns] = False
    assert torch.equal(wide[kept], before[kept])


# NeoX only: the pairing on strided views is the in-place test's, and interleaved out of
# place is the interleaved-partial-offset case.
@pytest.mark.parametrize(
    "columns",
    [
        pytest.param(slice(64, 136), marks=_SMOKE, id="neox-tail"),
        pytest.param(slice(8, 136, 2), marks=_FULL, id="neox-every-2nd"),
    ],
)
def test_rope_cos_sin_rotates_a_strided_view_into_a_new_tensor(columns) -> None:
    """A column view of wider heads read out of place: the wider heads are left as they were."""
    test, op, inputs = _strided_view_call(False, False, columns, 136)
    test.check(op, *inputs)
    wide = inputs[0]._base
    before = wide.clone()
    op(*inputs)
    assert torch.equal(wide, before)


_OFFSETS = _dense(2, 19, 3, 128, 64, P=32, some=["seqlen_offsets"])


@pytest.mark.parametrize(
    "row, dtypes, offsets",
    [
        pytest.param(_OFFSETS, {"O": "int64"}, [20, -3], marks=_SMOKE, id="int64"),
        # An int32 offset at its extremes, widened before it is added to a position, reads
        # the nearest row.
        pytest.param(
            _OFFSETS,
            {"O": "int32"},
            [2**31 - 1, -(2**31)],
            marks=_SMOKE,
            id="int32-extremes",
        ),
        pytest.param(
            _packed([5, 0, 9], 3, 64, 32, P=24, some=["seqlen_offsets"]),
            {"I": "int32", "O": "int32"},
            [2**31 - 1, 0, -(2**31)],
            marks=_FULL,
            id="packed-int32-extremes",
        ),
    ],
)
def test_rope_cos_sin_unchecked_row_outside_the_tables_reads_its_nearest_row(
    row, dtypes, offsets
) -> None:
    call = manifest_call("RoPECosSinFwdOp", {"T": "float16", **_F32_TABLES, **dtypes}, **row)
    test = RoPECosSinTest(call)
    x, cos, sin, cu_seqlens, seqlen_offsets = test.gen_inputs()
    seqlen_offsets.copy_(torch.tensor(offsets))
    test.check(RoPECosSinFwdOp(**test.arguments()), x, cos, sin, cu_seqlens, seqlen_offsets)


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(_OFFSETS, marks=_SMOKE, id="one-head"),
        # Units past half the resident threads: two heads a thread.
        pytest.param(
            _dense(2, 2048, 8, 128, 64, P=4096, some=["seqlen_offsets"]),
            marks=_FULL,
            id="two-heads",
        ),
    ],
)
def test_rope_cos_sin_int64_offset_at_its_maximum_reads_the_last_row(row) -> None:
    """Offset 2**63 - 1 plus a position past 0 would wrap below row 0 before its clamp. The op
    on the int64 extremes reads the rows the reference reads on P - 1 and -S, which reach the
    same rows without the wrapping sum the reference makes on the extremes."""
    call = manifest_call("RoPECosSinFwdOp", {"T": "float16", **_F32_TABLES, "O": "int64"}, **row)
    test = RoPECosSinTest(call)
    x, cos, sin, cu_seqlens, _ = test.gen_inputs()
    op = RoPECosSinFwdOp(**test.arguments())
    extreme = torch.tensor([2**63 - 1, -(2**63)], device=x.device)
    nearest = torch.tensor([cos.shape[0] - 1, -x.shape[1]], device=x.device)

    def on_extremes(x, cos, sin, cu_seqlens=None, seqlen_offsets=0):
        return op(x, cos, sin, cu_seqlens, extreme)

    test.check(op, x, cos, sin, cu_seqlens, nearest, runs=on_extremes)


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
@pytest.mark.parametrize("extreme", [2**63 - 1, -(2**63)], ids=["max", "min"])
def test_rope_cos_sin_kernel_int_offset_at_its_extremes_reads_the_end_rows(extreme) -> None:
    """The op takes an int offset in [0, P - S]; the kernel clamps any other to the rows it
    reaches, so an int64 extreme reads the rows P - 1 or -S read, without a wrapping sum."""
    row = _dense(1, 37, 4, 72, 36)
    call = manifest_call("RoPECosSinFwdOp", {"T": "float16", **_F32_TABLES}, **row)
    test = RoPECosSinTest(call)
    x, cos, sin, cu_seqlens, seqlen_offsets = test.gen_inputs()
    op = RoPECosSinFwdOp(**test.arguments())
    op(x, cos, sin, cu_seqlens, seqlen_offsets)
    (kernel,) = op.iter_kernels()
    nearest = cos.shape[0] - 1 if extreme > 0 else -x.shape[1]
    want = kernel.forward(x, cos, sin, cu_seqlens, nearest)
    got = kernel.forward(x, cos, sin, cu_seqlens, extreme)
    torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.smoke
def test_rope_cos_sin_in_place_leaves_the_memory_past_x_untouched() -> None:
    """The 1332 units of 37 tokens fill 52 threads of the last block; its other threads touch
    nothing, here the part of the buffer past x."""
    row = _dense(1, 37, 4, 72, 36, inplace=True)
    call = manifest_call("RoPECosSinFwdOp", {"T": "bfloat16", **_F32_TABLES}, **row)
    test = RoPECosSinTest(call)
    x, cos, sin, cu_seqlens, seqlen_offsets = test.gen_inputs()
    op = RoPECosSinFwdOp(**test.arguments())
    buffer = torch.randn(x.numel() + 128 * 4 * 72, dtype=x.dtype, device=x.device)
    head = buffer[: x.numel()].view(x.shape)
    rest = buffer[x.numel() :].clone()
    assert op(head, cos, sin, cu_seqlens, seqlen_offsets) is head
    assert torch.equal(buffer[x.numel() :], rest)


@pytest.mark.parametrize(
    "apart",
    [
        pytest.param(19 * 3 * 256, marks=_SMOKE, id="adjacent"),
        # Element offsets past int32: the program takes 64-bit indexes.
        pytest.param(2**31, marks=_FULL, id="past-int32"),
    ],
)
def test_rope_cos_sin_rotates_sequences_far_apart(apart) -> None:
    """Two sequences *apart* elements apart, every second column of 256-column heads:
    one-column accesses."""
    row = _dense(2, 19, 3, 128, 64, P=32)
    call = manifest_call(
        "RoPECosSinFwdOp", {"T": "float16", "C": "float16", "Cs": "float16"}, **row
    )
    test = RoPECosSinTest(call)
    x, cos, sin, cu_seqlens, seqlen_offsets = test.gen_inputs()
    storage = torch.zeros(apart + 19 * 3 * 256, dtype=x.dtype, device=x.device)
    view = storage.as_strided(x.shape, (apart, 3 * 256, 256, 2))
    view.copy_(x)
    test.check(RoPECosSinFwdOp(), view, cos, sin, cu_seqlens, seqlen_offsets)
