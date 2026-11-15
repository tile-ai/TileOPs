"""Tests for the rotary position embedding ops.

Each op rotates ``query`` and ``key`` at the cache rows ``positions`` names, as vLLM's
``RotaryEmbedding.forward_native(positions, query, key)`` does. The ops are checked against
an independent reference of the vLLM formulas, and against vLLM itself at positions that do
not start at 0.
"""

import pytest
import torch

from workloads.device import run_device
from workloads.numerics import compare_outputs
from workloads.rope import ref_rotate, rope_cache, rope_verification

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
    cos, sin, mscale = rope_cache(name, params)
    cos, sin = (
        (t * mscale).to(params["dtype"]).to(device=query.device, dtype=query.dtype)
        for t in (cos, sin)
    )
    index = positions if offsets is None else positions + offsets
    hs, rd, neox = params["head_size"], params["rotary_dim"], params["is_neox_style"]
    out_q = ref_rotate(query, index, cos, sin, hs, rd, neox)
    return out_q, None if key is None else ref_rotate(key, index, cos, sin, hs, rd, neox)


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


@pytest.mark.smoke
@pytest.mark.parametrize("neox", [True, False], ids=["neox", "gptj"])
@pytest.mark.parametrize("rotary_dim", [64, 32], ids=["full", "partial"])
def test_rope_rotates_the_leading_columns_in_either_convention(neox, rotary_dim) -> None:
    dtype = torch.float16
    op, params = _op("RoPEFwdOp", dtype, rotary_dim=rotary_dim, is_neox_style=neox)
    positions, query, key = _inputs("RoPEFwdOp", params, 64, 8, 2, dtype, first=4000)
    got_q, got_k = op(positions.flip(0), query, key)
    ref_q, ref_k = _reference("RoPEFwdOp", params, positions.flip(0), query, key)
    compare_outputs(got_q, ref_q, rope_verification())
    compare_outputs(got_k, ref_k, rope_verification())


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


@pytest.mark.smoke
@pytest.mark.parametrize("neox", [True, False], ids=["neox", "gptj"])
def test_rope_query_off_a_vector_boundary(neox) -> None:
    """A contiguous query starting one element past a 16-byte boundary matches the reference."""
    dtype = torch.float16
    op, params = _op("RoPEFwdOp", dtype, is_neox_style=neox)
    positions, aligned, _ = _inputs("RoPEFwdOp", params, 96, 8, None, dtype, first=7)
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
@pytest.mark.parametrize("first", [1, 3000], ids=["from-1", "from-3000"])
def test_rope_matches_vllm_at_positions_past_0(name, neox, first) -> None:
    """Decode at position p > 0: the rows vLLM reads for each token, not rows 0..S-1."""
    dtype = torch.bfloat16
    op, params = _op(name, dtype, is_neox_style=neox)
    reference = _vllm_rope(name, params).to(run_device())
    positions, query, key = _inputs(name, params, 64, 8, 2, dtype, first=first)
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
