"""Roofline cost-model functions for Tier 2 ops (attention, conv, MoE, etc.).

Each function returns a ``(flops, bytes)`` tuple of ints, matching the
``Op.eval_roofline(self) -> tuple[int, int]`` shape that codegen emits for ``func`` mode
(see ``docs/design/roofline.md`` §4.4.2). A parametric entry's function takes the checked
call; a legacy entry's takes the bound Op instance.

These are referenced from ``src/tileops/manifest/`` via the ``roofline.func``
field.
"""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tileops.manifest.workload import CallView
    from tileops.ops.op_base import Op

__all__ = [
    "adaptive_pool2d_roofline",
    "deltanet_inference_roofline",
    "fft_c2c_roofline",
    "fused_moe_fwd_roofline",
    "fused_moe_shared_expert_fwd_roofline",
    "gated_deltanet_fwd_roofline",
    "gqa_dense_fwd_roofline",
    "gqa_paged_fwd_roofline",
    "gqa_prefill_paged_cached_tokens",
    "gqa_prefill_paged_with_kv_cache_fwd_roofline",
    "gqa_prefill_varlen_fwd_roofline",
    "gqa_sliding_window_varlen_fwd_roofline",
    "gqa_varlen_fwd_roofline",
    "grouped_gemm_roofline",
    "moe_post_permute_roofline",
    "nsa_closed_chunk_pairs",
    "nsa_cmp_fwd_varlen_roofline",
    "nsa_fwd_varlen_roofline",
    "nsa_selected_block_loads",
    "nsa_topk_scored_pairs",
    "nsa_topk_varlen_roofline",
    "packed_visible_scores",
    "visible_scores",
]


_CALL_PAYLOAD_ATTR = "_roofline_kwargs"


def _shape_or_attrs(op: Any | None, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return formula inputs, with call-bound state overriding instance state.

    Raises:
        RuntimeError: The op declares call-bound state but has not run a call.
        ValueError: The call-bound state is neither a mapping nor ``None``.
    """
    if op is None:
        return kwargs
    if isinstance(op, dict):
        return op
    data = dict(vars(op))
    if _CALL_PAYLOAD_ATTR not in data:
        return data
    payload = data.pop(_CALL_PAYLOAD_ATTR)
    if payload is None:
        raise RuntimeError(f"{type(op).__name__}.eval_roofline() requires a prior forward() call")
    if not isinstance(payload, dict):
        raise ValueError(
            f"{type(op).__name__} stores {_CALL_PAYLOAD_ATTR} as {type(payload).__name__}; "
            "a formula reads it as a mapping of what the call bound"
        )
    data.update(payload)
    return data


def _dtype_itemsize(dtype: Any) -> int:
    if isinstance(dtype, (list, tuple)):
        dtype = dtype[0] if dtype else "float16"
    if hasattr(dtype, "itemsize"):
        return int(dtype.itemsize)
    dtype_name = str(dtype)
    if "complex128" in dtype_name:
        return 16
    if "complex64" in dtype_name:
        return 8
    if "float32" in dtype_name or "int32" in dtype_name:
        return 4
    if "float64" in dtype_name or "int64" in dtype_name:
        return 8
    if (
        "bool" in dtype_name
        or "int8" in dtype_name
        or "uint8" in dtype_name
        or "float8" in dtype_name
        or "fp8" in dtype_name
    ):
        return 1
    return 2


def _supplied(op: Any, name: str) -> bool:
    """Whether the call passed the ``optional: true`` input *name*.

    Mirrors the two bindings inline roofline synthesis accepts: the tensor on
    ``self.<name>``, or its shape on ``self.<name>_shape``.
    """
    if getattr(op, name, None) is not None:
        return True
    return getattr(op, f"{name}_shape", None) is not None


def deltanet_inference_roofline(op: Any | None = None, **kwargs: Any) -> tuple[int, int]:
    """Algorithmic lower bound for BTHD ungated DeltaNet inference."""
    data = _shape_or_attrs(op, kwargs)
    batch, seq_len, heads, dim_k = data["q_shape"]
    dim_v = data["v_shape"][-1]
    elem_bytes = _dtype_itemsize(data.get("dtype", data.get("dtypes", "float16")))
    flops = batch * seq_len * heads * (6 * dim_k * dim_v + 2 * dim_k)
    tensor_elements = batch * seq_len * heads * (2 * dim_k + 2 * dim_v + 1)
    cu_shape = data.get("cu_seqlens_shape")
    state_batch = cu_shape[0] - 1 if cu_shape is not None else batch
    state_elements = state_batch * heads * dim_k * dim_v
    nbytes = tensor_elements * elem_bytes + state_elements * 4
    if data.get("initial_state"):
        nbytes += state_elements * 4
    return int(flops), int(nbytes)


def gated_deltanet_fwd_roofline(op: Any | None = None, **kwargs: Any) -> tuple[int, int]:
    """Algorithmic lower bound for dense Gated DeltaNet inference."""
    data = _shape_or_attrs(op, kwargs)
    batch, seq_len, heads, dim_k = data["q_shape"]
    _batch, _seq_len, value_heads, dim_v = data["v_shape"]
    elem_bytes = _dtype_itemsize(data.get("dtype", data.get("dtypes", "float16")))

    # Per recurrent head and token: two state matvecs and one state
    # outer-product update (six FLOPs per state element), plus the elementwise
    # state decay (one multiply per state element).
    flops = batch * seq_len * value_heads * (7 * dim_k * dim_v)

    qk = 2 * batch * seq_len * heads * dim_k
    token_values = 2 * batch * seq_len * value_heads * dim_v  # v input and o output
    gates = 2 * batch * seq_len * value_heads
    cu_shape = data.get("cu_seqlens_shape")
    state_batch = cu_shape[0] - 1 if cu_shape is not None else batch
    state = state_batch * value_heads * dim_v * dim_k
    seeded = data.get("initial_state") is not None or data.get("initial_state_shape") is not None
    nbytes = (qk + token_values + gates) * elem_bytes
    nbytes += state * 4 * (2 if seeded else 1)
    return int(flops), int(nbytes)


def _chunkwise_dims_bshd(data: dict) -> tuple[int, int, int, int, int]:
    """``(batch, heads, seq_len, dim_k, dim_v)`` for an op declaring ``q [B, S, H, DK]``."""
    if "q_shape" in data:
        batch, seq_len, heads, dim_k = data["q_shape"]
        return batch, heads, seq_len, dim_k, data["v_shape"][3]
    return _chunkwise_dims_bound(data)


def _chunkwise_dims_bound(data: dict) -> tuple[int, int, int, int, int]:
    """The same five numbers off an instance that has run a call, in either order."""
    return (
        int(data["batch"]),
        int(data["heads"]),
        int(data["seq_len"]),
        int(data["dim_k"]),
        int(data["dim_v"]),
    )


def gla_fwd_roofline(op: Any | None = None, **kwargs: Any) -> tuple[int, int]:
    """Roofline for the chunked GLA forward, token-major: one state matmul pair per token."""
    data = _shape_or_attrs(op, kwargs)
    batch, heads, seq_len, dim_k, dim_v = _chunkwise_dims_bshd(data)
    elem_bytes = _dtype_itemsize(data.get("dtype", data.get("dtypes", "float16")))

    flops = 2 * batch * heads * seq_len * dim_k * dim_v
    tokens = batch * seq_len * heads
    cu_seqlens_shape = data.get("cu_seqlens_shape")
    state_batch = cu_seqlens_shape[0] - 1 if cu_seqlens_shape is not None else batch
    state = state_batch * heads * dim_k * dim_v
    seeded = data.get("initial_state") is not None or data.get("initial_state_shape") is not None
    # in: q, k, v, g and the fp32 state a caller may seed; out: o and the fp32 final state.
    nbytes = tokens * (3 * dim_k + 2 * dim_v) * elem_bytes + state * (2 if seeded else 1) * 4
    return int(flops), int(nbytes)


def _distribute_total(total: int, batch: int, max_len: int) -> list[int]:
    lengths = [0] * batch
    remaining = total
    for idx in range(batch):
        slots_left = batch - idx - 1
        value = min(max_len, remaining - slots_left)
        lengths[idx] = value
        remaining -= value
    return lengths


def moe_post_permute_roofline(call) -> tuple[int, int]:
    """One multiply-add per route and hidden element, and a scale per output element when the
    routing epilogue carries a factor other than one; each tensor moves once."""
    t, k, h = call.ix["T"], call.ix["K"], call.ix["H"]
    epilogue = call.ix["epilogue"]
    scaled = epilogue is not None and epilogue.routed_scaling_factor != 1.0
    flops = 2 * t * k * h + (t * h if scaled else 0)
    return flops, sum(call.bytes(name) for name in call.tensors)


def routed_active_experts(call) -> int:
    """The distinct experts a routed-experts call's ``topk_ids`` selects."""
    return len({e for row in call.values("topk_ids") for e in row})


def _expert_weight_bytes(call) -> int:
    """One expert's gate/up and down weights."""
    experts = call.ix["E"]
    return (call.bytes("w_gate_up") + call.bytes("w_down")) // experts


def routed_expert_mlp_roofline(call) -> tuple[int, int]:
    """The expert MLP of a call handed its routing: the experts ``topk_ids`` selects are read.

    FLOPs are the two GEMMs and the gated activation over every route, independent of which
    experts the routes land on.
    """
    t, k, f, h = call.ix["T"], call.ix["K"], call.ix["F"], call.ix["H"]
    flops = 6 * t * k * f * h
    nbytes = routed_active_experts(call) * _expert_weight_bytes(call)
    nbytes += 2 * call.bytes("hidden_states")  # the tokens read, the output written
    nbytes += call.bytes("topk_ids") + call.bytes("topk_weights")
    return flops, nbytes


def fused_moe_active_experts(call) -> int:
    """The experts a router's call reads: those its routed-experts stage was handed, from that
    stage's checked call, or the data-independent lower bound ``top_k`` where none exists."""
    stage_calls = (call.stages or {}).get("routed_experts") or ()
    if not stage_calls:
        return call.ix["top_k"]
    return len({e for c in stage_calls for row in c.values("topk_ids") for e in row})


def fused_moe_fwd_roofline(call) -> tuple[int, int]:
    """A router with its experts: the experts the routed-experts stage was handed are read.

    The routing is that stage's metadata input, read from its checked call in ``stages``
    (docs/design/roofline.md §4.7). Where no such call exists, the price takes the
    data-independent lower bound: each token selects ``top_k`` distinct experts.
    """
    t, f, h, top_k = call.ix["T"], call.ix["F"], call.ix["H"], call.ix["top_k"]
    flops = 6 * t * top_k * f * h
    nbytes = fused_moe_active_experts(call) * _expert_weight_bytes(call)
    nbytes += 2 * call.bytes("hidden_states")
    nbytes += call.bytes("gating_output")
    if call.present("correction_bias"):
        nbytes += call.bytes("correction_bias")
    return flops, nbytes


def fused_moe_shared_expert_fwd_roofline(call) -> tuple[int, int]:
    """The routed cost of :func:`fused_moe_fwd_roofline`, plus the shared expert's two GEMMs on
    this rank's shard, its own read of the hidden states and its write of ``shared_output``."""
    flops, nbytes = fused_moe_fwd_roofline(call)
    if not call.present("shared_w_gate_up"):
        return flops, nbytes
    t, h = call.ix["T"], call.ix["H"]
    shard = call.ix["S"] // call.ix["tp_size"]
    elem = call.bytes("hidden_states") // (t * h)
    weights = 3 * shard * h
    flops += 2 * t * weights
    nbytes += (weights + 2 * t * h) * elem
    return flops, nbytes


def grouped_gemm_roofline(op: "Op") -> tuple[int, int]:
    batch_sum = int(op.batch_sum)
    batch_count = int(op.batch_count)
    # The op carries both spellings and leaves one unset, so a default on the
    # missing name is not enough.
    n = int(getattr(op, "N", None) or getattr(op, "n", 0))
    k = int(getattr(op, "K", None) or getattr(op, "k", 0))
    elem = _dtype_itemsize(getattr(op, "dtype", "float16"))

    flops = 2 * batch_sum * n * k
    if not bool(op.transpose_a):
        memory_a = batch_sum * k
        memory_c = batch_sum * n
        memory_b = batch_count * n * k
    else:
        memory_a = batch_sum * n
        memory_c = batch_count * n * k
        memory_b = k * batch_sum if bool(op.transpose_b) else batch_sum * k
    # Two of the three int32 tensors: the kernels index batch_sizes and
    # batch_offsets, and take batch_padded_offsets without reading it -- the
    # templates pad nothing.
    metadata_bytes = 2 * batch_count * 4
    return int(flops), int((memory_a + memory_b + memory_c) * elem + metadata_bytes)


def fft_c2c_roofline(call: "CallView") -> tuple[int, int]:
    """1D complex FFT: ``5 * n * log2(n)`` FLOPs per transform, each tensor moved once."""
    n = call.ix["n"]
    flops = prod(call.ix["B"]) * 5 * n * (n.bit_length() - 1)
    return flops, sum(call.bytes(t) for t in call.tensors)


def _adaptive_scan(extent: int, out: int) -> int:
    """Rows an adaptive pool reads along one axis of *extent* pooled to *out* bins.

    Bin ``o`` covers ``[floor(o * extent / out), ceil((o + 1) * extent / out))``, so two
    adjacent bins share one row unless ``out`` divides ``j * extent``.
    """
    return extent + sum(1 for j in range(1, out) if (j * extent) % out)


def adaptive_pool2d_roofline(call: "CallView") -> tuple[int, int]:
    """Adaptive 2D pooling: one add or comparison per element each bin reads."""
    ix = call.ix
    scan = _adaptive_scan(ix["H_in"], ix["H_out"]) * _adaptive_scan(ix["W_in"], ix["W_out"])
    return prod(ix["B"]) * ix["C"] * scan, sum(call.bytes(t) for t in call.tensors)


# ---------------------------------------------------------------- attention


def _segments(call: "CallView", name: str) -> "list[int]":
    """Per-request lengths of the packed batch whose bounds are metadata tensor *name*."""
    bounds = call.values(name)
    return [end - start for start, end in zip(bounds, bounds[1:], strict=False)]


def _derived_bytes(call: "CallView") -> int:
    """Each tensor the call binds, moved once."""
    return sum(call.bytes(t) for t in call.tensors)


def visible_scores(q_len: int, kv_len: int, is_causal: bool, left: int, right: int) -> int:
    """Keys each query of one request sees under bottom-right alignment, summed over its queries.

    Query ``i`` sits at key position ``i + kv_len - q_len``; ``left`` and ``right`` bound the
    window around it, ``-1`` meaning unlimited.
    """
    if left < 0 and right < 0:
        if not is_causal:
            return q_len * kv_len
        rows = min(q_len, kv_len)
        return rows * kv_len - rows * (rows - 1) // 2
    offset = kv_len - q_len
    total = 0
    for i in range(q_len):
        position = i + offset
        if is_causal:
            hi = min(position, kv_len - 1)
        else:
            hi = min(position + right, kv_len - 1) if right >= 0 else kv_len - 1
        lo = max(0, position - left) if left >= 0 else 0
        total += max(0, hi - lo + 1)
    return total


def gqa_dense_fwd_roofline(call: "CallView") -> tuple[int, int]:
    """Dense GQA forward: two contractions per visible score; each tensor moved once."""
    ix = call.ix
    visible = visible_scores(
        ix["S_q"], ix["S_kv"], ix["is_causal"], ix["window_size_left"], ix["window_size_right"]
    )
    return 4 * ix["B"] * ix["H"] * visible * ix["D"], _derived_bytes(call)


def packed_visible_scores(call: "CallView", cu_kv: str) -> int:
    """Visible scores of a packed GQA call, summed over the requests its offsets carry."""
    ix = call.ix
    left, right = ix.get("window_size_left", -1), ix.get("window_size_right", -1)
    return sum(
        visible_scores(q, kv, ix["is_causal"], left, right)
        for q, kv in zip(_segments(call, "cu_seqlens_q"), _segments(call, cu_kv), strict=True)
    )


def _varlen_fwd(call: "CallView", cu_kv: str) -> tuple[int, int]:
    """Packed GQA forward: two contractions per visible score; each tensor moved once."""
    ix = call.ix
    return 4 * ix["H"] * packed_visible_scores(call, cu_kv) * ix["D"], _derived_bytes(call)


def gqa_varlen_fwd_roofline(call: "CallView") -> tuple[int, int]:
    """Unified packed GQA forward: two contractions per visible score."""
    return _varlen_fwd(call, "cu_seqlens_kv")


def gqa_prefill_varlen_fwd_roofline(call: "CallView") -> tuple[int, int]:
    """Packed GQA prefill: two contractions per visible score; each tensor moved once."""
    return _varlen_fwd(call, "cu_seqlens_kv")


def gqa_sliding_window_varlen_fwd_roofline(call: "CallView") -> tuple[int, int]:
    """Packed sliding-window GQA: two contractions per score inside the window."""
    return _varlen_fwd(call, "cu_seqlens_k")


def gqa_prefill_paged_with_kv_cache_fwd_roofline(call: "CallView") -> tuple[int, int]:
    """Paged GQA prefill with KV append, against the cached lengths the call carries.

    Each request's queries see its cached keys and, causally, the new ones before them. The
    cache is read as far as each request's cached tokens, the new tokens are appended into
    it, and the block table is read as far as each request's pages reach.
    """
    ix = call.ix
    heads_kv, dim, page_size = ix["H_kv"], ix["D"], ix["page_size"]
    q_lens, cache_lens = _segments(call, "cu_seqlens_q"), call.values("cache_seqlens")
    is_causal = ix["is_causal"]
    visible = sum(
        q * c + q * (q + 1) // 2 if is_causal else q * (c + q)
        for q, c in zip(q_lens, cache_lens, strict=True)
    )
    flops = 4 * ix["H"] * visible * dim
    cache_elem = call.bytes("k_pages") // max(1, prod(call.tensors["k_pages"][0]))
    old_kv = 2 * sum(cache_lens) * heads_kv * dim
    append = 2 * ix["T_q"] * heads_kv * dim
    pages_named = sum(-(-(c + q) // page_size) for q, c in zip(q_lens, cache_lens, strict=True))
    moved = call.bytes("q") + call.bytes("k_new") + call.bytes("v_new") + call.bytes("o")
    moved += (old_kv + append) * cache_elem
    moved += call.bytes("cu_seqlens_q") + call.bytes("cache_seqlens") + pages_named * 4
    if call.tensors["k_pages"][1] != call.tensors["q"][1]:
        # A narrower cache is dequantized, and then the call reads both scales.
        moved += call.bytes("k_scale") + call.bytes("v_scale")
    return flops, moved


def gqa_paged_fwd_roofline(call: "CallView") -> tuple[int, int]:
    """Read-only paged GQA: two contractions per visible score, each tensor moved once."""
    ix = call.ix
    q_lens, cache_lens = _segments(call, "cu_seqlens_q"), call.values("cache_seqlens")
    visible = sum(
        visible_scores(q, c, ix["is_causal"], ix["window_size_left"], ix["window_size_right"])
        for q, c in zip(q_lens, cache_lens, strict=True)
    )
    return 4 * ix["H"] * visible * ix["D"], _derived_bytes(call)


def _closed_chunks(length: int, block_size: int) -> int:
    """``sum(t // block_size for t in range(length))``, in closed form."""
    full, rest = divmod(length, block_size)
    return block_size * full * (full - 1) // 2 + rest * full


def nsa_closed_chunk_pairs(call: "CallView") -> int:
    """(token, chunk) pairs the NSA compression scores: ``(t + 1) // bs`` chunks at position ``t``."""
    return sum(_closed_chunks(n + 1, call.ix["bs"]) for n in _segments(call, "offsets"))


def nsa_cmp_fwd_varlen_roofline(call: "CallView") -> tuple[int, int]:
    """NSA compression forward: a QK and a PV contraction per scored (token, chunk) pair;
    every tensor is moved once."""
    ix = call.ix
    return 2 * nsa_closed_chunk_pairs(call) * ix["H"] * (ix["DK"] + ix["DV"]), _derived_bytes(call)


def nsa_topk_scored_pairs(call: "CallView") -> int:
    """(token, chunk) pairs the NSA selection scores over its two QK passes.

    The lse pass scores ``(t + 1) // bs`` chunks at position ``t``, the selection pass
    ``t // bs + 1``.
    """
    bs = call.ix["bs"]
    return sum(
        _closed_chunks(n + 1, bs) + _closed_chunks(n, bs) + n for n in _segments(call, "offsets")
    )


def nsa_topk_varlen_roofline(call: "CallView") -> tuple[int, int]:
    """NSA block selection: one QK contraction per scored pair, no PV. ``lse_in`` is passed
    and discarded, so it moves no bytes."""
    ix = call.ix
    flops = 2 * nsa_topk_scored_pairs(call) * ix["H"] * ix["D"]
    return flops, _derived_bytes(call) - call.bytes("lse_in")


def nsa_selected_block_loads(call: "CallView") -> int:
    """Block tiles the NSA sparse forward loads, summed over tokens and KV heads.

    The kernel walks ``block_counts`` entries of each row and skips a block starting past
    the token.
    """
    counts, picks = call.values("block_counts"), call.values("block_indices")
    tokens = call.values("token_indices")
    block_size = call.ix["block_size"]
    loads = 0
    for t, (row_counts, row_picks) in enumerate(zip(counts, picks, strict=True)):
        position = tokens[t][1]
        for n, blocks in zip(row_counts, row_picks, strict=True):
            loads += sum(1 for start in blocks[:n] if 0 <= start * block_size <= position)
    return loads


def nsa_fwd_varlen_roofline(call: "CallView") -> tuple[int, int]:
    """NSA sparse forward over the blocks the selection keeps.

    Scores come from the kept blocks, not the sequence length. In: q, the gathered k/v rows
    of those blocks and the selection metadata. Out: the attention output.
    """
    ix = call.ix
    rows = nsa_selected_block_loads(call) * ix["block_size"]
    elem = call.bytes("q") // max(1, prod(call.tensors["q"][0]))
    flops = 4 * rows * (ix["H"] // ix["H_kv"]) * ix["D"]
    moved = call.bytes("q") + call.bytes("o_slc") + 2 * rows * ix["D"] * elem
    moved += call.bytes("block_indices") + call.bytes("block_counts")
    moved += call.bytes("offsets") + call.bytes("token_indices")
    return flops, moved


def gqa_prefill_paged_cached_tokens(call: "CallView") -> int:
    """Cached tokens the paged prefill call reads, which its cache traffic follows."""
    return sum(call.values("cache_seqlens"))
