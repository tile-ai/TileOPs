"""Roofline cost-model functions for Tier 2 ops (attention, conv, MoE, etc.).

Each function returns a ``(flops, bytes)`` tuple of ints, the shape of
``Op.eval_roofline(self) -> tuple[int, int]``. A manifest ``roofline.func`` takes the checked
call (``docs/design/roofline.md`` § Formula Modes).

These are referenced from ``src/tileops/manifest/`` via the ``roofline.func``
field.
"""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

from tileops.manifest.dtype_rules import FLOAT8_DTYPES

if TYPE_CHECKING:
    from tileops.manifest.workload import CallView

__all__ = [
    "adaptive_pool2d_roofline",
    "attention_flops",
    "conv_roofline",
    "dsa_decode_roofline",
    "dsa_distinct_kv_rows",
    "dsa_paged_fwd_roofline",
    "dsa_selected_keys",
    "fft_c2c_roofline",
    "fp8_lightning_indexer_roofline",
    "fused_moe_fwd_roofline",
    "fused_moe_shared_expert_fwd_roofline",
    "fused_qk_norm_rope_roofline",
    "gqa_dense_fwd_roofline",
    "gqa_paged_cache_rows",
    "gqa_paged_fwd_roofline",
    "gqa_varlen_fwd_roofline",
    "hadamard_roofline",
    "lightning_indexer_scored_keys",
    "mla_kv_cache_write_roofline",
    "mla_paged_fwd_roofline",
    "mla_varlen_fwd_roofline",
    "moe_expert_mlp_roofline",
    "moe_grouped_gemm_roofline",
    "moe_layout_active_experts",
    "moe_layout_rows",
    "moe_post_permute_roofline",
    "nsa_closed_chunk_pairs",
    "nsa_compressed_fwd_varlen_roofline",
    "nsa_fwd_varlen_roofline",
    "nsa_selected_rows",
    "nsa_topk_scored_pairs",
    "nsa_topk_varlen_roofline",
    "paged_decode_cache_rows",
    "paged_decode_roofline",
    "paged_kv_cache_gather_roofline",
    "paged_kv_cache_write_roofline",
    "paged_rows",
    "pool_roofline",
    "top_k_mask_roofline",
    "top_k_top_p_mask_roofline",
    "topk_select_roofline",
    "topk_select_window_scores",
    "visible_score_rows",
    "visible_scores",
]


# Per gated element: the activation of the gate (silu: 5; the erf gelu: 5) and the multiply
# by the up projection.
_GATED_ACTIVATION = 6
# Per score: the scale, the running max, the subtraction, the exp and the sum of a softmax.
_SOFTMAX_PER_SCORE = 5
# Per score: the divide, tanh and multiply of a logit softcap.
_SOFTCAP_PER_SCORE = 3
# Per head score: the relu, the weight multiply and the add into the sum over heads.
_INDEXER_EPILOGUE_PER_SCORE = 3


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


def _routing_flops(call) -> int:
    """Routing as FusedTopKFwdOp prices it, per token: scoring (sigmoid 4 per logit; softmax 3,
    plus the row sum and a divide per kept weight unless renormalizing), top_k
    compare-and-selects per logit, the bias add, and 2 * top_k to renormalize."""
    ix = call.ix
    experts, top_k, renormalize = ix["E"], ix["top_k"], ix["renormalize"]
    if ix["scoring_func"] == "sigmoid":
        scoring = 4 * experts
    else:
        scoring = 3 * experts + (0 if renormalize else experts + top_k)
    per_token = scoring + top_k * experts + (2 * top_k if renormalize else 0)
    per_token += experts if call.present("correction_bias") else 0
    return ix["T"] * per_token


def _routed_flops(tokens: int, routes: int, ffn: int, hidden: int, scaled: bool) -> int:
    """The routed experts over ``routes`` (token, expert) pairs: the gate/up and down GEMMs,
    the gated activation, the weighted combine into each token (a multiply and an add per
    route and hidden element, as MoEPostPermuteFwdOp prices it) and the scale per output
    element when the scaling factor is not one."""
    flops = routes * (6 * ffn * hidden + _GATED_ACTIVATION * ffn + 2 * hidden)
    return flops + (tokens * hidden if scaled else 0)


def routed_expert_mlp_roofline(call) -> tuple[int, int]:
    """The expert MLP of a call handed its routing: the experts ``topk_ids`` selects are read.

    FLOPs are :func:`_routed_flops` over every route, independent of which experts the
    routes land on.
    """
    t, k, f, h = call.ix["T"], call.ix["K"], call.ix["F"], call.ix["H"]
    flops = _routed_flops(t, t * k, f, h, call.ix["routed_scaling_factor"] != 1.0)
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
    """A router with its experts: routing and the routed experts (:func:`_routed_flops`); the
    experts the routed-experts stage was handed are read.

    The routing is that stage's metadata input, read from its checked call in ``stages``
    (docs/design/roofline.md §4.7). Where no such call exists, the price takes the
    data-independent lower bound: each token selects ``top_k`` distinct experts.
    """
    ix = call.ix
    t, f, h, top_k = ix["T"], ix["F"], ix["H"], ix["top_k"]
    flops = _routing_flops(call)
    flops += _routed_flops(t, t * top_k, f, h, ix["routed_scaling_factor"] != 1.0)
    nbytes = fused_moe_active_experts(call) * _expert_weight_bytes(call)
    nbytes += 2 * call.bytes("hidden_states")
    nbytes += call.bytes("gating_output")
    if call.present("correction_bias"):
        nbytes += call.bytes("correction_bias")
    return flops, nbytes


def fused_moe_shared_expert_fwd_roofline(call) -> tuple[int, int]:
    """The routed cost of :func:`fused_moe_fwd_roofline`, plus the shared expert's two GEMMs and
    gated activation on this rank's shard, its weights and its write of ``shared_output``; the
    hidden states the routed path reads are the same storage."""
    flops, nbytes = fused_moe_fwd_roofline(call)
    if not call.present("shared_w_gate_up"):
        return flops, nbytes
    t, h = call.ix["T"], call.ix["H"]
    shard = call.ix["S"] // call.ix["tp_size"]
    elem = call.bytes("hidden_states") // (t * h)
    weights = 3 * shard * h
    flops += 2 * t * weights + _GATED_ACTIVATION * t * shard
    nbytes += (weights + t * h) * elem  # the shard's weights and shared_output
    return flops, nbytes


def moe_layout_rows(call: "CallView") -> int:
    """Expert rows a grouped layout's ``layout_metadata`` marks valid; padding rows are none.

    Masked metadata counts each expert's rows; tight contiguous rows are all valid; aligned
    physical-psum metadata ends each expert's run, which starts at the previous end rounded
    up to the alignment; aligned per-row metadata marks a padding row with ``E``.
    """
    layout, meta, experts = call.ix["layout"], call.values("layout_metadata"), call.ix["E"]
    if layout.kind == "masked":
        return sum(meta)
    if layout.packing.value == "tight":
        return call.ix["R"]
    if layout.metadata_kind.value == "per_row":
        return sum(1 for e in meta if e < experts)
    a, ends = layout.alignment, [0, *meta]
    return sum(end - -(-start // a) * a for start, end in zip(ends, ends[1:], strict=False))


def moe_layout_active_experts(call: "CallView") -> int:
    """Experts a grouped layout's ``layout_metadata`` gives at least one valid row, by the
    rules of :func:`moe_layout_rows`."""
    layout, meta, experts = call.ix["layout"], call.values("layout_metadata"), call.ix["E"]
    if layout.kind == "masked":
        return sum(1 for n in meta if n > 0)
    if layout.metadata_kind.value == "per_row":
        return len({e for e in meta if e < experts})
    a, ends = layout.alignment, [0, *meta]
    return sum(1 for start, end in zip(ends, ends[1:], strict=False) if end > -(-start // a) * a)


def _active_weight_bytes(call: "CallView", *weights: str) -> int:
    """Each tensor moved once, the per-expert *weights* only for the experts with valid rows."""
    active, experts = moe_layout_active_experts(call), call.ix["E"]
    moved = _derived_bytes(call)
    for name in weights:
        moved += call.bytes(name) * active // experts - call.bytes(name)
    return moved


def moe_grouped_gemm_roofline(call: "CallView") -> tuple[int, int]:
    """Grouped expert GEMM over the valid rows, and the gated activation when fused; an expert
    with no valid row reads no weight, every other tensor moves once."""
    ix = call.ix
    rows, fused = moe_layout_rows(call), call.ix["activation"] is not None
    flops = 2 * rows * (2 * ix["N"] if fused else ix["N"]) * ix["K"]
    flops += _GATED_ACTIVATION * rows * ix["N"] if fused else 0
    return flops, _active_weight_bytes(call, "b")


def moe_grouped_gemm_fp8_roofline(call: "CallView") -> tuple[int, int]:
    """Grouped expert GEMM over the valid rows, with no fused activation; an expert with no
    valid row reads neither its weight nor that weight's scale."""
    ix = call.ix
    flops = 2 * moe_layout_rows(call) * ix["N"] * ix["K"]
    return flops, _active_weight_bytes(call, "b", "b_scale")


def moe_expert_mlp_roofline(call: "CallView") -> tuple[int, int]:
    """Expert MLP over the valid rows: the gate/up and down GEMMs and the gated activation; an
    expert with no valid row reads no weight, every other tensor moves once."""
    f, h = call.ix["F"], call.ix["H"]
    flops = moe_layout_rows(call) * (6 * f * h + _GATED_ACTIVATION * f)
    return flops, _active_weight_bytes(call, "w_gate_up", "w_down")


def hadamard_roofline(call: "CallView") -> tuple[int, int]:
    """Fast Walsh-Hadamard transform along the last axis.

    The radix-2 stages are one add or subtract per element each, and there are
    ``log2(n // base_order)`` of them. A ``base_order`` above 1 ends with one dense
    ``base_order x base_order`` Hadamard product per group, two FLOPs per element-column.
    The ``1 / sqrt(n)`` scaling is one multiply per element. Each tensor moves once.
    """
    ix = call.ix
    n, order = ix["n"], ix["base_order"]
    radix2 = (n // order).bit_length() - 1
    flops = prod(ix["B"]) * n * (radix2 + 1 + (2 * order if order > 1 else 0))
    return flops, sum(call.bytes(t) for t in call.tensors)


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


def _window_taps(
    extent: int, out: int, kernel: int, stride: int, padding: int, dilation: int
) -> int:
    """Taps inside the input, summed over the windows of one pooled axis; padded taps are none."""
    return sum(
        1
        for o in range(out)
        for j in range(kernel)
        if 0 <= o * stride - padding + j * dilation < extent
    )


def _window_reads(
    extent: int, out: int, kernel: int, stride: int, padding: int, dilation: int
) -> int:
    """Distinct input positions the windows of one pooled axis read."""
    return len(
        {
            p
            for o in range(out)
            for j in range(kernel)
            if 0 <= (p := o * stride - padding + j * dilation) < extent
        }
    )


def _windowed_bytes(call: "CallView", axes: "list[tuple]") -> int:
    """Each tensor moved once, ``input`` at the positions some window reads.

    *axes* holds ``(extent, out, kernel, stride, padding, dilation)`` per windowed axis, the
    trailing axes of ``input``.
    """
    whole = prod(axis[0] for axis in axes)
    read = prod(_window_reads(*axis) for axis in axes)
    return _derived_bytes(call) - call.bytes("input") + call.bytes("input") * read // whole


def pool_roofline(call: "CallView") -> tuple[int, int]:
    """Fixed-window pooling: one add or comparison per input tap a window covers, and an
    average's division per output; ``input`` is read where some window reads it."""
    ix = call.ix
    if "L_in" in ix:
        axes = (("L", "W"),)
    elif "D_in" in ix:
        axes = (("D", "D"), ("H", "H"), ("W", "W"))
    else:
        axes = (("H", "H"), ("W", "W"))
    geometry = [
        (
            ix[f"{axis}_in"],
            ix[f"{axis}_out"],
            ix[f"k{k}"],
            ix[f"s{k}"],
            ix[f"p{k}"],
            ix.get(f"d{k}", 1),
        )
        for axis, k in axes
    ]
    taps = prod(_window_taps(*g) for g in geometry)
    flops = ix["N"] * ix["C"] * taps
    if "count_include_pad" in ix:  # an average divides each window's sum once
        flops += ix["N"] * ix["C"] * prod(ix[f"{axis}_out"] for axis, _ in axes)
    return flops, _windowed_bytes(call, geometry)


def conv_roofline(call: "CallView") -> tuple[int, int]:
    """Direct convolution: a multiply-add per input channel of the group and in-range tap of
    each output element, padded taps being none, and the bias add when present; ``input`` is
    read where some window reads it, every other tensor moves once."""
    ix = call.ix
    if "L_in" in ix:
        axes = (("L_in", "L_out", "W"),)
    elif "kD" in ix:
        axes = (("D", "out_D", "D"), ("H", "out_H", "H"), ("W", "out_W", "W"))
    else:
        axes = (("H", "out_H", "H"), ("W", "out_W", "W"))
    same = ix["padding"] == "same"
    geometry = []
    for extent, out, k in axes:
        kernel, dilation = ix[f"k{k}"], ix[f"d{k}"]
        # "same" pads the left side by half the dilated kernel span, rounded down.
        pad = dilation * (kernel - 1) // 2 if same else ix[f"p{k}"]
        geometry.append((ix[extent], ix[out], kernel, ix[f"s{k}"], pad, dilation))
    taps = prod(_window_taps(*g) for g in geometry)
    outputs = prod(g[1] for g in geometry)
    flops = 2 * ix["N"] * ix["C_out"] * ix["C_in_g"] * taps
    if call.present("bias"):
        flops += ix["N"] * ix["C_out"] * outputs
    return flops, _windowed_bytes(call, geometry)


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


def visible_score_rows(
    q_len: int, kv_len: int, is_causal: bool, left: int, right: int
) -> "tuple[int, int]":
    """``(scores, rows)`` of one request under bottom-right alignment: the keys its queries see,
    summed, and the queries that see at least one.

    Query ``i`` sits at key position ``i + kv_len - q_len``; ``left`` and ``right`` bound the
    window around it, ``-1`` meaning unlimited.
    """
    if left < 0 and right < 0:
        if kv_len <= 0:
            return 0, 0
        if not is_causal:
            return q_len * kv_len, q_len
        rows = min(q_len, kv_len)
        return rows * kv_len - rows * (rows - 1) // 2, rows
    offset = kv_len - q_len
    total = rows = 0
    for i in range(q_len):
        position = i + offset
        if is_causal:
            hi = min(position, kv_len - 1)
        else:
            hi = min(position + right, kv_len - 1) if right >= 0 else kv_len - 1
        lo = max(0, position - left) if left >= 0 else 0
        if hi >= lo:
            total += hi - lo + 1
            rows += 1
    return total, rows


def visible_scores(q_len: int, kv_len: int, is_causal: bool, left: int, right: int) -> int:
    """Keys each query of one request sees under bottom-right alignment, summed over its queries."""
    return visible_score_rows(q_len, kv_len, is_causal, left, right)[0]


def attention_flops(
    heads: int, scores: int, rows: int, qk_dim: int, v_dim: int, softcap: bool = False
) -> int:
    """Attention arithmetic of ``heads`` query heads over ``scores`` scores and ``rows`` query rows.

    Per score, a QK contraction over ``qk_dim`` and a PV contraction over ``v_dim`` (2 per
    multiply-add), the softmax and, when present, the softcap; per output element, the
    softmax's normalizing divide.
    """
    per_score = 2 * qk_dim + 2 * v_dim + _SOFTMAX_PER_SCORE
    per_score += _SOFTCAP_PER_SCORE if softcap else 0
    return heads * (scores * per_score + rows * v_dim)


def _softcap(call: "CallView") -> bool:
    """Whether the call caps its logits; an op may bind the absent cap as ``0.0``."""
    return bool(call.ix.get("softcap"))


def gqa_dense_fwd_roofline(call: "CallView") -> tuple[int, int]:
    """Dense GQA forward: the attention arithmetic of the visible scores; each tensor moved once."""
    ix = call.ix
    scores, rows = visible_score_rows(
        ix["S_q"], ix["S_kv"], ix["is_causal"], ix["window_size_left"], ix["window_size_right"]
    )
    flops = ix["B"] * attention_flops(ix["H"], scores, rows, ix["D"], ix["D"], _softcap(call))
    return flops, _derived_bytes(call)


def gqa_varlen_fwd_roofline(call: "CallView") -> tuple[int, int]:
    """Packed GQA forward: the attention arithmetic of the scores each request sees under its
    mask and window, summed over the requests the offsets carry; each tensor moved once."""
    ix = call.ix
    left, right = ix["window_size_left"], ix["window_size_right"]
    pairs = [
        visible_score_rows(q, kv, ix["is_causal"], left, right)
        for q, kv in zip(
            _segments(call, "cu_seqlens_q"), _segments(call, "cu_seqlens_kv"), strict=True
        )
    ]
    scores, rows = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
    flops = attention_flops(ix["H"], scores, rows, ix["D"], ix["D"], _softcap(call))
    return flops, _derived_bytes(call)


def paged_rows(table: list, lengths: list, page_size: int, starts: "list | None" = None):
    """Distinct pool rows a paged read covers, and the block-table entries it consults.

    Request ``b`` reads its rows ``[starts[b], lengths[b])``; row ``r`` lives at offset
    ``r % page_size`` of page ``table[b][r // page_size]``. Two requests naming one page
    read its shared rows once.
    """
    rows: dict = {}
    consulted = 0
    for b, end in enumerate(lengths):
        start = starts[b] if starts else 0
        if end <= start:
            continue
        first, last = start // page_size, (end - 1) // page_size
        consulted += last - first + 1
        for p in range(first, last + 1):
            lo = max(start - p * page_size, 0)
            hi = min(end - p * page_size, page_size)
            rows.setdefault(table[b][p], set()).update(range(lo, hi))
    return sum(len(r) for r in rows.values()), consulted


def _paged_kv_bytes(call: "CallView", k: str, v: str, table: str, lengths, page_size, starts=None):
    """``bytes`` with the key and value pools and the block table priced at what the call reads."""
    rows, consulted = paged_rows(call.values(table), lengths, page_size, starts)
    pool_rows = prod(call.tensors[k][0][:-2])  # [.., H_kv, D]: every leading axis indexes a row
    moved = _derived_bytes(call) - call.bytes(k) - call.bytes(v) - call.bytes(table)
    moved += rows * (call.bytes(k) + call.bytes(v)) // pool_rows
    return moved + consulted * call.bytes(table) // max(1, prod(call.tensors[table][0]))


def _gqa_paged_read_starts(call: "CallView") -> list:
    """The first cache row each paged GQA request reads.

    The earliest query sits at key ``c - q``; a left window cuts the rows before it, and a
    request with no query reads none.
    """
    left = call.ix["window_size_left"]
    return [
        c if q == 0 else max(0, c - q - left) if left >= 0 else 0
        for q, c in zip(_segments(call, "cu_seqlens_q"), call.values("cache_seqlens"), strict=True)
    ]


def gqa_paged_cache_rows(call: "CallView") -> int:
    """Distinct pool rows a paged GQA call reads through its page table."""
    page_size = call.tensors["k_pages"][0][1]
    lengths = call.values("cache_seqlens")
    return paged_rows(call.values("page_table"), lengths, page_size, _gqa_paged_read_starts(call))[
        0
    ]


def gqa_paged_fwd_roofline(call: "CallView") -> tuple[int, int]:
    """Read-only paged GQA: two contractions per visible score.

    The cache is read as far as some query of each request sees, from the pages its page
    table names; every other tensor moves once.
    """
    ix = call.ix
    q_lens, cache_lens = _segments(call, "cu_seqlens_q"), call.values("cache_seqlens")
    pairs = [
        visible_score_rows(q, c, ix["is_causal"], ix["window_size_left"], ix["window_size_right"])
        for q, c in zip(q_lens, cache_lens, strict=True)
    ]
    scores, rows = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
    page_size = call.tensors["k_pages"][0][1]
    moved = _paged_kv_bytes(
        call,
        "k_pages",
        "v_pages",
        "page_table",
        cache_lens,
        page_size,
        _gqa_paged_read_starts(call),
    )
    return attention_flops(ix["H"], scores, rows, ix["D"], ix["D"], _softcap(call)), moved


def paged_decode_cache_rows(call: "CallView") -> int:
    """Distinct pool rows a paged decode call reads through its block table."""
    lengths = call.values("real_seqlen_kv")
    return paged_rows(call.values("block_table"), lengths, call.ix["page_size"])[0]


def paged_decode_roofline(call: "CallView") -> tuple[int, int]:
    """Paged decode against the cached lengths the call carries.

    Each query row sees its request's ``real_seqlen_kv`` keys, causally aligned to the end
    when it carries several; the pool is read as far as those lengths reach through the
    block table.
    """
    ix = call.ix
    lengths = call.values("real_seqlen_kv")
    s_q = ix.get("S_q", 1)
    pairs = [visible_score_rows(s_q, n, ix.get("is_causal", False), -1, -1) for n in lengths]
    scores, rows = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
    moved = _paged_kv_bytes(call, "k", "v", "block_table", lengths, ix["page_size"])
    return attention_flops(ix["H"], scores, rows, ix["D"], ix["D"], _softcap(call)), moved


def _closed_chunks(length: int, block_size: int) -> int:
    """``sum(t // block_size for t in range(length))``, in closed form."""
    full, rest = divmod(length, block_size)
    return block_size * full * (full - 1) // 2 + rest * full


def nsa_closed_chunk_pairs(call: "CallView") -> int:
    """(token, chunk) pairs the NSA compression scores: ``(t + 1) // bs`` chunks at position ``t``."""
    return sum(_closed_chunks(n + 1, call.ix["bs"]) for n in _segments(call, "offsets"))


def nsa_compressed_fwd_varlen_roofline(call: "CallView") -> tuple[int, int]:
    """NSA compression forward: the attention arithmetic of each scored (token, chunk) pair,
    over the tokens that close at least one chunk; every tensor is moved once."""
    ix = call.ix
    bs = ix["bs"]
    rows = sum(max(0, n - bs + 1) for n in _segments(call, "offsets"))
    flops = attention_flops(ix["H"], nsa_closed_chunk_pairs(call), rows, ix["DK"], ix["DV"])
    return flops, _derived_bytes(call)


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
    """NSA block selection: one QK contraction per scored pair, no PV."""
    ix = call.ix
    flops = 2 * nsa_topk_scored_pairs(call) * ix["H"] * ix["D"]
    return flops, _derived_bytes(call)


def _nsa_selection(call: "CallView") -> "tuple[int, int, int]":
    """``(scored, distinct, rows)`` key rows of the NSA sparse forward, over tokens and KV heads.

    Token ``t`` scores the first ``block_counts`` blocks of its selection, each clipped at
    ``t`` when causal and at the sequence end otherwise; a block starting past that bound
    scores nothing. ``scored`` counts a row once per token that scores it, ``distinct``
    once per KV head, which is what is read; ``rows`` counts the (token, KV head) pairs
    that score at least one key.
    """
    counts, picks = call.values("block_counts"), call.values("block_indices")
    tokens, offsets = call.values("token_indices"), call.values("offsets")
    block_size, is_causal = call.ix["block_size"], call.ix["is_causal"]
    scored = rows = 0
    distinct: set = set()
    for (request, position), row_counts, row_picks in zip(tokens, counts, picks, strict=True):
        bos = offsets[request]
        end = position + 1 if is_causal else offsets[request + 1] - bos
        for head, (n, blocks) in enumerate(zip(row_counts, row_picks, strict=True)):
            before = scored
            for start in blocks[:n]:
                lo = start * block_size
                if 0 <= lo < end:
                    hi = min(lo + block_size, end)
                    scored += hi - lo
                    distinct.update((head, bos + r) for r in range(lo, hi))
            rows += scored > before
    return scored, len(distinct), rows


def nsa_selected_rows(call: "CallView") -> "tuple[int, int]":
    """``(scored, distinct)`` key rows of the NSA sparse forward (see ``_nsa_selection``)."""
    return _nsa_selection(call)[:2]


def nsa_fwd_varlen_roofline(call: "CallView") -> tuple[int, int]:
    """NSA sparse forward over the key rows the selection keeps.

    The attention arithmetic of each scored (token, key) pair per query head. In: q, the
    distinct k/v rows the selection reaches and the selection metadata. Out: the output.
    """
    ix = call.ix
    scored, distinct, rows = _nsa_selection(call)
    elem = call.bytes("q") // max(1, prod(call.tensors["q"][0]))
    flops = attention_flops(ix["H"] // ix["H_kv"], scored, rows, ix["D"], ix["D"])
    moved = call.bytes("q") + call.bytes("o_slc") + 2 * distinct * ix["D"] * elem
    moved += call.bytes("block_indices") + call.bytes("block_counts")
    moved += call.bytes("offsets") + call.bytes("token_indices")
    return flops, moved


def _dsa_selections(call: "CallView") -> "list[tuple[int, int, set]]":
    """``(batch, kv head, selected keys)`` of every query row the DeepSeek sparse decode scores.

    A top-k slot selects key ``j`` when ``0 <= j < S_kv`` and the key's last compressed
    position ``(j + 1) * stride_kv - 1`` is at or before the query's ``q_start_index_s + s``;
    padded and repeated slots select nothing more.
    """
    ix = call.ix
    stride, first, extent = ix["stride_kv"], ix["q_start_index_s"], ix["S_kv"]
    return [
        (b, g, {j for j in slots if 0 <= j < extent and (j + 1) * stride - 1 <= first + s})
        for b, batch in enumerate(call.values("indices"))
        for s, heads in enumerate(batch)
        for g, slots in enumerate(heads)
    ]


def dsa_selected_keys(call: "CallView") -> int:
    """(query, key) pairs the DeepSeek sparse decode scores, over batch, tokens and KV heads."""
    return sum(len(keys) for _b, _g, keys in _dsa_selections(call))


def dsa_distinct_kv_rows(call: "CallView") -> int:
    """The ``kv`` rows some query of the DeepSeek sparse decode selects, per batch and KV head."""
    return len({(b, g, j) for b, g, keys in _dsa_selections(call) for j in keys})


def dsa_decode_roofline(call: "CallView") -> tuple[int, int]:
    """DeepSeek sparse decode: the attention arithmetic of each selected key per query head, QK
    over ``D + dim_tail`` and PV over ``D``. ``kv`` is read at the rows some query selects;
    every other tensor moves once."""
    ix = call.ix
    selections = _dsa_selections(call)
    scores = sum(len(keys) for _b, _g, keys in selections)
    rows = sum(1 for _b, _g, keys in selections if keys)
    flops = attention_flops(ix["H"] // ix["H_kv"], scores, rows, ix["D"] + ix["dim_tail"], ix["D"])
    kv_row = call.bytes("kv") // max(1, prod(call.tensors["kv"][0][:-1]))
    return flops, _derived_bytes(call) - call.bytes("kv") + dsa_distinct_kv_rows(call) * kv_row


def lightning_indexer_scored_keys(call: "CallView") -> int:
    """Keys the lightning indexer scores per batch row: query ``s`` scores its window
    ``[cu_seqlen_ks[s], cu_seqlen_ke[s])``."""
    return sum(
        max(0, e - s)
        for s, e in zip(call.values("cu_seqlen_ks"), call.values("cu_seqlen_ke"), strict=True)
    )


def fp8_lightning_indexer_roofline(call: "CallView") -> tuple[int, int]:
    """Lightning indexer: per query head and windowed key, a D-long contraction and the relu,
    weight and head-sum epilogue; each tensor moves once, ``logits`` written whole."""
    ix = call.ix
    per_score = 2 * ix["D"] + _INDEXER_EPILOGUE_PER_SCORE
    return ix["B"] * ix["H"] * per_score * lightning_indexer_scored_keys(call), _derived_bytes(call)


def topk_select_window_scores(call: "CallView") -> int:
    """Scores in the windows ``[starts, ends)`` of every row ``(b, s)``, per group."""
    return sum(
        max(0, e - s)
        for srow, erow in zip(call.values("starts"), call.values("ends"), strict=True)
        for s, e in zip(srow, erow, strict=True)
    )


def topk_select_roofline(call: "CallView") -> tuple[int, int]:
    """Top-k selection: one comparison per score in a row's window, per group; only those
    scores are read."""
    ix = call.ix
    widths = topk_select_window_scores(call)
    shape = call.tensors["index_score"][0]
    read = widths * ix["G"] * call.bytes("index_score") // max(1, prod(shape))
    return ix["G"] * widths, _derived_bytes(call) - call.bytes("index_score") + read


# ---------------------------------------------------------------- paged caches and MLA


def _elem_bytes(call: "CallView", name: str) -> int:
    """Bytes of one element of tensor *name*."""
    return call.bytes(name) // max(1, prod(call.tensors[name][0]))


def _paged_cache_read_bytes(
    call: "CallView", cache: str, table: str, ends: list, starts: "list | None" = None
) -> int:
    """Bytes of the distinct rows of the paged *cache* ``[NP, PS, ...]`` that the requests'
    ranges ``[starts[b], ends[b])`` reach through *table*, plus the table entries consulted."""
    shape = call.tensors[cache][0]
    rows, consulted = paged_rows(call.values(table), ends, shape[1], starts)
    row_bytes = call.bytes(cache) // max(1, shape[0] * shape[1])
    return rows * row_bytes + consulted * _elem_bytes(call, table)


def mla_paged_fwd_roofline(call: "CallView") -> tuple[int, int]:
    """Paged MLA decode: QK over the whole latent row, PV over its first ``kv_lora_rank``
    columns, for the scores each query row sees in its request's cache.

    An FP8 cache's per-tensor scale folds into the score scale and, once per query row and
    head, into the softmax denominator. The cache is read at the rows the block table reaches;
    every other tensor moves once.
    """
    ix = call.ix
    lengths = call.values("cache_seqlens")
    pairs = [visible_score_rows(ix["S_q"], c, ix["is_causal"], -1, -1) for c in lengths]
    scores, rows = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
    flops = attention_flops(ix["H"], scores, rows, ix["DK"], ix["kv_lora_rank"])
    if call.tensors["kv_cache"][1] in FLOAT8_DTYPES:
        flops += ix["H"] * rows
    moved = _derived_bytes(call) - call.bytes("kv_cache") - call.bytes("block_table")
    moved += _paged_cache_read_bytes(call, "kv_cache", "block_table", lengths)
    return flops, moved


def mla_varlen_fwd_roofline(call: "CallView") -> tuple[int, int]:
    """Packed MLA prefill: QK over ``DN + PE``, PV over ``DV``, for the scores each request
    sees under its mask; every tensor moves once."""
    ix = call.ix
    pairs = [
        visible_score_rows(n, n, ix["is_causal"], -1, -1) for n in _segments(call, "cu_seqlens")
    ]
    scores, rows = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
    flops = attention_flops(ix["H"], scores, rows, ix["DN"] + ix["PE"], ix["DV"])
    return flops, _derived_bytes(call)


# FlashMLA DeepSeek-V3.2 FP8 cache row: the QK width and the value (latent) width.
_DSA_QK_DIM, _DSA_V_DIM = 576, 512


def dsa_paged_fwd_roofline(call: "CallView") -> tuple[int, int]:
    """Paged DeepSeek Sparse Attention (DSA) over an FP8 latent cache.

    Every index slot ``0 <= j < cache_seqlens[b]`` is a score, a repeated slot once per
    occurrence. The latent of each distinct cache row is dequantized once (a multiply per
    value) and read once; the block table is read at the pages the slots resolve to; ``q`` is
    read at the query rows with a score; every other tensor moves once.
    """
    ix = call.ix
    lengths, table = call.values("cache_seqlens"), call.values("block_table")
    page_size = call.tensors["kv_cache"][0][1]
    scores = rows = 0
    distinct: set = set()
    pages: set = set()
    for b, per_query in enumerate(call.values("indices")):
        for slots in per_query:
            valid = [j for j in slots if 0 <= j < lengths[b]]
            scores += len(valid)
            rows += bool(valid)
            for j in valid:
                pages.add((b, j // page_size))
                distinct.add((table[b][j // page_size], j % page_size))
    flops = attention_flops(ix["H"], scores, rows, _DSA_QK_DIM, _DSA_V_DIM)
    flops += _DSA_V_DIM * len(distinct)
    shape = call.tensors["kv_cache"][0]
    moved = _derived_bytes(call) - call.bytes("q") - call.bytes("kv_cache")
    moved -= call.bytes("block_table")
    moved += rows * ix["H"] * _DSA_QK_DIM * _elem_bytes(call, "q")
    moved += len(distinct) * shape[2] + len(pages) * _elem_bytes(call, "block_table")
    return flops, moved


def _written_slots(call: "CallView") -> int:
    """Tokens a cache write stores: the ``slot_mapping`` entries other than -1."""
    return sum(1 for s in call.values("slot_mapping") if s >= 0)


def paged_kv_cache_write_roofline(call: "CallView") -> tuple[int, int]:
    """Paged K/V cache write: each token with a slot reads its k and v rows and writes them
    into both caches; an FP8 cache scales and saturates each value (2 per value)."""
    shape = call.tensors["k"][0]
    tokens, row = _written_slots(call), prod(shape[1:])
    flops = 4 * tokens * row if call.tensors["k_pages"][1] in FLOAT8_DTYPES else 0
    moved = 2 * tokens * row * (_elem_bytes(call, "k") + _elem_bytes(call, "k_pages"))
    moved += call.bytes("slot_mapping")
    if tokens * row:
        moved += sum(call.bytes(s) for s in ("k_scale", "v_scale") if call.present(s))
    return flops, moved


def mla_kv_cache_write_roofline(call: "CallView") -> tuple[int, int]:
    """MLA latent-cache write: each token with a slot reads kv_c and k_pe and writes the
    concatenated row; an FP8 cache scales and saturates each value (2 per value), and a fused
    RoPE rotates k_pe (3 per value) from the cos/sin rows the tokens' positions name."""
    ix = call.ix
    width, pe = ix["DC"] + ix["PE"], ix["PE"]
    slots = call.values("slot_mapping")
    tokens = _written_slots(call)
    flops = 2 * tokens * width if call.tensors["kv_cache"][1] in FLOAT8_DTYPES else 0
    moved = tokens * width * (_elem_bytes(call, "kv_c") + _elem_bytes(call, "kv_cache"))
    moved += call.bytes("slot_mapping")
    moved += call.bytes("scale") if tokens * width and call.present("scale") else 0
    if ix["fuse_rope"]:
        flops += 3 * tokens * pe
        positions = [p for p, s in zip(call.values("positions"), slots, strict=True) if s >= 0]
        moved += len(positions) * _elem_bytes(call, "positions")
        moved += len(set(positions)) * pe * _elem_bytes(call, "cos_sin_cache")
    return flops, moved


def paged_kv_cache_gather_roofline(call: "CallView") -> tuple[int, int]:
    """Paged cache gather: the cache rows each request's range reaches are read once and
    written to ``dst``; an FP8 cache is dequantized with a multiply per value."""
    lengths = _segments(call, "cu_seq_lens")
    starts = call.values("seq_starts") if call.present("seq_starts") else [0] * len(lengths)
    ends = [s + n for s, n in zip(starts, lengths, strict=True)]
    fp8 = call.tensors["cache"][1] in FLOAT8_DTYPES
    flops = prod(call.tensors["dst"][0]) if fp8 else 0
    moved = _derived_bytes(call) - call.bytes("cache") - call.bytes("block_table")
    moved += _paged_cache_read_bytes(call, "cache", "block_table", ends, starts)
    if call.present("scale") and not prod(call.tensors["dst"][0]):
        moved -= call.bytes("scale")  # nothing is dequantized
    return flops, moved


def fused_qk_norm_rope_roofline(call: "CallView") -> tuple[int, int]:
    """Q/K RMSNorm and RoPE in place: per q and k value, RMSNorm with its weight (4), and 3
    per rotated value. The q and k columns are read and written, the v columns untouched,
    and the cos/sin rows the positions name read once."""
    ix = call.ix
    heads = ix["num_heads"] + ix["num_kv_heads"]
    values = ix["N"] * heads * ix["D"]
    flops = ix["N"] * heads * (4 * ix["D"] + 3 * ix["R"])
    moved = 2 * values * _elem_bytes(call, "qkv")
    moved += sum(call.bytes(t) for t in ("q_weight", "k_weight", "positions"))
    moved += len(set(call.values("positions"))) * ix["R"] * _elem_bytes(call, "cos_sin_cache")
    return flops, moved


# ---------------------------------------------------------------- sampling


def top_k_mask_roofline(call: "CallView") -> tuple[int, int]:
    """Top-k logit mask: a threshold selection (1 per logit) and the mask (1 per logit) on
    each row ``k`` restricts; a row with ``k >= V`` is the identity. Every tensor moves once."""
    vocab = call.ix["V"]
    return 2 * vocab * sum(1 for k in call.values("k") if k < vocab), _derived_bytes(call)


def top_k_top_p_mask_roofline(call: "CallView") -> tuple[int, int]:
    """Top-k then top-p logit mask, per row: the top-k threshold (1 per logit) where ``k``
    restricts the row; over the ``min(k, V)`` survivors the softmax numerator (max,
    subtract, exp, sum) and the weighted threshold selection (compare, accumulate); ``p``
    times the sum; the final mask (1 per logit). Every tensor moves once."""
    vocab = call.ix["V"]
    flops = sum(
        (vocab if k < vocab else 0) + 6 * min(k, vocab) + 1 + vocab for k in call.values("k")
    )
    return flops, _derived_bytes(call)
