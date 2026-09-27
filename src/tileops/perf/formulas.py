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

if TYPE_CHECKING:
    from tileops.manifest.workload import CallView

__all__ = [
    "adaptive_pool2d_roofline",
    "conv_roofline",
    "dsa_decode_roofline",
    "dsa_selected_keys",
    "fft_c2c_roofline",
    "fp8_lightning_indexer_roofline",
    "fused_moe_fwd_roofline",
    "fused_moe_shared_expert_fwd_roofline",
    "gqa_dense_fwd_roofline",
    "gqa_paged_cache_rows",
    "gqa_paged_fwd_roofline",
    "gqa_prefill_paged_cache_rows",
    "gqa_prefill_paged_with_kv_cache_fwd_roofline",
    "gqa_prefill_varlen_fwd_roofline",
    "gqa_sliding_window_varlen_fwd_roofline",
    "gqa_varlen_fwd_roofline",
    "lightning_indexer_scored_keys",
    "moe_expert_mlp_roofline",
    "moe_grouped_gemm_roofline",
    "moe_layout_rows",
    "moe_post_permute_roofline",
    "nsa_closed_chunk_pairs",
    "nsa_cmp_fwd_varlen_roofline",
    "nsa_fwd_varlen_roofline",
    "nsa_selected_rows",
    "nsa_topk_scored_pairs",
    "nsa_topk_varlen_roofline",
    "packed_visible_scores",
    "paged_decode_cache_rows",
    "paged_decode_roofline",
    "paged_rows",
    "pool_roofline",
    "topk_selector_roofline",
    "topk_selector_window_scores",
    "visible_scores",
]


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
    this rank's shard, its weights and its write of ``shared_output``; the hidden states the
    routed path reads are the same storage."""
    flops, nbytes = fused_moe_fwd_roofline(call)
    if not call.present("shared_w_gate_up"):
        return flops, nbytes
    t, h = call.ix["T"], call.ix["H"]
    shard = call.ix["S"] // call.ix["tp_size"]
    elem = call.bytes("hidden_states") // (t * h)
    weights = 3 * shard * h
    flops += 2 * t * weights
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


def moe_grouped_gemm_roofline(call: "CallView") -> tuple[int, int]:
    """Grouped expert GEMM over the valid rows, and the gated activation when fused; each
    tensor moves once."""
    ix = call.ix
    rows, fused = moe_layout_rows(call), call.ix["activation"] is not None
    flops = 2 * rows * (2 * ix["N"] if fused else ix["N"]) * ix["K"]
    return flops + (6 * rows * ix["N"] if fused else 0), _derived_bytes(call)


def moe_expert_mlp_roofline(call: "CallView") -> tuple[int, int]:
    """Expert MLP over the valid rows: the gate/up and down GEMMs and the gated activation;
    each tensor moves once."""
    f, h = call.ix["F"], call.ix["H"]
    return moe_layout_rows(call) * (6 * f * h + 6 * f), _derived_bytes(call)


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


def pool_roofline(call: "CallView") -> tuple[int, int]:
    """Fixed-window pooling: one add or comparison per input tap a window covers, and an
    average's division per output."""
    ix = call.ix
    if "L_in" in ix:
        axes = (("L", "W"),)
    elif "D_in" in ix:
        axes = (("D", "D"), ("H", "H"), ("W", "W"))
    else:
        axes = (("H", "H"), ("W", "W"))
    taps = prod(
        _window_taps(
            ix[f"{axis}_in"],
            ix[f"{axis}_out"],
            ix[f"k{k}"],
            ix[f"s{k}"],
            ix[f"p{k}"],
            ix.get(f"d{k}", 1),
        )
        for axis, k in axes
    )
    flops = ix["N"] * ix["C"] * taps
    if "count_include_pad" in ix:  # an average divides each window's sum once
        flops += ix["N"] * ix["C"] * prod(ix[f"{axis}_out"] for axis, _ in axes)
    return flops, sum(call.bytes(t) for t in call.tensors)


def conv_roofline(call: "CallView") -> tuple[int, int]:
    """Direct convolution: a multiply-add per input channel of the group and in-range tap of
    each output element, padded taps being none, and the bias add when present; each tensor
    moves once."""
    ix = call.ix
    if "L_in" in ix:
        axes = (("L_in", "L_out", "W"),)
    elif "kD" in ix:
        axes = (("D", "out_D", "D"), ("H", "out_H", "H"), ("W", "out_W", "W"))
    else:
        axes = (("H", "out_H", "H"), ("W", "out_W", "W"))
    same = ix["padding"] == "same"
    taps = outputs = 1
    for extent, out, k in axes:
        kernel, dilation = ix[f"k{k}"], ix[f"d{k}"]
        # "same" pads the left side by half the dilated kernel span, rounded down.
        pad = dilation * (kernel - 1) // 2 if same else ix[f"p{k}"]
        taps *= _window_taps(ix[extent], ix[out], kernel, ix[f"s{k}"], pad, dilation)
        outputs *= ix[out]
    flops = 2 * ix["N"] * ix["C_out"] * ix["C_in_g"] * taps
    if call.present("bias"):
        flops += ix["N"] * ix["C_out"] * outputs
    return flops, sum(call.bytes(t) for t in call.tensors)


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
    distinct cache rows the requests' cached tokens name are read once, the new tokens are
    appended into the cache, and the block table is read as far as each request's pages reach.
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
    old_kv = 2 * gqa_prefill_paged_cache_rows(call) * heads_kv * dim
    append = 2 * ix["T_q"] * heads_kv * dim
    pages_named = sum(-(-(c + q) // page_size) for q, c in zip(q_lens, cache_lens, strict=True))
    moved = call.bytes("q") + call.bytes("k_new") + call.bytes("v_new") + call.bytes("o")
    moved += (old_kv + append) * cache_elem
    moved += call.bytes("cu_seqlens_q") + call.bytes("cache_seqlens") + pages_named * 4
    if call.tensors["k_pages"][1] != call.tensors["q"][1]:
        # A narrower cache is dequantized, and then the call reads both scales.
        moved += call.bytes("k_scale") + call.bytes("v_scale")
    return flops, moved


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
    visible = sum(
        visible_scores(q, c, ix["is_causal"], ix["window_size_left"], ix["window_size_right"])
        for q, c in zip(q_lens, cache_lens, strict=True)
    )
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
    return 4 * ix["H"] * visible * ix["D"], moved


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
    visible = sum(visible_scores(s_q, n, ix.get("is_causal", False), -1, -1) for n in lengths)
    moved = _paged_kv_bytes(call, "k", "v", "block_table", lengths, ix["page_size"])
    return 4 * ix["H"] * visible * ix["D"], moved


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


def nsa_selected_rows(call: "CallView") -> "tuple[int, int]":
    """``(scored, distinct)`` key rows of the NSA sparse forward, over tokens and KV heads.

    Token ``t`` scores the first ``block_counts`` blocks of its selection; a block starting
    past it scores nothing, and a causal call clips each block at ``t``, a non-causal one at
    the sequence end. ``scored`` counts a row once per token that scores it, ``distinct``
    once per KV head, which is what is read.
    """
    counts, picks = call.values("block_counts"), call.values("block_indices")
    tokens, offsets = call.values("token_indices"), call.values("offsets")
    block_size, is_causal = call.ix["block_size"], call.ix["is_causal"]
    scored = 0
    distinct: set = set()
    for (request, position), row_counts, row_picks in zip(tokens, counts, picks, strict=True):
        bos = offsets[request]
        end = position + 1 if is_causal else offsets[request + 1] - bos
        for head, (n, blocks) in enumerate(zip(row_counts, row_picks, strict=True)):
            for start in blocks[:n]:
                lo = start * block_size
                if 0 <= lo <= position:
                    hi = min(lo + block_size, end)
                    scored += hi - lo
                    distinct.update((head, bos + r) for r in range(lo, hi))
    return scored, len(distinct)


def nsa_fwd_varlen_roofline(call: "CallView") -> tuple[int, int]:
    """NSA sparse forward over the key rows the selection keeps.

    A QK and a PV contraction per scored (token, key) pair and query head. In: q, the
    distinct k/v rows the selection reaches and the selection metadata. Out: the output.
    """
    ix = call.ix
    scored, distinct = nsa_selected_rows(call)
    elem = call.bytes("q") // max(1, prod(call.tensors["q"][0]))
    flops = 4 * scored * (ix["H"] // ix["H_kv"]) * ix["D"]
    moved = call.bytes("q") + call.bytes("o_slc") + 2 * distinct * ix["D"] * elem
    moved += call.bytes("block_indices") + call.bytes("block_counts")
    moved += call.bytes("offsets") + call.bytes("token_indices")
    return flops, moved


def dsa_selected_keys(call: "CallView") -> int:
    """(query, key) pairs the DeepSeek sparse decode scores, over batch, tokens and KV heads.

    A top-k slot selects key ``j`` when ``0 <= j < S_kv`` and the key's last compressed
    position ``(j + 1) * stride_kv - 1`` is at or before the query's ``q_start_index_s + s``;
    padded and repeated slots select nothing more.
    """
    ix = call.ix
    stride, first, extent = ix["stride_kv"], ix["q_start_index_s"], ix["S_kv"]
    return sum(
        len({j for j in slots if 0 <= j < extent and (j + 1) * stride - 1 <= first + s})
        for batch in call.values("indices")
        for s, heads in enumerate(batch)
        for slots in heads
    )


def dsa_decode_roofline(call: "CallView") -> tuple[int, int]:
    """DeepSeek sparse decode: per selected key and query head, a QK contraction over
    ``D + dim_tail`` and a PV contraction over ``D``; each tensor moves once."""
    ix = call.ix
    per_key = 2 * (ix["H"] // ix["H_kv"]) * (2 * ix["D"] + ix["dim_tail"])
    return per_key * dsa_selected_keys(call), _derived_bytes(call)


def lightning_indexer_scored_keys(call: "CallView") -> int:
    """Keys the lightning indexer scores per batch row: query ``s`` scores its window
    ``[cu_seqlen_ks[s], cu_seqlen_ke[s])``."""
    return sum(
        max(0, e - s)
        for s, e in zip(call.values("cu_seqlen_ks"), call.values("cu_seqlen_ke"), strict=True)
    )


def fp8_lightning_indexer_roofline(call: "CallView") -> tuple[int, int]:
    """Lightning indexer: one D-long contraction per query head and windowed key; each
    tensor moves once, ``logits`` written whole."""
    ix = call.ix
    return 2 * ix["B"] * ix["H"] * ix["D"] * lightning_indexer_scored_keys(call), _derived_bytes(
        call
    )


def topk_selector_window_scores(call: "CallView") -> int:
    """Scores in the windows ``[starts, ends)`` of every row ``(b, s)``, per group."""
    return sum(
        max(0, e - s)
        for srow, erow in zip(call.values("starts"), call.values("ends"), strict=True)
        for s, e in zip(srow, erow, strict=True)
    )


def topk_selector_roofline(call: "CallView") -> tuple[int, int]:
    """Top-k selection: one comparison per score in a row's window, per group; only those
    scores are read."""
    ix = call.ix
    widths = topk_selector_window_scores(call)
    shape = call.tensors["index_score"][0]
    read = widths * ix["G"] * call.bytes("index_score") // max(1, prod(shape))
    return ix["G"] * widths, _derived_bytes(call) - call.bytes("index_score") + read


def gqa_prefill_paged_cache_rows(call: "CallView") -> int:
    """Distinct cache rows the paged prefill call reads, which its cache traffic follows."""
    q_lens, cache_lens = _segments(call, "cu_seqlens_q"), call.values("cache_seqlens")
    # A request with no new token attends nothing and reads none of its cache.
    read_lens = [c if q else 0 for q, c in zip(q_lens, cache_lens, strict=True)]
    return paged_rows(call.values("block_table"), read_lens, call.ix["page_size"])[0]
