"""Roofline cost-model functions for Tier 2 ops (attention, conv, MoE, etc.).

Each function takes the bound Op instance and returns a ``(flops, bytes)``
tuple of ints, matching the ``Op.eval_roofline(self) -> tuple[int, int]``
shape that codegen emits for ``func`` mode (see ``docs/design/roofline.md`` §4.4.2).

These are referenced from ``tileops/manifest/`` via the ``roofline.func``
field, e.g.::

    roofline:
      func: "tileops.perf.formulas.mha_fwd_roofline"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tileops.ops.op_base import Op

__all__ = [
    "add_fwd_roofline",
    "bitwise_and_fwd_roofline",
    "bitwise_or_fwd_roofline",
    "bitwise_xor_fwd_roofline",
    "clamp_fwd_roofline",
    "clamp_max_fwd_roofline",
    "clamp_min_fwd_roofline",
    "deepseek_dsa_decode_roofline",
    "deepseek_mla_decode_roofline",
    "div_fwd_roofline",
    "dropout_roofline",
    "engram_decode_roofline",
    "engram_gate_conv_bwd_roofline",
    "engram_gate_conv_fwd_roofline",
    "eq_fwd_roofline",
    "fft_c2c_roofline",
    "floor_divide_fwd_roofline",
    "fp8_lightning_indexer_roofline",
    "fp8_quant_roofline",
    "fused_moe_fwd_bytes",
    "gated_deltanet_prefill_fwd_roofline",
    "ge_fwd_roofline",
    "gemm_fwd_roofline",
    "gqa_bwd_roofline",
    "gqa_decode_paged_roofline",
    "gqa_decode_roofline",
    "gqa_fwd_roofline",
    "gqa_prefill_fp8_tensor_core_roofline",
    "gqa_prefill_fwd_roofline",
    "gqa_prefill_paged_with_kv_cache_fwd_roofline",
    "gqa_prefill_varlen_fwd_roofline",
    "gqa_prefill_with_kv_cache_fwd_roofline",
    "gqa_sliding_window_fwd_roofline",
    "gqa_sliding_window_varlen_fwd_roofline",
    "grouped_gemm_roofline",
    "gt_fwd_roofline",
    "le_fwd_roofline",
    "lerp_fwd_roofline",
    "lerp_tensor_fwd_roofline",
    "logical_and_fwd_roofline",
    "logical_or_fwd_roofline",
    "lt_fwd_roofline",
    "masked_fill_fwd_roofline",
    "maximum_fwd_roofline",
    "mha_bwd_roofline",
    "mha_decode_paged_roofline",
    "mha_decode_roofline",
    "mha_fwd_roofline",
    "mhc_post_roofline",
    "mhc_pre_roofline",
    "minimum_fwd_roofline",
    "mul_fwd_roofline",
    "ne_fwd_roofline",
    "pow_fwd_roofline",
    "remainder_fwd_roofline",
    "rope_position_ids_roofline",
    "rope_roofline",
    "sub_fwd_roofline",
    "topk_selector_roofline",
    "where_fwd_roofline",
]


# ---------------------------------------------------------------------------
# MHA prefill
# ---------------------------------------------------------------------------


def _shape_or_attrs(op: Any | None, kwargs: dict[str, Any]) -> dict[str, Any]:
    if op is not None and not isinstance(op, dict):
        return vars(op)
    if isinstance(op, dict):
        return op
    return kwargs


def mha_fwd_roofline(op: Any | None = None, **kwargs: Any) -> tuple[int, int]:
    """Roofline for multi-head attention forward (prefill)."""
    data = _shape_or_attrs(op, kwargs)
    if "q_shape" in data:
        batch, seq_len, heads, dim = data["q_shape"]
    else:
        batch, seq_len, heads, dim = (
            data["batch"],
            data["seq_len"],
            data["heads"],
            data["dim"],
        )
    is_causal = bool(data.get("is_causal", True))
    elem_bytes = _dtype_itemsize(data.get("dtype", data.get("dtypes", "float16")))

    flops = 4 * batch * heads * seq_len * seq_len * dim
    if is_causal:
        flops //= 2
    q_elems = batch * seq_len * heads * dim
    kv_elems = q_elems
    return int(flops), int(2 * (q_elems + kv_elems) * elem_bytes)


def mha_bwd_roofline(op: Any | None = None, **kwargs: Any) -> tuple[int, int]:
    """Roofline for multi-head attention backward."""
    data = _shape_or_attrs(op, kwargs)
    if "q_shape" in data:
        batch, seq_len, heads, dim = data["q_shape"]
    else:
        batch, seq_len, heads, dim = (
            data["batch"],
            data["seq_len"],
            data["heads"],
            data["dim"],
        )
    is_causal = bool(data.get("is_causal", True))
    elem_bytes = _dtype_itemsize(data.get("dtype", data.get("dtypes", "float16")))

    flops = 10 * batch * heads * seq_len * seq_len * dim
    if is_causal:
        flops //= 2
    nbytes = batch * 7 * heads * seq_len * dim * elem_bytes
    return int(flops), int(nbytes)


# ---------------------------------------------------------------------------
# GQA prefill
# ---------------------------------------------------------------------------


def gqa_fwd_roofline(op: Any | None = None, **kwargs: Any) -> tuple[int, int]:
    """Roofline for grouped-query attention forward (prefill)."""
    data = _shape_or_attrs(op, kwargs)
    if "q_shape" in data:
        batch, seq_len, heads, dim = data["q_shape"]
        _, _, heads_kv, _ = data["kv_shape"]
    else:
        batch, seq_len, heads, heads_kv, dim = (
            data["batch"],
            data["seq_len"],
            data["heads"],
            data["heads_kv"],
            data["dim"],
        )
    is_causal = bool(data.get("is_causal", True))
    elem_bytes = _dtype_itemsize(data.get("dtype", data.get("dtypes", "float16")))

    flops = 4 * batch * heads * seq_len * seq_len * dim
    if is_causal:
        flops //= 2
    q_elems = batch * seq_len * heads * dim
    kv_elems = batch * seq_len * heads_kv * dim
    return int(flops), int(2 * (q_elems + kv_elems) * elem_bytes)


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
    if "bool" in dtype_name or "int8" in dtype_name or "uint8" in dtype_name:
        return 1
    return 2


def _causal_prefill_visible_scores(seq_len_q: int, seq_len_kv: int) -> int:
    return seq_len_q * seq_len_kv - seq_len_q * (seq_len_q - 1) // 2


def gqa_prefill_fwd_roofline(op: Any | None = None, **kwargs: Any) -> tuple[int, int]:
    """Conservative roofline for dense GQA prefill.

    Workloads bind ``q_shape`` and ``kv_shape``. Causal mode uses bottom-right
    alignment with ``S_q <= S_kv``.
    """
    data = _shape_or_attrs(op, kwargs)
    if "q_shape" in data:
        q_shape = data["q_shape"]
        kv_shape = data.get("kv_shape", data.get("k_shape"))
        batch, seq_len_q, heads, dim = q_shape
        _, seq_len_kv, heads_kv, _ = kv_shape
    else:
        batch, seq_len_q, seq_len_kv, heads, heads_kv, dim = (
            data["batch"],
            data["seq_len_q"],
            data["seq_len_kv"],
            data["heads"],
            data["heads_kv"],
            data["dim"],
        )
    is_causal = bool(data.get("is_causal", True))
    elem_bytes = _dtype_itemsize(data.get("dtype", data.get("dtypes", "float16")))

    visible = (
        _causal_prefill_visible_scores(seq_len_q, seq_len_kv)
        if is_causal
        else seq_len_q * seq_len_kv
    )
    flops = 4 * batch * heads * visible * dim

    q_elems = batch * seq_len_q * heads * dim
    kv_elems = batch * seq_len_kv * heads_kv * dim
    o_elems = q_elems
    nbytes = (q_elems + 2 * kv_elems + o_elems) * elem_bytes
    return int(flops), int(nbytes)


def gated_deltanet_prefill_fwd_roofline(op: Any | None = None, **kwargs: Any) -> tuple[int, int]:
    """Approximate roofline for Gated DeltaNet zero-state prefill.

    This models the dominant chunkwise matmul work in the current
    implementation. It is intentionally conservative; the helper exists so the
    manifest and benchmark share one explicit cost-model hook.
    """
    data = _shape_or_attrs(op, kwargs)
    if "q_shape" in data:
        layout = str(data.get("layout", "bthd")).lower()
        q_shape = data["q_shape"]
        v_shape = data["v_shape"]
        if layout == "bthd":
            batch, seq_len, heads, dim_k = q_shape
            _, v_seq_len, v_heads, dim_v = v_shape
        elif layout in ("bhtd", "bhsd"):
            batch, heads, seq_len, dim_k = q_shape
            _, v_heads, v_seq_len, dim_v = v_shape
        else:
            raise ValueError(f"Unsupported GDN prefill layout: {layout}")
        if v_seq_len != seq_len or v_heads != heads:
            raise ValueError(
                "GDN prefill q_shape and v_shape must share seq_len and heads"
            )
        chunk_size = data.get("chunk_size", 64) or 64
    else:
        batch, heads, seq_len, dim_k, dim_v, chunk_size = (
            data["batch"],
            data["heads"],
            data["seq_len"],
            data["dim_k"],
            data["dim_v"],
            data["chunk_size"] or 64,
        )
    elem_bytes = _dtype_itemsize(data.get("dtype", data.get("dtypes", "float16")))

    num_chunks = seq_len // chunk_size
    state_flops = 4 * batch * heads * num_chunks * chunk_size * dim_k * dim_v
    intra_flops = 4 * batch * heads * num_chunks * chunk_size * chunk_size * (
        dim_k + dim_v
    )
    flops = state_flops + intra_flops

    input_elems = (
        3 * batch * heads * seq_len * dim_k
        + batch * heads * seq_len * dim_v
        + 2 * batch * heads * seq_len
    )
    output_elems = batch * heads * seq_len * dim_v + batch * heads * dim_k * dim_v
    nbytes = (input_elems + output_elems) * elem_bytes
    return int(flops), int(nbytes)


def gqa_prefill_fp8_tensor_core_roofline(
    op: Any | None = None,
    **kwargs: Any,
) -> tuple[int, int]:
    """Roofline for dense no-cache FP8 Tensor Core GQA prefill."""
    data = _shape_or_attrs(op, kwargs)
    if "q_shape" in data:
        q_shape = data["q_shape"]
        kv_shape = data.get("kv_shape", data.get("k_shape"))
        batch, seq_len, heads, dim = q_shape
        _, _, heads_kv, _ = kv_shape
    else:
        batch, seq_len, heads, heads_kv, dim = (
            data["batch"],
            data["seq_len"],
            data["heads"],
            data["heads_kv"],
            data["dim"],
        )
    out_bytes = _dtype_itemsize(data.get("dtype", data.get("dtypes", "float16")))

    flops = 4 * batch * heads * seq_len * seq_len * dim
    q_elems = batch * seq_len * heads * dim
    kv_elems = batch * seq_len * heads_kv * dim
    # Public FP8 descales follow the FA3-compatible [batch, heads_kv] contract.
    descale_bytes = 3 * batch * heads_kv * 4
    nbytes = q_elems + 2 * kv_elems + q_elems * out_bytes + descale_bytes
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


def gqa_prefill_varlen_fwd_roofline(
    op: Any | None = None,
    **kwargs: Any,
) -> tuple[int, int]:
    """Conservative roofline for packed-varlen GQA prefill.

    Preferred workload binding supplies explicit ``q_lens`` and ``kv_lens``.
    If they are absent, this falls back to a deterministic fill from aggregate
    totals and ``max_seqlen_*`` so benchmark metadata remains reproducible.
    Causal mode uses bottom-right alignment independently per request.
    """
    data = _shape_or_attrs(op, kwargs)
    if "q_shape" not in data and data.get("_roofline_kwargs") is not None:
        data = dict(data["_roofline_kwargs"])
    q_shape = data["q_shape"]
    k_shape = data["k_shape"]
    total_q, heads, dim = q_shape
    total_kv, heads_kv, _ = k_shape
    batch = int(data["batch"])
    max_seqlen_q = int(data["max_seqlen_q"])
    max_seqlen_kv = int(data["max_seqlen_kv"])
    is_causal = bool(data.get("is_causal", True))
    elem_bytes = _dtype_itemsize(data.get("dtype", data.get("dtypes", "float16")))

    q_lens = data.get("q_lens")
    kv_lens = data.get("kv_lens")
    if q_lens is None and (cu_seqlens_q := data.get("cu_seqlens_q")) is not None:
        values = [int(x) for x in cu_seqlens_q.detach().cpu().tolist()]
        q_lens = [values[idx + 1] - values[idx] for idx in range(len(values) - 1)]
    if kv_lens is None and (cu_seqlens_kv := data.get("cu_seqlens_kv")) is not None:
        values = [int(x) for x in cu_seqlens_kv.detach().cpu().tolist()]
        kv_lens = [values[idx + 1] - values[idx] for idx in range(len(values) - 1)]
    if q_lens is None:
        q_lens = _distribute_total(total_q, batch, max_seqlen_q)
    if kv_lens is None:
        kv_lens = _distribute_total(total_kv, batch, max_seqlen_kv)

    visible = 0
    for q_len, kv_len in zip(q_lens, kv_lens, strict=True):
        visible += (
            _causal_prefill_visible_scores(int(q_len), int(kv_len))
            if is_causal
            else int(q_len) * int(kv_len)
        )
    flops = 4 * heads * visible * dim

    q_elems = total_q * heads * dim
    kv_elems = total_kv * heads_kv * dim
    o_elems = q_elems
    cu_bytes = 2 * (batch + 1) * 4
    nbytes = (q_elems + 2 * kv_elems + o_elems) * elem_bytes + cu_bytes
    return int(flops), int(nbytes)


def gqa_prefill_with_kv_cache_fwd_roofline(
    op: Any | None = None,
    **kwargs: Any,
) -> tuple[int, int]:
    """Conservative roofline for contiguous-cache GQA prefill.

    The benchmark workload convention uses
    ``old_len = S_kv_cap - S_new`` for every batch item.
    """
    data = _shape_or_attrs(op, kwargs)
    if "q_shape" in data:
        q_shape = data["q_shape"]
        k_new_shape = data["k_new_shape"]
        k_cache_shape = data["k_cache_shape"]
        batch, seq_len_new, heads, dim = q_shape
        _, _, heads_kv, _ = k_new_shape
        _, seq_len_cap, _, _ = k_cache_shape
    else:
        seq_len_cap = data["seq_len_cap"] if "seq_len_cap" in data else data["seqlen_kv"]
        batch, seq_len_new, heads, heads_kv, dim = (
            data["batch"],
            data["seq_len_new"],
            data["heads"],
            data["heads_kv"],
            data["dim"],
        )
    old_len = seq_len_cap - seq_len_new
    is_causal = bool(data.get("is_causal", True))
    elem_bytes = _dtype_itemsize(data.get("dtype", data.get("dtypes", "float16")))

    visible = (
        seq_len_new * old_len + seq_len_new * (seq_len_new + 1) // 2
        if is_causal
        else seq_len_new * (old_len + seq_len_new)
    )
    flops = 4 * batch * heads * visible * dim

    q_elems = batch * seq_len_new * heads * dim
    old_kv_elems = 2 * batch * old_len * heads_kv * dim
    new_kv_elems = 2 * batch * seq_len_new * heads_kv * dim
    append_kv_elems = new_kv_elems
    o_elems = q_elems
    cache_seqlens_bytes = batch * 4
    nbytes = (
        q_elems + old_kv_elems + new_kv_elems + append_kv_elems + o_elems
    ) * elem_bytes + cache_seqlens_bytes
    return int(flops), int(nbytes)


def gqa_prefill_paged_with_kv_cache_fwd_roofline(
    op: Any | None = None,
    **kwargs: Any,
) -> tuple[int, int]:
    """Conservative roofline for paged-cache GQA prefill.

    Paged workloads should bind explicit per-request ``q_lens`` and
    ``cache_lens``. If they are absent, fall back to a deterministic fill from
    aggregate metadata so older workload entries remain evaluable.
    """
    data = _shape_or_attrs(op, kwargs)
    total_q = int(data["total_q"]) if "total_q" in data else None
    batch = int(data["batch"])
    heads = int(data["heads"])
    heads_kv = int(data["heads_kv"])
    dim = int(data["dim"])
    max_pages_per_req = int(data["max_pages_per_req"])
    page_size = int(data["page_size"])
    max_seqlen_q = int(data.get("max_seqlen_q", max_pages_per_req * page_size))
    is_causal = bool(data.get("is_causal", True))
    elem_bytes = _dtype_itemsize(data.get("dtype", data.get("dtypes", "float16")))

    q_lens = data.get("q_lens")
    if total_q is None and q_lens is not None:
        total_q = int(sum(q_lens))
    if total_q is None:
        total_q = batch * max_seqlen_q
    if q_lens is None:
        q_lens = _distribute_total(total_q, batch, max_seqlen_q)
    cache_lens = data.get("cache_lens")
    if cache_lens is None:
        max_position = data.get("max_position")
        max_total_len = max_pages_per_req * page_size if max_position is None else int(max_position)
        cache_lens = [max(max_total_len - int(q_len), 0) for q_len in q_lens]

    visible = 0
    old_kv_tokens = 0
    for q_len, old_len in zip(q_lens, cache_lens, strict=True):
        q_len = int(q_len)
        old_len = int(old_len)
        old_kv_tokens += old_len
        visible += (
            q_len * old_len + q_len * (q_len + 1) // 2 if is_causal else q_len * (old_len + q_len)
        )
    flops = 4 * heads * visible * dim

    q_elems = total_q * heads * dim
    old_kv_elems = 2 * old_kv_tokens * heads_kv * dim
    new_kv_elems = 2 * total_q * heads_kv * dim
    append_kv_elems = new_kv_elems
    o_elems = q_elems
    metadata_bytes = (batch + 1) * 4 + batch * 4 + batch * max_pages_per_req * 4
    nbytes = (
        q_elems + old_kv_elems + new_kv_elems + append_kv_elems + o_elems
    ) * elem_bytes + metadata_bytes
    return int(flops), int(nbytes)


def gqa_bwd_roofline(op: Any | None = None, **kwargs: Any) -> tuple[int, int]:
    """Roofline for grouped-query attention backward."""
    data = _shape_or_attrs(op, kwargs)
    if "q_shape" in data:
        batch, seq_len, heads, dim = data["q_shape"]
        _, _, heads_kv, _ = data["kv_shape"]
    else:
        batch, seq_len, heads, heads_kv, dim = (
            data["batch"],
            data["seq_len"],
            data["heads"],
            data["heads_kv"],
            data["dim"],
        )
    is_causal = bool(data.get("is_causal", True))
    elem_bytes = _dtype_itemsize(data.get("dtype", data.get("dtypes", "float16")))

    flops = 10 * batch * heads * seq_len * seq_len * dim
    if is_causal:
        flops //= 2
    nbytes = batch * (3 * heads + 4 * heads_kv) * seq_len * dim * elem_bytes
    return int(flops), int(nbytes)


# ---------------------------------------------------------------------------
# MHA decode
# ---------------------------------------------------------------------------


def mha_decode_roofline(op: Any | None = None, **kwargs: Any) -> tuple[int, int]:
    """Roofline for MHA decode with KV cache."""
    data = _shape_or_attrs(op, kwargs)
    if "q_shape" in data:
        batch, seqlen_q, heads, dim = data["q_shape"]
        _, seqlen_kv, _, _ = data["kv_shape"]
    else:
        batch, seqlen_q, heads, seqlen_kv, dim = (
            data["batch"],
            data["seqlen_q"],
            data["heads"],
            data["seqlen_kv"],
            data["dim"],
        )
    elem_bytes = _dtype_itemsize(data.get("dtype", data.get("dtypes", "float16")))
    flops = 4 * batch * heads * seqlen_q * seqlen_kv * dim
    q_elems = batch * seqlen_q * heads * dim
    kv_elems = batch * seqlen_kv * heads * dim
    nbytes = (q_elems + 2 * kv_elems + q_elems) * elem_bytes
    return int(flops), int(nbytes)


def mha_decode_paged_roofline(op: Any | None = None, **kwargs: Any) -> tuple[int, int]:
    """Roofline for paged MHA decode with KV cache."""
    data = _shape_or_attrs(op, kwargs)
    if "q_shape" in data:
        batch, seqlen_q, heads, dim = data["q_shape"]
        seqlen_kv, _, _ = data["kv_shape"]
    else:
        batch, seqlen_q, heads, seqlen_kv, dim = (
            data["batch"],
            data["seqlen_q"],
            data["heads"],
            data["seqlen_kv"],
            data["dim"],
        )
    elem_bytes = _dtype_itemsize(data.get("dtype", data.get("dtypes", "float16")))
    flops = 4 * batch * heads * seqlen_q * seqlen_kv * dim
    q_elems = batch * seqlen_q * heads * dim
    kv_elems = seqlen_kv * heads * dim
    metadata_bytes = (
        batch * 4
        + batch * max(1, (seqlen_kv + int(data["page_size"]) - 1) // int(data["page_size"])) * 4
    )
    nbytes = (q_elems + 2 * kv_elems + q_elems) * elem_bytes + metadata_bytes
    return int(flops), int(nbytes)


# ---------------------------------------------------------------------------
# GQA decode
# ---------------------------------------------------------------------------


def gqa_decode_roofline(op: Any | None = None, **kwargs: Any) -> tuple[int, int]:
    """Roofline for GQA decode with KV cache."""
    data = _shape_or_attrs(op, kwargs)
    if "q_shape" in data:
        batch, heads, dim = data["q_shape"]
        _, seqlen_kv, heads_kv, _ = data["kv_shape"]
    else:
        batch, heads, heads_kv, seqlen_kv, dim = (
            data["batch"],
            data["heads"],
            data["heads_kv"],
            data["seqlen_kv"],
            data["dim"],
        )
    elem_bytes = _dtype_itemsize(data.get("dtype", data.get("dtypes", "float16")))
    flops = 4 * batch * heads * seqlen_kv * dim
    q_elems = batch * heads * dim
    kv_elems = batch * seqlen_kv * heads_kv * dim
    nbytes = (q_elems + 2 * kv_elems + q_elems) * elem_bytes
    return int(flops), int(nbytes)


def gqa_decode_paged_roofline(op: Any | None = None, **kwargs: Any) -> tuple[int, int]:
    """Roofline for paged GQA decode with KV cache."""
    data = _shape_or_attrs(op, kwargs)
    if "q_shape" in data:
        batch, heads, dim = data["q_shape"]
        seqlen_kv, heads_kv, _ = data["kv_shape"]
    else:
        batch, heads, heads_kv, seqlen_kv, dim = (
            data["batch"],
            data["heads"],
            data["heads_kv"],
            data["seqlen_kv"],
            data["dim"],
        )
    elem_bytes = _dtype_itemsize(data.get("dtype", data.get("dtypes", "float16")))
    flops = 4 * batch * heads * seqlen_kv * dim
    q_elems = batch * heads * dim
    kv_elems = seqlen_kv * heads_kv * dim
    page_size = int(data["page_size"])
    metadata_bytes = batch * 4 + batch * max(1, (seqlen_kv + page_size - 1) // page_size) * 4
    nbytes = (q_elems + 2 * kv_elems + q_elems) * elem_bytes + metadata_bytes
    return int(flops), int(nbytes)


# ---------------------------------------------------------------------------
# Sliding window attention
# ---------------------------------------------------------------------------


def gqa_sliding_window_fwd_roofline(op: Any | None = None, **kwargs: Any) -> tuple[int, int]:
    """Roofline for GQA sliding window forward."""
    data = _shape_or_attrs(op, kwargs)
    if "q_shape" in data:
        batch, seq_len, heads, dim = data["q_shape"]
        _, _, heads_kv, _ = data["kv_shape"]
    else:
        batch, seq_len, heads, heads_kv, dim = (
            data["batch"],
            data["seq_len"],
            data["heads"],
            data["heads_kv"],
            data["dim"],
        )
    is_causal = bool(data.get("is_causal", True))
    wl = int(data.get("window_size_left", -1))
    wr = int(data.get("window_size_right", -1))
    elem_bytes = _dtype_itemsize(data.get("dtype", data.get("dtypes", "float16")))

    total_attended = 0
    for q_idx in range(seq_len):
        hi = q_idx if is_causal else (min(seq_len - 1, q_idx + wr) if wr >= 0 else seq_len - 1)
        lo = max(0, q_idx - wl) if wl >= 0 else 0
        total_attended += hi - lo + 1
    flops = 4 * batch * heads * total_attended * dim
    nbytes = 2 * batch * seq_len * (heads + heads_kv) * dim * elem_bytes
    return int(flops), int(nbytes)


def gqa_sliding_window_varlen_fwd_roofline(
    op: Any | None = None,
    **kwargs: Any,
) -> tuple[int, int]:
    """Roofline for variable-length GQA sliding window forward."""
    data = _shape_or_attrs(op, kwargs)
    batch = int(data["batch"])
    heads = int(data["heads"])
    heads_kv = int(data["heads_kv"])
    dim = int(data["dim"])
    total_q = int(data.get("total_q", 0))
    total_k = int(data.get("total_k", 0))
    max_seqlen_q = int(data.get("max_seqlen_q", total_q // batch if batch else total_q))
    q_lens = data.get("q_lens")
    k_lens = data.get("k_lens")
    if q_lens is None:
        q_lens = _distribute_total(total_q, batch, max_seqlen_q)
    if k_lens is None:
        max_seqlen_k = int(data.get("max_seqlen_k", total_k // batch if batch else total_k))
        k_lens = _distribute_total(total_k, batch, max_seqlen_k)
    total_q = sum(q_lens)
    total_k = sum(k_lens)
    is_causal = bool(data.get("is_causal", True))
    wl = int(data.get("window_size_left", -1))
    wr = int(data.get("window_size_right", -1))
    elem_bytes = _dtype_itemsize(data.get("dtype", data.get("dtypes", "float16")))

    total_attended = 0
    for sq, sk in zip(q_lens, k_lens, strict=True):
        offset = int(sk) - int(sq)
        for q_pos in range(int(sq)):
            hi = (
                min(q_pos + offset, int(sk) - 1)
                if is_causal
                else (min(q_pos + offset + wr, int(sk) - 1) if wr >= 0 else int(sk) - 1)
            )
            lo = max(0, q_pos + offset - wl) if wl >= 0 else 0
            total_attended += max(0, hi - lo + 1)
    flops = 4 * heads * total_attended * dim
    nbytes = (
        total_q * heads * dim + 2 * total_k * heads_kv * dim + total_q * heads * dim
    ) * elem_bytes
    return int(flops), int(nbytes)


# ---------------------------------------------------------------------------
# DeepSeek MLA / DSA decode
# ---------------------------------------------------------------------------


def deepseek_mla_decode_roofline(op: Any | None = None, **kwargs: Any) -> tuple[int, int]:
    """Roofline for DeepSeek MLA decode with KV cache."""
    data = _shape_or_attrs(op, kwargs)
    if "q_shape" in data:
        batch, heads, dim = data["q_shape"]
        _, seqlen_kv, heads_kv, _ = data["kv_shape"]
        pe_dim = data["pe_dim"]
    else:
        batch, heads, heads_kv, seqlen_kv, dim, pe_dim = (
            data["batch"],
            data["heads"],
            data["heads_kv"],
            data["seqlen_kv"],
            data["dim"],
            data["pe_dim"],
        )
    elem_bytes = _dtype_itemsize(data.get("dtype", data.get("dtypes", "float16")))
    flops = 2 * batch * heads * seqlen_kv * (2 * dim + pe_dim)
    nbytes = (
        batch * heads * (dim + pe_dim)
        + batch * seqlen_kv * heads_kv * (dim + pe_dim)
        + batch * heads * dim
    ) * elem_bytes
    return int(flops), int(nbytes)


def deepseek_dsa_decode_roofline(op: Any | None = None, **kwargs: Any) -> tuple[int, int]:
    """Roofline for DeepSeek sparse attention decode."""
    data = _shape_or_attrs(op, kwargs)
    if "q_shape" in data:
        batch, seq_len, heads, q_dim = data["q_shape"]
        _, seq_len_kv, heads_kv, _ = data["kv_shape"]
        dim_tail = data["dim_tail"]
        dim = q_dim - dim_tail
        topk = data["topk"]
    else:
        batch, seq_len, heads, seq_len_kv, dim, dim_tail, topk, heads_kv = (
            data["batch"],
            data["seq_len"],
            data["heads"],
            data["seq_len_kv"],
            data["dim"],
            data["dim_tail"],
            data["topk"],
            data["heads_kv"],
        )
    elem_bytes = _dtype_itemsize(data.get("dtype", data.get("dtypes", "float16")))
    flops = 2 * batch * seq_len * heads * topk * (2 * dim + dim_tail)
    q_elems = batch * seq_len * heads * (dim + dim_tail)
    kv_elems = batch * seq_len_kv * heads_kv * (dim + dim_tail)
    o_elems = batch * seq_len * heads * dim
    index_bytes = batch * seq_len * heads_kv * topk * 4
    nbytes = (q_elems + kv_elems + o_elems) * elem_bytes + index_bytes
    return int(flops), int(nbytes)


# ---------------------------------------------------------------------------
# Elementwise — mixed-dtype ops requiring func-mode roofline
# ---------------------------------------------------------------------------


def where_fwd_roofline(op: "Op") -> tuple[int, int]:
    """Roofline for ``torch.where`` forward (bool condition + float input/other).

    Func-mode is required because the byte accounting mixes a 1-byte bool
    condition with the float input/other dtype (see manifest comment on
    ``WhereFwdOp.roofline``). Inline mode binds ``elem_bytes`` to a single
    dtype and cannot express that.

    ``N_total`` follows the post-broadcast convention used by
    ``WhereFwdOp.shape_rules``:
    ``N_total = product(broadcast_shapes(condition.shape, input.shape,
    other.shape))``. The function reads ``op.N_total`` directly when the
    bound Op exposes it (current ``WhereFwdOp`` stores the flattened
    element count there); when the conformed Op grows
    ``condition``/``input``/``other`` shape attributes per the spec, the
    same value can be derived as ``op.condition.shape`` /
    ``op.input.shape`` / ``op.other.shape`` broadcast together, and a
    ``static_dims``-driven update will flow in via codegen.

    Byte traffic is approximated as
    ``N_total + 3 * N_total * elem_bytes`` — a 1-byte condition read
    (logical bytes; the bool is broadcast to ``N_total``) plus input,
    other, and out at the float ``elem_bytes`` each. This matches the
    "logical bytes, post-broadcast" convention used elsewhere in the
    elementwise manifest entries.

    Args:
        op: The bound ``WhereFwdOp`` instance. Must expose ``N_total``
            (int) and ``dtype`` (``torch.dtype``).

    Returns:
        ``(flops, bytes)`` as ints, with ``flops == N_total`` (one
        predicated select per output element) and
        ``bytes == N_total + 3 * N_total * elem_bytes``.
    """
    n_total = int(op.N_total)
    elem_bytes = op.dtype.itemsize
    flops = n_total
    nbytes = n_total + 3 * n_total * elem_bytes
    return flops, nbytes


# ---------------------------------------------------------------------------
# Clamp family (Tensor-bound variants)
# ---------------------------------------------------------------------------
#
# Func mode is required for the broadcasted Tensor-bound clamp variants
# because ``N_total`` follows the post-broadcast convention
# ``product(broadcast_shapes(input.shape, ...))`` and ``broadcast_shapes``
# is not in the inline-roofline vars-layer namespace
# (``docs/design/roofline.md`` §4.4.4). Codegen would fail to bind it.
# ``ClampScalarFwdOp`` keeps inline mode because its ``N_total`` reduces
# to ``product(input.shape)`` (no broadcasting).


def clamp_fwd_roofline(op: "Op") -> tuple[int, int]:
    """Roofline for ``ClampFwdOp`` (Tensor-bound double-sided clamp).

    Models ``torch.clamp(input, min: Tensor, max: Tensor)`` with
    broadcasting across all three operands. Reads ``op.N_total`` (the
    post-broadcast element count) and ``op.dtype.itemsize``.

    Per ``docs/design/roofline.md`` §1.3, two-sided clamp collapses to
    one fused compare-and-select = ``flops = N_total``. Bytes: read
    input + read min + read max + write out, all post-broadcast at
    ``elem_bytes`` each → ``bytes = 4 * N_total * elem_bytes``.

    Args:
        op: bound ``ClampFwdOp`` instance exposing ``N_total`` and
            ``dtype``.

    Returns:
        ``(flops, bytes)`` ints.
    """
    n_total = int(op.N_total)
    elem_bytes = op.dtype.itemsize
    return n_total, 4 * n_total * elem_bytes


def clamp_min_fwd_roofline(op: "Op") -> tuple[int, int]:
    """Roofline for ``ClampMinFwdOp`` (Tensor lower bound only).

    Models ``torch.clamp_min(input, min: Tensor)`` with broadcasting.
    Per output element: one ``max(input, min)``. Bytes: read input +
    read min + write out, all post-broadcast.

    Args:
        op: bound ``ClampMinFwdOp`` instance exposing ``N_total`` and
            ``dtype``.

    Returns:
        ``(flops, bytes)`` ints with ``flops == N_total`` and
        ``bytes == 3 * N_total * elem_bytes``.
    """
    n_total = int(op.N_total)
    elem_bytes = op.dtype.itemsize
    return n_total, 3 * n_total * elem_bytes


def clamp_max_fwd_roofline(op: "Op") -> tuple[int, int]:
    """Roofline for ``ClampMaxFwdOp`` (Tensor upper bound only).

    Models ``torch.clamp_max(input, max: Tensor)`` with broadcasting.
    Per output element: one ``min(input, max)``. Bytes: read input +
    read max + write out, all post-broadcast.

    Args:
        op: bound ``ClampMaxFwdOp`` instance exposing ``N_total`` and
            ``dtype``.

    Returns:
        ``(flops, bytes)`` ints with ``flops == N_total`` and
        ``bytes == 3 * N_total * elem_bytes``.
    """
    n_total = int(op.N_total)
    elem_bytes = op.dtype.itemsize
    return n_total, 3 * n_total * elem_bytes


def lerp_tensor_fwd_roofline(op: "Op") -> tuple[int, int]:
    """Roofline for ``LerpTensorFwdOp`` (Tensor-weight ``torch.lerp``).

    Per output element: 3 flops (sub + mul + add); 3 reads + 1 write at
    post-broadcast ``N_total``.
    """
    n_total = int(op.N_total)
    elem_bytes = op.dtype.itemsize
    return 3 * n_total, 4 * n_total * elem_bytes


# ---------------------------------------------------------------------------
# MaskedFill family
# ---------------------------------------------------------------------------
#
# Func mode is required because ``N_total`` follows the post-broadcast
# convention ``product(broadcast_shapes(input.shape, mask.shape))`` —
# out-of-place ``Tensor.masked_fill`` returns a tensor whose shape is the
# bidirectional broadcast of ``input`` and ``mask`` (verified empirically:
# ``torch.zeros((2,1)).masked_fill(mask=(2,3), 1.0)`` → shape ``(2,3)``).
# ``broadcast_shapes`` is not in the inline-roofline vars-layer namespace
# (``docs/design/roofline.md`` §4.4.4). One function serves both the
# Tensor-value primary and the Scalar-value variant — the 0-dim value
# read is negligible vs ``N_total`` and is folded into the per-element
# write cost.


def masked_fill_fwd_roofline(op: "Op") -> tuple[int, int]:
    """Roofline for ``MaskedFillFwdOp`` and ``MaskedFillScalarFwdOp``.

    Models out-of-place ``Tensor.masked_fill`` whose output shape is the
    bidirectional broadcast of ``input`` and ``mask``. Reads
    ``op.N_total`` (post-broadcast element count) and ``op.dtype.itemsize``.

    Per output element: one predicated select → ``flops = N_total``.
    Bytes: 1-byte mask read (broadcast to ``N_total``) + input read at
    ``elem_bytes`` + out write at ``elem_bytes`` →
    ``bytes = N_total + 2 * N_total * elem_bytes``. The 0-dim ``value``
    Tensor read (Tensor-value variant only) is one ``elem_bytes`` and is
    negligible vs ``N_total``.

    Args:
        op: bound ``MaskedFillFwdOp`` or ``MaskedFillScalarFwdOp``
            instance exposing ``N_total`` and ``dtype``.

    Returns:
        ``(flops, bytes)`` ints.
    """
    n_total = int(op.N_total)
    elem_bytes = op.dtype.itemsize
    flops = n_total
    nbytes = n_total + 2 * n_total * elem_bytes
    return flops, nbytes


def _binary_broadcast_roofline(
    op: "Op", *, flops_per_elem: int, bool_output: bool
) -> tuple[int, int]:
    """Shared core for the broadcast-binary roofline family."""
    a_numel = int(op.a_numel)
    b_numel = int(op.b_numel)
    n_total = int(op.N_total)
    elem_bytes = op.dtype.itemsize
    out_elem_bytes = 1 if bool_output else elem_bytes
    flops = flops_per_elem * n_total
    nbytes = (a_numel + b_numel) * elem_bytes + n_total * out_elem_bytes
    return flops, nbytes


def add_fwd_roofline(op: "Op") -> tuple[int, int]:
    return _binary_broadcast_roofline(op, flops_per_elem=2, bool_output=False)


def sub_fwd_roofline(op: "Op") -> tuple[int, int]:
    return _binary_broadcast_roofline(op, flops_per_elem=2, bool_output=False)


def mul_fwd_roofline(op: "Op") -> tuple[int, int]:
    return _binary_broadcast_roofline(op, flops_per_elem=1, bool_output=False)


def div_fwd_roofline(op: "Op") -> tuple[int, int]:
    return _binary_broadcast_roofline(op, flops_per_elem=1, bool_output=False)


def remainder_fwd_roofline(op: "Op") -> tuple[int, int]:
    return _binary_broadcast_roofline(op, flops_per_elem=4, bool_output=False)


def pow_fwd_roofline(op: "Op") -> tuple[int, int]:
    return _binary_broadcast_roofline(op, flops_per_elem=3, bool_output=False)


def floor_divide_fwd_roofline(op: "Op") -> tuple[int, int]:
    return _binary_broadcast_roofline(op, flops_per_elem=2, bool_output=False)


def lerp_fwd_roofline(op: "Op") -> tuple[int, int]:
    return _binary_broadcast_roofline(op, flops_per_elem=3, bool_output=False)


def maximum_fwd_roofline(op: "Op") -> tuple[int, int]:
    return _binary_broadcast_roofline(op, flops_per_elem=1, bool_output=False)


def minimum_fwd_roofline(op: "Op") -> tuple[int, int]:
    return _binary_broadcast_roofline(op, flops_per_elem=1, bool_output=False)


def eq_fwd_roofline(op: "Op") -> tuple[int, int]:
    return _binary_broadcast_roofline(op, flops_per_elem=1, bool_output=True)


def ne_fwd_roofline(op: "Op") -> tuple[int, int]:
    return _binary_broadcast_roofline(op, flops_per_elem=1, bool_output=True)


def gt_fwd_roofline(op: "Op") -> tuple[int, int]:
    return _binary_broadcast_roofline(op, flops_per_elem=1, bool_output=True)


def lt_fwd_roofline(op: "Op") -> tuple[int, int]:
    return _binary_broadcast_roofline(op, flops_per_elem=1, bool_output=True)


def ge_fwd_roofline(op: "Op") -> tuple[int, int]:
    return _binary_broadcast_roofline(op, flops_per_elem=1, bool_output=True)


def le_fwd_roofline(op: "Op") -> tuple[int, int]:
    return _binary_broadcast_roofline(op, flops_per_elem=1, bool_output=True)


def logical_and_fwd_roofline(op: "Op") -> tuple[int, int]:
    return _binary_broadcast_roofline(op, flops_per_elem=3, bool_output=True)


def logical_or_fwd_roofline(op: "Op") -> tuple[int, int]:
    return _binary_broadcast_roofline(op, flops_per_elem=3, bool_output=True)


def bitwise_and_fwd_roofline(op: "Op") -> tuple[int, int]:
    return _binary_broadcast_roofline(op, flops_per_elem=1, bool_output=False)


def bitwise_or_fwd_roofline(op: "Op") -> tuple[int, int]:
    return _binary_broadcast_roofline(op, flops_per_elem=1, bool_output=False)


def bitwise_xor_fwd_roofline(op: "Op") -> tuple[int, int]:
    return _binary_broadcast_roofline(op, flops_per_elem=1, bool_output=False)


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------


def fused_moe_fwd_bytes(op: "Op") -> tuple[int, int]:
    """Roofline for FusedMoeFwdOp / FusedMoeFwdCbFwdOp.

    Func-mode: byte traffic mixes float32 gating (and float32 correction bias
    for the Cb variant) with hidden-states / weights at ``op.dtype``, so a
    single ``elem_bytes`` cannot express the total.
    """
    num_tokens = int(op.num_tokens)
    num_experts = int(op.num_experts)
    top_k = int(op.top_k)
    hidden_size = int(op.hidden_size)
    ffn_size = int(op.ffn_size)
    elem_bytes = _dtype_itemsize(op.dtype)

    flops = num_tokens * top_k * 6 * ffn_size * hidden_size
    weight_bytes = num_experts * 3 * ffn_size * hidden_size * elem_bytes
    token_bytes = 2 * num_tokens * hidden_size * elem_bytes
    gating_bytes = num_tokens * num_experts * 4  # float32 logits
    bias_bytes = num_experts * 4 if getattr(op, "with_correction_bias", False) else 0
    return flops, weight_bytes + token_bytes + gating_bytes + bias_bytes


def gemm_fwd_roofline(op: "Op") -> tuple[int, int]:
    """Roofline for dense ``GemmOp`` (``d = a @ b``, fp16/bf16).

    ``GemmOp`` is input-inferred, so the logical dims ``m/n/k`` and the dtype
    are bound on the op during ``forward()``; this reads them directly, which
    stays correct across all ``trans_a``/``trans_b`` layouts (the logical dims
    are transpose-independent). Valid only after the first ``forward()``.

    Raises:
        RuntimeError: If called before ``forward()`` has bound the dims.
    """
    if getattr(op, "m", None) is None or getattr(op, "dtype", None) is None:
        raise RuntimeError(
            "GemmOp.eval_roofline() is valid only after the first forward(); "
            "m/n/k and dtype are inferred from the inputs."
        )
    m, n, k = op.m, op.n, op.k
    elem_bytes = op.dtype.itemsize
    flops = 2 * m * n * k
    nbytes = (m * k + n * k + m * n) * elem_bytes
    return int(flops), int(nbytes)


def gemm_fp8_fwd_roofline(op: "Op") -> tuple[int, int]:
    """Roofline for dense FP8 ``GemmFp8Op``."""
    if getattr(op, "m", None) is None or getattr(op, "dtype", None) is None:
        raise RuntimeError(
            "GemmFp8Op.eval_roofline() is valid only after the first forward(); "
            "m/n/k and dtype are inferred from the inputs."
        )
    m, n, k = op.m, op.n, op.k
    input_bytes = op.dtype.itemsize
    out_bytes = op.out_dtype.itemsize
    scale_a_shape = getattr(op, "scale_a_shape", (1, 1))
    scale_b_shape = getattr(op, "scale_b_shape", (1, 1))
    scale_elems = scale_a_shape[0] * scale_a_shape[1] + scale_b_shape[0] * scale_b_shape[1]
    flops = 2 * m * n * k
    nbytes = (m * k + n * k) * input_bytes + m * n * out_bytes + scale_elems * 4
    if getattr(op, "has_bias", False):
        nbytes += n * out_bytes
    return int(flops), int(nbytes)


def grouped_gemm_roofline(op: "Op") -> tuple[int, int]:
    batch_sum = int(op.batch_sum)
    batch_count = int(op.batch_count)
    n = int(getattr(op, "N", getattr(op, "n", 0)))
    k = int(getattr(op, "K", getattr(op, "k", 0)))
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
    return int(flops), int((memory_a + memory_b + memory_c) * elem)


def rope_roofline(op: "Op") -> tuple[int, int]:
    seq_len = int(op.seq_len)
    head_dim = int(op.head_dim)
    batch = int(getattr(op, "batch", 1))
    num_heads = int(getattr(op, "num_heads", 1))
    layout = getattr(op, "layout", "1d")
    elem = _dtype_itemsize(getattr(op, "dtype", "float16"))
    outer = batch * num_heads if layout == "2d" else 1
    x_elems = outer * seq_len * head_dim
    cos_sin_elems = seq_len * (head_dim // 2) * 2
    flops = 4 * x_elems
    nbytes = (2 * x_elems + cos_sin_elems) * elem
    return int(flops), int(nbytes)


def rope_position_ids_roofline(op: "Op") -> tuple[int, int]:
    num_tokens = int(op.num_tokens)
    num_heads = int(op.num_heads)
    head_dim = int(op.head_dim)
    rotary_dim = int(getattr(op, "rotary_dim", head_dim) or head_dim)
    max_position = int(op.max_position)
    elem = _dtype_itemsize(getattr(op, "dtype", "float16"))
    x_elems = num_tokens * num_heads * head_dim
    cos_sin_elems = max_position * (rotary_dim // 2) * 2
    pos_elems = num_tokens
    return int(4 * x_elems), int((2 * x_elems + cos_sin_elems) * elem + pos_elems * 4)


def dropout_roofline(op: "Op") -> tuple[int, int]:
    n_total = int(op.N_total)
    elem = _dtype_itemsize(getattr(op, "dtype", "float16"))
    if not bool(getattr(op, "training", True)) or float(getattr(op, "p", 0.5)) == 0.0:
        return 0, int(2 * n_total * elem)
    if float(getattr(op, "p", 0.5)) == 1.0:
        return 0, int(n_total * elem)
    return int(n_total), int(2 * n_total * elem)


def fp8_quant_roofline(op: "Op") -> tuple[int, int]:
    batch = int(op.batch)
    seq_len_kv = int(op.seq_len_kv)
    kv_group = int(op.kv_group)
    index_dim = int(op.index_dim)
    in_elem = _dtype_itemsize(getattr(op, "in_dtype", "float16"))
    groups = batch * seq_len_kv * kv_group
    elems = groups * index_dim
    flops = 6 * elems + groups
    nbytes = elems * in_elem + elems * 1 + groups * 4
    return int(flops), int(nbytes)


def fp8_lightning_indexer_roofline(op: "Op") -> tuple[int, int]:
    batch = int(op.batch)
    seq_len = int(op.seq_len)
    heads = int(op.heads)
    index_dim = int(op.index_dim)
    seq_len_kv = int(op.seq_len_kv)
    kv_group = int(op.kv_group)
    scores = batch * seq_len * seq_len_kv * kv_group
    q_elems = batch * seq_len * heads * index_dim
    k_elems = batch * seq_len_kv * kv_group * index_dim
    weights = seq_len * heads
    # The public forward accepts either bf16 tensors that are quantized inside
    # the op or pre-quantized fp8 tensors. The op does not currently retain
    # the observed input dtype, so default to bf16 for conservative bandwidth.
    index_elem = _dtype_itemsize(getattr(op, "dtype", "bfloat16"))
    flops = 2 * scores * index_dim
    nbytes = (q_elems + k_elems) * index_elem + batch * seq_len_kv * kv_group * 4
    nbytes += weights * 4
    nbytes += 2 * seq_len * 4 + scores * 4
    return int(flops), int(nbytes)


def topk_selector_roofline(op: "Op") -> tuple[int, int]:
    batch = int(op.batch)
    seq_len = int(op.seq_len)
    seq_len_kv = int(op.seq_len_kv)
    kv_group = int(op.kv_group)
    topk = int(op.topk)
    in_elem = _dtype_itemsize(getattr(op, "in_dtype", "float32"))
    out_elem = _dtype_itemsize(getattr(op, "out_dtype", "int32"))
    comparisons = batch * seq_len * kv_group * seq_len_kv
    nbytes = comparisons * in_elem + batch * seq_len * 2 * out_elem
    nbytes += batch * seq_len * kv_group * topk * out_elem
    return int(comparisons), int(nbytes)


def _engram_elem_bytes(op: "Op") -> int:
    return _dtype_itemsize(getattr(op, "dtype", "float16"))


def engram_gate_conv_fwd_roofline(op: "Op") -> tuple[int, int]:
    m = int(op.M)
    seq_len = int(op.seq_len)
    d = int(getattr(op, "d_padded", op.d))
    elem = _engram_elem_bytes(op)
    flops = m * seq_len * (24 * d) + 20 * m * seq_len
    nbytes = (5 * m * seq_len * d) * elem + 4 * m * seq_len * 4 + 6 * d * elem
    return int(flops), int(nbytes)


def engram_gate_conv_bwd_roofline(op: "Op") -> tuple[int, int]:
    m = int(op.M)
    seq_len = int(op.seq_len)
    d = int(getattr(op, "d_padded", op.d))
    elem = _engram_elem_bytes(op)
    fwd_flops = m * seq_len * (24 * d) + 20 * m * seq_len
    read_bytes = 5 * m * seq_len * d * elem + 6 * d * elem + 4 * m * seq_len * 4
    write_bytes = 3 * m * seq_len * d * elem + 10 * d * 4 + m * seq_len * d * 4
    return int(fwd_flops * 2.5), int(read_bytes + write_bytes)


def engram_decode_roofline(op: "Op") -> tuple[int, int]:
    batch = int(op.batch)
    d_mem = int(op.d_mem)
    d = int(getattr(op, "d_padded", op.d))
    max_conv_len = int(op.max_conv_len)
    conv_kernel_size = int(op.conv_kernel_size)
    elem = _engram_elem_bytes(op)
    flops = (
        4 * batch * d_mem * d
        + batch * (16 * d + conv_kernel_size * 2 * d)
        + 20 * batch
    )
    nbytes = (
        batch * d_mem
        + batch * d
        + 2 * batch * max_conv_len * d
        + 2 * d_mem * d
        + 2 * d
        + conv_kernel_size * d
        + batch * d
    ) * elem
    return int(flops), int(nbytes)


def fft_c2c_roofline(op: "Op") -> tuple[int, int]:
    import math

    n = int(op.n)
    elem = _dtype_itemsize(getattr(op, "dtype", "complex64"))
    # Runtime batch is inferred from the input and kernel cache. Before a
    # forward call, the constructed default kernel represents batch=1.
    batch = int(getattr(getattr(op, "kernel", None), "batch_size", 1) or 1)
    return int(batch * 5 * n * math.log2(n)), int(batch * 2 * n * elem)


def mhc_pre_roofline(op: "Op") -> tuple[int, int]:
    batch = int(op.batch)
    n_expand = int(op.n_expand)
    c_x = int(op.c_x)
    x_dim = n_expand * c_x
    phi_dim = n_expand * n_expand + 2 * n_expand
    x_elem = _dtype_itemsize(getattr(op, "dtype", "bfloat16"))

    x_phi_flops = 2 * batch * x_dim * phi_dim
    x_layer_flops = 2 * batch * c_x * n_expand
    x_res_flops = 2 * batch * n_expand * c_x * n_expand
    flops = x_phi_flops + x_layer_flops + x_res_flops

    phi_bytes = x_dim * phi_dim * 4
    b_bytes = phi_dim * 4
    x_bytes = batch * x_dim * x_elem
    output_bytes = batch * (x_dim + c_x) * x_elem
    nbytes = phi_bytes + b_bytes + x_bytes + output_bytes
    return int(flops), int(nbytes)


def mhc_post_roofline(op: "Op") -> tuple[int, int]:
    batch = int(op.batch)
    n_expand = int(op.n_expand)
    c_x = int(op.c_x)
    x_elem = _dtype_itemsize(getattr(op, "dtype", "bfloat16"))
    flops = 2 * batch * n_expand * c_x
    x_layer_out_bytes = batch * c_x * x_elem
    h_post_bytes = batch * n_expand * 4
    x_res_bytes = batch * n_expand * c_x * x_elem
    x_out_bytes = batch * n_expand * c_x * x_elem
    nbytes = x_layer_out_bytes + h_post_bytes + x_res_bytes + x_out_bytes
    return int(flops), int(nbytes)

def bmm_fwd_roofline(op: "Op") -> tuple[int, int]:
    """Roofline for batched GEMM ``BmmFwdOp`` (``d[i] = a[i] @ b[i]``).

    Models ``torch.bmm`` exactly: strict 3D-3D with no broadcasting. Each of
    the ``B`` batch items is an independent ``(M, K) @ (K, N)`` GEMM whose
    flops/bytes are ``B``-scaled totals. ``BmmFwdOp`` is input-inferred, so
    the logical dims ``batch/m/n/k`` and the dtype are bound on the op during
    ``forward()``; this reads them directly, matching ``gemm_fwd_roofline``'s
    contract. Valid only after the first ``forward()``.

    Raises:
        RuntimeError: If called before ``forward()`` has bound the dims.
    """
    if getattr(op, "m", None) is None or getattr(op, "dtype", None) is None:
        raise RuntimeError(
            "BmmFwdOp.eval_roofline() is valid only after the first forward(); "
            "batch/m/n/k and dtype are inferred from the inputs."
        )
    batch, m, n, k = op.batch, op.m, op.n, op.k
    elem_bytes = op.dtype.itemsize
    flops = 2 * batch * m * n * k
    nbytes = batch * (m * k + n * k + m * n) * elem_bytes
    return int(flops), int(nbytes)
