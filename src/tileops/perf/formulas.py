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
    "deepseek_dsa_decode_roofline",
    "deepseek_mla_decode_roofline",
    "deltanet_inference_roofline",
    "fft_c2c_roofline",
    "fp8_lightning_indexer_roofline",
    "fp8_quant_roofline",
    "fused_moe_fwd_roofline",
    "fused_moe_shared_expert_fwd_roofline",
    "gated_deltanet_fwd_roofline",
    "gemm_fwd_roofline",
    "gemm_w4a16_fwd_roofline",
    "gqa_bwd_roofline",
    "gqa_decode_paged_roofline",
    "gqa_fwd_roofline",
    "gqa_prefill_paged_with_kv_cache_fwd_roofline",
    "gqa_prefill_varlen_fwd_roofline",
    "gqa_sliding_window_varlen_fwd_roofline",
    "gqa_varlen_fwd_roofline",
    "grouped_gemm_roofline",
    "mha_bwd_roofline",
    "mha_decode_paged_roofline",
    "moe_post_permute_roofline",
    "topk_selector_roofline",
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
    # q, k, v, o and do read; dq, dk and dv written.
    nbytes = batch * 8 * heads * seq_len * dim * elem_bytes
    nbytes += batch * heads * seq_len * 4  # lse
    return int(flops), int(nbytes)


def gqa_fwd_roofline(op: Any | None = None, **kwargs: Any) -> tuple[int, int]:
    """Roofline for dense grouped-query attention forward."""
    data = _shape_or_attrs(op, kwargs)
    if "q_shape" in data:
        batch, seq_len_q, heads, dim = data["q_shape"]
        kv_shape = data.get("k_shape", data.get("kv_shape"))
        if kv_shape is None:
            raise KeyError("dense GQA roofline requires k_shape or kv_shape")
        _, seq_len_kv, heads_kv, _ = kv_shape
    else:
        batch, seq_len_q, seq_len_kv, heads, heads_kv, dim = (
            data["batch"],
            data.get("seq_len_q", data.get("seq_len")),
            data.get("seq_len_kv", data.get("seq_len")),
            data["heads"],
            data["heads_kv"],
            data["dim"],
        )
    is_causal = bool(data.get("is_causal", True))
    elem_bytes = _dtype_itemsize(data.get("dtype", data.get("dtypes", "float16")))
    # FP8 input has no FP8 output, so the write is priced on its own dtype.
    out_bytes = _dtype_itemsize(data.get("out_dtype", data.get("dtype", "float16")))

    visible_scores = seq_len_q * seq_len_kv
    if is_causal:
        visible_scores = seq_len_q * (seq_len_kv - seq_len_q) + seq_len_q * (seq_len_q + 1) // 2
    flops = 4 * batch * heads * visible_scores * dim
    q_elems = batch * seq_len_q * heads * dim
    kv_elems = batch * seq_len_kv * heads_kv * dim
    # Per-KV-head scales and RoPE tables the call passed, each read once.
    optional_bytes = sum(
        prod(shape) * _dtype_itemsize(dtype) for shape, dtype in data.get("optional_shapes", ())
    )
    read_bytes = (q_elems + 2 * kv_elems) * elem_bytes + optional_bytes
    return int(flops), int(read_bytes + q_elems * out_bytes)


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


def _causal_prefill_visible_scores(seq_len_q: int, seq_len_kv: int) -> int:
    # Bottom-right alignment: when the query run is longer than the key run, its
    # leading queries see no keys at all, so only the last seq_len_kv rows count.
    rows = min(seq_len_q, seq_len_kv)
    return rows * seq_len_kv - rows * (rows - 1) // 2


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
    q_shape = data["q_shape"]
    k_shape = data["k_shape"]
    total_q, heads, dim = q_shape
    total_kv, heads_kv, _ = k_shape
    batch = int(data["batch"])
    max_seqlen_q = int(data.get("max_seqlen_q", total_q))
    max_seqlen_kv = int(data.get("max_seqlen_kv", total_kv))
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

    # The cache may hold a narrower dtype than the query, and then the call also
    # reads the two scales that dequantize it.
    cache_bytes = _dtype_itemsize(data.get("cache_dtype") or data.get("dtype", "float16"))
    quantized = cache_bytes != elem_bytes

    q_elems = total_q * heads * dim
    old_kv_elems = 2 * old_kv_tokens * heads_kv * dim
    new_kv_elems = 2 * total_q * heads_kv * dim
    append_kv_elems = new_kv_elems
    o_elems = q_elems
    # The call indexes the block table only as far as each request's pages reach,
    # and the rest of the row is capacity the algorithm never reads.
    pages_named = sum(
        -(-(int(old_len) + int(q_len)) // page_size)
        for q_len, old_len in zip(q_lens, cache_lens, strict=True)
    )
    metadata_bytes = (batch + 1) * 4 + batch * 4 + pages_named * 4
    if quantized:
        metadata_bytes += 2 * 4
    nbytes = (q_elems + new_kv_elems + o_elems) * elem_bytes
    nbytes += (old_kv_elems + append_kv_elems) * cache_bytes
    nbytes += metadata_bytes
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
    # q, o, do and dq carry all heads; k, v, dk and dv carry the KV heads.
    nbytes = batch * (4 * heads + 4 * heads_kv) * seq_len * dim * elem_bytes
    nbytes += batch * heads * seq_len * 4  # lse
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
    total_k = int(data.get("total_k", data.get("total_kv", 0)))
    max_seqlen_q = int(data.get("max_seqlen_q", total_q // batch if batch else total_q))
    q_lens = data.get("q_lens")
    k_lens = data.get("k_lens", data.get("kv_lens"))
    if q_lens is None and (cu_seqlens_q := data.get("cu_seqlens_q")) is not None:
        values = [int(x) for x in cu_seqlens_q.detach().cpu().tolist()]
        q_lens = [values[idx + 1] - values[idx] for idx in range(len(values) - 1)]
    if k_lens is None and (cu_seqlens_kv := data.get("cu_seqlens_kv")) is not None:
        values = [int(x) for x in cu_seqlens_kv.detach().cpu().tolist()]
        k_lens = [values[idx + 1] - values[idx] for idx in range(len(values) - 1)]
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
    # The two cumulative-length tensors the kernel walks to find each request's
    # bounds, one bound per request plus the zero. The packed prefill sibling
    # counts them; this one did not.
    nbytes += 2 * (batch + 1) * 4
    return int(flops), int(nbytes)


def gqa_varlen_fwd_roofline(op: Any | None = None, **kwargs: Any) -> tuple[int, int]:
    """Route unified Varlen GQA workloads to the matching cost model."""
    data = dict(_shape_or_attrs(op, kwargs))
    if "q_shape" not in data:
        data["q_shape"] = (data["total_q"], data["heads"], data["dim"])
        data["k_shape"] = (
            data.get("total_kv", data.get("total_k")),
            data["heads_kv"],
            data["dim"],
        )
    if "kv_lens" in data and "k_lens" not in data:
        data["k_lens"] = data["kv_lens"]
    if int(data.get("window_size_left", -1)) != -1 or int(data.get("window_size_right", -1)) != -1:
        return gqa_sliding_window_varlen_fwd_roofline(**data)
    return gqa_prefill_varlen_fwd_roofline(**data)


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


def gemm_fwd_roofline(op: "Op") -> tuple[int, int]:
    """Roofline for dense ``GemmFwdOp`` (``d = a @ b``, fp16/bf16).

    ``GemmFwdOp`` is input-inferred, so the logical dims ``m/n/k`` and the dtype
    are bound on the op during ``forward()``; this reads them directly, which
    stays correct across all ``trans_a``/``trans_b`` layouts (the logical dims
    are transpose-independent). Valid only after the first ``forward()``.

    Raises:
        RuntimeError: If called before ``forward()`` has bound the dims.
    """
    if getattr(op, "m", None) is None or getattr(op, "dtype", None) is None:
        raise RuntimeError(
            "GemmFwdOp.eval_roofline() is valid only after the first forward(); "
            "m/n/k and dtype are inferred from the inputs."
        )
    m, n, k = op.m, op.n, op.k
    elem_bytes = op.dtype.itemsize
    flops = 2 * m * n * k
    nbytes = (m * k + n * k + m * n) * elem_bytes
    return int(flops), int(nbytes)


def gemm_fp8_fwd_roofline(op: "Op") -> tuple[int, int]:
    """Roofline for dense FP8 ``GemmFp8FwdOp``."""
    if getattr(op, "m", None) is None or getattr(op, "dtype", None) is None:
        raise RuntimeError(
            "GemmFp8FwdOp.eval_roofline() is valid only after the first forward(); "
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


def gemm_w4a16_fwd_roofline(op: "Op") -> tuple[int, int]:
    """Roofline for dense W4A16 ``GemmW4A16FwdOp``."""
    if getattr(op, "m", None) is None or getattr(op, "dtype", None) is None:
        raise RuntimeError(
            "GemmW4A16FwdOp.eval_roofline() is valid only after the first forward(); "
            "m/n/k and dtype are inferred from the inputs."
        )
    m, n, k = op.m, op.n, op.k
    elem_bytes = op.dtype.itemsize
    group_size = int(getattr(op, "group_size", 128))
    groups = k // group_size
    flops = 2 * m * n * k
    activation_bytes = m * k * elem_bytes
    packed_weight_bytes = n * k // 2
    # One scale in the activation dtype plus one UINT8 zero point per
    # (row, group), matching the manifest's weight_scale / weight_zero dtypes.
    metadata_bytes = n * groups * (elem_bytes + 1)
    output_bytes = m * n * elem_bytes
    return int(flops), int(activation_bytes + packed_weight_bytes + metadata_bytes + output_bytes)


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
    # The call decides all three terms below. Handed bf16 tensors, the op
    # quantizes them itself and produces the scale, so the fp8 tensors and the
    # scale are intermediates and the public reads are bf16. Handed fp8 tensors,
    # the caller supplies the scale and it is a read of its own. index_q and
    # index_k carry their own dtypes: the signature lets them differ, and only
    # the pre-quantized path requires both to be fp8.
    q_elem = _dtype_itemsize(getattr(op, "dtype", "bfloat16"))
    k_elem = _dtype_itemsize(getattr(op, "index_k_dtype", None) or "bfloat16")
    flops = 2 * scores * index_dim
    nbytes = q_elems * q_elem + k_elems * k_elem
    if _supplied(op, "index_k_scale"):
        nbytes += batch * seq_len_kv * kv_group * 4
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


def bmm_fp8_fwd_roofline(op: "Op") -> tuple[int, int]:
    """Roofline for batched FP8 GEMM ``BmmFp8FwdOp``.

    Layout matches the fp16 ``bmm_fwd_roofline``: ``a``: $[B \\times M \\times K]$,
    ``b``: $[B \\times K \\times N]$. Per-tensor scales only and no fused bias, so bytes are
    ``B*M*K`` (A, fp8) + ``B*K*N`` (B, fp8) + ``B*M*N`` (C, ``out_dtype``)
    + 8 for the two batch-independent fp32 scales.

    Valid only after the first ``forward()`` binds dims and dtype.
    """
    if getattr(op, "m", None) is None or getattr(op, "dtype", None) is None:
        raise RuntimeError(
            "BmmFp8FwdOp.eval_roofline() is valid only after the first forward(); "
            "batch/m/n/k and dtype are inferred from the inputs."
        )
    batch, m, n, k = op.batch, op.m, op.n, op.k
    input_bytes = op.dtype.itemsize
    out_bytes = op.out_dtype.itemsize
    flops = 2 * batch * m * n * k
    nbytes = batch * ((m * k + n * k) * input_bytes + m * n * out_bytes) + 8
    return int(flops), int(nbytes)


def _nsa_request_lens(data: dict[str, Any], c_seq_len: int, seq_num: int) -> list[int]:
    """Token count per request, read off ``offsets`` when the call supplied it.

    The manifest path has only the aggregate, so it falls back to an even split; the
    benchmark path has the tensor and gets the lengths that were actually run.
    """
    offsets = data.get("offsets")
    if offsets is not None:
        values = [int(x) for x in offsets.detach().cpu().tolist()]
        return [values[i + 1] - values[i] for i in range(len(values) - 1)]
    return _distribute_total(c_seq_len, seq_num, c_seq_len)


def _nsa_ragged_index_bytes(seq_num: int, c_seq_len: int, chunk_offsets: bool = True) -> int:
    """The int32 metadata every NSA pass reads to walk a packed batch.

    ``offsets`` and ``token_indices`` always; ``chunk_offsets`` for the two passes that
    index the compressed chunks.
    """
    bounds = 2 * (seq_num + 1) if chunk_offsets else seq_num + 1
    return (bounds + c_seq_len * 2) * 4


def _nsa_closed_chunks(length: int, block_size: int) -> int:
    """``sum(t // block_size for t in range(length))``, in closed form."""
    full, rest = divmod(length, block_size)
    return block_size * full * (full - 1) // 2 + rest * full


def nsa_cmp_fwd_varlen_roofline(op: Any | None = None, **kwargs: Any) -> tuple[int, int]:
    """Roofline for the NSA compression forward over a packed batch.

    Each token attends to its own request's closed chunks. In: q and the compressed
    k/v. Out: the attention output and the lse the top-k pass scores against.
    """
    data = _shape_or_attrs(op, kwargs)
    if "q_shape" in data:
        c_seq_len, heads, dim_k = data["q_shape"]
        chunk_num, head_kv, _ = data["k_cmp_shape"]
        dim_v = data["v_cmp_shape"][2]
        seq_num = data["offsets_shape"][0] - 1
    else:
        c_seq_len, heads, dim_k = data["c_seq_len"], data["heads"], data["dim_k"]
        chunk_num, head_kv = data["chunk_num"], data["head_kv"]
        dim_v, seq_num = data["dim_v"], data["seq_num"]
    block_size = int(data["bs"])
    elem_bytes = _dtype_itemsize(data.get("dtype", data.get("dtypes", "float16")))

    lens = _nsa_request_lens(data, c_seq_len, seq_num)
    # The kernel scores `(t + 1) // bs` chunks for the token at position `t`.
    pairs = sum(_nsa_closed_chunks(length + 1, block_size) for length in lens)
    flops = 2 * pairs * heads * (dim_k + dim_v)
    nbytes = (
        c_seq_len * heads * (dim_k + dim_v + 1) + chunk_num * head_kv * (dim_k + dim_v)
    ) * elem_bytes
    nbytes += _nsa_ragged_index_bytes(seq_num, c_seq_len)
    return int(flops), int(nbytes)


def nsa_topk_varlen_roofline(op: Any | None = None, **kwargs: Any) -> tuple[int, int]:
    """Roofline for the NSA block selection over a packed batch.

    Two QK passes over the compressed chunks, no PV. Out: one int32 block id per
    token, KV head and kept block.
    """
    data = _shape_or_attrs(op, kwargs)
    if "q_shape" in data:
        c_seq_len, heads, dim = data["q_shape"]
        chunk_num, head_kv, _ = data["k_cmp_shape"]
        seq_num = data["offsets_shape"][0] - 1
    else:
        c_seq_len, heads, dim = data["c_seq_len"], data["heads"], data["dim"]
        chunk_num, head_kv, seq_num = data["chunk_num"], data["head_kv"], data["seq_num"]
    block_size = int(data["bs"])
    selected = int(data["selected_block_num"])
    elem_bytes = _dtype_itemsize(data.get("dtype", data.get("dtypes", "float16")))

    lens = _nsa_request_lens(data, c_seq_len, seq_num)
    # Two QK passes over slightly different chunk counts: the lse pass scores
    # `(t + 1) // bs` chunks for the token at position `t`, the selection pass `t // bs + 1`.
    pairs = sum(
        _nsa_closed_chunks(length + 1, block_size) + _nsa_closed_chunks(length, block_size) + length
        for length in lens
    )
    flops = 2 * pairs * heads * dim
    # `lse_in` produces no read: the top-k kernel recomputes the lse itself and
    # discards the argument, and a declared input the algorithm does not read
    # moves no bytes.
    nbytes = (c_seq_len * heads * dim + chunk_num * head_kv * dim) * elem_bytes
    nbytes += c_seq_len * head_kv * selected * 4
    nbytes += _nsa_ragged_index_bytes(seq_num, c_seq_len)
    return int(flops), int(nbytes)


def _nsa_selected_block_loads(
    data: dict[str, Any], c_seq_len: int, head_kv: int, selected: int, block_size: int
) -> int:
    """Block tiles the sparse forward loads, summed over tokens and KV heads.

    The kernel walks ``block_counts`` entries and skips a block starting past the token.
    The manifest path has only shapes, so it takes the ``selected``-per-token bound.
    """
    counts = data.get("block_counts")
    indices = data.get("block_indices")
    token_indices = data.get("token_indices")
    if counts is None or indices is None or token_indices is None:
        return c_seq_len * head_kv * selected

    # Read as lists, not reduced on device: this module imports where torch does not.
    kept = counts.reshape(-1).tolist()
    blocks = indices.reshape(-1, selected).tolist()
    positions = token_indices[:, 1].tolist()
    return sum(
        sum(1 for start in row[:n] if 0 <= start * block_size <= positions[i // head_kv])
        for i, (n, row) in enumerate(zip(kept, blocks, strict=True))
    )


def nsa_fwd_varlen_roofline(op: Any | None = None, **kwargs: Any) -> tuple[int, int]:
    """Roofline for the NSA sparse forward over a packed batch.

    Each token attends to the blocks of ``block_size`` tokens its selection kept, so the
    score count comes from that selection rather than from the sequence length. In: q and
    the gathered k/v of those blocks, plus the selection. Out: the attention output.
    """
    data = _shape_or_attrs(op, kwargs)
    if "q_shape" in data:
        c_seq_len, heads, dim = data["q_shape"]
        head_kv = data["k_shape"][1]
        selected = data["block_indices_shape"][2]
    else:
        c_seq_len, heads, dim = data["c_seq_len"], data["heads"], data["dim"]
        head_kv, selected = data["head_kv"], data["selected_blocks"]
    block_size = int(data["block_size"])
    elem_bytes = _dtype_itemsize(data.get("dtype", data.get("dtypes", "float16")))

    loads = _nsa_selected_block_loads(data, c_seq_len, head_kv, selected, block_size)
    rows = loads * block_size
    flops = 4 * rows * (heads // head_kv) * dim
    nbytes = (2 * c_seq_len * heads * dim + 2 * rows * dim) * elem_bytes
    nbytes += c_seq_len * head_kv * (selected + 1) * 4
    batch = int(data["offsets_shape"][0]) - 1 if "q_shape" in data else int(data["batch"])
    nbytes += _nsa_ragged_index_bytes(batch, c_seq_len, chunk_offsets=False)
    return int(flops), int(nbytes)
