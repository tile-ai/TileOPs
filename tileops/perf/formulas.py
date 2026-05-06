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
    "eq_fwd_roofline",
    "floor_divide_fwd_roofline",
    "fused_moe_fwd_bytes",
    "ge_fwd_roofline",
    "gqa_bwd_roofline",
    "gqa_decode_paged_roofline",
    "gqa_decode_roofline",
    "gqa_fwd_roofline",
    "gqa_prefill_fwd_roofline",
    "gqa_prefill_paged_with_kv_cache_fwd_roofline",
    "gqa_prefill_varlen_fwd_roofline",
    "gqa_prefill_with_kv_cache_fwd_roofline",
    "gqa_sliding_window_fwd_roofline",
    "gqa_sliding_window_varlen_fwd_roofline",
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
    "minimum_fwd_roofline",
    "mul_fwd_roofline",
    "ne_fwd_roofline",
    "pow_fwd_roofline",
    "remainder_fwd_roofline",
    "sub_fwd_roofline",
    "where_fwd_roofline",
]


# ---------------------------------------------------------------------------
# MHA prefill
# ---------------------------------------------------------------------------


def mha_fwd_roofline(**kwargs: Any) -> dict[str, int]:
    """Roofline for multi-head attention forward (prefill).

    TODO: implement full formula based on B, S, H, D, is_causal.
    """
    raise NotImplementedError


def mha_bwd_roofline(**kwargs: Any) -> dict[str, int]:
    """Roofline for multi-head attention backward.

    TODO: implement full formula based on B, S, H, D, is_causal.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# GQA prefill
# ---------------------------------------------------------------------------


def gqa_fwd_roofline(**kwargs: Any) -> dict[str, int]:
    """Roofline for grouped-query attention forward (prefill).

    TODO: implement full formula based on B, S, H, H_kv, D, is_causal.
    """
    raise NotImplementedError


def _dtype_itemsize(dtype: Any) -> int:
    if isinstance(dtype, (list, tuple)):
        dtype = dtype[0] if dtype else "float16"
    if hasattr(dtype, "itemsize"):
        return int(dtype.itemsize)
    dtype_name = str(dtype)
    if "float32" in dtype_name or "int32" in dtype_name:
        return 4
    if "float64" in dtype_name or "int64" in dtype_name:
        return 8
    if "bool" in dtype_name or "int8" in dtype_name or "uint8" in dtype_name:
        return 1
    return 2


def _causal_prefill_visible_scores(seq_len_q: int, seq_len_kv: int) -> int:
    return seq_len_q * seq_len_kv - seq_len_q * (seq_len_q - 1) // 2


def gqa_prefill_fwd_roofline(**kwargs: Any) -> dict[str, int]:
    """Conservative roofline for dense GQA prefill.

    Workloads bind ``q_shape`` and ``kv_shape``. Causal mode uses bottom-right
    alignment with ``S_q <= S_kv``.
    """
    q_shape = kwargs["q_shape"]
    kv_shape = kwargs.get("kv_shape", kwargs.get("k_shape"))
    batch, seq_len_q, heads, dim = q_shape
    _, seq_len_kv, heads_kv, _ = kv_shape
    is_causal = bool(kwargs.get("is_causal", True))
    elem_bytes = _dtype_itemsize(kwargs.get("dtype", kwargs.get("dtypes", "float16")))

    visible = (
        _causal_prefill_visible_scores(seq_len_q, seq_len_kv)
        if is_causal else seq_len_q * seq_len_kv)
    flops = 4 * batch * heads * visible * dim

    q_elems = batch * seq_len_q * heads * dim
    kv_elems = batch * seq_len_kv * heads_kv * dim
    o_elems = q_elems
    nbytes = (q_elems + 2 * kv_elems + o_elems) * elem_bytes
    return {"flops": int(flops), "bytes": int(nbytes)}


def _distribute_total(total: int, batch: int, max_len: int) -> list[int]:
    lengths = [0] * batch
    remaining = total
    for idx in range(batch):
        slots_left = batch - idx - 1
        value = min(max_len, remaining - slots_left)
        lengths[idx] = value
        remaining -= value
    return lengths


def gqa_prefill_varlen_fwd_roofline(**kwargs: Any) -> dict[str, int]:
    """Conservative roofline for packed-varlen GQA prefill.

    Preferred workload binding supplies explicit ``q_lens`` and ``kv_lens``.
    If they are absent, this falls back to a deterministic fill from aggregate
    totals and ``max_seqlen_*`` so benchmark metadata remains reproducible.
    Causal mode uses bottom-right alignment independently per request.
    """
    q_shape = kwargs["q_shape"]
    k_shape = kwargs["k_shape"]
    total_q, heads, dim = q_shape
    total_kv, heads_kv, _ = k_shape
    batch = int(kwargs["batch"])
    max_seqlen_q = int(kwargs["max_seqlen_q"])
    max_seqlen_kv = int(kwargs["max_seqlen_kv"])
    is_causal = bool(kwargs.get("is_causal", True))
    elem_bytes = _dtype_itemsize(kwargs.get("dtype", kwargs.get("dtypes", "float16")))

    q_lens = kwargs.get("q_lens")
    kv_lens = kwargs.get("kv_lens")
    if q_lens is None:
        q_lens = _distribute_total(total_q, batch, max_seqlen_q)
    if kv_lens is None:
        kv_lens = _distribute_total(total_kv, batch, max_seqlen_kv)

    visible = 0
    for q_len, kv_len in zip(q_lens, kv_lens, strict=True):
        visible += (
            _causal_prefill_visible_scores(int(q_len), int(kv_len))
            if is_causal else int(q_len) * int(kv_len))
    flops = 4 * heads * visible * dim

    q_elems = total_q * heads * dim
    kv_elems = total_kv * heads_kv * dim
    o_elems = q_elems
    cu_bytes = 2 * (batch + 1) * 4
    nbytes = (q_elems + 2 * kv_elems + o_elems) * elem_bytes + cu_bytes
    return {"flops": int(flops), "bytes": int(nbytes)}


def gqa_prefill_with_kv_cache_fwd_roofline(**kwargs: Any) -> dict[str, int]:
    """Conservative roofline for contiguous-cache GQA prefill.

    The benchmark workload convention uses
    ``old_len = S_kv_cap - S_new`` for every batch item.
    """
    q_shape = kwargs["q_shape"]
    k_new_shape = kwargs["k_new_shape"]
    k_cache_shape = kwargs["k_cache_shape"]
    batch, seq_len_new, heads, dim = q_shape
    _, _, heads_kv, _ = k_new_shape
    _, seq_len_cap, _, _ = k_cache_shape
    old_len = seq_len_cap - seq_len_new
    is_causal = bool(kwargs.get("is_causal", True))
    elem_bytes = _dtype_itemsize(kwargs.get("dtype", kwargs.get("dtypes", "float16")))

    visible = (
        seq_len_new * old_len + seq_len_new * (seq_len_new + 1) // 2
        if is_causal else seq_len_new * (old_len + seq_len_new))
    flops = 4 * batch * heads * visible * dim

    q_elems = batch * seq_len_new * heads * dim
    old_kv_elems = 2 * batch * old_len * heads_kv * dim
    new_kv_elems = 2 * batch * seq_len_new * heads_kv * dim
    append_kv_elems = new_kv_elems
    o_elems = q_elems
    cache_seqlens_bytes = batch * 4
    nbytes = (
        (q_elems + old_kv_elems + new_kv_elems + append_kv_elems + o_elems) * elem_bytes
        + cache_seqlens_bytes)
    return {"flops": int(flops), "bytes": int(nbytes)}


def gqa_prefill_paged_with_kv_cache_fwd_roofline(**kwargs: Any) -> dict[str, int]:
    """Conservative roofline for paged-cache GQA prefill.

    Paged workloads should bind explicit per-request ``q_lens`` and
    ``cache_lens``. If they are absent, fall back to a deterministic fill from
    aggregate metadata so older workload entries remain evaluable.
    """
    total_q = int(kwargs["total_q"])
    batch = int(kwargs["batch"])
    heads = int(kwargs["heads"])
    heads_kv = int(kwargs["heads_kv"])
    dim = int(kwargs["dim"])
    max_pages_per_req = int(kwargs["max_pages_per_req"])
    page_size = int(kwargs["page_size"])
    max_seqlen_q = int(kwargs["max_seqlen_q"])
    is_causal = bool(kwargs.get("is_causal", True))
    elem_bytes = _dtype_itemsize(kwargs.get("dtype", kwargs.get("dtypes", "float16")))

    q_lens = kwargs.get("q_lens")
    if q_lens is None:
        q_lens = _distribute_total(total_q, batch, max_seqlen_q)
    cache_lens = kwargs.get("cache_lens")
    if cache_lens is None:
        max_total_len = int(
            kwargs.get("max_position") or max_pages_per_req * page_size
        )
        cache_lens = [max(max_total_len - int(q_len), 0) for q_len in q_lens]

    visible = 0
    old_kv_tokens = 0
    for q_len, old_len in zip(q_lens, cache_lens, strict=True):
        q_len = int(q_len)
        old_len = int(old_len)
        old_kv_tokens += old_len
        visible += (
            q_len * old_len + q_len * (q_len + 1) // 2
            if is_causal else q_len * (old_len + q_len)
        )
    flops = 4 * heads * visible * dim

    q_elems = total_q * heads * dim
    old_kv_elems = 2 * old_kv_tokens * heads_kv * dim
    new_kv_elems = 2 * total_q * heads_kv * dim
    append_kv_elems = new_kv_elems
    o_elems = q_elems
    metadata_bytes = (
        (batch + 1) * 4
        + batch * 4
        + batch * max_pages_per_req * 4
    )
    nbytes = (
        (q_elems + old_kv_elems + new_kv_elems + append_kv_elems + o_elems)
        * elem_bytes
        + metadata_bytes
    )
    return {"flops": int(flops), "bytes": int(nbytes)}


def gqa_bwd_roofline(**kwargs: Any) -> dict[str, int]:
    """Roofline for grouped-query attention backward.

    TODO: implement full formula based on B, S, H, H_kv, D, is_causal.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# MHA decode
# ---------------------------------------------------------------------------


def mha_decode_roofline(**kwargs: Any) -> dict[str, int]:
    """Roofline for MHA decode with KV cache.

    TODO: implement full formula based on B, H, N_kv, D.
    """
    raise NotImplementedError


def mha_decode_paged_roofline(**kwargs: Any) -> dict[str, int]:
    """Roofline for paged MHA decode with KV cache.

    TODO: implement full formula based on B, H, N_kv, D, page_size.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# GQA decode
# ---------------------------------------------------------------------------


def gqa_decode_roofline(**kwargs: Any) -> dict[str, int]:
    """Roofline for GQA decode with KV cache.

    TODO: implement full formula based on B, H, H_kv, N_kv, D.
    """
    raise NotImplementedError


def gqa_decode_paged_roofline(**kwargs: Any) -> dict[str, int]:
    """Roofline for paged GQA decode with KV cache.

    TODO: implement full formula based on B, H, H_kv, N_kv, D, page_size.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Sliding window attention
# ---------------------------------------------------------------------------


def gqa_sliding_window_fwd_roofline(**kwargs: Any) -> dict[str, int]:
    """Roofline for GQA sliding window forward.

    TODO: implement full formula based on B, S, H, H_kv, D, window_size.
    """
    raise NotImplementedError


def gqa_sliding_window_varlen_fwd_roofline(**kwargs: Any) -> dict[str, int]:
    """Roofline for variable-length GQA sliding window forward.

    TODO: implement full formula based on B, H, H_kv, D, window_size.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# DeepSeek MLA / DSA decode
# ---------------------------------------------------------------------------


def deepseek_mla_decode_roofline(**kwargs: Any) -> dict[str, int]:
    """Roofline for DeepSeek MLA decode with KV cache.

    TODO: implement full formula based on B, H, H_kv, N_kv, D, pe_dim.
    """
    raise NotImplementedError


def deepseek_dsa_decode_roofline(**kwargs: Any) -> dict[str, int]:
    """Roofline for DeepSeek sparse attention decode.

    TODO: implement full formula based on B, H, S, S_kv, D, D_tail, topk.
    """
    raise NotImplementedError


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

    Per-output element: one ``max(input, min)`` then one ``min(.., max)``
    → ``flops = 2 * N_total``. Bytes: read input + read min + read max +
    write out, all post-broadcast at ``elem_bytes`` each →
    ``bytes = 4 * N_total * elem_bytes``.

    Args:
        op: bound ``ClampFwdOp`` instance exposing ``N_total`` and
            ``dtype``.

    Returns:
        ``(flops, bytes)`` ints.
    """
    n_total = int(op.N_total)
    elem_bytes = op.dtype.itemsize
    return 2 * n_total, 4 * n_total * elem_bytes


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


def fused_moe_fwd_bytes(**kwargs: Any) -> dict[str, int]:
    """Roofline for FusedMoeFwdOp / FusedMoeFwdCbFwdOp.

    Mixed-dtype inputs: hidden_states (bf16/fp16) + gating_output (float32).
    elem_bytes applies only to the bf16/fp16 tensors.
    """
    num_tokens = kwargs["num_tokens"]
    num_experts = kwargs["num_experts"]
    top_k = kwargs["top_k"]
    hidden_size = kwargs["hidden_size"]
    ffn_size = kwargs["ffn_size"]
    elem_bytes = kwargs.get("elem_bytes", 2)  # bf16 default

    flops = num_tokens * top_k * 6 * ffn_size * hidden_size
    weight_bytes = num_experts * 3 * ffn_size * hidden_size * elem_bytes
    token_bytes = 2 * num_tokens * hidden_size * elem_bytes
    gating_bytes = num_tokens * num_experts * 4  # float32
    return {"flops": flops, "bytes": weight_bytes + token_bytes + gating_bytes}
