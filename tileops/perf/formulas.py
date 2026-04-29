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
    "clamp_fwd_roofline",
    "clamp_max_fwd_roofline",
    "clamp_min_fwd_roofline",
    "deepseek_dsa_decode_roofline",
    "deepseek_mla_decode_roofline",
    "fused_moe_fwd_bytes",
    "gqa_bwd_roofline",
    "gqa_decode_paged_roofline",
    "gqa_decode_roofline",
    "gqa_fwd_roofline",
    "gqa_sliding_window_fwd_roofline",
    "gqa_sliding_window_varlen_fwd_roofline",
    "masked_fill_fwd_roofline",
    "mha_bwd_roofline",
    "mha_decode_paged_roofline",
    "mha_decode_roofline",
    "mha_fwd_roofline",
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
