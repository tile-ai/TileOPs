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
    "deepseek_dsa_decode_roofline",
    "deepseek_mla_decode_roofline",
    "fused_moe_fwd_bytes",
    "gqa_bwd_roofline",
    "gqa_decode_paged_roofline",
    "gqa_decode_roofline",
    "gqa_fwd_roofline",
    "gqa_sliding_window_fwd_roofline",
    "gqa_sliding_window_varlen_fwd_roofline",
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

    TODO: implement the formula once ``WhereFwdOp`` flips from
    ``status: spec-only`` to ``implemented``.
    """
    raise NotImplementedError


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
