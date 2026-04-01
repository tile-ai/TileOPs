"""Roofline cost-model functions for Tier 2 ops (attention, conv, MoE, etc.).

Each function accepts keyword arguments matching the workload dimensions
and returns a dict with ``"flops"`` and ``"bytes"`` keys (both int).

These are referenced from ``ops_manifest.yaml`` via the ``roofline.func``
field, e.g.::

    roofline:
      func: "tileops.perf.formulas.mha_fwd_roofline"
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "mha_fwd_roofline",
    "mha_bwd_roofline",
    "gqa_fwd_roofline",
    "gqa_bwd_roofline",
    "mha_decode_roofline",
    "mha_decode_paged_roofline",
    "gqa_decode_roofline",
    "gqa_decode_paged_roofline",
    "gqa_sliding_window_fwd_roofline",
    "gqa_sliding_window_varlen_fwd_roofline",
    "deepseek_mla_decode_roofline",
    "deepseek_dsa_decode_roofline",
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
