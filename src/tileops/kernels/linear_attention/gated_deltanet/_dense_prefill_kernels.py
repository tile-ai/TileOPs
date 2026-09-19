# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under the MIT License.
# Adapted and modified for TileOps GatedDeltaNet prefill integration.
"""Compatibility facade for private dense-prefill stages."""

from ._dense_prefill_forward import fused_gdr_fwd
from ._dense_prefill_prepare import (
    _prefill_blocksolve_A_bthd,
    _prefill_chunk_local_cumsum_bthd_tl,
    correct_initial_states,
    fused_gdr_h,
    get_warmup_chunks,
)

__all__ = [
    "_prefill_blocksolve_A_bthd",
    "_prefill_chunk_local_cumsum_bthd_tl",
    "correct_initial_states",
    "fused_gdr_fwd",
    "fused_gdr_h",
    "get_warmup_chunks",
]
