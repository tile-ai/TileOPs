# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under the MIT License; see THIRD_PARTY_NOTICES.md for details.
# Adapted and modified for TileOps GatedDeltaNet prefill integration.

from .tilelang_compat import install_gemm_v1_compat

install_gemm_v1_compat()

from .cp_fwd import correct_initial_states, get_warmup_chunks  # noqa: E402
from .fused_fwd import fused_gdr_fwd  # noqa: E402
from .prepare_h import fused_gdr_h  # noqa: E402

__all__ = [
    "correct_initial_states",
    "fused_gdr_fwd",
    "fused_gdr_h",
    "get_warmup_chunks",
]
