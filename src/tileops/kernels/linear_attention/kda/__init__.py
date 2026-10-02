"""Kimi Delta Attention: the gated delta rule with a per-key-channel decay."""

from tileops.kernels.linear_attention.kda.decode import (
    KimiDeltaAttentionRecurrentDecodeFwdKernel,
)
from tileops.kernels.linear_attention.kda.prefill import (
    KimiDeltaAttentionChunkPrefillFwdKernel,
)
from tileops.kernels.linear_attention.kda.prefill_fused import (
    KimiDeltaAttentionFusedPrefillFwdKernel,
)

__all__ = [
    "KimiDeltaAttentionChunkPrefillFwdKernel",
    "KimiDeltaAttentionFusedPrefillFwdKernel",
    "KimiDeltaAttentionRecurrentDecodeFwdKernel",
]
