"""Kimi Delta Attention (KDA): the gated delta rule with a per-key-channel decay."""

from tileops.kernels.linear_attention.kda.decode import (
    KDARecurrentDecodeFwdKernel,
)
from tileops.kernels.linear_attention.kda.prefill import (
    KDAChunkPrefillFwdKernel,
)
from tileops.kernels.linear_attention.kda.prefill_fused import (
    KDAFusedPrefillFwdKernel,
)

__all__ = [
    "KDAChunkPrefillFwdKernel",
    "KDAFusedPrefillFwdKernel",
    "KDARecurrentDecodeFwdKernel",
]
