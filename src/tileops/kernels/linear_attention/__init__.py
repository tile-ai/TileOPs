"""Linear-attention kernels: DeltaNet, Gated DeltaNet and Gated Linear Attention (GLA).

Kimi Delta Attention lives in the ``kda`` subpackage: the gated delta rule whose
decay is one log-space value per key channel.

The DeltaNet and GLA chunked kernels share the V-tile width rule (``v_tile``); the
DeltaNet forward also tunes through ``autotune``. The chunkwise kernels live in the
per-variant subpackages; the DeltaNet and GLA single-token decode kernels are the
``*_recurrence`` modules.
"""

from tileops.kernels.linear_attention.call_spec import (
    DeltaNetBwdInterface,
    DeltaNetChunkCall,
    DeltaNetDecodeCall,
    DeltaNetDecodeFwdInterface,
    DeltaNetFwdInterface,
    DeltaNetInferenceCall,
    DeltaNetInferenceFwdInterface,
    GatedDeltaNetCall,
    GatedDeltaNetFwdInterface,
    GLABwdInterface,
    GLAChunkCall,
    GLADecodeCall,
    GLADecodeFwdInterface,
    GLAFwdInterface,
    KimiDeltaAttentionCall,
    KimiDeltaAttentionFwdInterface,
)
from tileops.kernels.linear_attention.deltanet import (
    DeltaNetBwdKernel,
    DeltaNetDenseDecodeFwdKernel,
    DeltaNetDensePrefillFwdKernel,
    DeltaNetFwdKernel,
)
from tileops.kernels.linear_attention.deltanet_recurrence import (
    DeltaNetDecodeFP32Kernel,
    DeltaNetDecodeKernel,
    DeltaNetDecodeRawCudaFlaStyleKernel,
)
from tileops.kernels.linear_attention.gated_deltanet import (
    GatedDeltaNetDenseDecodeFwdKernel,
    GatedDeltaNetDensePrefillFwdKernel,
)
from tileops.kernels.linear_attention.gla import (
    GLABwdKernel,
    GLADensePrefillFwdKernel,
    GLADensePrefillSubchunkKernel,
    GLAFwdKernel,
)
from tileops.kernels.linear_attention.gla_recurrence import GLADecodeFP32Kernel, GLADecodeKernel
from tileops.kernels.linear_attention.kda import (
    KimiDeltaAttentionChunkPrefillFwdKernel,
    KimiDeltaAttentionFusedPrefillFwdKernel,
    KimiDeltaAttentionRecurrentDecodeFwdKernel,
)

__all__ = [
    "DeltaNetBwdInterface",
    "DeltaNetBwdKernel",
    "DeltaNetChunkCall",
    "DeltaNetDecodeCall",
    "DeltaNetDecodeFP32Kernel",
    "DeltaNetDecodeFwdInterface",
    "DeltaNetDecodeKernel",
    "DeltaNetDecodeRawCudaFlaStyleKernel",
    "DeltaNetDenseDecodeFwdKernel",
    "DeltaNetDensePrefillFwdKernel",
    "DeltaNetFwdInterface",
    "DeltaNetFwdKernel",
    "DeltaNetInferenceCall",
    "DeltaNetInferenceFwdInterface",
    "GLABwdInterface",
    "GLABwdKernel",
    "GLAChunkCall",
    "GLADecodeCall",
    "GLADecodeFP32Kernel",
    "GLADecodeFwdInterface",
    "GLADecodeKernel",
    "GLADensePrefillFwdKernel",
    "GLADensePrefillSubchunkKernel",
    "GLAFwdInterface",
    "GLAFwdKernel",
    "GatedDeltaNetCall",
    "GatedDeltaNetDenseDecodeFwdKernel",
    "GatedDeltaNetDensePrefillFwdKernel",
    "GatedDeltaNetFwdInterface",
    "KimiDeltaAttentionCall",
    "KimiDeltaAttentionChunkPrefillFwdKernel",
    "KimiDeltaAttentionFusedPrefillFwdKernel",
    "KimiDeltaAttentionFwdInterface",
    "KimiDeltaAttentionRecurrentDecodeFwdKernel",
]
