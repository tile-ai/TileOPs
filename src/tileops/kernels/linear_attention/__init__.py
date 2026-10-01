"""Linear-attention kernels: DeltaNet, Gated DeltaNet and Gated Linear Attention (GLA).

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
]
