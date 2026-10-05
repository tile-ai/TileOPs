"""Linear-attention kernels: DeltaNet, Gated DeltaNet (GDN) and Gated Linear Attention (GLA).

Kimi Delta Attention (KDA) lives in the ``kda`` subpackage: the gated delta rule whose
decay is one log-space value per key channel.

The DeltaNet and GLA chunked kernels share the V-tile width rule (``v_tile``); the
DeltaNet forward also tunes through ``deltanet.autotune``. Each algorithm's chunk,
recurrent and inference implementations live together in its subpackage. The
gated and ungated DeltaNet inference kernels share ``delta_decode``.
"""

from tileops.kernels.linear_attention.call_spec import (
    DeltaNetBwdInterface,
    DeltaNetChunkCall,
    DeltaNetDecodeCall,
    DeltaNetDecodeFwdInterface,
    DeltaNetFwdInterface,
    DeltaNetInferenceCall,
    DeltaNetInferenceFwdInterface,
    GDNCall,
    GDNFwdInterface,
    GLABwdInterface,
    GLAChunkCall,
    GLADecodeCall,
    GLADecodeFwdInterface,
    GLAFwdInterface,
    KDACall,
    KDAFwdInterface,
)
from tileops.kernels.linear_attention.deltanet import (
    DeltaNetBwdKernel,
    DeltaNetDenseDecodeFwdKernel,
    DeltaNetDensePrefillFwdKernel,
    DeltaNetFwdKernel,
)
from tileops.kernels.linear_attention.deltanet.recurrent import (
    DeltaNetDecodeFP32Kernel,
    DeltaNetDecodeKernel,
    DeltaNetDecodeRawCudaFlaStyleKernel,
)
from tileops.kernels.linear_attention.gdn import (
    GDNDenseDecodeFwdKernel,
    GDNDensePrefillFwdKernel,
)
from tileops.kernels.linear_attention.gla import (
    GLABwdKernel,
    GLADensePrefillFwdKernel,
    GLADensePrefillSubchunkKernel,
    GLAFwdKernel,
)
from tileops.kernels.linear_attention.gla.recurrent import GLADecodeFP32Kernel, GLADecodeKernel
from tileops.kernels.linear_attention.kda import (
    KDAChunkPrefillFwdKernel,
    KDAFusedPrefillFwdKernel,
    KDARecurrentDecodeFwdKernel,
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
    "GDNCall",
    "GDNDenseDecodeFwdKernel",
    "GDNDensePrefillFwdKernel",
    "GDNFwdInterface",
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
    "KDACall",
    "KDAChunkPrefillFwdKernel",
    "KDAFusedPrefillFwdKernel",
    "KDAFwdInterface",
    "KDARecurrentDecodeFwdKernel",
]
