"""Linear-attention kernels: DeltaNet, Gated DeltaNet (GDN) and Gated Linear Attention (GLA).

Kimi Delta Attention (KDA) lives in the ``kda`` subpackage: the gated delta rule whose
decay is one log-space value per key channel.

The DeltaNet and GLA chunked kernels share the V-tile width rule (``v_tile``); the
DeltaNet forward also tunes through ``deltanet.autotune``. Each algorithm's chunk,
recurrent and inference implementations live together in its subpackage. The
gated and ungated DeltaNet inference kernels share ``delta_decode``.
"""

from tileops.kernels.linear_attention.call_spec import (
    DeltaNetCall,
    DeltaNetChunkBwdInterface,
    DeltaNetChunkCall,
    DeltaNetChunkFwdInterface,
    DeltaNetDecodeCall,
    DeltaNetDecodeFwdInterface,
    DeltaNetFwdInterface,
    GDNCall,
    GDNFwdInterface,
    GLAChunkBwdInterface,
    GLAChunkCall,
    GLAChunkFwdInterface,
    GLADecodeCall,
    GLADecodeFwdInterface,
    KDACall,
    KDAFwdInterface,
)
from tileops.kernels.linear_attention.deltanet import (
    DeltaNetChunkBwdKernel,
    DeltaNetChunkFwdKernel,
    DeltaNetDenseDecodeFwdKernel,
    DeltaNetDensePrefillFwdKernel,
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
    GLAChunkBwdKernel,
    GLAChunkFwdKernel,
    GLADensePrefillFwdKernel,
    GLADensePrefillSubchunkKernel,
)
from tileops.kernels.linear_attention.gla.recurrent import GLADecodeFP32Kernel, GLADecodeKernel
from tileops.kernels.linear_attention.kda import (
    KDAChunkPrefillFwdKernel,
    KDAFusedPrefillFwdKernel,
    KDARecurrentDecodeFwdKernel,
)

__all__ = [
    "DeltaNetCall",
    "DeltaNetChunkBwdInterface",
    "DeltaNetChunkBwdKernel",
    "DeltaNetChunkCall",
    "DeltaNetChunkFwdInterface",
    "DeltaNetChunkFwdKernel",
    "DeltaNetDecodeCall",
    "DeltaNetDecodeFP32Kernel",
    "DeltaNetDecodeFwdInterface",
    "DeltaNetDecodeKernel",
    "DeltaNetDecodeRawCudaFlaStyleKernel",
    "DeltaNetDenseDecodeFwdKernel",
    "DeltaNetDensePrefillFwdKernel",
    "DeltaNetFwdInterface",
    "GDNCall",
    "GDNDenseDecodeFwdKernel",
    "GDNDensePrefillFwdKernel",
    "GDNFwdInterface",
    "GLAChunkBwdInterface",
    "GLAChunkBwdKernel",
    "GLAChunkCall",
    "GLAChunkFwdInterface",
    "GLAChunkFwdKernel",
    "GLADecodeCall",
    "GLADecodeFP32Kernel",
    "GLADecodeFwdInterface",
    "GLADecodeKernel",
    "GLADensePrefillFwdKernel",
    "GLADensePrefillSubchunkKernel",
    "KDACall",
    "KDAChunkPrefillFwdKernel",
    "KDAFusedPrefillFwdKernel",
    "KDAFwdInterface",
    "KDARecurrentDecodeFwdKernel",
]
