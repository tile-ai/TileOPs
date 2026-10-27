"""Linear-attention kernels: DeltaNet, Gated DeltaNet (GDN) and Gated Linear Attention (GLA).

Kimi Delta Attention (KDA) lives in the ``kda`` subpackage: the gated delta rule whose
decay is one log-space value per key channel.

The DeltaNet and GLA chunked kernels share the V-tile width rule (``v_tile``); the
DeltaNet forward also tunes through ``deltanet.autotune``. Each algorithm's chunk,
recurrent and inference implementations live together in its subpackage. The
gated and ungated DeltaNet inference kernels share ``delta_decode``.
"""

from tileops._kernels.linear_attention.call_spec import (
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
from tileops._kernels.linear_attention.deltanet import (
    DeltaNetChunkBwdKernel,
    DeltaNetChunkFwdKernel,
    DeltaNetChunkPrefillFwdKernel,
    DeltaNetDenseDecodeFwdKernel,
    DeltaNetDensePrefillFwdKernel,
)
from tileops._kernels.linear_attention.deltanet.recurrent import (
    DeltaNetDecodeFP32Kernel,
    DeltaNetDecodeKernel,
    DeltaNetDecodeWarpKernel,
)
from tileops._kernels.linear_attention.gdn import (
    GDNChunkPrefillFwdKernel,
    GDNDenseDecodeFwdKernel,
    GDNDensePrefillFwdKernel,
    GDNSplitPrefillFwdKernel,
)
from tileops._kernels.linear_attention.gla import (
    GLAChunkBwdKernel,
    GLAChunkFwdKernel,
    GLADensePrefillSubchunkKernel,
)
from tileops._kernels.linear_attention.gla.recurrent import GLADecodeFP32Kernel, GLADecodeKernel
from tileops._kernels.linear_attention.kda import (
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
    "DeltaNetChunkPrefillFwdKernel",
    "DeltaNetDecodeCall",
    "DeltaNetDecodeFP32Kernel",
    "DeltaNetDecodeFwdInterface",
    "DeltaNetDecodeKernel",
    "DeltaNetDecodeWarpKernel",
    "DeltaNetDenseDecodeFwdKernel",
    "DeltaNetDensePrefillFwdKernel",
    "DeltaNetFwdInterface",
    "GDNCall",
    "GDNChunkPrefillFwdKernel",
    "GDNDenseDecodeFwdKernel",
    "GDNDensePrefillFwdKernel",
    "GDNFwdInterface",
    "GDNSplitPrefillFwdKernel",
    "GLAChunkBwdInterface",
    "GLAChunkBwdKernel",
    "GLAChunkCall",
    "GLAChunkFwdInterface",
    "GLAChunkFwdKernel",
    "GLADecodeCall",
    "GLADecodeFP32Kernel",
    "GLADecodeFwdInterface",
    "GLADecodeKernel",
    "GLADensePrefillSubchunkKernel",
    "KDACall",
    "KDAChunkPrefillFwdKernel",
    "KDAFusedPrefillFwdKernel",
    "KDAFwdInterface",
    "KDARecurrentDecodeFwdKernel",
]
