"""Linear-attention kernels: DeltaNet, Gated DeltaNet and Gated Linear Attention (GLA).

The DeltaNet and GLA chunked kernels share the V-tile width rule (``v_tile``); the
DeltaNet forward also tunes through ``autotune``. The chunkwise kernels live in the
per-variant subpackages; the DeltaNet and GLA single-token decode kernels are the
``*_recurrence`` modules.
"""

from .deltanet import DeltaNetBwdKernel, DeltaNetDensePrefillFwdKernel, DeltaNetFwdKernel
from .deltanet_recurrence import (
    DeltaNetDecodeFP32Kernel,
    DeltaNetDecodeKernel,
    DeltaNetDecodeRawCudaFlaStyleKernel,
)
from .gated_deltanet import (
    GatedDeltaNetDenseDecodeFwdKernel,
    GatedDeltaNetDensePrefillFwdKernel,
)
from .gla import GLABwdKernel, GLADensePrefillFwdKernel, GLADensePrefillSubchunkKernel, GLAFwdKernel
from .gla_recurrence import GLADecodeFP32Kernel, GLADecodeKernel

__all__ = [
    "DeltaNetBwdKernel",
    "DeltaNetDecodeFP32Kernel",
    "DeltaNetDecodeKernel",
    "DeltaNetDecodeRawCudaFlaStyleKernel",
    "DeltaNetDensePrefillFwdKernel",
    "DeltaNetFwdKernel",
    "GLABwdKernel",
    "GLADecodeFP32Kernel",
    "GLADecodeKernel",
    "GLADensePrefillFwdKernel",
    "GLADensePrefillSubchunkKernel",
    "GLAFwdKernel",
    "GatedDeltaNetDenseDecodeFwdKernel",
    "GatedDeltaNetDensePrefillFwdKernel",
]
