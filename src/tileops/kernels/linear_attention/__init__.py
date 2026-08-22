"""Linear-attention kernels: DeltaNet, Gated DeltaNet and Gated Linear Attention (GLA).

The three share the chunked-recurrence formulation, so they share the V-tile width
rule (``v_tile``) and the autotune search space (``autotune``). The chunkwise
kernels live in the per-variant subpackages; the single-token decode kernels are the
``*_recurrence`` modules.
"""

from .deltanet import DeltaNetBwdKernel, DeltaNetFwdKernel
from .deltanet_recurrence import (
    DeltaNetDecodeFP32Kernel,
    DeltaNetDecodeKernel,
    DeltaNetDecodeRawCudaFlaStyleKernel,
)
from .gated_deltanet import (
    GatedDeltaNetBwdKernel,
    GatedDeltaNetFwdKernel,
    GatedDeltaNetFwdProductionKernel,
    GatedDeltaNetPrefillFwdKernel,
)
from .gated_deltanet_recurrence import (
    GatedDeltaNetDecodeFP32Kernel,
    GatedDeltaNetDecodeKernel,
    GatedDeltaNetDecodeRawCudaFlaStyleKernel,
)
from .gla import GLABwdKernel, GLAFwdKernel
from .gla_recurrence import GLADecodeFP32Kernel, GLADecodeKernel

__all__ = [
    "DeltaNetBwdKernel",
    "DeltaNetDecodeFP32Kernel",
    "DeltaNetDecodeKernel",
    "DeltaNetDecodeRawCudaFlaStyleKernel",
    "DeltaNetFwdKernel",
    "GLABwdKernel",
    "GLADecodeFP32Kernel",
    "GLADecodeKernel",
    "GLAFwdKernel",
    "GatedDeltaNetBwdKernel",
    "GatedDeltaNetDecodeFP32Kernel",
    "GatedDeltaNetDecodeKernel",
    "GatedDeltaNetDecodeRawCudaFlaStyleKernel",
    "GatedDeltaNetFwdKernel",
    "GatedDeltaNetFwdProductionKernel",
    "GatedDeltaNetPrefillFwdKernel",
]
