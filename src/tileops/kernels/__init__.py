from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # type checkers and IDEs do not run __getattr__
    from tileops.kernels.attention import (
        FlashAttnBwdPreprocessKernel,
        GQABwdWgmmaPipelinedKernel,
        GQADecodeKernel,
        GQADecodePagedKernel,
        GQADenseFP8Kernel,
        GQADenseSlidingWindowKernel,
        GQADenseWsKernel,
        GQAPrefillPagedWithFP8KVCacheFwdKernel,
        GQAPrefillPagedWithKVCacheFwdKernel,
        GQAPrefillPagedWithKVCacheRopeFwdKernel,
        GQASlidingWindowVarlenFwdWgmmaPipelinedKernel,
        MLADecodeWsKernel,
        NSACmpFwdVarlenKernel,
        NSAFwdVarlenKernel,
        NSATopkVarlenKernel,
        SparseMlaBasicKernel,
        SparseMlaKernel,
    )
    from tileops.kernels.convolution import (
        Conv1dKernel,
        Conv1dPointwiseKernel,
        Conv2d1x1Kernel,
        Conv2dKernel,
        Conv3dKernel,
        DepthwiseConv1dKernel,
        DepthwiseConv2dKernel,
        GroupConv1dKernel,
        GroupConv2dKernel,
        GroupConv3dKernel,
    )
    from tileops.kernels.dropout import DropoutKernel
    from tileops.kernels.elementwise import BinaryKernel, FusedGatedKernel, UnaryKernel
    from tileops.kernels.engram import (
        EngramDecodeKernel,
        EngramGateConvBwdKernel,
        EngramGateConvFwdKernel,
    )
    from tileops.kernels.fft import FFTC2CDecomposedKernel, FFTC2COneCTAKernel
    from tileops.kernels.fp8_lightning_indexer import FP8LightningIndexerKernel
    from tileops.kernels.fp8_quant import FP8QuantKernel
    from tileops.kernels.gemm import (
        BmmFp8Kernel,
        BmmFp8PersistentKernel,
        BmmFp8TransposeKernel,
        BmmFp8WsKernel,
        BmmKernel,
        BmmPersistentKernel,
        GemmCpAsyncKernel,
        GemmFp8BlockScaleKernel,
        GemmFp8TensorScaleKernel,
        GemmFp81D2DFwdKernel,
        GemmTmaKernel,
        GemvKernel,
    )
    from tileops.kernels.grouped_gemm import GroupedGemmKernel, GroupedGemmPersistentKernel
    from tileops.kernels.kernel_base import Kernel
    from tileops.kernels.linear_attention import (
        DeltaNetBwdKernel,
        DeltaNetDecodeFP32Kernel,
        DeltaNetDecodeKernel,
        DeltaNetDecodeRawCudaFlaStyleKernel,
        DeltaNetDensePrefillFwdKernel,
        DeltaNetFwdKernel,
        GatedDeltaNetDenseDecodeFwdKernel,
        GatedDeltaNetDensePrefillFwdKernel,
        GLABwdKernel,
        GLADecodeFP32Kernel,
        GLADecodeKernel,
        GLADensePrefillFwdKernel,
        GLAFwdKernel,
    )
    from tileops.kernels.mhc import MHCPostKernel, MHCPreKernel
    from tileops.kernels.moe import MoePermuteAlignKernel
    from tileops.kernels.norm import (
        BatchNormBwdKernel,
        BatchNormFwdInferKernel,
        BatchNormFwdTrainKernel,
        GroupNormKernel,
        LayerNormKernel,
        RMSNormKernel,
    )
    from tileops.kernels.pool import (
        AdaptiveAvgPool2dKernel,
        AdaptiveMaxPool2dKernel,
        AdaptiveMaxPool2dWithIndicesKernel,
        AvgPool1dKernel,
        AvgPool2dKernel,
        AvgPool3dKernel,
        MaxPool1dKernel,
        MaxPool1dWithIndicesKernel,
        MaxPool2dKernel,
        MaxPool2dWithIndicesKernel,
        MaxPool3dKernel,
        MaxPool3dWithIndicesKernel,
        MeanPoolingFwdKernel,
    )
    from tileops.kernels.rope import (
        RopeNeoxKernel,
        RopeNeoxPositionIdsKernel,
        RopeNonNeoxKernel,
    )
    from tileops.kernels.topk_selector import TopkSelectorKernel

# Public name -> the submodule that defines it; `__all__` follows this order.
_LAZY = {
    "AdaptiveAvgPool2dKernel": ".pool",
    "AdaptiveMaxPool2dKernel": ".pool",
    "AdaptiveMaxPool2dWithIndicesKernel": ".pool",
    "AvgPool1dKernel": ".pool",
    "AvgPool2dKernel": ".pool",
    "AvgPool3dKernel": ".pool",
    "BatchNormBwdKernel": ".norm",
    "BatchNormFwdInferKernel": ".norm",
    "BatchNormFwdTrainKernel": ".norm",
    "BinaryKernel": ".elementwise",
    "BmmFp8Kernel": ".gemm",
    "BmmFp8PersistentKernel": ".gemm",
    "BmmFp8TransposeKernel": ".gemm",
    "BmmFp8WsKernel": ".gemm",
    "BmmKernel": ".gemm",
    "BmmPersistentKernel": ".gemm",
    "Conv1dKernel": ".convolution",
    "Conv1dPointwiseKernel": ".convolution",
    "Conv2d1x1Kernel": ".convolution",
    "Conv2dKernel": ".convolution",
    "Conv3dKernel": ".convolution",
    "DepthwiseConv1dKernel": ".convolution",
    "DepthwiseConv2dKernel": ".convolution",
    "DeltaNetBwdKernel": ".linear_attention",
    "DeltaNetDecodeFP32Kernel": ".linear_attention",
    "DeltaNetDecodeKernel": ".linear_attention",
    "DeltaNetDecodeRawCudaFlaStyleKernel": ".linear_attention",
    "DeltaNetDensePrefillFwdKernel": ".linear_attention",
    "DeltaNetFwdKernel": ".linear_attention",
    "DropoutKernel": ".dropout",
    "EngramDecodeKernel": ".engram",
    "EngramGateConvBwdKernel": ".engram",
    "EngramGateConvFwdKernel": ".engram",
    "FFTC2CDecomposedKernel": ".fft",
    "FFTC2COneCTAKernel": ".fft",
    "FP8LightningIndexerKernel": ".fp8_lightning_indexer",
    "FP8QuantKernel": ".fp8_quant",
    "FlashAttnBwdPreprocessKernel": ".attention",
    "FusedGatedKernel": ".elementwise",
    "GLABwdKernel": ".linear_attention",
    "GLADecodeFP32Kernel": ".linear_attention",
    "GLADecodeKernel": ".linear_attention",
    "GLADensePrefillFwdKernel": ".linear_attention",
    "GLAFwdKernel": ".linear_attention",
    "GQABwdWgmmaPipelinedKernel": ".attention",
    "GQADecodeKernel": ".attention",
    "GQADecodePagedKernel": ".attention",
    "GQADenseFP8Kernel": ".attention",
    "GQADenseWsKernel": ".attention",
    "GQADenseSlidingWindowKernel": ".attention",
    "GQAPrefillPagedWithFP8KVCacheFwdKernel": ".attention",
    "GQAPrefillPagedWithKVCacheFwdKernel": ".attention",
    "GQAPrefillPagedWithKVCacheRopeFwdKernel": ".attention",
    "GQASlidingWindowVarlenFwdWgmmaPipelinedKernel": ".attention",
    "GatedDeltaNetDenseDecodeFwdKernel": ".linear_attention",
    "GatedDeltaNetDensePrefillFwdKernel": ".linear_attention",
    "GemmCpAsyncKernel": ".gemm",
    "GemmFp81D2DFwdKernel": ".gemm",
    "GemmFp8BlockScaleKernel": ".gemm",
    "GemmFp8TensorScaleKernel": ".gemm",
    "GemmTmaKernel": ".gemm",
    "GemvKernel": ".gemm",
    "GroupConv1dKernel": ".convolution",
    "GroupConv2dKernel": ".convolution",
    "GroupConv3dKernel": ".convolution",
    "GroupNormKernel": ".norm",
    "GroupedGemmKernel": ".grouped_gemm",
    "GroupedGemmPersistentKernel": ".grouped_gemm",
    "Kernel": ".kernel_base",
    "LayerNormKernel": ".norm",
    "MHCPostKernel": ".mhc",
    "MHCPreKernel": ".mhc",
    "MLADecodeWsKernel": ".attention",
    "MaxPool1dKernel": ".pool",
    "MaxPool1dWithIndicesKernel": ".pool",
    "MaxPool2dKernel": ".pool",
    "MaxPool2dWithIndicesKernel": ".pool",
    "MaxPool3dKernel": ".pool",
    "MaxPool3dWithIndicesKernel": ".pool",
    "MeanPoolingFwdKernel": ".pool",
    "MoePermuteAlignKernel": ".moe",
    "NSACmpFwdVarlenKernel": ".attention",
    "NSAFwdVarlenKernel": ".attention",
    "NSATopkVarlenKernel": ".attention",
    "RMSNormKernel": ".norm",
    "RopeNeoxKernel": ".rope",
    "RopeNeoxPositionIdsKernel": ".rope",
    "RopeNonNeoxKernel": ".rope",
    "SparseMlaBasicKernel": ".attention",
    "SparseMlaKernel": ".attention",
    "TopkSelectorKernel": ".topk_selector",
    "UnaryKernel": ".elementwise",
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> Any:
    import importlib

    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __package__), name)
    globals()[name] = value  # later reads hit the module dict, not this function
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})
