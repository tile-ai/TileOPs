from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # type checkers and IDEs do not run __getattr__
    from tileops._kernels.attention import (
        DSADecodeBasicKernel,
        DSADecodeKernel,
        GQABwdPreprocessKernel,
        GQABwdWGMMAPipelinedKernel,
        GQABwdWSKernel,
        GQADecodeKernel,
        GQADecodePagedKernel,
        GQADenseFP8Kernel,
        GQADensePersistentWSKernel,
        GQADenseSlidingWindowBandKernel,
        GQADenseSlidingWindowKernel,
        GQADenseWSKernel,
        GQASlidingWindowVarlenFwdWGMMAPipelinedKernel,
        MLADecodeWSKernel,
        NSACompressedFwdVarlenKernel,
        NSAFwdVarlenKernel,
        NSATopKVarlenKernel,
    )
    from tileops._kernels.attention.fp8_lightning_indexer import (
        FP8LightningIndexerKernel,
        FP8LightningIndexerWSKernel,
    )
    from tileops._kernels.attention.topk_select import TopKSelectKernel
    from tileops._kernels.convolution import (
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
    from tileops._kernels.elementwise import BinaryKernel, FusedGatedKernel, UnaryKernel
    from tileops._kernels.elementwise.dropout import DropoutKernel
    from tileops._kernels.fft import FFTC2CFourStepKernel, FFTC2COneCTAKernel
    from tileops._kernels.gemm import (
        BMMFP8Kernel,
        BMMFP8PersistentKernel,
        BMMFP8TransposeKernel,
        BMMFP8WSKernel,
        BMMKernel,
        BMMPersistentKernel,
        GEMMCpAsyncKernel,
        GEMMFP8Block128WaveFwdKernel,
        GEMMFP8BlockScaleKernel,
        GEMMFP8SwapABKernel,
        GEMMFP8TensorScaleKernel,
        GEMMFP81D2DFwdKernel,
        GEMMTMAKernel,
        GemvKernel,
    )
    from tileops._kernels.gemm.grouped import GroupedGEMMKernel, GroupedGEMMPersistentKernel
    from tileops._kernels.kernel_base import Kernel
    from tileops._kernels.linear_attention import (
        DeltaNetChunkBwdKernel,
        DeltaNetChunkFwdKernel,
        DeltaNetDecodeFP32Kernel,
        DeltaNetDecodeKernel,
        DeltaNetDecodeRawCudaFlaStyleKernel,
        DeltaNetDenseDecodeFwdKernel,
        DeltaNetDensePrefillFwdKernel,
        GDNDenseDecodeFwdKernel,
        GDNDensePrefillFwdKernel,
        GLAChunkBwdKernel,
        GLAChunkFwdKernel,
        GLADecodeFP32Kernel,
        GLADecodeKernel,
    )
    from tileops._kernels.moe import MoEPermuteAlignKernel
    from tileops._kernels.norm import (
        BatchNormBwdKernel,
        BatchNormFwdInferKernel,
        BatchNormFwdTrainKernel,
        GroupNormKernel,
        LayerNormKernel,
        LayerNormWarpRowKernel,
        RMSNormKernel,
    )
    from tileops._kernels.pool import (
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
    from tileops._kernels.quantization.fp8_quant import FP8QuantKernel
    from tileops._kernels.rope import (
        RoPENeoxKernel,
        RoPENeoxPositionIdsKernel,
        RoPENonNeoxKernel,
    )
    from tileops._kernels.sequence_modeling.engram import (
        EngramDecodeKernel,
        EngramGateConvBwdKernel,
        EngramGateConvFwdKernel,
    )
    from tileops._kernels.sequence_modeling.mhc import MHCPostKernel, MHCPreKernel
    from tileops._kernels.transform import (
        HadamardBaseExchangeKernel,
        HadamardExchangeKernel,
        HadamardMultiPassKernel,
        HadamardSharedRowsKernel,
        HadamardWarpRowsKernel,
    )

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
    "BMMFP8Kernel": ".gemm",
    "BMMFP8PersistentKernel": ".gemm",
    "BMMFP8TransposeKernel": ".gemm",
    "BMMFP8WSKernel": ".gemm",
    "BMMKernel": ".gemm",
    "BMMPersistentKernel": ".gemm",
    "Conv1dKernel": ".convolution",
    "Conv1dPointwiseKernel": ".convolution",
    "Conv2d1x1Kernel": ".convolution",
    "Conv2dKernel": ".convolution",
    "Conv3dKernel": ".convolution",
    "DepthwiseConv1dKernel": ".convolution",
    "DepthwiseConv2dKernel": ".convolution",
    "DeltaNetChunkBwdKernel": ".linear_attention",
    "DeltaNetDecodeFP32Kernel": ".linear_attention",
    "DeltaNetDecodeKernel": ".linear_attention",
    "DeltaNetDecodeRawCudaFlaStyleKernel": ".linear_attention",
    "DeltaNetDenseDecodeFwdKernel": ".linear_attention",
    "DeltaNetDensePrefillFwdKernel": ".linear_attention",
    "DeltaNetChunkFwdKernel": ".linear_attention",
    "DropoutKernel": ".elementwise.dropout",
    "EngramDecodeKernel": ".sequence_modeling.engram",
    "EngramGateConvBwdKernel": ".sequence_modeling.engram",
    "EngramGateConvFwdKernel": ".sequence_modeling.engram",
    "FFTC2CFourStepKernel": ".fft",
    "FFTC2COneCTAKernel": ".fft",
    "FP8LightningIndexerKernel": ".attention.fp8_lightning_indexer",
    "FP8LightningIndexerWSKernel": ".attention.fp8_lightning_indexer",
    "FP8QuantKernel": ".quantization.fp8_quant",
    "GQABwdPreprocessKernel": ".attention",
    "FusedGatedKernel": ".elementwise",
    "GLAChunkBwdKernel": ".linear_attention",
    "GLADecodeFP32Kernel": ".linear_attention",
    "GLADecodeKernel": ".linear_attention",
    "GLAChunkFwdKernel": ".linear_attention",
    "GQABwdWGMMAPipelinedKernel": ".attention",
    "GQABwdWSKernel": ".attention",
    "GQADecodeKernel": ".attention",
    "GQADecodePagedKernel": ".attention",
    "GQADenseFP8Kernel": ".attention",
    "GQADensePersistentWSKernel": ".attention",
    "GQADenseWSKernel": ".attention",
    "GQADenseSlidingWindowBandKernel": ".attention",
    "GQADenseSlidingWindowKernel": ".attention",
    "GQASlidingWindowVarlenFwdWGMMAPipelinedKernel": ".attention",
    "GDNDenseDecodeFwdKernel": ".linear_attention",
    "GDNDensePrefillFwdKernel": ".linear_attention",
    "GEMMCpAsyncKernel": ".gemm",
    "GEMMFP81D2DFwdKernel": ".gemm",
    "GEMMFP8Block128WaveFwdKernel": ".gemm",
    "GEMMFP8BlockScaleKernel": ".gemm",
    "GEMMFP8SwapABKernel": ".gemm",
    "GEMMFP8TensorScaleKernel": ".gemm",
    "GEMMTMAKernel": ".gemm",
    "HadamardBaseExchangeKernel": ".transform",
    "HadamardExchangeKernel": ".transform",
    "HadamardMultiPassKernel": ".transform",
    "HadamardSharedRowsKernel": ".transform",
    "HadamardWarpRowsKernel": ".transform",
    "GemvKernel": ".gemm",
    "GroupConv1dKernel": ".convolution",
    "GroupConv2dKernel": ".convolution",
    "GroupConv3dKernel": ".convolution",
    "GroupNormKernel": ".norm",
    "GroupedGEMMKernel": ".gemm.grouped",
    "GroupedGEMMPersistentKernel": ".gemm.grouped",
    "Kernel": ".kernel_base",
    "LayerNormKernel": ".norm",
    "LayerNormWarpRowKernel": ".norm",
    "MHCPostKernel": ".sequence_modeling.mhc",
    "MHCPreKernel": ".sequence_modeling.mhc",
    "MLADecodeWSKernel": ".attention",
    "MaxPool1dKernel": ".pool",
    "MaxPool1dWithIndicesKernel": ".pool",
    "MaxPool2dKernel": ".pool",
    "MaxPool2dWithIndicesKernel": ".pool",
    "MaxPool3dKernel": ".pool",
    "MaxPool3dWithIndicesKernel": ".pool",
    "MeanPoolingFwdKernel": ".pool",
    "MoEPermuteAlignKernel": ".moe",
    "NSACompressedFwdVarlenKernel": ".attention",
    "NSAFwdVarlenKernel": ".attention",
    "NSATopKVarlenKernel": ".attention",
    "RMSNormKernel": ".norm",
    "RoPENeoxKernel": ".rope",
    "RoPENeoxPositionIdsKernel": ".rope",
    "RoPENonNeoxKernel": ".rope",
    "DSADecodeBasicKernel": ".attention",
    "DSADecodeKernel": ".attention",
    "TopKSelectKernel": ".attention.topk_select",
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
