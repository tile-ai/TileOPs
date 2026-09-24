from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # type checkers and IDEs do not run __getattr__
    from .attention import (
        FlashAttnBwdPreprocessKernel,
        GQABwdWgmmaPipelinedKernel,
        GQADecodeKernel,
        GQADecodePagedKernel,
        GQADenseFP8Kernel,
        GQADenseSlidingWindowKernel,
        GQADenseWsKernel,
        GQAFwdWgmmaPipelinedKernel,
        GQAPrefillFwdKernel,
        GQAPrefillPagedWithFP8KVCacheFwdKernel,
        GQAPrefillPagedWithKVCacheFwdKernel,
        GQAPrefillPagedWithKVCacheRopeAppendKernel,
        GQAPrefillPagedWithKVCacheRopeFwdKernel,
        GQASlidingWindowVarlenFwdWgmmaPipelinedKernel,
        MHADecodeKernel,
        MHADecodePagedKernel,
        MLADecodeWsKernel,
        NSACmpFwdVarlenKernel,
        NSAFwdVarlenKernel,
        NSATopkVarlenKernel,
        SparseMlaBasicKernel,
        SparseMlaKernel,
    )
    from .convolution import (
        Conv1dKernel,
        Conv1dPointwiseKernel,
        Conv2d1x1Kernel,
        Conv2dKernel,
        Conv3dKernel,
        GroupConv1dKernel,
        GroupConv2dKernel,
        GroupConv3dKernel,
    )
    from .dropout import DropoutKernel
    from .elementwise import (
        BinaryKernel,
        FusedGatedKernel,
        UnaryKernel,
    )
    from .engram import (
        EngramDecodeKernel,
        EngramGateConvBwdKernel,
        EngramGateConvFwdKernel,
    )
    from .fft_c2c import FFTC2CDecomposedKernel, FFTC2COneCTAKernel
    from .fp8_lightning_indexer import FP8LightningIndexerKernel
    from .fp8_quant import FP8QuantKernel
    from .gemm import (
        BmmFp8Kernel,
        BmmFp8TransposeKernel,
        BmmKernel,
        BmmPersistentKernel,
        GemmCpAsyncKernel,
        GemmFp8BlockScaleKernel,
        GemmFp8TensorScaleKernel,
        GemmTmaKernel,
        GemvKernel,
    )
    from .grouped_gemm import (
        GroupedGemmKernel,
        GroupedGemmPersistentKernel,
    )
    from .kernel_base import Kernel
    from .linear_attention import (
        DeltaNetBwdKernel,
        DeltaNetDecodeFP32Kernel,
        DeltaNetDecodeKernel,
        DeltaNetDecodeRawCudaFlaStyleKernel,
        DeltaNetDensePrefillFwdKernel,
        DeltaNetFwdKernel,
        GatedDeltaNetBwdKernel,
        GatedDeltaNetDecodeFP32Kernel,
        GatedDeltaNetDecodeKernel,
        GatedDeltaNetDecodeRawCudaFlaStyleKernel,
        GatedDeltaNetDenseDecodeFwdKernel,
        GatedDeltaNetDensePrefillFwdKernel,
        GatedDeltaNetFwdKernel,
        GatedDeltaNetFwdProductionKernel,
        GLABwdKernel,
        GLADecodeFP32Kernel,
        GLADecodeKernel,
        GLADensePrefillFwdKernel,
        GLAFwdKernel,
    )
    from .mhc import (
        MHCPostKernel,
        MHCPreKernel,
    )
    from .moe import MoePermuteAlignKernel
    from .norm import (
        BatchNormBwdKernel,
        BatchNormFwdInferKernel,
        BatchNormFwdTrainKernel,
        GroupNormKernel,
        LayerNormKernel,
        RMSNormKernel,
    )
    from .pool import (
        AdaptiveAvgPool2dKernel,
        AdaptiveMaxPool2dKernel,
        AdaptiveMaxPool2dWithIndicesKernel,
        AvgPool1dKernel,
        AvgPool1dSpatialKernel,
        AvgPool2dKernel,
        AvgPool2dSpatialKernel,
        AvgPool3dKernel,
        AvgPool3dSpatialKernel,
        MaxPool1dKernel,
        MaxPool1dWithIndicesKernel,
        MaxPool2dKernel,
        MaxPool2dWithIndicesKernel,
        MaxPool3dKernel,
        MaxPool3dWithIndicesKernel,
        MeanPoolingFwdKernel,
    )
    from .rope import (
        RopeLlama31Kernel,
        RopeLongRopeKernel,
        RopeNeoxKernel,
        RopeNeoxPositionIdsKernel,
        RopeNonNeoxKernel,
        RopeYarnKernel,
    )
    from .topk_selector import TopkSelectorKernel

# Public name -> the submodule that defines it; `__all__` follows this order.
_LAZY = {
    "AdaptiveAvgPool2dKernel": ".pool",
    "AdaptiveMaxPool2dKernel": ".pool",
    "AdaptiveMaxPool2dWithIndicesKernel": ".pool",
    "AvgPool1dKernel": ".pool",
    "AvgPool1dSpatialKernel": ".pool",
    "AvgPool2dKernel": ".pool",
    "AvgPool2dSpatialKernel": ".pool",
    "AvgPool3dKernel": ".pool",
    "AvgPool3dSpatialKernel": ".pool",
    "BatchNormBwdKernel": ".norm",
    "BatchNormFwdInferKernel": ".norm",
    "BatchNormFwdTrainKernel": ".norm",
    "BinaryKernel": ".elementwise",
    "BmmFp8Kernel": ".gemm",
    "BmmFp8TransposeKernel": ".gemm",
    "BmmKernel": ".gemm",
    "BmmPersistentKernel": ".gemm",
    "Conv1dKernel": ".convolution",
    "Conv1dPointwiseKernel": ".convolution",
    "Conv2d1x1Kernel": ".convolution",
    "Conv2dKernel": ".convolution",
    "Conv3dKernel": ".convolution",
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
    "FFTC2CDecomposedKernel": ".fft_c2c",
    "FFTC2COneCTAKernel": ".fft_c2c",
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
    "GQAFwdWgmmaPipelinedKernel": ".attention",
    "GQAPrefillFwdKernel": ".attention",
    "GQAPrefillPagedWithFP8KVCacheFwdKernel": ".attention",
    "GQAPrefillPagedWithKVCacheFwdKernel": ".attention",
    "GQAPrefillPagedWithKVCacheRopeAppendKernel": ".attention",
    "GQAPrefillPagedWithKVCacheRopeFwdKernel": ".attention",
    "GQASlidingWindowVarlenFwdWgmmaPipelinedKernel": ".attention",
    "GatedDeltaNetBwdKernel": ".linear_attention",
    "GatedDeltaNetDenseDecodeFwdKernel": ".linear_attention",
    "GatedDeltaNetDecodeFP32Kernel": ".linear_attention",
    "GatedDeltaNetDecodeKernel": ".linear_attention",
    "GatedDeltaNetDecodeRawCudaFlaStyleKernel": ".linear_attention",
    "GatedDeltaNetDensePrefillFwdKernel": ".linear_attention",
    "GatedDeltaNetFwdKernel": ".linear_attention",
    "GatedDeltaNetFwdProductionKernel": ".linear_attention",
    "GemmCpAsyncKernel": ".gemm",
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
    "MHADecodeKernel": ".attention",
    "MHADecodePagedKernel": ".attention",
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
    "RopeLlama31Kernel": ".rope",
    "RopeLongRopeKernel": ".rope",
    "RopeNeoxKernel": ".rope",
    "RopeNeoxPositionIdsKernel": ".rope",
    "RopeNonNeoxKernel": ".rope",
    "RopeYarnKernel": ".rope",
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
