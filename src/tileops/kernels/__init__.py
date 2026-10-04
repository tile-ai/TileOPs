from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # type checkers and IDEs do not run __getattr__
    from tileops.kernels.attention import (
        DSADecodeBasicKernel,
        DSADecodeKernel,
        GQABwdPreprocessKernel,
        GQABwdWGMMAPipelinedKernel,
        GQADecodeKernel,
        GQADecodePagedKernel,
        GQADenseFP8Kernel,
        GQADenseSlidingWindowKernel,
        GQADenseWSKernel,
        GQAPrefillPagedWithFP8KVCacheFwdKernel,
        GQAPrefillPagedWithKVCacheFwdKernel,
        GQAPrefillPagedWithKVCacheRoPEFwdKernel,
        GQASlidingWindowVarlenFwdWGMMAPipelinedKernel,
        MLADecodeWSKernel,
        NSACompressedFwdVarlenKernel,
        NSAFwdVarlenKernel,
        NSATopKVarlenKernel,
    )
    from tileops.kernels.attention.fp8_lightning_indexer import FP8LightningIndexerKernel
    from tileops.kernels.attention.topk_select import TopKSelectKernel
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
    from tileops.kernels.elementwise import BinaryKernel, FusedGatedKernel, UnaryKernel
    from tileops.kernels.elementwise.dropout import DropoutKernel
    from tileops.kernels.fft import FFTC2CFourStepKernel, FFTC2COneCTAKernel
    from tileops.kernels.gemm import (
        BmmFP8Kernel,
        BmmFP8PersistentKernel,
        BmmFP8TransposeKernel,
        BmmFP8WSKernel,
        BmmKernel,
        BmmPersistentKernel,
        GemmCpAsyncKernel,
        GemmFP8BlockScaleKernel,
        GemmFP8TensorScaleKernel,
        GemmFP81D2DFwdKernel,
        GemmTMAKernel,
        GemvKernel,
    )
    from tileops.kernels.gemm.grouped import GroupedGemmKernel, GroupedGemmPersistentKernel
    from tileops.kernels.kernel_base import Kernel
    from tileops.kernels.linear_attention import (
        DeltaNetBwdKernel,
        DeltaNetDecodeFP32Kernel,
        DeltaNetDecodeKernel,
        DeltaNetDecodeRawCudaFlaStyleKernel,
        DeltaNetDenseDecodeFwdKernel,
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
    from tileops.kernels.moe import MoEPermuteAlignKernel
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
    from tileops.kernels.quantization.fp8_quant import FP8QuantKernel
    from tileops.kernels.rope import (
        RoPENeoxKernel,
        RoPENeoxPositionIdsKernel,
        RoPENonNeoxKernel,
    )
    from tileops.kernels.sequence_modeling.engram import (
        EngramDecodeKernel,
        EngramGateConvBwdKernel,
        EngramGateConvFwdKernel,
    )
    from tileops.kernels.sequence_modeling.mhc import MHCPostKernel, MHCPreKernel

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
    "BmmFP8Kernel": ".gemm",
    "BmmFP8PersistentKernel": ".gemm",
    "BmmFP8TransposeKernel": ".gemm",
    "BmmFP8WSKernel": ".gemm",
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
    "DeltaNetDenseDecodeFwdKernel": ".linear_attention",
    "DeltaNetDensePrefillFwdKernel": ".linear_attention",
    "DeltaNetFwdKernel": ".linear_attention",
    "DropoutKernel": ".elementwise.dropout",
    "EngramDecodeKernel": ".sequence_modeling.engram",
    "EngramGateConvBwdKernel": ".sequence_modeling.engram",
    "EngramGateConvFwdKernel": ".sequence_modeling.engram",
    "FFTC2CFourStepKernel": ".fft",
    "FFTC2COneCTAKernel": ".fft",
    "FP8LightningIndexerKernel": ".attention.fp8_lightning_indexer",
    "FP8QuantKernel": ".quantization.fp8_quant",
    "GQABwdPreprocessKernel": ".attention",
    "FusedGatedKernel": ".elementwise",
    "GLABwdKernel": ".linear_attention",
    "GLADecodeFP32Kernel": ".linear_attention",
    "GLADecodeKernel": ".linear_attention",
    "GLADensePrefillFwdKernel": ".linear_attention",
    "GLAFwdKernel": ".linear_attention",
    "GQABwdWGMMAPipelinedKernel": ".attention",
    "GQADecodeKernel": ".attention",
    "GQADecodePagedKernel": ".attention",
    "GQADenseFP8Kernel": ".attention",
    "GQADenseWSKernel": ".attention",
    "GQADenseSlidingWindowKernel": ".attention",
    "GQAPrefillPagedWithFP8KVCacheFwdKernel": ".attention",
    "GQAPrefillPagedWithKVCacheFwdKernel": ".attention",
    "GQAPrefillPagedWithKVCacheRoPEFwdKernel": ".attention",
    "GQASlidingWindowVarlenFwdWGMMAPipelinedKernel": ".attention",
    "GatedDeltaNetDenseDecodeFwdKernel": ".linear_attention",
    "GatedDeltaNetDensePrefillFwdKernel": ".linear_attention",
    "GemmCpAsyncKernel": ".gemm",
    "GemmFP81D2DFwdKernel": ".gemm",
    "GemmFP8BlockScaleKernel": ".gemm",
    "GemmFP8TensorScaleKernel": ".gemm",
    "GemmTMAKernel": ".gemm",
    "GemvKernel": ".gemm",
    "GroupConv1dKernel": ".convolution",
    "GroupConv2dKernel": ".convolution",
    "GroupConv3dKernel": ".convolution",
    "GroupNormKernel": ".norm",
    "GroupedGemmKernel": ".gemm.grouped",
    "GroupedGemmPersistentKernel": ".gemm.grouped",
    "Kernel": ".kernel_base",
    "LayerNormKernel": ".norm",
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
