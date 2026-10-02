from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # type checkers and IDEs do not run __getattr__
    from tileops.ops.attention import (
        DeepSeekSparseAttentionDecodeWithKVCacheFwdOp,
        GroupedQueryAttentionBwdOp,
        GroupedQueryAttentionDenseFwdOp,
        GroupedQueryAttentionPagedFwdOp,
        GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp,
        GroupedQueryAttentionVarlenFwdOp,
        MultiHeadAttentionDecodePagedWithKVCacheFwdOp,
        MultiHeadLatentAttentionDecodeWithKVCacheFwdOp,
        MultiHeadLatentAttentionVarlenFwdOp,
        NSACmpVarlenFwdOp,
        NSATopkVarlenFwdOp,
        NSAVarlenFwdOp,
    )
    from tileops.ops.convolution import Conv1dFwdOp, Conv2dFwdOp, Conv3dFwdOp
    from tileops.ops.dropout import DropoutFwdOp
    from tileops.ops.elementwise import BinaryOp, FusedGatedOp, UnaryOp
    from tileops.ops.fft import FFTC2CFwdOp
    from tileops.ops.fp8_lightning_indexer import FP8LightningIndexerFwdOp
    from tileops.ops.fp8_quant import FP8QuantFwdOp
    from tileops.ops.gemm import (
        BmmFp8FwdOp,
        BmmFwdOp,
        GemmFp8FwdOp,
        GemmFwdOp,
        GemmW4A16FwdOp,
        GroupedGemmFwdOp,
    )
    from tileops.ops.linear_attention import (
        DeltaNetAutogradFwdOp,
        DeltaNetBwdOp,
        DeltaNetDecodeFwdOp,
        DeltaNetFwdOp,
        DeltaNetInferenceFwdOp,
        GatedDeltaNetFwdOp,
        GLABwdOp,
        GLADecodeFwdOp,
        GLAFwdOp,
        GLAInferenceFwdOp,
        KimiDeltaAttentionFwdOp,
    )
    from tileops.ops.mamba import (
        DaCumsumFwdOp,
        Mamba2FwdOp,
        SSDChunkScanFwdOp,
        SSDChunkStateFwdOp,
        SSDDecodeFwdOp,
        SSDStatePassingFwdOp,
    )
    from tileops.ops.moe import (
        MoeExpertMLPFwdOp,
        MoeGroupedGemmFwdOp,
        MoePermuteAlignFwdOp,
        MoePostPermuteFwdOp,
        MoePrePermuteFwdOp,
    )
    from tileops.ops.norm import (
        AdaLayerNormFwdOp,
        AdaLayerNormZeroFwdOp,
        BatchNormBwdOp,
        BatchNormFwdOp,
        FusedAddLayerNormFwdOp,
        FusedAddRMSNormFwdOp,
        GroupNormFwdOp,
        InstanceNormFwdOp,
        LayerNormFwdOp,
        RMSNormFwdOp,
    )
    from tileops.ops.op_base import Op
    from tileops.ops.pool import (
        AdaptiveAvgPool2dFwdOp,
        AdaptiveMaxPool2dFwdOp,
        AdaptiveMaxPool2dIndicesFwdOp,
        AvgPool1dFwdOp,
        AvgPool2dFwdOp,
        AvgPool3dFwdOp,
        MaxPool1dFwdOp,
        MaxPool1dIndicesFwdOp,
        MaxPool2dFwdOp,
        MaxPool2dIndicesFwdOp,
        MaxPool3dFwdOp,
        MaxPool3dIndicesFwdOp,
        MeanPoolingFwdOp,
    )
    from tileops.ops.quantization import (
        FP8QuantPerBlockFwdOp,
        INT4QuantPerGroupFwdOp,
        INT8DequantPerBlockFwdOp,
        INT8DequantPerChannelFwdOp,
        INT8DequantPerTensorFwdOp,
        INT8QuantPerBlockFwdOp,
        INT8QuantPerChannelFwdOp,
        INT8QuantPerTensorFwdOp,
        SmoothQuantFwdOp,
    )
    from tileops.ops.reduction import (
        AllFwdOp,
        AmaxFwdOp,
        AminFwdOp,
        AnyFwdOp,
        ArgmaxFwdOp,
        ArgminFwdOp,
        CountNonzeroFwdOp,
        CumprodFwdOp,
        CumsumFwdOp,
        InfNormFwdOp,
        L1NormFwdOp,
        L2NormFwdOp,
        LogSoftmaxFwdOp,
        LogSumExpFwdOp,
        MeanFwdOp,
        ProdFwdOp,
        SoftmaxFwdOp,
        StdFwdOp,
        SumFwdOp,
        VarFwdOp,
        VarMeanFwdOp,
    )
    from tileops.ops.rope import (
        RopeLlama31FwdOp,
        RopeLongRopeFwdOp,
        RopeNeoxFwdOp,
        RopeNeoxPositionIdsFwdOp,
        RopeNonNeoxFwdOp,
        RopeYarnFwdOp,
    )
    from tileops.ops.sampling import (
        ChainSpeculativeSamplingFwdOp,
        MinPMaskFwdOp,
        SamplingFromProbsFwdOp,
        TopKMaskFwdOp,
        TopKTopPMaskFwdOp,
        TopPMaskFwdOp,
    )
    from tileops.ops.sequence_modeling import MHCPostFwdOp, MHCPreFwdOp
    from tileops.ops.topk_selector import TopkSelectorFwdOp

# Public name -> the submodule that defines it; `__all__` follows this order.
# Grouped by op family, simple to composite; within a group, base case before variants.
_LAZY = {
    # Base class
    "Op": ".op_base",
    # Elementwise
    "UnaryOp": ".elementwise",
    "BinaryOp": ".elementwise",
    "FusedGatedOp": ".elementwise",
    "DropoutFwdOp": ".dropout",
    # Reduction
    "SumFwdOp": ".reduction",
    "MeanFwdOp": ".reduction",
    "ProdFwdOp": ".reduction",
    "AmaxFwdOp": ".reduction",
    "AminFwdOp": ".reduction",
    "VarFwdOp": ".reduction",
    "VarMeanFwdOp": ".reduction",
    "StdFwdOp": ".reduction",
    "ArgmaxFwdOp": ".reduction",
    "ArgminFwdOp": ".reduction",
    "SoftmaxFwdOp": ".reduction",
    "LogSoftmaxFwdOp": ".reduction",
    "LogSumExpFwdOp": ".reduction",
    "L1NormFwdOp": ".reduction",
    "L2NormFwdOp": ".reduction",
    "InfNormFwdOp": ".reduction",
    "CumsumFwdOp": ".reduction",
    "CumprodFwdOp": ".reduction",
    "AllFwdOp": ".reduction",
    "AnyFwdOp": ".reduction",
    "CountNonzeroFwdOp": ".reduction",
    # Normalization
    "LayerNormFwdOp": ".norm",
    "FusedAddLayerNormFwdOp": ".norm",
    "RMSNormFwdOp": ".norm",
    "FusedAddRMSNormFwdOp": ".norm",
    "AdaLayerNormFwdOp": ".norm",
    "AdaLayerNormZeroFwdOp": ".norm",
    "BatchNormFwdOp": ".norm",
    "BatchNormBwdOp": ".norm",
    "GroupNormFwdOp": ".norm",
    "InstanceNormFwdOp": ".norm",
    # Quantization
    "FP8QuantFwdOp": ".fp8_quant",
    "INT8DequantPerTensorFwdOp": ".quantization",
    "INT8DequantPerChannelFwdOp": ".quantization",
    "INT8DequantPerBlockFwdOp": ".quantization",
    "INT8QuantPerTensorFwdOp": ".quantization",
    "INT8QuantPerChannelFwdOp": ".quantization",
    "INT8QuantPerBlockFwdOp": ".quantization",
    "FP8QuantPerBlockFwdOp": ".quantization",
    "INT4QuantPerGroupFwdOp": ".quantization",
    "SmoothQuantFwdOp": ".quantization",
    # GEMM
    "GemmFwdOp": ".gemm",
    "GemmFp8FwdOp": ".gemm",
    "GemmW4A16FwdOp": ".gemm",
    "BmmFwdOp": ".gemm",
    "BmmFp8FwdOp": ".gemm",
    "GroupedGemmFwdOp": ".gemm",
    # Pooling
    "AvgPool1dFwdOp": ".pool",
    "AvgPool2dFwdOp": ".pool",
    "AvgPool3dFwdOp": ".pool",
    "MaxPool1dFwdOp": ".pool",
    "MaxPool1dIndicesFwdOp": ".pool",
    "MaxPool2dFwdOp": ".pool",
    "MaxPool2dIndicesFwdOp": ".pool",
    "MaxPool3dFwdOp": ".pool",
    "MaxPool3dIndicesFwdOp": ".pool",
    "AdaptiveAvgPool2dFwdOp": ".pool",
    "AdaptiveMaxPool2dFwdOp": ".pool",
    "AdaptiveMaxPool2dIndicesFwdOp": ".pool",
    "MeanPoolingFwdOp": ".pool",
    # Convolution
    "Conv1dFwdOp": ".convolution",
    "Conv2dFwdOp": ".convolution",
    "Conv3dFwdOp": ".convolution",
    # FFT
    "FFTC2CFwdOp": ".fft",
    # Mixture of experts
    "MoePrePermuteFwdOp": ".moe",
    "MoePermuteAlignFwdOp": ".moe",
    "MoeGroupedGemmFwdOp": ".moe",
    "MoeExpertMLPFwdOp": ".moe",
    "MoePostPermuteFwdOp": ".moe",
    # Sampling
    "TopKMaskFwdOp": ".sampling",
    "MinPMaskFwdOp": ".sampling",
    "TopPMaskFwdOp": ".sampling",
    "TopKTopPMaskFwdOp": ".sampling",
    "SamplingFromProbsFwdOp": ".sampling",
    "ChainSpeculativeSamplingFwdOp": ".sampling",
    # Rotary position embedding
    "RopeNeoxFwdOp": ".rope",
    "RopeNeoxPositionIdsFwdOp": ".rope",
    "RopeNonNeoxFwdOp": ".rope",
    "RopeLlama31FwdOp": ".rope",
    "RopeYarnFwdOp": ".rope",
    "RopeLongRopeFwdOp": ".rope",
    # Attention
    "MultiHeadAttentionDecodePagedWithKVCacheFwdOp": ".attention",
    "GroupedQueryAttentionBwdOp": ".attention",
    "GroupedQueryAttentionDenseFwdOp": ".attention",
    "GroupedQueryAttentionPagedFwdOp": ".attention",
    "GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp": ".attention",
    "GroupedQueryAttentionVarlenFwdOp": ".attention",
    "MultiHeadLatentAttentionDecodeWithKVCacheFwdOp": ".attention",
    "MultiHeadLatentAttentionVarlenFwdOp": ".attention",
    "NSACmpVarlenFwdOp": ".attention",
    "NSATopkVarlenFwdOp": ".attention",
    "NSAVarlenFwdOp": ".attention",
    "DeepSeekSparseAttentionDecodeWithKVCacheFwdOp": ".attention",
    "FP8LightningIndexerFwdOp": ".fp8_lightning_indexer",
    "TopkSelectorFwdOp": ".topk_selector",
    # Linear attention
    "DeltaNetAutogradFwdOp": ".linear_attention",
    "DeltaNetFwdOp": ".linear_attention",
    "DeltaNetBwdOp": ".linear_attention",
    "DeltaNetInferenceFwdOp": ".linear_attention",
    "DeltaNetDecodeFwdOp": ".linear_attention",
    "GatedDeltaNetFwdOp": ".linear_attention",
    "GLAFwdOp": ".linear_attention",
    "GLAInferenceFwdOp": ".linear_attention",
    "GLABwdOp": ".linear_attention",
    "GLADecodeFwdOp": ".linear_attention",
    "KimiDeltaAttentionFwdOp": ".linear_attention",
    # Mamba
    "Mamba2FwdOp": ".mamba",
    "DaCumsumFwdOp": ".mamba",
    "SSDChunkStateFwdOp": ".mamba",
    "SSDStatePassingFwdOp": ".mamba",
    "SSDChunkScanFwdOp": ".mamba",
    "SSDDecodeFwdOp": ".mamba",
    # mHC (Manifold-Constrained Hyper-Connections)
    "MHCPreFwdOp": ".sequence_modeling",
    "MHCPostFwdOp": ".sequence_modeling",
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
