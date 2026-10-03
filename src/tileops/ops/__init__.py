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
        NSACompressedVarlenFwdOp,
        NSATopKVarlenFwdOp,
        NSAVarlenFwdOp,
    )
    from tileops.ops.attention.fp8_lightning_indexer import FP8LightningIndexerFwdOp
    from tileops.ops.attention.topk_select import TopKSelectFwdOp
    from tileops.ops.convolution import Conv1dFwdOp, Conv2dFwdOp, Conv3dFwdOp
    from tileops.ops.elementwise import BinaryOp, FusedGatedOp, UnaryOp
    from tileops.ops.elementwise.dropout import DropoutFwdOp
    from tileops.ops.fft import FFTC2CFwdOp
    from tileops.ops.gemm import (
        BmmFP8FwdOp,
        BmmFwdOp,
        GemmFP8FwdOp,
        GemmFwdOp,
        GemmW4A16FwdOp,
        GroupedGemmFwdOp,
    )
    from tileops.ops.linear_attention import (
        DeltaNetChunkBwdOp,
        DeltaNetChunkFwdOp,
        DeltaNetInferenceFwdOp,
        DeltaNetRecurrentFwdOp,
        GatedDeltaNetFwdOp,
        GLAChunkBwdOp,
        GLAChunkFwdOp,
        GLAInferenceFwdOp,
        GLARecurrentFwdOp,
    )
    from tileops.ops.mamba import (
        Mamba2FwdOp,
        SSDChunkCumsumFwdOp,
        SSDChunkScanFwdOp,
        SSDChunkStateFwdOp,
        SSDRecurrentFwdOp,
        SSDStatePassingFwdOp,
    )
    from tileops.ops.moe import (
        MoEExpertMLPFwdOp,
        MoEGroupedGemmFwdOp,
        MoEPermuteAlignFwdOp,
        MoEPostPermuteFwdOp,
        MoEPrePermuteFwdOp,
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
    from tileops.ops.pool.adaptive_pool import (
        AdaptiveAvgPool2dFwdOp,
        AdaptiveMaxPool2dFwdOp,
        AdaptiveMaxPool2dIndicesFwdOp,
    )
    from tileops.ops.pool.avg_pool import AvgPool1dFwdOp, AvgPool2dFwdOp, AvgPool3dFwdOp
    from tileops.ops.pool.max_pool import (
        MaxPool1dFwdOp,
        MaxPool1dIndicesFwdOp,
        MaxPool2dFwdOp,
        MaxPool2dIndicesFwdOp,
        MaxPool3dFwdOp,
        MaxPool3dIndicesFwdOp,
    )
    from tileops.ops.pool.mean_pooling import MeanPoolingFwdOp
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
    from tileops.ops.quantization.fp8_quant import FP8QuantFwdOp
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
        LogSoftmaxFwdOp,
        LogSumExpFwdOp,
        MeanFwdOp,
        ProdFwdOp,
        SoftmaxFwdOp,
        StdFwdOp,
        SumFwdOp,
        VarFwdOp,
        VarMeanFwdOp,
        VectorNormFwdOp,
    )
    from tileops.ops.rope import (
        RopeFwdOp,
        RopeLlama31FwdOp,
        RopeLongRopeFwdOp,
        RopeNeoxPositionIdsFwdOp,
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

# Public name -> the submodule that defines it; `__all__` follows this order.
# Grouped by op family, simple to composite; within a group, base case before variants.
_LAZY = {
    # Base class
    "Op": ".op_base",
    # Elementwise
    "UnaryOp": ".elementwise",
    "BinaryOp": ".elementwise",
    "FusedGatedOp": ".elementwise",
    "DropoutFwdOp": ".elementwise.dropout",
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
    "VectorNormFwdOp": ".reduction",
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
    "FP8QuantFwdOp": ".quantization.fp8_quant",
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
    "GemmFP8FwdOp": ".gemm",
    "GemmW4A16FwdOp": ".gemm",
    "BmmFwdOp": ".gemm",
    "BmmFP8FwdOp": ".gemm",
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
    "MoEPrePermuteFwdOp": ".moe",
    "MoEPermuteAlignFwdOp": ".moe",
    "MoEGroupedGemmFwdOp": ".moe",
    "MoEExpertMLPFwdOp": ".moe",
    "MoEPostPermuteFwdOp": ".moe",
    # Sampling
    "TopKMaskFwdOp": ".sampling",
    "MinPMaskFwdOp": ".sampling",
    "TopPMaskFwdOp": ".sampling",
    "TopKTopPMaskFwdOp": ".sampling",
    "SamplingFromProbsFwdOp": ".sampling",
    "ChainSpeculativeSamplingFwdOp": ".sampling",
    # Rotary position embedding
    "RopeFwdOp": ".rope",
    "RopeNeoxPositionIdsFwdOp": ".rope",
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
    "NSACompressedVarlenFwdOp": ".attention",
    "NSATopKVarlenFwdOp": ".attention",
    "NSAVarlenFwdOp": ".attention",
    "DeepSeekSparseAttentionDecodeWithKVCacheFwdOp": ".attention",
    "FP8LightningIndexerFwdOp": ".attention.fp8_lightning_indexer",
    "TopKSelectFwdOp": ".attention.topk_select",
    # Linear attention
    "DeltaNetChunkFwdOp": ".linear_attention",
    "DeltaNetChunkBwdOp": ".linear_attention",
    "DeltaNetInferenceFwdOp": ".linear_attention",
    "DeltaNetRecurrentFwdOp": ".linear_attention",
    "GatedDeltaNetFwdOp": ".linear_attention",
    "GLAChunkFwdOp": ".linear_attention",
    "GLAInferenceFwdOp": ".linear_attention",
    "GLAChunkBwdOp": ".linear_attention",
    "GLARecurrentFwdOp": ".linear_attention",
    # Mamba
    "Mamba2FwdOp": ".mamba",
    "SSDChunkCumsumFwdOp": ".mamba",
    "SSDChunkStateFwdOp": ".mamba",
    "SSDStatePassingFwdOp": ".mamba",
    "SSDChunkScanFwdOp": ".mamba",
    "SSDRecurrentFwdOp": ".mamba",
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
