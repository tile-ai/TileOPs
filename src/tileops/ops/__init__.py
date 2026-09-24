from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # type checkers and IDEs do not run __getattr__
    from .attention import (
        DeepSeekSparseAttentionDecodeWithKVCacheFwdOp,
        GroupedQueryAttentionBwdOp,
        GroupedQueryAttentionDecodePagedWithKVCacheFwdOp,
        GroupedQueryAttentionDenseFwdOp,
        GroupedQueryAttentionPagedFwdOp,
        GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp,
        GroupedQueryAttentionVarlenFwdOp,
        MultiHeadAttentionBwdOp,
        MultiHeadAttentionDecodePagedWithKVCacheFwdOp,
        MultiHeadLatentAttentionDecodeWithKVCacheFwdOp,
        NSACmpFwdVarlenOp,
        NSAFwdVarlenOp,
        NSATopkVarlenOp,
    )
    from .convolution import (
        Conv1dFwdOp,
        Conv2dFwdOp,
        Conv3dFwdOp,
    )
    from .dropout import DropoutFwdOp
    from .elementwise import (
        BinaryOp,
        FusedGatedOp,
        UnaryOp,
    )
    from .fft import FFTC2CFwdOp
    from .fp8_lightning_indexer import FP8LightningIndexerFwdOp
    from .fp8_quant import FP8QuantFwdOp
    from .gemm import (
        BmmFp8FwdOp,
        BmmFwdOp,
        GemmFp8FwdOp,
        GemmFwdOp,
        GemmW4A16FwdOp,
        GroupedGemmFwdOp,
    )
    from .linear_attention import (
        DeltaNetAutogradOp,
        DeltaNetBwdOp,
        DeltaNetDecodeFwdOp,
        DeltaNetFwdOp,
        DeltaNetInferenceFwdOp,
        GatedDeltaNetFwdOp,
        GLABwdOp,
        GLADecodeFwdOp,
        GLAFwdOp,
    )
    from .mamba import (
        DaCumsumFwdOp,
        Mamba2FwdOp,
        SSDChunkScanFwdOp,
        SSDChunkStateFwdOp,
        SSDDecodeFwdOp,
        SSDStatePassingFwdOp,
    )
    from .moe import (
        MoeExpertMLPFwdOp,
        MoeGroupedGemmFwdOp,
        MoePermuteAlignFwdOp,
        MoePostPermuteFwdOp,
        MoePrePermuteFwdOp,
    )
    from .norm import (
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
    from .op_base import Op
    from .pool import (
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
    from .reduction import (
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
    from .rope import (
        RopeLlama31FwdOp,
        RopeLongRopeFwdOp,
        RopeNeoxFwdOp,
        RopeNeoxPositionIdsFwdOp,
        RopeNonNeoxFwdOp,
        RopeYarnFwdOp,
    )
    from .sequence_modeling import (
        MHCPostFwdOp,
        MHCPreFwdOp,
    )
    from .topk_selector import TopkSelectorFwdOp

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
    # Rotary position embedding
    "RopeNeoxFwdOp": ".rope",
    "RopeNeoxPositionIdsFwdOp": ".rope",
    "RopeNonNeoxFwdOp": ".rope",
    "RopeLlama31FwdOp": ".rope",
    "RopeYarnFwdOp": ".rope",
    "RopeLongRopeFwdOp": ".rope",
    # Attention
    "MultiHeadAttentionBwdOp": ".attention",
    "MultiHeadAttentionDecodePagedWithKVCacheFwdOp": ".attention",
    "GroupedQueryAttentionBwdOp": ".attention",
    "GroupedQueryAttentionDenseFwdOp": ".attention",
    "GroupedQueryAttentionPagedFwdOp": ".attention",
    "GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp": ".attention",
    "GroupedQueryAttentionDecodePagedWithKVCacheFwdOp": ".attention",
    "GroupedQueryAttentionVarlenFwdOp": ".attention",
    "MultiHeadLatentAttentionDecodeWithKVCacheFwdOp": ".attention",
    "NSACmpFwdVarlenOp": ".attention",
    "NSATopkVarlenOp": ".attention",
    "NSAFwdVarlenOp": ".attention",
    "DeepSeekSparseAttentionDecodeWithKVCacheFwdOp": ".attention",
    "FP8LightningIndexerFwdOp": ".fp8_lightning_indexer",
    "TopkSelectorFwdOp": ".topk_selector",
    # Linear attention
    "DeltaNetAutogradOp": ".linear_attention",
    "DeltaNetFwdOp": ".linear_attention",
    "DeltaNetBwdOp": ".linear_attention",
    "DeltaNetInferenceFwdOp": ".linear_attention",
    "DeltaNetDecodeFwdOp": ".linear_attention",
    "GatedDeltaNetFwdOp": ".linear_attention",
    "GLAFwdOp": ".linear_attention",
    "GLABwdOp": ".linear_attention",
    "GLADecodeFwdOp": ".linear_attention",
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
