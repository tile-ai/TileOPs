from .attention import (
    DeepSeekSparseAttentionDecodeWithKVCacheFwdOp,
    GroupedQueryAttentionBwdOp,
    GroupedQueryAttentionDecodePagedWithKVCacheFwdOp,
    GroupedQueryAttentionDecodeWithKVCacheFwdOp,
    GroupedQueryAttentionDenseFwdOp,
    GroupedQueryAttentionFwdOp,
    GroupedQueryAttentionPrefillFwdOp,
    GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp,
    GroupedQueryAttentionPrefillVarlenFwdOp,
    GroupedQueryAttentionSlidingWindowFwdOp,
    GroupedQueryAttentionSlidingWindowVarlenFwdOp,
    MultiHeadAttentionBwdOp,
    MultiHeadAttentionDecodePagedWithKVCacheFwdOp,
    MultiHeadAttentionDecodeWithKVCacheFwdOp,
    MultiHeadAttentionFwdOp,
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
from .elementwise import BinaryOp, FusedGatedOp, UnaryOp
from .fft import FFTC2CFwdOp
from .fp8_lightning_indexer import FP8LightningIndexerFwdOp
from .fp8_quant import FP8QuantFwdOp
from .gemm import (
    BmmFp8KNFwdOp,
    BmmFp8NKFwdOp,
    BmmFwdOp,
    GemmFp8FwdOp,
    GemmFwdOp,
    GemmW4A16FwdOp,
    GroupedGemmFwdOp,
)
from .linear_attention import (
    DeltaNetBwdOp,
    DeltaNetDecodeFwdOp,
    DeltaNetFwdOp,
    DeltaNetOp,
    GatedDeltaNetBHTDFwdOp,
    GatedDeltaNetBTHDFwdOp,
    GatedDeltaNetBwdOp,
    GatedDeltaNetDecodeFwdOp,
    GatedDeltaNetOp,
    GatedDeltaNetPrefillBHTDFwdOp,
    GatedDeltaNetPrefillBTHDFwdOp,
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
    MeanPoolingForwardOp,
)

# --- Reduction ops (uncomment as sub-category PRs land) ---
from .reduction import (
    AllFwdOp,
    AmaxFwdOp,  # ReduceMaxOp
    AminFwdOp,  # ReduceMinOp
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
    MeanFwdOp,  # ReduceMeanOp
    ProdFwdOp,  # ReduceProdOp
    SoftmaxFwdOp,
    StdFwdOp,
    SumFwdOp,  # ReduceSumOp
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
from .sequence_modeling import MHCPostFwdOp, MHCPreFwdOp
from .topk_selector import TopkSelectorFwdOp

# Ordered simple to composite, not alphabetically: pointwise transforms, then the
# axis reductions and the normalizations built on them, then the matmul, then the
# windowed and spectral transforms, and the sequence-model kernels built on all of
# the above. Within a group, the base case comes first and its variants follow.
# mHC sits at the end, outside that progression: it is one algorithm, not a family.
__all__ = [
    # Base class
    "Op",
    # Elementwise
    "UnaryOp",
    "BinaryOp",
    "FusedGatedOp",
    # Dropout
    "DropoutFwdOp",
    # Reduction
    "SumFwdOp",
    "MeanFwdOp",
    "ProdFwdOp",
    "AmaxFwdOp",
    "AminFwdOp",
    "VarFwdOp",
    "VarMeanFwdOp",
    "StdFwdOp",
    "ArgmaxFwdOp",
    "ArgminFwdOp",
    "SoftmaxFwdOp",
    "LogSoftmaxFwdOp",
    "LogSumExpFwdOp",
    "L1NormFwdOp",
    "L2NormFwdOp",
    "InfNormFwdOp",
    "CumsumFwdOp",
    "CumprodFwdOp",
    "AllFwdOp",
    "AnyFwdOp",
    "CountNonzeroFwdOp",
    # Normalization
    "LayerNormFwdOp",
    "FusedAddLayerNormFwdOp",
    "RMSNormFwdOp",
    "FusedAddRMSNormFwdOp",
    "AdaLayerNormFwdOp",
    "AdaLayerNormZeroFwdOp",
    "BatchNormFwdOp",
    "BatchNormBwdOp",
    "GroupNormFwdOp",
    "InstanceNormFwdOp",
    # Quantization
    "FP8QuantFwdOp",
    # Top-k selection
    "TopkSelectorFwdOp",
    # GEMM
    "GemmFwdOp",
    "GemmFp8FwdOp",
    "GemmW4A16FwdOp",
    "BmmFwdOp",
    "BmmFp8KNFwdOp",
    "BmmFp8NKFwdOp",
    "GroupedGemmFwdOp",
    # Pooling
    "AvgPool1dFwdOp",
    "AvgPool2dFwdOp",
    "AvgPool3dFwdOp",
    "MaxPool1dFwdOp",
    "MaxPool1dIndicesFwdOp",
    "MaxPool2dFwdOp",
    "MaxPool2dIndicesFwdOp",
    "MaxPool3dFwdOp",
    "MaxPool3dIndicesFwdOp",
    "AdaptiveAvgPool2dFwdOp",
    "AdaptiveMaxPool2dFwdOp",
    "AdaptiveMaxPool2dIndicesFwdOp",
    "MeanPoolingForwardOp",
    # Convolution
    "Conv1dFwdOp",
    "Conv2dFwdOp",
    "Conv3dFwdOp",
    # FFT
    "FFTC2CFwdOp",
    # Mixture of experts
    "MoePrePermuteFwdOp",
    "MoePermuteAlignFwdOp",
    "MoeGroupedGemmFwdOp",
    "MoeExpertMLPFwdOp",
    "MoePostPermuteFwdOp",
    # Rotary position embedding
    "RopeNeoxFwdOp",
    "RopeNeoxPositionIdsFwdOp",
    "RopeNonNeoxFwdOp",
    "RopeLlama31FwdOp",
    "RopeYarnFwdOp",
    "RopeLongRopeFwdOp",
    # Attention
    "MultiHeadAttentionFwdOp",
    "MultiHeadAttentionBwdOp",
    "MultiHeadAttentionDecodeWithKVCacheFwdOp",
    "MultiHeadAttentionDecodePagedWithKVCacheFwdOp",
    "GroupedQueryAttentionFwdOp",
    "GroupedQueryAttentionBwdOp",
    "GroupedQueryAttentionDenseFwdOp",
    "GroupedQueryAttentionPrefillFwdOp",
    "GroupedQueryAttentionPrefillVarlenFwdOp",
    "GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp",
    "GroupedQueryAttentionDecodeWithKVCacheFwdOp",
    "GroupedQueryAttentionDecodePagedWithKVCacheFwdOp",
    "GroupedQueryAttentionSlidingWindowFwdOp",
    "GroupedQueryAttentionSlidingWindowVarlenFwdOp",
    "MultiHeadLatentAttentionDecodeWithKVCacheFwdOp",
    "NSACmpFwdVarlenOp",
    "NSATopkVarlenOp",
    "NSAFwdVarlenOp",
    "DeepSeekSparseAttentionDecodeWithKVCacheFwdOp",
    "FP8LightningIndexerFwdOp",
    # Linear attention
    "DeltaNetOp",
    "DeltaNetFwdOp",
    "DeltaNetBwdOp",
    "DeltaNetDecodeFwdOp",
    "GatedDeltaNetOp",
    "GatedDeltaNetBTHDFwdOp",
    "GatedDeltaNetBHTDFwdOp",
    "GatedDeltaNetPrefillBTHDFwdOp",
    "GatedDeltaNetPrefillBHTDFwdOp",
    "GatedDeltaNetDecodeFwdOp",
    "GatedDeltaNetBwdOp",
    "GLAFwdOp",
    "GLABwdOp",
    "GLADecodeFwdOp",
    # Mamba
    "Mamba2FwdOp",
    "DaCumsumFwdOp",
    "SSDChunkStateFwdOp",
    "SSDStatePassingFwdOp",
    "SSDChunkScanFwdOp",
    "SSDDecodeFwdOp",
    # mHC (Manifold-Constrained Hyper-Connections) — one algorithm, not an op family
    # like the groups above
    "MHCPreFwdOp",
    "MHCPostFwdOp",
]
