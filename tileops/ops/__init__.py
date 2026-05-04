from .attention import (
    DeepSeekSparseAttentionDecodeWithKVCacheFwdOp,
    GroupedQueryAttentionBwdOp,
    GroupedQueryAttentionDecodePagedWithKVCacheFwdOp,
    GroupedQueryAttentionDecodeWithKVCacheFwdOp,
    GroupedQueryAttentionFwdOp,
    GroupedQueryAttentionPrefillFwdOp,
    GroupedQueryAttentionPrefillVarlenFwdOp,
    GroupedQueryAttentionPrefillWithKVCacheFwdOp,
    GroupedQueryAttentionSlidingWindowFwdOp,
    GroupedQueryAttentionSlidingWindowVarlenFwdOp,
    MeanPoolingForwardOp,
    MultiHeadAttentionBwdOp,
    MultiHeadAttentionDecodePagedWithKVCacheFwdOp,
    MultiHeadAttentionDecodeWithKVCacheFwdOp,
    MultiHeadAttentionFwdOp,
    MultiHeadLatentAttentionDecodeWithKVCacheFwdOp,
    NSACmpFwdVarlenOp,
    NSAFwdVarlenOp,
    NSATopkVarlenOp,
)
from .convolution import Conv1dBiasFwdOp, Conv1dFwdOp, Conv2dOp, Conv3dOp
from .da_cumsum import DaCumsumFwdOp
from .deltanet import DeltaNetBwdOp, DeltaNetFwdOp, DeltaNetOp
from .deltanet_recurrence import DeltaNetDecodeOp
from .dropout import DropoutOp
from .elementwise import BinaryOp, FusedGatedOp, UnaryOp
from .fft import FFTC2COp
from .fp8_lighting_indexer import FP8LightingIndexerOp
from .fp8_quant import FP8QuantOp
from .gated_deltanet import (
    GatedDeltaNetBwdOp,
    GatedDeltaNetDecodeOp,
    GatedDeltaNetFwdOp,
    GatedDeltaNetOp,
)
from .gated_linear_attn import GLADecodeOp
from .gemm import GemmOp
from .gla import GLABwdOp, GLAFwdOp
from .grouped_gemm import GroupedGemmOp
from .mhc import MHCPostOp, MHCPreOp
from .moe import MoePermuteAlignFwdOp
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
from .pool import AvgPool1dOp, AvgPool2dOp, AvgPool3dOp

# --- Reduction ops (uncomment as sub-category PRs land) ---
from .reduction import (
    AllFwdOp,
    AmaxFwdOp,  # ReduceMaxOp
    AminFwdOp,  # ReduceMinOp
    AnyFwdOp,
    ArgmaxFwdOp,
    ArgminFwdOp,
    CountNonzeroFwdOp,
    # CummaxOp,
    # CumminOp,
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
    RopeLlama31Op,
    RopeLongRopeOp,
    RopeNeoxOp,
    RopeNonNeoxOp,
    RopeYarnOp,
)
from .ssd_chunk_scan import SSDChunkScanFwdOp
from .ssd_chunk_state import SSDChunkStateFwdOp
from .ssd_decode import SSDDecodeOp
from .ssd_state_passing import SSDStatePassingFwdOp
from .topk_selector import TopkSelectorOp

__all__ = [
    "BinaryOp",
    "AvgPool1dOp",
    "AvgPool2dOp",
    "AvgPool3dOp",
    "AdaLayerNormFwdOp",
    "AdaLayerNormZeroFwdOp",
    "BatchNormBwdOp",
    "BatchNormFwdOp",
    "Conv1dBiasFwdOp",
    "Conv1dFwdOp",
    "Conv2dOp",
    "Conv3dOp",
    "DaCumsumFwdOp",
    "DeepSeekSparseAttentionDecodeWithKVCacheFwdOp",
    "DropoutOp",
    "FFTC2COp",
    "FP8LightingIndexerOp",
    "FP8QuantOp",
    "FusedAddLayerNormFwdOp",
    "FusedAddRMSNormFwdOp",
    "FusedGatedOp",
    "DeltaNetBwdOp",
    "DeltaNetDecodeOp",
    "DeltaNetFwdOp",
    "DeltaNetOp",
    "GatedDeltaNetBwdOp",
    "GatedDeltaNetDecodeOp",
    "GatedDeltaNetFwdOp",
    "GatedDeltaNetOp",
    "GLABwdOp",
    "GLADecodeOp",
    "GLAFwdOp",
    "GemmOp",
    "GroupedQueryAttentionSlidingWindowFwdOp",
    "GroupedQueryAttentionSlidingWindowVarlenFwdOp",
    "GroupedQueryAttentionBwdOp",
    "GroupedQueryAttentionDecodePagedWithKVCacheFwdOp",
    "GroupedQueryAttentionDecodeWithKVCacheFwdOp",
    "GroupedQueryAttentionFwdOp",
    "GroupedQueryAttentionPrefillFwdOp",
    "GroupedQueryAttentionPrefillVarlenFwdOp",
    "GroupedQueryAttentionPrefillWithKVCacheFwdOp",
    "GroupNormFwdOp",
    "GroupedGemmOp",
    "InstanceNormFwdOp",
    "LayerNormFwdOp",
    "MHCPostOp",
    "MHCPreOp",
    "MeanPoolingForwardOp",
    "MultiHeadAttentionBwdOp",
    "MultiHeadAttentionDecodePagedWithKVCacheFwdOp",
    "MultiHeadAttentionDecodeWithKVCacheFwdOp",
    "MultiHeadAttentionFwdOp",
    "MultiHeadLatentAttentionDecodeWithKVCacheFwdOp",
    "NSACmpFwdVarlenOp",
    "NSAFwdVarlenOp",
    "NSATopkVarlenOp",
    "Op",
    "MoePermuteAlignFwdOp",
    "RMSNormFwdOp",
    "SSDChunkScanFwdOp",
    "SSDChunkStateFwdOp",
    "SSDDecodeOp",
    "SSDStatePassingFwdOp",
    "RopeLlama31Op",
    "RopeLongRopeOp",
    "RopeNeoxOp",
    "RopeNonNeoxOp",
    "RopeYarnOp",
    "UnaryOp",
    "TopkSelectorOp",
    # --- Reduction ops (uncomment as sub-category PRs land) ---
    "AllFwdOp",
    "AmaxFwdOp",
    "AminFwdOp",
    "AnyFwdOp",
    "ArgmaxFwdOp",
    "ArgminFwdOp",
    "CountNonzeroFwdOp",
    # "CummaxOp",
    # "CumminOp",
    "CumprodFwdOp",
    "CumsumFwdOp",
    "InfNormFwdOp",
    "L1NormFwdOp",
    "L2NormFwdOp",
    "LogSoftmaxFwdOp",
    "LogSumExpFwdOp",
    "MeanFwdOp",
    "ProdFwdOp",
    # "ReduceMaxOp",
    # "ReduceMeanOp",
    # "ReduceMinOp",
    # "ReduceProdOp",
    # "ReduceSumOp",
    "SoftmaxFwdOp",
    "StdFwdOp",
    "SumFwdOp",
    "VarMeanFwdOp",
    "VarFwdOp",
]
