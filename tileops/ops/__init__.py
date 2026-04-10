from .avg_pool1d import AvgPool1dOp
from .avg_pool2d import AvgPool2dOp
from .avg_pool3d import AvgPool3dOp
from .conv1d import Conv1dBiasFwdOp, Conv1dFwdOp
from .conv2d import Conv2dOp
from .conv3d import Conv3dOp
from .da_cumsum_fwd import DaCumsumFwdOp
from .deepseek_dsa_decode import DeepSeekSparseAttentionDecodeWithKVCacheFwdOp
from .deepseek_mla_decode import MultiHeadLatentAttentionDecodeWithKVCacheFwdOp
from .deepseek_nsa import (
    MeanPoolingForwardOp,
    NSACmpFwdVarlenOp,
    NSAFwdVarlenOp,
    NSATopkVarlenOp,
)
from .deltanet_chunkwise import DeltaNetBwdOp, DeltaNetFwdOp, DeltaNetOp
from .deltanet_recurrence import DeltaNetDecodeOp
from .dropout import DropoutOp
from .elementwise import BinaryOp, FusedGatedOp, UnaryOp
from .fft import FFTC2COp
from .fp8_lighting_indexer import FP8LightingIndexerOp
from .fp8_quant import FP8QuantOp
from .gated_deltanet_chunkwise import GatedDeltaNetBwdOp, GatedDeltaNetFwdOp, GatedDeltaNetOp
from .gated_deltanet_recurrence import GatedDeltaNetDecodeOp
from .gemm import GemmOp
from .gla_chunkwise import GLABwdOp, GLAFwdOp
from .gla_recurrence import GLADecodeOp
from .gqa import GroupedQueryAttentionBwdOp, GroupedQueryAttentionFwdOp
from .gqa_decode import GroupedQueryAttentionDecodeWithKVCacheFwdOp
from .gqa_decode_paged import GroupedQueryAttentionDecodePagedWithKVCacheFwdOp
from .gqa_sliding_window_fwd import GqaSlidingWindowFwdOp
from .gqa_sliding_window_varlen_fwd import GqaSlidingWindowVarlenFwdOp
from .grouped_gemm import GroupedGemmOp
from .mha import MultiHeadAttentionBwdOp, MultiHeadAttentionFwdOp
from .mha_decode import MultiHeadAttentionDecodeWithKVCacheFwdOp
from .mha_decode_paged import MultiHeadAttentionDecodePagedWithKVCacheFwdOp
from .mhc_post import MHCPostOp
from .mhc_pre import MHCPreOp
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
from .op import Op

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
from .ssd_chunk_scan_fwd import SsdChunkScanFwdOp
from .ssd_chunk_state_fwd import SsdChunkStateFwdOp
from .ssd_decode import SsdDecodeOp
from .ssd_state_passing_fwd import SsdStatePassingFwdOp
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
    "GqaSlidingWindowFwdOp",
    "GqaSlidingWindowVarlenFwdOp",
    "GroupedQueryAttentionBwdOp",
    "GroupedQueryAttentionDecodePagedWithKVCacheFwdOp",
    "GroupedQueryAttentionDecodeWithKVCacheFwdOp",
    "GroupedQueryAttentionFwdOp",
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
    "SsdChunkScanFwdOp",
    "SsdChunkStateFwdOp",
    "SsdDecodeOp",
    "SsdStatePassingFwdOp",
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
