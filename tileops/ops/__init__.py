from .deepseek_dsa_decode import DeepSeekSparseAttentionDecodeWithKVCacheOp
from .deepseek_mla_decode import MultiHeadLatentAttentionDecodeWithKVCacheOp
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
from .fft import FFTC2CLUTOp, FFTC2COp
from .fp8_lighting_indexer import Fp8LightingIndexerOp
from .fp8_quant import Fp8QuantOp
from .gated_deltanet_chunkwise import GatedDeltaNetBwdOp, GatedDeltaNetFwdOp, GatedDeltaNetOp
from .gated_deltanet_recurrence import GatedDeltaNetDecodeOp
from .gemm import GemmOp
from .gla_chunkwise import GLABwdOp, GLAFwdOp
from .gla_recurrence import GLADecodeOp
from .gqa import GroupQueryAttentionBwdOp, GroupQueryAttentionFwdOp
from .gqa_decode import GroupQueryAttentionDecodeWithKVCacheOp
from .gqa_decode_paged import GroupQueryAttentionDecodePagedWithKVCacheOp
from .gqa_sliding_window_fwd import GqaSlidingWindowFwdOp
from .gqa_sliding_window_varlen_fwd import GqaSlidingWindowVarlenFwdOp
from .grouped_gemm import GroupedGemmOp
from .mha import MultiHeadAttentionBwdOp, MultiHeadAttentionFwdOp
from .mha_decode import MultiHeadAttentionDecodeWithKVCacheOp
from .mha_decode_paged import MultiHeadAttentionDecodePagedWithKVCacheOp
from .mhc_post import ManifoldConstrainedHyperConnectionPostOp
from .mhc_pre import ManifoldConstrainedHyperConnectionPreOp
from .moe import MoePermuteAlignOp
from .norm import (
    AdaLayerNormOp,
    AdaLayerNormZeroOp,
    BatchNormBwdOp,
    BatchNormFwdOp,
    FusedAddLayerNormOp,
    FusedAddRmsNormOp,
    GroupNormOp,
    InstanceNormOp,
    LayerNormOp,
    RmsNormOp,
)
from .op import Op

# --- Reduction ops (uncomment as sub-category PRs land) ---
from .reduction import (
    AllOp,
    AmaxOp,  # ReduceMaxOp
    AminOp,  # ReduceMinOp
    AnyOp,
    ArgmaxOp,
    ArgminOp,
    CountNonzeroOp,
    # CummaxOp,
    # CumminOp,
    CumprodOp,
    CumsumOp,
    InfNormOp,
    L1NormOp,
    L2NormOp,
    LogSoftmaxOp,
    LogSumExpOp,
    MeanOp,  # ReduceMeanOp
    ProdOp,  # ReduceProdOp
    SoftmaxOp,
    StdOp,
    SumOp,  # ReduceSumOp
    VarMeanOp,
    VarOp,
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
from .ssd_state_passing_fwd import SsdStatePassingFwdOp
from .topk_selector import TopkSelectorOp

__all__ = [
    "BinaryOp",
    "AdaLayerNormOp",
    "AdaLayerNormZeroOp",
    "BatchNormBwdOp",
    "BatchNormFwdOp",
    "DeepSeekSparseAttentionDecodeWithKVCacheOp",
    "DropoutOp",
    "FFTC2CLUTOp",
    "FFTC2COp",
    "Fp8LightingIndexerOp",
    "Fp8QuantOp",
    "FusedAddLayerNormOp",
    "FusedAddRmsNormOp",
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
    "GroupQueryAttentionBwdOp",
    "GroupQueryAttentionDecodePagedWithKVCacheOp",
    "GroupQueryAttentionDecodeWithKVCacheOp",
    "GroupQueryAttentionFwdOp",
    "GroupNormOp",
    "GroupedGemmOp",
    "InstanceNormOp",
    "LayerNormOp",
    "ManifoldConstrainedHyperConnectionPostOp",
    "ManifoldConstrainedHyperConnectionPreOp",
    "MeanPoolingForwardOp",
    "MultiHeadAttentionBwdOp",
    "MultiHeadAttentionDecodePagedWithKVCacheOp",
    "MultiHeadAttentionDecodeWithKVCacheOp",
    "MultiHeadAttentionFwdOp",
    "MultiHeadLatentAttentionDecodeWithKVCacheOp",
    "NSACmpFwdVarlenOp",
    "NSAFwdVarlenOp",
    "NSATopkVarlenOp",
    "Op",
    "MoePermuteAlignOp",
    "RmsNormOp",
    "SsdChunkScanFwdOp",
    "SsdChunkStateFwdOp",
    "SsdStatePassingFwdOp",
    "RopeLlama31Op",
    "RopeLongRopeOp",
    "RopeNeoxOp",
    "RopeNonNeoxOp",
    "RopeYarnOp",
    "UnaryOp",
    "TopkSelectorOp",
    # --- Reduction ops (uncomment as sub-category PRs land) ---
    "AllOp",
    "AmaxOp",
    "AminOp",
    "AnyOp",
    "ArgmaxOp",
    "ArgminOp",
    "CountNonzeroOp",
    # "CummaxOp",
    # "CumminOp",
    "CumprodOp",
    "CumsumOp",
    "InfNormOp",
    "L1NormOp",
    "L2NormOp",
    "LogSoftmaxOp",
    "LogSumExpOp",
    "MeanOp",
    "ProdOp",
    # "ReduceMaxOp",
    # "ReduceMeanOp",
    # "ReduceMinOp",
    # "ReduceProdOp",
    # "ReduceSumOp",
    "SoftmaxOp",
    "StdOp",
    "SumOp",
    "VarMeanOp",
    "VarOp",
]
