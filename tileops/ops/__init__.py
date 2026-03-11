from .deepseek_dsa_decode import DeepSeekSparseAttentionDecodeWithKVCacheOp
from .deepseek_mla_decode import MultiHeadLatentAttentionDecodeWithKVCacheOp
from .deepseek_nsa import (
    MeanPoolingForwardOp,
    NSACmpFwdVarlenOp,
    NSAFwdVarlenOp,
    NSATopkVarlenOp,
)
from .fft import FFTC2CLUTOp, FFTC2COp
from .fp8_lighting_indexer import Fp8LightingIndexerOp
from .fp8_quant import Fp8QuantOp
from .gated_deltanet import GatedDeltaNetBwdOp, GatedDeltaNetFwdOp
from .gated_deltanet_decode import GatedDeltaNetDecodeOp
from .gemm import GemmOp
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
# from .reduction import (
#     AllOp,
#     AnyOp,
#     ArgmaxOp,
#     ArgminOp,
#     CountNonzeroOp,
#     CummaxOp,
#     CumminOp,
#     CumprodOp,
#     CumsumOp,
#     InfNormOp,
#     L1NormOp,
#     L2NormOp,
#     LogSoftmaxOp,
#     LogSumExpOp,
#     ReduceMaxOp,
#     ReduceMeanOp,
#     ReduceMinOp,
#     ReduceProdOp,
#     ReduceSumOp,
#     SoftmaxOp,
# )
from .topk_selector import TopkSelectorOp

__all__ = [
    "AdaLayerNormOp",
    "AdaLayerNormZeroOp",
    "BatchNormBwdOp",
    "BatchNormFwdOp",
    "DeepSeekSparseAttentionDecodeWithKVCacheOp",
    "FFTC2CLUTOp",
    "FFTC2COp",
    "Fp8LightingIndexerOp",
    "Fp8QuantOp",
    "FusedAddLayerNormOp",
    "FusedAddRmsNormOp",
    "GatedDeltaNetBwdOp",
    "GatedDeltaNetDecodeOp",
    "GatedDeltaNetFwdOp",
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
    "RmsNormOp",
    "TopkSelectorOp",
    # --- Reduction ops (uncomment as sub-category PRs land) ---
    # "AllOp",
    # "AnyOp",
    # "ArgmaxOp",
    # "ArgminOp",
    # "CountNonzeroOp",
    # "CummaxOp",
    # "CumminOp",
    # "CumprodOp",
    # "CumsumOp",
    # "InfNormOp",
    # "L1NormOp",
    # "L2NormOp",
    # "LogSoftmaxOp",
    # "LogSumExpOp",
    # "ReduceMaxOp",
    # "ReduceMeanOp",
    # "ReduceMinOp",
    # "ReduceProdOp",
    # "ReduceSumOp",
    # "SoftmaxOp",
]
