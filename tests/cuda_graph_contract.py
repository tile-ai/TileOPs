"""Ops with H200 CUDA Graph evidence; marked tests define the qualified calls."""

QUALIFIED_OPS = frozenset(
    {
        "CausalConv1dFwdOp",
        "CausalConv1dUpdateFwdOp",
        "Conv1dFwdOp",
        "FusedMoEExpertsFwdOp",
        "GQAPagedFwdOp",
        "GQAVarlenFwdOp",
        "GEMMW4A8FwdOp",
        "KDAFwdOp",
        "MergeAttentionStatesFwdOp",
        "NGramHashFwdOp",
        "PagedKVCacheGatherFwdOp",
        "RMSNormFwdOp",
        "RoPEFwdOp",
    }
)
