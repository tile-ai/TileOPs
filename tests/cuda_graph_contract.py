"""Ops with H200 CUDA Graph evidence; marked tests define the qualified calls."""

QUALIFIED_OPS = frozenset(
    {
        "Conv1dFwdOp",
        "FusedMoEExpertsFwdOp",
        "GQAPagedFwdOp",
        "GQAVarlenFwdOp",
        "GEMMW4A8FwdOp",
        "HyperConnectionPreFwdOp",
        "KDAFwdOp",
        "MergeAttentionStatesFwdOp",
        "NGramHashFwdOp",
        "PagedKVCacheGatherFwdOp",
        "RMSNormFwdOp",
        "RoPEFwdOp",
    }
)
