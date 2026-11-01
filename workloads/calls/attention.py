"""Call workloads of the attention family."""

from typing import Any

from workloads.attention.dsa import DSADecodeCall, DSAPagedCall
from workloads.attention.fp8_lightning_indexer import FP8LightningIndexerCall
from workloads.attention.gqa.bwd import GQABwdCall
from workloads.attention.gqa.dense import (
    GQADenseDecodeCall,
    GQADensePrefillCall,
    GQADenseWithLSECall,
)
from workloads.attention.gqa.paged import GQAPagedCall
from workloads.attention.gqa.varlen import GQAVarlenCall, GQAVarlenScaledCall
from workloads.attention.merge_states import MergeAttentionStatesCall
from workloads.attention.mha import MHADecodePagedCall
from workloads.attention.mla import (
    MLADecodeCall,
    MLADenseBwdCall,
    MLAPagedCall,
    MLAVarlenBwdCall,
    MLAVarlenCall,
)
from workloads.attention.mla_cache_write import MLAKVCacheWriteCall
from workloads.attention.nsa import NSACompressedFwdCall, NSAFwdCall, NSATopKCall
from workloads.attention.paged_cache_gather import PagedKVCacheGatherCall
from workloads.attention.paged_cache_write import PagedKVCacheWriteCall
from workloads.attention.topk_select import TopKSelectCall


def _gqa_dense_workload(call: Any) -> Any:
    """Use the full dense reference for prefill, windows, or optional inputs."""
    if (
        call.indices["S_q"] != 1
        or call.present("q_scale")
        or call.present("rope_cos")
        or call.present("sinks")
        or call.params["window_size_left"] >= 0
        or call.params["window_size_right"] >= 0
    ):
        return GQADensePrefillCall(call)
    return GQADenseDecodeCall(call)


def _gqa_varlen_workload(call: Any) -> Any:
    """A call passing the FP8 scales takes the scaled reference.

    A rotating 16-bit call keeps the unscaled one: its rotation is a baseline concern, not a
    different reference.
    """
    if call.present("q_scale"):
        return GQAVarlenScaledCall(call)
    return GQAVarlenCall(call)


WORKLOADS = {
    "PagedKVCacheGatherFwdOp": PagedKVCacheGatherCall,
    "MergeAttentionStatesFwdOp": MergeAttentionStatesCall,
    "GQADenseBwdOp": GQABwdCall,
    "GQADenseFwdOp": _gqa_dense_workload,
    "GQADenseWithLSEFwdOp": GQADenseWithLSECall,
    "GQAVarlenFwdOp": _gqa_varlen_workload,
    "GQAPagedFwdOp": GQAPagedCall,
    "PagedKVCacheWriteFwdOp": PagedKVCacheWriteCall,
    "MHAPagedFwdOp": MHADecodePagedCall,
    "MLADenseDecodeFwdOp": MLADecodeCall,
    "MLAVarlenFwdOp": MLAVarlenCall,
    "MLADenseBwdOp": MLADenseBwdCall,
    "MLAVarlenBwdOp": MLAVarlenBwdCall,
    "MLAKVCacheWriteFwdOp": MLAKVCacheWriteCall,
    "MLAPagedFwdOp": MLAPagedCall,
    "DSADenseFwdOp": DSADecodeCall,
    "DSAPagedFwdOp": DSAPagedCall,
    "NSACompressedVarlenFwdOp": NSACompressedFwdCall,
    "NSATopKVarlenFwdOp": NSATopKCall,
    "NSAVarlenFwdOp": NSAFwdCall,
    "TopKSelectFwdOp": TopKSelectCall,
    "FP8LightningIndexerFwdOp": FP8LightningIndexerCall,
}
