"""Case factories of the attention family."""

from typing import Any

from benchmarks._cases import Entry
from workloads.attention.dsa import DSADecodeCall
from workloads.attention.fp8_lightning_indexer import FP8LightningIndexerCall
from workloads.attention.gqa.bwd import GQABwdCall
from workloads.attention.gqa.dense import GQADenseDecodeCall, GQADensePrefillCall
from workloads.attention.gqa.paged import GQAPagedCall
from workloads.attention.gqa.varlen import GQAVarlenCall, GQAVarlenScaledCall
from workloads.attention.mha import MHADecodePagedCall
from workloads.attention.mla import MLADecodeCall
from workloads.attention.nsa import NSACompressedFwdCall, NSAFwdCall, NSATopKCall
from workloads.attention.topk_select import TopKSelectCall


def _gqa_dense_workload(call: Any) -> Any:
    """A call passing FP8 scales or RoPE tables is a prefill; any other is a decode step."""
    if call.present("q_scale") or call.present("rope_cos"):
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


ENTRIES = {
    "GQABwdOp": Entry(GQABwdCall),
    "GQADenseFwdOp": Entry(_gqa_dense_workload, count_copies=True),
    "GQAVarlenFwdOp": Entry(_gqa_varlen_workload),
    "GQAPagedFwdOp": Entry(GQAPagedCall),
    "MHADecodePagedWithKVCacheFwdOp": Entry(MHADecodePagedCall),
    "MLADecodeWithKVCacheFwdOp": Entry(MLADecodeCall, count_copies=True),
    "DSADecodeWithKVCacheFwdOp": Entry(DSADecodeCall),
    "NSACompressedVarlenFwdOp": Entry(NSACompressedFwdCall),
    "NSATopKVarlenFwdOp": Entry(NSATopKCall),
    "NSAVarlenFwdOp": Entry(NSAFwdCall),
    "TopKSelectFwdOp": Entry(TopKSelectCall),
    "FP8LightningIndexerFwdOp": Entry(FP8LightningIndexerCall, count_copies=True),
}
