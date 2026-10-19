"""Case factories of the attention family."""

from typing import Any

from benchmarks._cases import Entry
from benchmarks.api import Implementation
from benchmarks.baselines import private_inputs
from workloads.attention.dsa import DSADecodeCall, DSAPagedCall
from workloads.attention.fp8_lightning_indexer import FP8LightningIndexerCall
from workloads.attention.gqa.bwd import GQABwdCall
from workloads.attention.gqa.dense import GQADenseDecodeCall, GQADensePrefillCall
from workloads.attention.gqa.paged import GQAPagedCall
from workloads.attention.gqa.prefill_paged_kv_append import (
    GQAPrefillPagedWithKVCacheFwdCall,
    paged_prefill_result,
)
from workloads.attention.gqa.varlen import GQAVarlenCall, GQAVarlenScaledCall
from workloads.attention.merge_states import MergeAttentionStatesCall
from workloads.attention.mha import MHADecodePagedCall
from workloads.attention.mla import MLADecodeCall, MLAPagedCall, MLAVarlenCall
from workloads.attention.mla_cache_write import MLAKVCacheWriteCall, mla_cache_write_result
from workloads.attention.nsa import NSACompressedFwdCall, NSAFwdCall, NSATopKCall
from workloads.attention.paged_cache_gather import PagedKVCacheGatherCall, paged_cache_gather_result
from workloads.attention.paged_cache_write import (
    PagedKVCacheWriteCall,
    paged_cache_write_result,
)
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


def _gqa_paged_append_binder(op: Any, case: Any) -> Implementation:
    """The op on private pages it appends k_new and v_new into, returning output and pages."""
    return private_inputs(lambda *args: paged_prefill_result(op, *args), case.inputs, 3, 4)


def _paged_cache_gather_binder(op, case):
    """Time the write-only gather on its private destination."""
    return private_inputs(lambda *args: paged_cache_gather_result(op, *args), case.inputs, 0)


def _paged_cache_write_binder(op: Any, case: Any) -> Implementation:
    """The op on private pages it scatters k and v into, returning both caches."""
    return private_inputs(lambda *args: paged_cache_write_result(op, *args), case.inputs, 2, 3)


def _mla_cache_write_binder(op: Any, case: Any) -> Implementation:
    """The op on a private cache it writes kv_c and k_pe into, returning the cache."""
    return private_inputs(lambda *args: mla_cache_write_result(op, *args), case.inputs, 2)


ENTRIES = {
    "PagedKVCacheGatherFwdOp": Entry(
        PagedKVCacheGatherCall, binder=_paged_cache_gather_binder, count_copies=True
    ),
    "MergeAttentionStatesFwdOp": Entry(MergeAttentionStatesCall, count_copies=True),
    "GQABwdOp": Entry(GQABwdCall),
    "GQADenseFwdOp": Entry(_gqa_dense_workload, count_copies=True),
    "GQAVarlenFwdOp": Entry(_gqa_varlen_workload),
    "GQAPrefillPagedWithKVCacheFwdOp": Entry(
        GQAPrefillPagedWithKVCacheFwdCall, binder=_gqa_paged_append_binder
    ),
    "GQAPagedFwdOp": Entry(GQAPagedCall),
    "PagedKVCacheWriteFwdOp": Entry(PagedKVCacheWriteCall, binder=_paged_cache_write_binder),
    "MHADecodePagedWithKVCacheFwdOp": Entry(MHADecodePagedCall),
    "MLADecodeWithKVCacheFwdOp": Entry(MLADecodeCall, count_copies=True),
    "MLAVarlenFwdOp": Entry(MLAVarlenCall),
    "MLAKVCacheWriteFwdOp": Entry(MLAKVCacheWriteCall, binder=_mla_cache_write_binder),
    "MLAPagedFwdOp": Entry(MLAPagedCall),
    "DSADecodeWithKVCacheFwdOp": Entry(DSADecodeCall),
    "DSAPagedFwdOp": Entry(DSAPagedCall),
    "NSACompressedVarlenFwdOp": Entry(NSACompressedFwdCall),
    "NSATopKVarlenFwdOp": Entry(NSATopKCall),
    "NSAVarlenFwdOp": Entry(NSAFwdCall),
    "TopKSelectFwdOp": Entry(TopKSelectCall),
    "FP8LightningIndexerFwdOp": Entry(FP8LightningIndexerCall, count_copies=True),
}
