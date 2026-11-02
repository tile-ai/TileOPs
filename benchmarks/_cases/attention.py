"""How the attention family's benchmarks call their ops."""

from typing import Any

from benchmarks._cases import Entry, default_binder
from benchmarks.api import Implementation
from benchmarks.baselines import private_inputs
from workloads.attention.mla_cache_write import mla_cache_write_result
from workloads.attention.paged_cache_write import paged_cache_write_result


def _paged_cache_gather_binder(op, case):
    """Time the write-only gather on its private destination."""
    return private_inputs(default_binder(op, case).run, case.inputs, len(case.inputs) - 1)


def _paged_cache_write_binder(op: Any, case: Any) -> Implementation:
    """The op on private pages it scatters k and v into, returning both caches."""
    return private_inputs(lambda *args: paged_cache_write_result(op, *args), case.inputs, 2, 3)


def _mla_cache_write_binder(op: Any, case: Any) -> Implementation:
    """The op on a private cache it writes kv_c and k_pe into, returning the cache."""
    return private_inputs(lambda *args: mla_cache_write_result(op, *args), case.inputs, 2)


ENTRIES = {
    "PagedKVCacheGatherFwdOp": Entry(binder=_paged_cache_gather_binder, count_copies=True),
    "MergeAttentionStatesFwdOp": Entry(count_copies=True),
    "GQADenseFwdOp": Entry(count_copies=True),
    "PagedKVCacheWriteFwdOp": Entry(binder=_paged_cache_write_binder),
    "MLADenseDecodeFwdOp": Entry(count_copies=True),
    "MLAKVCacheWriteFwdOp": Entry(binder=_mla_cache_write_binder),
    "FP8LightningIndexerFwdOp": Entry(count_copies=True),
}
