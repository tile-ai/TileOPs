"""What the two masks that select a row's k-th largest key share.

A cluster of CTAs holds a row in registers and settles the k-th largest key one 8-bit digit
at a time. The pieces here are the ones both kernels hold in the same form: the cluster
reduction of a pass's counts, the rank search over those counts, and the launch policy that
decides which rows fit and how wide a row spreads.

What they hold in different forms stays in each file: how a slot is addressed and stored,
which the fused mask reads in a different order, and the buffers the kernel body allocates.
"""

from typing import Optional

import tilelang.language as T
import torch

from tileops._kernels.constants import MAX_PORTABLE_CLUSTER_BLOCKS, VECTOR_ACCESS_BYTES
from tileops._utils import WARP_LANES, WARP_SHUFFLE_STAGES

__all__ = [
    "cluster_limit",
    "cluster_plan",
    "merge_counts",
    "rank_in_bins",
    "row_refusal",
    "widest_row",
]


@T.macro
def merge_counts(hist, total, sum32, q, tx, cluster: int, slabs: int, threads: int):
    """Sum slab ``q`` of every CTA of the cluster into ``total``."""
    if cluster == 1:
        T.sync_threads()
        for i in T.serial(-(-slabs // threads)):
            if i * threads + tx < slabs:
                total[i * threads + tx] = hist[q * slabs + i * threads + tx]
    else:
        T.cluster_sync()
        for i in T.serial(-(-slabs // threads)):
            if i * threads + tx < slabs:
                sum32[0] = T.uint32(0)
                for c in T.serial(cluster):
                    sum32[0] = sum32[0] + T.call_extern(
                        "uint32",
                        "tileops::cluster_load_u32",
                        T.address_of(hist[q * slabs + i * threads + tx]),
                        c,
                    )
                total[i * threads + tx] = sum32[0]
    T.sync_threads()


@T.macro
def rank_in_bins(total, target, acc, tx, span: int):
    """Warp 0: into ``acc``, the bin of ``total`` holding rank ``target`` counted from
    the largest, that rank inside it, and the bin's count.

    Lane ``tx`` owns ``span`` bins. ``acc[0]`` stays -1 when the bins hold fewer than
    ``target`` keys.
    """
    acc[0] = -1
    acc[3] = 0
    for i in T.serial(span):
        acc[3] = acc[3] + T.cast(total[tx * span + i], "int32")
    acc[1] = acc[3]
    for stage in T.unroll(WARP_SHUFFLE_STAGES):
        up = T.shfl_down(acc[1], 1 << stage, width=WARP_LANES)
        if tx + (1 << stage) < WARP_LANES:
            acc[1] = acc[1] + up
    # This lane's bins hold the ranks in (acc[1] - acc[3], acc[1]].
    acc[2] = acc[1] - acc[3]
    if (acc[2] < target) & (target <= acc[1]):
        for i in T.serial(span):
            b = tx * span + span - 1 - i
            count = T.cast(total[b], "int32")
            if (acc[2] < target) & (target <= acc[2] + count):
                acc[0] = b
                acc[1] = target - acc[2]
                acc[3] = count
            acc[2] = acc[2] + count


def cluster_limit(arch: int) -> int:
    """CTAs one row may span: thread-block clusters start at SM90."""
    return MAX_PORTABLE_CLUSTER_BLOCKS if arch >= 90 else 1


def widest_row(dtype: torch.dtype, threads: int, max_slots: int, arch: int) -> int:
    """The longest row the launch policy holds at ``max_slots`` a thread."""
    return cluster_limit(arch) * threads * (VECTOR_ACCESS_BYTES // dtype.itemsize) * max_slots


def row_refusal(call, threads: int, max_slots: int, max_slots_one_cta: int) -> Optional[str]:
    """Why a launch of ``threads`` a CTA cannot hold the call's rows, or ``None``.

    ``max_slots_one_cta`` bounds a thread's slots where ``cluster_limit`` keeps a row in one
    CTA, and ``max_slots`` bounds them where a cluster spreads it.
    """
    if call.batch * call.vocab > 2**31 - 1:
        return f"indexes elements with int32, and B * V = {call.batch * call.vocab}"
    slots = max_slots if cluster_limit(call.arch) > 1 else max_slots_one_cta
    widest = widest_row(call.dtype, threads, slots, call.arch)
    if call.vocab > widest:
        return f"supports rows of at most {widest} values, and V = {call.vocab}"
    return None


def cluster_plan(call, threads: int, max_slots: int) -> dict:
    """As many CTAs per row as it takes to fill the device, each holding an equal run.

    A cluster costs a barrier across its CTAs on every pass, so a batch that already fills
    the device puts one CTA on each row and only a batch short of it spreads a row wider; a
    row too long to hold in ``max_slots`` per thread spreads wider still. The run is then
    exactly the row's share, since a slot no key lands in is a register the passes scan for
    nothing.
    """
    vec = VECTOR_ACCESS_BYTES // call.dtype.itemsize
    limit = cluster_limit(call.arch)
    cluster = 1
    while cluster < limit and 2 * cluster * call.batch <= call.sm_count:
        cluster *= 2
    slots = -(-call.vocab // (cluster * threads * vec))
    while cluster < limit and slots > max_slots:
        cluster *= 2
        slots = -(-call.vocab // (cluster * threads * vec))
    return {"threads": threads, "cluster": cluster, "slots": slots}
