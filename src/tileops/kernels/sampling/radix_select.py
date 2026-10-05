"""What the two masks that select a row's k-th largest key share.

A cluster of CTAs holds a row in registers and settles the k-th largest key one 8-bit digit
at a time. The pieces here are the ones both kernels hold in the same form: the bracket the
samples give, the cluster reduction of a pass's counts, the rank search over those counts,
and the launch policy that decides how wide a row spreads.

What they hold in different forms stays in each file: how a slot is addressed and stored,
which the fused mask reads in a different order, and the buffers the kernel body allocates.
"""

import tilelang.language as T
import torch

from tileops.kernels.constants import MAX_PORTABLE_CLUSTER_BLOCKS, VECTOR_ACCESS_BYTES
from tileops.utils import WARP_LANES, WARP_SHUFFLE_STAGES

__all__ = [
    "BRACKET_SIGMAS",
    "BRACKET_SLACK",
    "cluster_limit",
    "cluster_plan",
    "merge_counts",
    "rank_in_bins",
    "widest_row",
]

# How far the bracket taken from the samples reaches around the expected sample rank of the
# k-th key: this many standard deviations of a binomial count, plus a constant that covers
# the ranks whose expected count is a handful. A bracket that misses costs one more pass and
# changes no result, so the two trade retries against keys counted. Re-fit by timing the
# manifest rows over sigmas in {2, 3, 4, 6} and slack in {1, 3, 8}.
BRACKET_SIGMAS: float = 4.0
BRACKET_SLACK: float = 3.0


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
                        "tl::tileops_cluster_load_u32",
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
    """The longest row the launch policy holds without spilling a thread's registers."""
    return cluster_limit(arch) * threads * (VECTOR_ACCESS_BYTES // dtype.itemsize) * max_slots


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
