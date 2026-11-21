"""XOR-butterfly reduction across the lanes of a warp, shared by the norm and reduction kernels."""

import tilelang.language as T

from tileops._utils import WARP_LANES

__all__ = ["warp_reduce"]


@T.macro
def warp_reduce(
    acc,
    combine,
    count: int = 1,
    first: int = 0,
    lanes: int = WARP_LANES,
    high_first: bool = False,
):
    """Combine each of ``acc[first:first + count]`` across each aligned group of *lanes* lanes.

    *lanes* is a power of two of at most a warp. Every lane of a group ends with the group's
    result. The XOR masks run from 1 up, or from ``lanes // 2`` down when *high_first*; the
    order fixes how a floating-point sum or product rounds. The values interleave within a
    stage, so one value's shuffle latency hides behind the next's.

    Args:
        acc: Local buffer holding the values.
        combine: ``combine(own, peer)``, the expression folding a peer lane's value in.
        count: Values to combine.
        first: Index of the first value in *acc*.
        lanes: Lanes in a group.
        high_first: Run the masks from ``lanes // 2`` down to 1.
    """
    for st in T.unroll(lanes.bit_length() - 1):
        for j in T.unroll(first, first + count):
            if high_first:
                acc[j] = combine(
                    acc[j], T.shfl_xor(acc[j], T.int32(lanes // 2) >> st, width=WARP_LANES)
                )
            else:
                acc[j] = combine(acc[j], T.shfl_xor(acc[j], T.shift_left(1, st), width=WARP_LANES))
