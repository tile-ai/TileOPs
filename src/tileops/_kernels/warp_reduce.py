"""XOR-butterfly reduction across the lanes of a warp, shared by the norm and reduction kernels."""

import tilelang.language as T

from tileops._utils import WARP_LANES

__all__ = ["warp_reduce"]


@T.macro
def _fold_values(acc, combine, count, first, mask):
    for j in T.unroll(first, first + count):
        acc[j] = combine(acc[j], T.shfl_xor(acc[j], mask, width=WARP_LANES))


@T.macro
def _shuffle_values(acc, peer, count, first, mask):
    for j in T.unroll(count, explicit=True):
        peer[j] = T.shfl_xor(acc[first + j], mask, width=WARP_LANES)


def _stage(acc, combine, count, first, mask, peer):
    # Plain Python: a macro's own ``for`` only takes TIR loops, not a tuple of buffers.
    buffers = acc if isinstance(acc, tuple) else (acc,)
    if peer is None:
        for buffer in buffers:
            _fold_values(buffer, combine, count, first, mask)
    else:
        peers = peer if isinstance(peer, tuple) else (peer,)
        for buffer, out in zip(buffers, peers, strict=True):
            _shuffle_values(buffer, out, count, first, mask)
        combine(acc, peer)


@T.macro
def warp_reduce(
    acc,
    combine,
    count: int = 1,
    first: int = 0,
    lanes: int = WARP_LANES,
    high_first: bool = False,
    serial: bool = False,
    peer=None,
):
    """Combine each of ``acc[first:first + count]`` across each aligned group of *lanes* lanes.

    *lanes* is a power of two of at most a warp. Every lane of a group ends with the group's
    result. The XOR masks run from 1 up, or from ``lanes // 2`` down when *high_first*; the
    order fixes how a floating-point sum or product rounds. The values interleave within a
    stage, so one value's shuffle latency hides behind the next's.

    Args:
        acc: Local buffer holding the values, or a tuple of such buffers, each holding
            values at the same indices.
        combine: ``combine(own, peer)``, the expression folding a peer lane's value in.
            With *peer*, instead ``combine(acc, peer)``, the statements folding every
            shuffled value in at once.
        count: Values to combine per buffer.
        first: Index of the first value in each buffer.
        lanes: Lanes in a group.
        high_first: Run the masks from ``lanes // 2`` down to 1.
        serial: Loop over the stages with ``T.serial``, not ``T.unroll``.
        peer: Local buffer, or tuple matching *acc*, receiving each stage's shuffled
            values at indices ``0:count``.
    """
    stages = lanes.bit_length() - 1
    for st in T.serial(stages) if serial else T.unroll(stages):
        if high_first:
            _stage(acc, combine, count, first, T.int32(lanes // 2) >> st, peer)
        else:
            _stage(acc, combine, count, first, T.shift_left(1, st), peer)
