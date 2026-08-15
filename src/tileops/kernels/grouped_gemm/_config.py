"""Schedule selection shared by the persistent 3WG grouped-GEMM kernels.

Both kernels of this family — the plain grouped GEMM and the MoE fused-activation
variant — choose their tile schedule from one fact: how many routed rows land on a
local expert. Few rows per expert (decode) leave the default tiles half empty and
want a shorter, narrower schedule; many rows (prefill) want the default.

Measured on H200 (SM90), bf16, both GEMM roles of the MoE pipeline, comparing the
default schedule against the decode one:

    rows/expert   role      default    decode
    32            gate_up   2.0466 ms  1.9032 ms      decode 7.0% faster
    32            down      1.0224 ms  0.9817 ms      decode 4.0% faster
    16            gate_up   4.0370 ms  3.7484 ms      decode 7.1% faster
    16            down      1.9877 ms  1.9150 ms      decode 3.7% faster
    256           gate_up   3.0204 ms  4.9874 ms      default 39.4% faster
    256           down      1.6418 ms  2.6102 ms      default 37.1% faster

Two things follow. The regime is a property of the shape, not of which GEMM asks:
both roles move the same direction at both ends, so the selector takes no role.
And the thresholds are the largest row counts measured decode-positive, not a
margin below them: applying the decode schedule outside its regime costs far more
(34-39%) than staying in it gains (4-7%), so the rule is not to extrapolate above
what was measured, while keeping the win at the boundary itself.
"""

__all__ = [
    "ASSUMED_SM_COUNT",
    "DECODE_MAX_ROWS_PER_EXPERT",
    "DECODE_SPARSE_MAX_ROWS_PER_EXPERT",
    "decode_regime",
    "launches_enough_ctas",
]

#: Multiprocessors assumed when no device has been probed — the count on the SM90
#: parts these schedules were tuned on. An op is constructed wherever it is
#: imported, so a caller deciding at construction time has no device to ask; a
#: smaller part than this simply misses the decode schedule, which costs
#: performance rather than correctness.
ASSUMED_SM_COUNT = 132

#: Largest rows-per-expert measured faster under a decode schedule.
DECODE_MAX_ROWS_PER_EXPERT = 32

#: At or below this the tile is too thin for the cooperative split; the kernels
#: that offer a sparse schedule use it here.
DECODE_SPARSE_MAX_ROWS_PER_EXPERT = 16


def decode_regime(numel: int, num_experts: int) -> str | None:
    """Which decode schedule family this routing shape wants, if any.

    Args:
        numel: Routed rows in total (tokens x top_k).
        num_experts: Local experts the rows are spread over.

    Returns:
        ``"sparse"``, ``"dense"``, or ``None`` for shapes outside the decode
        regime, which keep the default schedule.
    """
    rows_per_expert = numel / max(1, num_experts)
    if rows_per_expert <= DECODE_SPARSE_MAX_ROWS_PER_EXPERT:
        return "sparse"
    if rows_per_expert <= DECODE_MAX_ROWS_PER_EXPERT:
        return "dense"
    return None


def launches_enough_ctas(
    numel: int, n: int, block_m: int, block_n: int, sm_count: int,
) -> bool:
    """Whether the schedule still fills the device when routing is worst-case.

    The lower bound assumes every routed row lands on one local expert, which is
    the fewest CTA tiles this shape can produce. Two full waves is what the
    persistent kernels need before their occupancy pays for the tail.
    """
    cta_tiles = ((numel + block_m - 1) // block_m) * ((n + block_n - 1) // block_n)
    return cta_tiles >= 2 * sm_count
