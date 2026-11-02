"""How the GEMM family's benchmarks call their ops."""

from benchmarks._cases import Entry
from benchmarks.api import Implementation


def _prepacked_weight(op, case) -> Implementation:
    """The op called on its own repack of the case's row-major weight, made once, untimed."""
    activation, packed_weight, *scales = case.inputs
    return Implementation(run=op, args=(activation, op.repack(packed_weight), *scales))


ENTRIES = {
    "GEMMFP8FwdOp": Entry(count_copies=True),
    "GEMMW4A8FwdOp": Entry(binder=_prepacked_weight),
}
