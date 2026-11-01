"""How the elementwise family's benchmarks call their ops."""

from benchmarks._cases import Entry

ENTRIES = {
    # The reference clones then fills in place; the clone is a device copy and counts.
    "MaskedFillTensorFwdOp": Entry(count_copies=True),
    "MaskedFillScalarFwdOp": Entry(count_copies=True),
}
