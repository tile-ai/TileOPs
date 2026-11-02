"""How the pool family's benchmarks call their ops."""

from benchmarks._cases import Entry

ENTRIES = {
    "MeanPoolingFwdOp": Entry(count_copies=True),
}
