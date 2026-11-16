"""The indexing ops, at the public path ``tileops.indexing``.

Ops that gather, scatter or join tensor elements by index or mask.
"""

from tileops._ops.indexing import ConcatFwdOp

__all__ = [
    "ConcatFwdOp",
]
