"""Call workloads of the indexing family."""

import torch

from workloads.indexing import ConcatWorkload


def _concat(call) -> ConcatWorkload:
    shapes, dtypes = zip(*call.tensors["tensors"].items, strict=True)
    return ConcatWorkload(shapes, call.arguments()["dim"], [getattr(torch, d) for d in dtypes])


WORKLOADS = {
    "ConcatFwdOp": _concat,
}
