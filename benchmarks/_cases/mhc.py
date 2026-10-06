"""Case factories of the mhc family."""

import torch

from benchmarks._cases import Entry
from workloads.sequence_modeling.mhc import MHCPostWorkload, MHCPreWorkload


def _pre(call) -> MHCPreWorkload:
    # The manifest workload is the authority for the scaling params, so the case
    # is built with them rather than with the ones the generator would draw.
    return MHCPreWorkload(
        call.ix["B"],
        call.ix["n"],
        call.ix["c_x"],
        getattr(torch, call.tensors["x"][1]),
        **call.arguments({}),
    )


def _post(call) -> MHCPostWorkload:
    return MHCPostWorkload(
        call.ix["B"], call.ix["n"], call.ix["c_x"], getattr(torch, call.tensors["x_res"][1])
    )


ENTRIES = {
    "MHCPreFwdOp": Entry(_pre),
    "MHCPostFwdOp": Entry(_post),
}
