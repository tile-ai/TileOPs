"""Case factories of the mhc family."""

import torch

from benchmarks._cases import Entry
from workloads.sequence_modeling.mhc import MHCPostWorkload, MHCPreWorkload


def _pre(call) -> MHCPreWorkload:
    # The scaling params come from the manifest row, not from the generator.
    return MHCPreWorkload(
        call.indices["B"],
        call.indices["n"],
        call.indices["c_x"],
        getattr(torch, call.tensors["x"][1]),
        **call.arguments({}),
    )


def _post(call) -> MHCPostWorkload:
    return MHCPostWorkload(
        call.indices["B"],
        call.indices["n"],
        call.indices["c_x"],
        getattr(torch, call.tensors["x_res"][1]),
    )


ENTRIES = {
    "MHCPreFwdOp": Entry(_pre),
    "MHCPostFwdOp": Entry(_post),
}
