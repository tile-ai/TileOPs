"""How the linear_attention family's benchmarks call their ops."""

from typing import Any

import torch

from benchmarks._cases import Entry, default_binder
from benchmarks.api import Implementation
from benchmarks.baselines import private_inputs
from workloads.linear_attention.deltanet import DeltaNetChunkwiseCall
from workloads.linear_attention.gla import GLAChunkwiseCall


def _deltanet_bwd_inputs(workload: DeltaNetChunkwiseCall) -> tuple:
    """The backward's inputs with the saved tensors a forward run produces for them."""
    from tileops.linear_attention import DeltaNetChunkFwdOp

    do, q, k, v, beta, *_saved = workload.gen_inputs()
    fwd_op = DeltaNetChunkFwdOp(workload.arguments()["chunk_size"])
    _o, h, Aw, Au, w, u = fwd_op(q, k, v, beta)
    return do, q, k, v, beta, h, Aw, Au, w, u


def _gla_bwd_inputs(workload: GLAChunkwiseCall) -> tuple:
    """The backward's inputs with the chunk states a forward run produces and a zero final-state
    gradient; the forward starts from a random state when the backward has one."""
    from tileops.linear_attention import GLAChunkFwdOp

    do, _dfinal_state, q, k, v, g, _h = workload.gen_inputs()
    arguments = workload.arguments()
    fwd_op = GLAChunkFwdOp(arguments["chunk_size"], arguments["scale"])
    initial_state = (
        torch.randn_like(_dfinal_state) * 0.1 if arguments["has_initial_state"] else None
    )
    _o, _final_state, h = fwd_op(q, k, v, g, initial_state)
    dfinal_state = torch.zeros_like(_dfinal_state)
    return do, dfinal_state, q, k, v, g, h


def _state_pool_binder(op: Any, case: Any) -> Implementation:
    """The op; given ``state_indices``, on a private copy of the state pool it updates in place."""
    implementation = default_binder(op, case)
    if case.inputs[10] is None:
        return implementation
    return private_inputs(implementation.run, case.inputs, 5)


ENTRIES = {
    "DeltaNetChunkBwdOp": Entry(inputs=_deltanet_bwd_inputs),
    "GDNFwdOp": Entry(binder=_state_pool_binder),
    "GLAChunkBwdOp": Entry(inputs=_gla_bwd_inputs),
}
