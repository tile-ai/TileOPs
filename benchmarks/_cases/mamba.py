"""How the mamba family's benchmarks call their ops."""

from typing import Any

from benchmarks._cases import Entry
from benchmarks.api import Implementation
from benchmarks.baselines import private_inputs
from workloads.mamba import conv_state_result, ssd_decode_result


def _ssd_decode_binder(op: Any, case: Any) -> Implementation:
    """The op on a private copy of the recurrent state it updates, returning output and state."""
    return private_inputs(lambda *args: ssd_decode_result(op, *args), case.inputs, 5)


def _conv_state_binder(op: Any, case: Any) -> Implementation:
    """The op on a private copy of the convolution state it overwrites, returning output and
    state."""
    return private_inputs(lambda *args: conv_state_result(op, *args), case.inputs, 2)


ENTRIES = {
    "CausalConv1dFwdOp": Entry(binder=_conv_state_binder),
    "CausalConv1dUpdateFwdOp": Entry(binder=_conv_state_binder),
    "SSDRecurrentFwdOp": Entry(binder=_ssd_decode_binder),
}
