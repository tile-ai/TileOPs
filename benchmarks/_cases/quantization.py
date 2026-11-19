"""How the quantization family's benchmarks call their ops."""

from typing import Any

from benchmarks._cases import Entry, default_binder
from benchmarks.api import Implementation
from benchmarks.baselines import private_inputs


def _fp8_per_token_group_binder(op: Any, case: Any) -> Implementation:
    """The op on a private ``out_q`` it writes the codes into, where the call passes one, and
    on a private ``x`` where it fake-quantizes ``x`` in place."""
    if case.arguments["inplace"]:
        return private_inputs(default_binder(op, case).run, case.inputs, 0)
    return private_inputs(default_binder(op, case).run, case.inputs, *range(1, len(case.inputs)))


def _fp4_per_group_binder(op: Any, case: Any) -> Implementation:
    """The op on a private ``x`` where it fake-quantizes ``x`` in place."""
    if case.arguments["inplace"]:
        return private_inputs(default_binder(op, case).run, case.inputs, 0)
    return default_binder(op, case)


ENTRIES = {
    "FP4QuantPerGroupFwdOp": Entry(binder=_fp4_per_group_binder),
    "FP8QuantPerTokenGroupFwdOp": Entry(binder=_fp8_per_token_group_binder),
}
