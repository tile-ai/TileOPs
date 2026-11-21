"""How the quantization family's benchmarks call their ops."""

from typing import Any

from benchmarks._cases import Entry, default_binder
from benchmarks.api import Implementation
from benchmarks.baselines import private_inputs


def _quantize_binder(op: Any, case: Any) -> Implementation:
    """The op on private copies of what it writes: ``x`` where it fake-quantizes ``x`` in
    place, otherwise any ``out_q`` the call passes after ``x``."""
    positions = (0,) if case.arguments["inplace"] else range(1, len(case.inputs))
    return private_inputs(default_binder(op, case).run, case.inputs, *positions)


ENTRIES = {
    "FP4QuantPerGroupFwdOp": Entry(binder=_quantize_binder),
    "FP8QuantPerTokenGroupFwdOp": Entry(binder=_quantize_binder),
}
