"""How the rope family's benchmarks call their ops."""

from benchmarks._cases import Entry, default_binder
from benchmarks.api import Implementation
from benchmarks.baselines import private_inputs


def _rope_cos_sin(op, case) -> Implementation:
    """The op on a private ``x`` where it rotates ``x`` in place."""
    if not case.arguments["inplace"]:
        return default_binder(op, case)
    return private_inputs(op, case.inputs, 0)


ENTRIES = {"RoPECosSinFwdOp": Entry(binder=_rope_cos_sin)}
