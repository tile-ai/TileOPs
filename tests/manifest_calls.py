"""The manifest ops and calls the whole-manifest target tests run."""

import math

from tileops.manifest import load_manifest
from tileops.manifest._plan import EntryPlan
from tileops.manifest._workload import Call, instantiate

__all__ = ["implemented_ops", "smallest_call"]


def implemented_ops() -> list[str]:
    """The names of the manifest's implemented ops, sorted."""
    return sorted(n for n, e in load_manifest().items() if e.get("status") == "implemented")


def smallest_call(plan: EntryPlan, rows: list, *, metadata: bool = True) -> Call:
    """The call of *rows* (workload rows of *plan*'s entry) whose tensors hold the fewest
    elements, over every dtype case."""
    calls = (
        instantiate(plan, row, case, metadata=metadata)
        for row in rows
        for case in row.get("dtype_cases") or [{}]
    )
    return min(calls, key=lambda call: sum(math.prod(s) for s, _ in call.tensors.values()))
