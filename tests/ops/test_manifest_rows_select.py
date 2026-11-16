"""Every workload row of an implemented entry selects an implementation or is refused, and
some row of each entry selects one.

A row may declare a call no in-tree implementation serves yet, such as a pilot model's mode;
every implementation's refusal then declines it, and the op raises ``OpNotAvailableError``.
Any other error is a defect. Shapes and dtypes come from meta tensors; selection reads the
run device, and nothing is built.
"""

import pytest
import torch

from tileops._kernels.call_spec import CallSpec
from tileops._ops.op_base import Op
from tileops.backend import OpNotAvailableError
from tileops.manifest import load_adts, load_manifest
from tileops.manifest._plan import entry_plan
from tileops.manifest._registry import op_class
from tileops.manifest._workload import instantiate
from workloads.device import run_device

pytestmark = [pytest.mark.smoke, pytest.mark.in_tree_kernels, pytest.mark.sm90]

_MANIFEST = load_manifest()
_IMPLEMENTED = sorted(n for n, e in _MANIFEST.items() if e.get("status") == "implemented")


class _SelectedError(Exception):
    """Selection named an implementation."""


@pytest.mark.parametrize("name", _IMPLEMENTED)
def test_every_row_selects_an_implementation_or_is_refused(name, monkeypatch):
    device = torch.device(run_device())

    def select(self, interface, call):
        if isinstance(call, CallSpec):
            call = call._on_device(device)
        self.key_for(interface, call)
        raise _SelectedError

    monkeypatch.setattr(Op, "_entry", select)
    entry = _MANIFEST[name]
    plan = entry_plan(name, entry, load_adts())
    cls = op_class(name, entry)
    outcomes = set()
    for row in entry["workloads"]:
        for case in row.get("dtype_cases") or [{}]:
            call = instantiate(plan, row, case)
            tensors = call.materialize("meta")
            args, keywords = call.signature.forward_arguments(
                tensors[t] for t in call.signature.inputs
            )
            try:
                # A composite op whose stages complete on meta selects nothing itself.
                cls(**call.arguments()).forward(*args, **keywords)
                outcomes.add("ran")
            except _SelectedError:
                outcomes.add("selected")
            except OpNotAvailableError:
                outcomes.add("refused")
    assert outcomes != {"refused"}, f"{name}: every implementation refuses every row"
