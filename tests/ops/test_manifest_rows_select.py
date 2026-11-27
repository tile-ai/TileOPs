"""Every workload row of an implemented entry selects an implementation or is refused, and
some row of each entry selects one.

A row may declare a call no in-tree implementation serves yet, such as a pilot model's mode;
every implementation's refusal then declines it, and the op raises ``OpNotAvailableError``.
Any other error is a defect. Shapes and dtypes come from meta tensors; selection reads the
run device, and nothing is built. Separately, a few manifest rows run on the kernel the SM90
selection picks for them, against their workload references.
"""

import pytest
import torch

from tests.manifest_calls import implemented_ops
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


class _SelectedError(Exception):
    """Selection named an implementation."""


@pytest.mark.parametrize("name", implemented_ops())
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


@pytest.mark.parametrize(
    "name,case_id,expected_kernel,config,program_checks",
    [
        ("DeltaNetRecurrentFwdOp", "serving-b1-h8-bfloat16", "DeltaNetDecodeColumnKernel", {}, ()),
        (
            "GEMMFP8FwdOp",
            "qwen3-8b-o-tp8-b8-float8_e4m3fn-bfloat16",
            "GEMMFP8SwapABKernel",
            {},
            ("_uses_short_panels",),
        ),
        ("BMMFwdOp", "square-128-bfloat16", "BMMKernel", {"block_k": 64, "threads": 64}, ()),
        ("BMMFwdOp", "square-128-float16", "BMMKernel", {"block_k": 64, "threads": 64}, ()),
        (
            "SSDChunkStateFwdOp",
            "mamba2-780m-bfloat16-float32",
            "SSDChunkStateFwdKernel",
            {},
            ("_uses_tma_panels",),
        ),
    ],
)
def test_sm90_workload_selects_and_runs_its_specialized_kernel(
    name, case_id, expected_kernel, config, program_checks
):
    """Row *case_id* selects *expected_kernel* with *config* and matches its reference."""
    from tileops.backend import BUILTIN
    from workloads.calls import call_workload
    from workloads.numerics import compare_outputs

    entry = _MANIFEST[name]
    plan = entry_plan(name, entry, load_adts())
    calls = (
        instantiate(plan, row, dtype_case)
        for row in entry["workloads"]
        for dtype_case in row.get("dtype_cases") or [{}]
    )
    call = next(c for c in calls if c.case_id == case_id)
    workload = call_workload(name)(call)
    arguments = workload.arguments() if hasattr(workload, "arguments") else call.arguments()
    op = op_class(name, entry)(**arguments, target=BUILTIN)
    inputs = workload.gen_inputs()
    expected = workload.ref_program(*inputs)
    args, keywords = call.signature.forward_arguments(inputs)
    compare_outputs(op(*args, **keywords), expected, workload.verification(*inputs))
    (kernel,) = op.iter_kernels()
    assert type(kernel).__name__ == expected_kernel
    assert (kernel.config or {}).items() >= config.items()
    for check in program_checks:
        assert getattr(kernel, check)(), check
