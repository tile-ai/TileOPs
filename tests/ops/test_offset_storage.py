"""A call tensor off a 16-byte boundary runs as an aligned one does
(docs/design/ops-design.md § Call protocol).

The rule is decided in ``Op`` for every family. Each family's own tests hold its offset cases;
these are the families that had none: GEMM and BMM, read through TMA descriptors, and the
selective scan, read in vectors.
"""

import pytest
import torch

from tileops._ops.gemm.bmm import BMMFwdOp
from tileops._ops.gemm.gemm import GEMMFP8FwdOp, GEMMFwdOp
from tileops.mamba import SelectiveScanFwdOp
from workloads.device import run_device, run_device_available
from workloads.mamba import SelectiveScanFwdCall
from workloads.workload_base import manifest_call

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(not run_device_available(), reason="the run device is not available"),
]


def _shifted(t: torch.Tensor) -> torch.Tensor:
    """*t*'s values in a contiguous view starting one element past a 16-byte boundary."""
    view = torch.empty(t.numel() + 1, dtype=t.dtype, device=t.device)[1:].view(t.shape)
    assert view.data_ptr() % 16
    return view.copy_(t)


def _gemm():
    a = torch.randn(256, 512, device=run_device()).bfloat16()
    b = torch.randn(384, 512, device=run_device()).bfloat16()
    return GEMMFwdOp(trans_b=True), [a, b]


def _bmm():
    a = torch.randn(4, 256, 512, device=run_device()).bfloat16()
    b = torch.randn(4, 512, 384, device=run_device()).bfloat16()
    return BMMFwdOp(), [a, b]


def _gemm_fp8_block_scaled():
    a = torch.randn(256, 512, device=run_device()).to(torch.float8_e4m3fn)
    b = torch.randn(384, 512, device=run_device()).to(torch.float8_e4m3fn)
    a_scale = torch.rand(256, 4, device=run_device()) + 0.5
    b_scale = torch.rand(3, 4, device=run_device()) + 0.5
    return GEMMFP8FwdOp(), [a, b, a_scale, b_scale]


def _selective_scan():
    sizes = {"Bsz": 2, "S": 1024, "Di": 64, "N": 16, "G": 1, "delta_softplus": True}
    call = manifest_call("SelectiveScanFwdOp", {"T": "bfloat16"}, **sizes, some=["D", "z"])
    return SelectiveScanFwdOp(**call.arguments({})), list(SelectiveScanFwdCall(call).gen_inputs())


@pytest.mark.parametrize(
    "case, shifted",
    [
        pytest.param(_gemm, 0, id="gemm-a"),
        pytest.param(_gemm, 1, id="gemm-b"),
        pytest.param(_bmm, 0, id="bmm-a"),
        pytest.param(_bmm, 1, id="bmm-b"),
        pytest.param(_gemm_fp8_block_scaled, 2, id="gemm-fp8-a_scale"),
        pytest.param(_selective_scan, 0, id="selective-scan-u"),
    ],
)
def test_an_offset_operand_runs_as_an_aligned_one(case, shifted):
    op, inputs = case()
    expected = op(*inputs)
    inputs[shifted] = _shifted(inputs[shifted])
    torch.testing.assert_close(op(*inputs), expected, rtol=0, atol=0)
