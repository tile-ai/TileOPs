"""Reduction kernels validate the config they are built with."""

import pytest
import torch

from tileops.kernels.reduction.logical_reduce import LogicalReduceKernel
from workloads.device import run_device_available


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_reduce_caller_tile_n_validated() -> None:
    """A caller-chosen tile_n is accepted only if it actually builds.

    Without the check the width reaches TileLang and surfaces as an ICHECK on
    min_reg_num, which names nothing the caller can act on.  128 threads at
    fp16 is a 1024-column thread-block pass; a width must divide it or be a
    multiple of it.
    """
    from tileops.kernels.reduction.call_spec import ReduceCall
    from tileops.kernels.reduction.reduce import ReduceKernel

    m, n, dtype = 8, 102400, torch.float16
    x = torch.randn(m, n, dtype=dtype, device="cuda")
    call = ReduceCall(device=x.device, shape=(m, n), axes=(1,), op_kind="sum", dtype=dtype)

    def run(tile_n: int, block_m: int = 2) -> None:
        kernel = ReduceKernel(call, config={"block_m": block_m, "threads": 128, "tile_n": tile_n})
        kernel.forward(x)  # construction defers the build; forward triggers it

    for accepted in (512, 1024, 2048):  # divides the pass, or a multiple of it
        run(accepted)
    run(1536, block_m=1)  # one row cannot shift, so nothing constrains it
    run(0)  # 0 is the "derive it for me" sentinel, not a width

    for rejected, why in (
        (768, "neither divides nor is a multiple"),
        (1536, "neither divides nor is a multiple"),
        (257, "must be positive and a multiple"),
        (65536, "exceeds"),
    ):
        with pytest.raises(ValueError, match=why):
            run(rejected)


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="CUDA required")
def test_logical_reduce_rejects_width_its_reduction_cannot_fold() -> None:
    """A block width that is not a power of two would drop warps from the reduction."""
    kernel = LogicalReduceKernel(
        (4, 4096), (1,), "count_nonzero", torch.float16, config={"threads": 96}
    )
    with pytest.raises(ValueError, match="power of two"):
        kernel(torch.ones(4, 4096, dtype=torch.float16, device="cuda"))
