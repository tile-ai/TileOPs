"""A rasterized traced kernel on the GPU: each CTA's events land in its own row."""

import pytest
import tilelang
import tilelang.language as T
import torch

from tileops.trace import trace
from workloads.device import run_device

pytestmark = [pytest.mark.smoke, pytest.mark.cuda_only]

# An 8 x 8 grid rasterized in panels of 10 rows remaps (bx, by) to (by, bx).
_GRID = 8
_EVENTS = 4
_MAX_EVENTS = 16


@tilelang.jit(out_idx=trace.out_idx(1, traced=True))
def _swizzled():
    @T.prim_func
    def main(out: T.Tensor((_GRID * _GRID,), "int32")):
        with T.Kernel(_GRID, _GRID, threads=128) as (bx, by):
            T.use_swizzle(10)
            tx = T.get_thread_binding()
            for i in T.serial(_EVENTS):
                trace.record("e", payload=i)
            if tx == 0:
                out[by * _GRID + bx] = 1

    return trace.finalize(main, traced=True, max_events=_MAX_EVENTS)


def test_every_cta_of_a_swizzled_kernel_fills_exactly_its_own_row():
    shape = (_GRID * _GRID, 1, (_MAX_EVENTS + 1) * 2)
    # A header the kernel reads before zeroing it would hold this: its low 32 bits pass the
    # count checks, and as an int64 it addresses memory far outside slots.
    poison = torch.full(shape, -(2**32), dtype=torch.int64, device=run_device())
    del poison

    out, slots = _swizzled()()

    assert torch.equal(out, torch.ones_like(out))
    assert slots[:, 0, 0].tolist() == [_EVENTS] * (_GRID * _GRID)
