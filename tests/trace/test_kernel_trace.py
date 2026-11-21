"""Traced TileLang kernels run on the GPU: each CTA's row, and the payloads its events carry."""

import pytest
import tilelang
import tilelang.language as T
import torch

from tileops.trace import trace
from workloads.device import run_device

pytestmark = pytest.mark.cuda_only("cuda-kernel")

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


@pytest.mark.smoke
def test_every_cta_of_a_swizzled_kernel_fills_exactly_its_own_row():
    shape = (_GRID * _GRID, 1, (_MAX_EVENTS + 1) * 2)
    # A header the kernel reads before zeroing it would hold this: its low 32 bits pass the
    # count checks, and as an int64 it addresses memory far outside slots.
    poison = torch.full(shape, -(2**32), dtype=torch.int64, device=run_device())
    del poison

    out, slots = _swizzled()()

    assert torch.equal(out, torch.ones_like(out))
    assert slots[:, 0, 0].tolist() == [_EVENTS] * (_GRID * _GRID)


@pytest.fixture
def preserve_trace_state():
    """Restore the trace switch and output directory a test changes."""
    original_enabled = trace.enabled
    original_output = trace.output
    try:
        yield
    finally:
        if original_enabled:
            trace.enable(output=original_output)
        else:
            trace.disable()


def _traced_events(kernel, expected: torch.Tensor, name: str) -> list:
    """Run *kernel*, check its output against *expected*, and decode the events named *name*."""
    result = kernel()
    assert isinstance(result, (tuple, list)), "a traced kernel returns (output, slots)"
    output, slots = result
    assert torch.allclose(output, expected)
    return [e for e in trace.decode(kernel, slots) if e.name == name]


@pytest.mark.full
def test_ranges_with_and_without_a_payload_build_with_tracing_off(preserve_trace_state):
    trace.disable()

    @tilelang.jit(out_idx=trace.out_idx(1))
    def build():
        @T.prim_func
        def kernel(out: T.Tensor((16,), "float32")):
            with T.Kernel(1, threads=16):
                with trace.range("test", payload=0):
                    pass
                with trace.range("test2"):
                    pass
                with trace.range("test3", lane="compute"):
                    pass

        return trace.finalize(kernel)

    assert build() is not None


@pytest.mark.full
def test_payload_with_range_start_end(preserve_trace_state, tmp_path):
    trace.enable(output=str(tmp_path))

    @tilelang.jit(out_idx=trace.out_idx(1))
    def build():
        @T.prim_func
        def kernel(out: T.Tensor((16,), "float32")):
            with T.Kernel(1, threads=16):
                tx = T.get_thread_binding()
                tok = trace.range_start("test_range", payload=42)
                out[tx] = T.float32(tx)
                trace.range_end(tok)

        return trace.finalize(kernel)

    expected = torch.arange(16, dtype=torch.float32, device="cuda")
    slices = _traced_events(build(), expected, "test_range")
    assert [s.payload for s in slices] == [42]


@pytest.mark.full
def test_a_range_records_its_payload_in_a_kernel_binding_no_thread_index(
    preserve_trace_state, tmp_path
):
    """``tileops::trace_thread_idx_x`` elects the thread that writes the markers."""
    trace.enable(output=str(tmp_path))

    @tilelang.jit(out_idx=trace.out_idx(1))
    def build():
        @T.prim_func
        def kernel(out: T.Tensor((16,), "float32")):
            # T.Kernel(..., threads=16) binds no threadIdx.x.
            with T.Kernel(1, threads=16):
                tx = T.get_thread_binding()
                with trace.range("test_range", payload=42):
                    out[tx] = T.float32(tx)

        return trace.finalize(kernel)

    expected = torch.arange(16, dtype=torch.float32, device="cuda")
    slices = _traced_events(build(), expected, "test_range")
    assert [s.payload for s in slices] == [42]


@pytest.mark.full
def test_a_payload_may_be_a_runtime_expression(preserve_trace_state, tmp_path):
    """The loop index tags each range: a payload is what the caller passes, not an index the
    tracer recovers."""
    trace.enable(output=str(tmp_path))

    @tilelang.jit(out_idx=trace.out_idx(1))
    def build():
        @T.prim_func
        def kernel(out: T.Tensor((4,), "float32")):
            with T.Kernel(1, threads=4):
                tx = T.get_thread_binding()
                for i in range(4):
                    with trace.range("loop_iter", payload=i):
                        if tx == 0:
                            out[i] = T.float32(i)

        return trace.finalize(kernel)

    expected = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32, device="cuda")
    slices = _traced_events(build(), expected, "loop_iter")
    assert len(slices) >= 4, f"expected at least 4 loop_iter slices, got {len(slices)}"
    assert sorted(s.payload for s in slices[:4]) == [0, 1, 2, 3]
