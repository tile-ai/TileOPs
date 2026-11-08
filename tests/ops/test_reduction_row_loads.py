"""The L2 policy of Softmax's input-row loads (``keeps_input_rows``): evict-last where a
kernel reads each input row once into one CTA's registers and writes an output row for it,
on an input past L2; the default load on an input within L2.

The inputs past L2 take 128 MB, past the L2 of any supported device."""

import re

import pytest
import torch
from tilelang.jit.kernel import JITKernel

from tests.workload_test_base import TestBase
from tileops._kernels.kernel_base import CompiledProgram
from tileops._ops.reduction.softmax import SoftmaxFwdOp
from workloads.reduction import SoftmaxCase


class SoftmaxTest(SoftmaxCase, TestBase):
    pass


def _launched_sources(monkeypatch) -> list:
    """The CUDA source of every program launched or built from here on."""
    sources = []
    call, build = JITKernel.__call__, CompiledProgram.__init__

    def launched(self, *args, **kwargs):
        sources.append(self.get_kernel_source())
        return call(self, *args, **kwargs)

    def built(self, jit):
        sources.append(jit.get_kernel_source())
        build(self, jit)

    monkeypatch.setattr(JITKernel, "__call__", launched)
    monkeypatch.setattr(CompiledProgram, "__init__", built)
    return sources


def _run(monkeypatch, shape, dtype, kernel) -> list:
    """The CUDA sources of the programs Softmax runs on *shape*, which must be *kernel*'s;
    the output is checked against the reference."""
    sources = _launched_sources(monkeypatch)
    test = SoftmaxTest(shape, dtype)
    op = SoftmaxFwdOp(dim=-1)
    test.check(op, *test.gen_inputs())
    assert [type(k).__name__ for k in op.iter_kernels()] == [kernel]
    assert sources
    return sources


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "shape, dtype, kernel",
    [
        pytest.param((131072, 512), torch.float16, "SoftmaxWarpRowsKernel", id="warp-rows"),
        pytest.param((32768, 2048), torch.bfloat16, "SoftmaxKernel", id="held-row"),
    ],
)
def test_rows_past_l2_load_evict_last(monkeypatch, shape, dtype, kernel) -> None:
    """Every 16-byte input-row load, on an input past L2, is evict-last in L2."""
    for source in _run(monkeypatch, shape, dtype, kernel):
        reads = [line for line in source.splitlines() if re.search(r"\bx\[|\(x \+ ", line)]
        assert reads
        assert all("tileops::load16_evict_last(" in line for line in reads)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "shape, dtype, kernel",
    [
        pytest.param((1024, 512), torch.float16, "SoftmaxWarpRowsKernel", id="warp-rows"),
        pytest.param((1024, 2048), torch.bfloat16, "SoftmaxKernel", id="held-row"),
    ],
)
def test_rows_within_l2_keep_the_default_load(monkeypatch, shape, dtype, kernel) -> None:
    """On an input within L2, the input rows load without an L2 eviction hint."""
    assert not any("evict_last" in source for source in _run(monkeypatch, shape, dtype, kernel))
