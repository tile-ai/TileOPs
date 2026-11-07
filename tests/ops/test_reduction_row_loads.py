"""The L2 policy of the input-row loads of the reductions built on ``reduction._primitives``:
evict-last where a kernel reads each input row once into one CTA's registers and writes an
output row for it, on an input past L2 (``keeps_input_rows``); the default load everywhere
else.

The inputs that exceed L2 take 128 MB, past the L2 of any supported device."""

import re

import pytest
import torch
from tilelang.jit.kernel import JITKernel

from tileops._kernels.kernel_base import CompiledProgram
from tileops._ops.reduction.argreduce import ArgmaxFwdOp
from tileops._ops.reduction.logical_reduce import AnyFwdOp
from tileops._ops.reduction.reduce import ProdFwdOp, SumFwdOp
from tileops._ops.reduction.softmax import LogSoftmaxFwdOp, LogSumExpFwdOp, SoftmaxFwdOp
from workloads.device import run_device


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


def _run(monkeypatch, make, shape, dtype, kernel) -> list:
    """The CUDA sources of the programs one call of *make()* on *shape* runs, which must be
    *kernel*'s."""
    sources = _launched_sources(monkeypatch)
    op = make()
    op(torch.randn(shape, device=run_device()).to(dtype))
    assert [type(k).__name__ for k in op.iter_kernels()] == [kernel]
    assert sources
    return sources


def _softmax(dim=-1):
    return SoftmaxFwdOp(dim=dim)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "make, shape, dtype, kernel",
    [
        pytest.param(
            _softmax, (131072, 512), torch.float16, "SoftmaxWarpRowsKernel", id="warp-rows"
        ),
        pytest.param(
            lambda: LogSoftmaxFwdOp(dim=-1),
            (131072, 512),
            torch.float16,
            "SoftmaxWarpRowsKernel",
            id="log-softmax-warp-rows",
        ),
        pytest.param(_softmax, (32768, 2048), torch.bfloat16, "SoftmaxKernel", id="held-row"),
    ],
)
def test_register_held_rows_past_l2_load_evict_last(
    monkeypatch, make, shape, dtype, kernel
) -> None:
    """Every 16-byte input-row load of a kernel reading each row once into one CTA's
    registers and writing an output row for it, on an input past L2, is evict-last in L2."""
    sources = _run(monkeypatch, make, shape, dtype, kernel)
    for source in sources:
        # Every line reading the input ``x`` reads it through the hinted load.
        reads = [line for line in source.splitlines() if re.search(r"\bx\[|\(x \+ ", line)]
        assert reads
        assert all("tileops::load16_evict_last(" in line for line in reads)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "make, shape, dtype, kernel",
    [
        # Row-writing kernels on an input within L2, splitting a row across a CTA cluster, or
        # reading a row twice.
        pytest.param(
            _softmax, (1024, 512), torch.float16, "SoftmaxWarpRowsKernel", id="warp-rows-in-l2"
        ),
        pytest.param(
            _softmax,
            (8, 65536),
            torch.bfloat16,
            "SoftmaxClusterSplitKernel",
            id="cluster-split-in-l2",
            marks=pytest.mark.sm90,
        ),
        pytest.param(_softmax, (1024, 2048), torch.bfloat16, "SoftmaxKernel", id="held-row-in-l2"),
        pytest.param(
            _softmax,
            (2048, 32768),
            torch.bfloat16,
            "SoftmaxOnChipKernel",
            id="on-chip-split-row",
            marks=pytest.mark.sm90,
        ),
        pytest.param(
            _softmax, (688, 100000), torch.bfloat16, "SoftmaxStreamingKernel", id="streaming"
        ),
        # Kernels folding each row to one value.
        pytest.param(
            lambda: SumFwdOp(dim=-1), (1024, 4096), torch.bfloat16, "ReduceFoldKernel", id="sum"
        ),
        pytest.param(
            lambda: ProdFwdOp(dim=0),
            (4, 128, 4096),
            torch.float16,
            "ReduceLeadingKernel",
            id="prod-leading",
        ),
        pytest.param(
            lambda: ArgmaxFwdOp(dim=-1),
            (1024, 4096),
            torch.bfloat16,
            "ArgreduceKernel",
            id="argmax",
        ),
        pytest.param(
            lambda: AnyFwdOp(dim=-1), (1024, 4096), torch.float16, "LogicalReduceKernel", id="any"
        ),
        pytest.param(
            lambda: LogSumExpFwdOp(dim=-1),
            (4096, 16384),
            torch.bfloat16,
            "LogSumExpStreamingKernel",
            id="logsumexp-streaming",
        ),
    ],
)
def test_other_reductions_keep_the_default_load(monkeypatch, make, shape, dtype, kernel) -> None:
    """Every other reduction kernel reads its input without an L2 eviction hint."""
    sources = _run(monkeypatch, make, shape, dtype, kernel)
    assert not any("evict_last" in source for source in sources)
