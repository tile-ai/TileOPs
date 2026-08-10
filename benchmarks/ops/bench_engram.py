"""Benchmarks for the Engram gate-conv and decode ops.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from each op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.

One ``test_*_bench`` per op, so the validator's L4 AST check can tie each
``load_workloads("<OpName>")`` call to its manifest entry.
"""

import pytest
import torch

from benchmarks.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workload_field_params,
)
from tileops.manifest import load_workloads
from tileops.ops.engram import EngramGateConvBwdOp, EngramGateConvFwdOp
from tileops.ops.engram_decode import EngramDecodeOp
from workloads.engram import (
    EngramDecodeWorkload,
    EngramGateConvBwdWorkload,
    EngramGateConvFwdWorkload,
)

# Autotuning is a bench-run policy, not a workload property; manifest
# workloads do not carry it.
_TUNE = True


_ENGRAM_GATE_CONV_FWD_OP = "EngramGateConvFwdOp"
_ENGRAM_GATE_CONV_FWD_PARAMS = workload_field_params(
    load_workloads(_ENGRAM_GATE_CONV_FWD_OP), ("M", "seq_len", "d", "dtype"),
)


@pytest.mark.parametrize("M, seq_len, d, dtype", _ENGRAM_GATE_CONV_FWD_PARAMS)
def test_engram_gate_conv_fwd_bench(M, seq_len, d, dtype):
    test = EngramGateConvFwdWorkload(M, seq_len, d, dtype)
    inputs = test.gen_inputs()

    op = EngramGateConvFwdOp(M, seq_len, d, tune=_TUNE)
    bm = ManifestBenchmark(_ENGRAM_GATE_CONV_FWD_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


_ENGRAM_GATE_CONV_BWD_OP = "EngramGateConvBwdOp"
_ENGRAM_GATE_CONV_BWD_PARAMS = workload_field_params(
    load_workloads(_ENGRAM_GATE_CONV_BWD_OP), ("M", "seq_len", "d", "dtype"),
)


@pytest.mark.parametrize("M, seq_len, d, dtype", _ENGRAM_GATE_CONV_BWD_PARAMS)
def test_engram_gate_conv_bwd_bench(M, seq_len, d, dtype):
    test = EngramGateConvBwdWorkload(M, seq_len, d, dtype)
    inputs = test.gen_inputs()

    op = EngramGateConvBwdOp(M, seq_len, d, tune=_TUNE)
    bm = ManifestBenchmark(_ENGRAM_GATE_CONV_BWD_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    @torch.enable_grad()
    def ref_with_grad(*args):
        return test.ref_program(*args)

    result_bl = bm.profile(ref_with_grad, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


_ENGRAM_DECODE_OP = "EngramDecodeOp"
_ENGRAM_DECODE_PARAMS = workload_field_params(
    load_workloads(_ENGRAM_DECODE_OP),
    ("batch", "d_mem", "d", "max_conv_len", "conv_kernel_size", "dilation", "dtype"),
)


@pytest.mark.parametrize(
    "batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype",
    _ENGRAM_DECODE_PARAMS,
)
def test_engram_decode_bench(batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype):
    test = EngramDecodeWorkload(batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype)
    inputs = test.gen_inputs()

    op = EngramDecodeOp(
        batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, tune=_TUNE,
    )
    bm = ManifestBenchmark(_ENGRAM_DECODE_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
