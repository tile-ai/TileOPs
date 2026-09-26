"""Benchmarks for the MHC pre/post ops.

Workload shapes and the pre-op scaling params come from the ops manifest; roofline
FLOP and byte counts come from each op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.
"""

import pytest
import torch

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops import MHCPostFwdOp, MHCPreFwdOp
from workloads.mhc import MHCPostWorkload, MHCPreWorkload

# Autotuning is a bench-run policy, not a workload property; manifest
# workloads do not carry it.
_TUNE = True


@pytest.mark.parametrize("call", manifest_calls(MHCPreFwdOp))
def test_mhc_pre_bench(call) -> None:
    # The manifest workload is the authority for the scaling params, so the case
    # is built with them rather than with the ones the generator would draw.
    params = call.arguments({})
    test = MHCPreWorkload(
        call.ix["B"],
        call.ix["n"],
        call.ix["c_x"],
        getattr(torch, call.tensors["x"][1]),
        **params,
    )
    inputs = test.gen_inputs()

    op = MHCPreFwdOp(**params, tune=_TUNE)
    bm = ManifestBenchmark(op, test)

    bm.compare(
        {
            "tileops": op,
            "torch-ref": test.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(test.ref_program),
        },
        *inputs,
    )


@pytest.mark.parametrize("call", manifest_calls(MHCPostFwdOp))
def test_mhc_post_bench(call) -> None:
    test = MHCPostWorkload(
        call.ix["B"], call.ix["n"], call.ix["c_x"], getattr(torch, call.tensors["x_res"][1])
    )
    inputs = test.gen_inputs()

    op = MHCPostFwdOp(**call.arguments({}), tune=_TUNE)
    bm = ManifestBenchmark(op, test)

    bm.compare(
        {
            "tileops": op,
            "torch-ref": test.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(test.ref_program),
        },
        *inputs,
    )
