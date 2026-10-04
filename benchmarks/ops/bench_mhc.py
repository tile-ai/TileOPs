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
from workloads.numerics import Exact
from workloads.sequence_modeling.mhc import MHCPostWorkload, MHCPreWorkload

# Autotuning is a bench-run policy, not a workload property; manifest
# workloads do not carry it.
_TUNE = True


@pytest.mark.parametrize("call", manifest_calls(MHCPreFwdOp))
def test_mhc_pre_bench(call) -> None:
    # The manifest workload is the authority for the scaling params, so the case
    # is built with them rather than with the ones the generator would draw.
    params = call.arguments({})
    workload = MHCPreWorkload(
        call.ix["B"],
        call.ix["n"],
        call.ix["c_x"],
        getattr(torch, call.tensors["x"][1]),
        **params,
    )
    inputs = workload.gen_inputs()

    op = MHCPreFwdOp(**params, tune=_TUNE)
    bm = ManifestBenchmark(op, workload)

    bm.compare(
        {
            "tileops": op,
            "torch-ref": workload.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
        },
        *inputs,
        # The MHC unit-test bound includes low-precision projection intermediates.
        evidence={tag: Exact(atol=1e-2, rtol=1e-2) for tag in ("tileops", TORCH_COMPILE_TAG)},
    )


@pytest.mark.parametrize("call", manifest_calls(MHCPostFwdOp))
def test_mhc_post_bench(call) -> None:
    workload = MHCPostWorkload(
        call.ix["B"], call.ix["n"], call.ix["c_x"], getattr(torch, call.tensors["x_res"][1])
    )
    inputs = workload.gen_inputs()

    op = MHCPostFwdOp(**call.arguments({}), tune=_TUNE)
    bm = ManifestBenchmark(op, workload)

    bm.compare(
        {
            "tileops": op,
            "torch-ref": workload.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
        },
        *inputs,
    )
