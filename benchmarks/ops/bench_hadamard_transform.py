"""Benchmark the TileOPs Hadamard transform, one case per manifest call.

Against fast_hadamard_transform (Dao-AILab), its 20N kernel on a base order of 20, and
HadaCore, a tensor-core power-of-two transform, which installs as faster_hadamard_transform.
"""

import math

import pytest

from benchmarks import api as bench
from benchmarks.api import Implementation
from benchmarks.baselines import baseline_module
from tileops.transform import HadamardTransformFwdOp

FAST_HADAMARD_TAG = "fast-hadamard-transform"
HADACORE_TAG = "hadacore"


@pytest.mark.parametrize("case", bench.cases(HadamardTransformFwdOp), ids=lambda case: case.id)
def test_hadamard_transform_bench(case) -> None:
    op = HadamardTransformFwdOp(**case.arguments)
    base_order = case.arguments.get("base_order", 1)
    n = case.params["x"][0][-1]
    scale = 1.0 / math.sqrt(n)
    dao = baseline_module(
        "fast_hadamard_transform.fast_hadamard_transform_interface", FAST_HADAMARD_TAG
    )
    if base_order == 1:
        hadacore = baseline_module("faster_hadamard_transform", HADACORE_TAG)
        comparators = {
            FAST_HADAMARD_TAG: lambda x: dao.hadamard_transform(x, scale),
            HADACORE_TAG: Implementation(
                run=hadacore.hadamard_transform,
                noncomparable_reason=(
                    "rounds each order-16 tensor-core stage to the 16-bit input dtype; the "
                    "transform rounds once after float32 accumulation"
                ),
            ),
        }
    else:
        # HadaCore serves powers of two only.
        transform = getattr(dao, f"hadamard_transform_{base_order}N")
        comparators = {FAST_HADAMARD_TAG: lambda x: transform(x, scale)}
    bench.Runner(op, case).compare(comparators)
